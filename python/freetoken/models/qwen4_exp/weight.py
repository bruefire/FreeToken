"""Qwen3.8-Flash-Next (RadixArk NVFP4) checkpoint reader.

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, into the offload cache's source banks.

Dropped: ``model.visual.*`` (served text-only), and ``mtp.*`` (speculative head) unless
FREETOKEN_QWEN4_MTP=1; the stacked ``mtp.layers.0.mlp.experts.*`` always load separately
via :func:`load_mtp_expert_banks`.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import download_hf_weight, init_logger
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

from .config import _mtp_enabled

logger = init_logger(__name__)

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        if not _mtp_enabled():
            return None
        if ".mlp.experts." in raw_name:
            return None  # stacked predictor experts: load_mtp_expert_banks
        return raw_name
    if raw_name.startswith(("model.visual.", "visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: the modelopt
    ``ignore`` list covers everything except those experts, so attention, GDN, HC, PLE, the shared
    expert and lm_head are all plain bf16 (the n-gram hash constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from :func:`load_nvfp4_expert_sources`.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
    shard_bytes = rows * cols
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
) -> dict:
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


# ======================================================================================
# MTP predictor experts
# ======================================================================================


def _quantize_fp8_block(
    source: torch.Tensor,
    weight_out: torch.Tensor,
    scale_out: torch.Tensor,
    *,
    device: torch.device,
    block: int = 128,
    expert_chunk: int = 8,
) -> None:
    if source.ndim != 3 or weight_out.shape != source.shape:
        raise ValueError("MTP FP8 expert banks must have matching [experts, out, in] shapes")
    experts, out_features, in_features = source.shape
    if out_features % block or in_features % block:
        raise ValueError("MTP FP8 expert dimensions must be divisible by the block size")
    expected_scales = (experts, out_features // block, in_features // block)
    if scale_out.shape != expected_scales:
        raise ValueError(
            f"MTP FP8 scale bank has shape {tuple(scale_out.shape)}, expected {expected_scales}"
        )
    if weight_out.dtype != torch.float8_e4m3fn or scale_out.dtype != torch.bfloat16:
        raise ValueError("MTP FP8 banks require float8_e4m3fn weights and BF16 scales")
    if expert_chunk <= 0:
        raise ValueError("MTP FP8 expert chunk must be positive")

    for start in range(0, experts, expert_chunk):
        end = min(start + expert_chunk, experts)
        count = end - start
        weight = source[start:end].to(device=device, dtype=torch.float32)
        blocks = weight.view(
            count,
            out_features // block,
            block,
            in_features // block,
            block,
        ).permute(0, 1, 3, 2, 4)
        scale = blocks.abs().amax(dim=(3, 4)).clamp_min(1e-10) / 448.0
        scale = scale.to(torch.bfloat16)
        quantized = (blocks / scale.float()[..., None, None]).clamp(-448.0, 448.0)
        quantized = quantized.to(torch.float8_e4m3fn).permute(0, 1, 3, 2, 4).reshape(
            count, out_features, in_features
        )
        weight_out[start:end].copy_(quantized)
        scale_out[start:end].copy_(scale)


def _mtp_expert_bank_specs(
    experts: int,
    hidden: int,
    intermediate: int,
    expert_quant: str,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    if expert_quant == "bf16":
        return {
            "gate_up": ((experts, 2 * intermediate, hidden), torch.bfloat16),
            "down": ((experts, hidden, intermediate), torch.bfloat16),
        }
    if expert_quant != "fp8_block":
        raise ValueError(f"Unsupported Qwen4-Exp MTP expert quantization {expert_quant!r}")
    from freetoken.moe.offload_cache import fp8_block_scale_pad

    block = 128
    if hidden % block or intermediate % block:
        raise ValueError("Qwen4-Exp MTP FP8 expert dimensions must be divisible by 128")
    # Scale banks keep the offload cache's padded trailing dim (fp8_block_scale_pad);
    # the GEMMs read the real columns through explicit strides.
    return {
        "gate_up": ((experts, 2 * intermediate, hidden), torch.float8_e4m3fn),
        "gate_up_scale": (
            (
                experts,
                2 * intermediate // block,
                fp8_block_scale_pad(2 * intermediate // block, hidden // block),
            ),
            torch.bfloat16,
        ),
        "down": ((experts, hidden, intermediate), torch.float8_e4m3fn),
        "down_scale": (
            (
                experts,
                hidden // block,
                fp8_block_scale_pad(hidden // block, intermediate // block),
            ),
            torch.bfloat16,
        ),
    }


def load_mtp_expert_banks(
    model_path: str,
    config,
    *,
    dummy: bool = False,
    expert_quant: str = "bf16",
    device: torch.device | None = None,
):
    """Read the stacked ``mtp.layers.0.mlp.experts.*`` tensors into pinned host banks.

    ``expert_quant='fp8_block'`` converts the BF16 checkpoint experts to 128x128
    block-FP8 while loading (halves the pinned bank and the predictor cache slot).
    """
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import PinPipeline, alloc_layer_banks

    if not config.qwen4_args.mtp_enabled:
        raise ValueError("Qwen4-Exp MTP expert loading requested while MTP is disabled")
    if config.qwen4_args.mtp_num_hidden_layers != 1:
        raise NotImplementedError("Qwen4-Exp currently supports one MTP layer")

    experts = config.num_experts
    hidden = config.hidden_size
    intermediate = config.moe_intermediate_size
    block = 128
    specs = _mtp_expert_bank_specs(experts, hidden, intermediate, expert_quant)
    host_banks = alloc_layer_banks(specs, 1)
    if dummy:
        for name, per_layer in host_banks.items():
            per_layer[0].tensor.fill_(1 if name.endswith("_scale") else 0)
        with PinPipeline() as pins:
            pins(0, {name: per_layer[0] for name, per_layer in host_banks.items()})
    else:
        folder = download_hf_weight(model_path)
        with open(
            os.path.join(folder, "model.safetensors.index.json"),
            encoding="utf-8",
        ) as index_file:
            weight_map = json.load(index_file)["weight_map"]
        keys = {
            "gate_up": "mtp.layers.0.mlp.experts.gate_up_proj",
            "down": "mtp.layers.0.mlp.experts.down_proj",
        }
        paths = {
            name: os.path.join(folder, weight_map[key]) for name, key in keys.items()
        }
        with PinPipeline() as pins:
            for bank_name, key in keys.items():
                path = paths[bank_name]
                with safetensors.safe_open(
                    path, framework="pt", device="cpu"
                ) as handle:
                    tensor = handle.get_tensor(key)
                    destination = host_banks[bank_name][0].tensor
                    if tensor.dtype != torch.bfloat16 or tensor.shape != destination.shape:
                        raise RuntimeError(
                            f"Unexpected Qwen4-Exp MTP expert tensor {key}: "
                            f"{tensor.dtype} {tuple(tensor.shape)}"
                        )
                    if expert_quant == "bf16":
                        destination.copy_(tensor)
                    else:
                        quant_device = device or torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        scale_bank = host_banks[f"{bank_name}_scale"][0].tensor
                        real_cols = destination.shape[2] // block
                        _quantize_fp8_block(
                            tensor,
                            destination,
                            scale_bank[:, :, :real_cols],
                            device=quant_device,
                        )
                pins.submit(host_banks[bank_name][0])
                if expert_quant == "fp8_block":
                    pins.submit(host_banks[f"{bank_name}_scale"][0])
        for path in set(paths.values()):
            drop_page_cache(path)

    return ExpertBanks(
        expert_quant,
        {
            name: [per_layer[0].tensor]
            for name, per_layer in host_banks.items()
        },
    )


__all__ = [
    "PleTable",
    "iter_weights",
    "load_mtp_expert_banks",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
]
