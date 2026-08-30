"""MTP speculative predictor for Qwen3.8-Flash-Next (checkpoint ``mtp.*`` module).

One extra decoder layer plus small fuse projections. The predictor consumes the
target's pre-mixer hidden state ``R [T, hc_count*hidden]`` and the embedding of the
token the target just accepted, and proposes the following token. Its routed experts
live in a private :class:`OffloadMoeCache` (``FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE``
slots, optionally block-FP8 via ``FREETOKEN_QWEN4_MTP_EXPERT_QUANT``), budgeted
before the main MoE/KV pools. The predictor layer's KV rides the shared QSA pool at
layer id ``num_layers`` (config.py appends it to the full-attention group).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList

from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .model import Qwen4ExpDecoderLayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def _mtp_moe_cache_size() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE", "64").strip()
    try:
        size = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE must be a positive integer"
        ) from error
    if size <= 0:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE must be a positive integer"
        )
    return size


def _mtp_expert_quant() -> str:
    value = os.getenv("FREETOKEN_QWEN4_MTP_EXPERT_QUANT", "bf16").strip().lower()
    if value not in ("bf16", "fp8_block"):
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_EXPERT_QUANT must be 'bf16' or 'fp8_block'"
        )
    return value


def _mtp_max_drafts() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "1").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1, 2, or 3"
        ) from error
    if value not in (1, 2, 3):
        raise ValueError("FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1, 2, or 3")
    return value


class MTPModule(BaseOP):
    """Weights and forward of the ``mtp.*`` predictor stack (state-dict prefix ``mtp.``)."""

    def __init__(self, config: ModelConfig) -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(
            config.hidden_size, config.rms_norm_eps, 1
        )
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.hc_count * config.hidden_size, config.rms_norm_eps, args.hc_count
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        first_layer_id = config.num_layers
        layers = [
            Qwen4ExpDecoderLayer(config, first_layer_id + index)
            for index in range(args.mtp_num_hidden_layers)
        ]
        # The predictor expert cache is its own OffloadMoeCache with num_layers ==
        # mtp_num_hidden_layers, so its layers index that cache from zero.
        for index, layer in enumerate(layers):
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None and hasattr(experts, "layer_id"):
                experts.layer_id = index
        self.layers = OPList(layers)
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def fuse_inputs(
        self, expanded_hidden: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if expanded_hidden.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                "Qwen4-Exp MTP hidden width does not match the hyper-connection stream"
            )
        embedded = self.fc_embedding.forward(
            self.pre_fc_norm_embedding.forward(token_embeddings)
        )
        hidden = self.pre_fc_norm_hidden.forward(expanded_hidden)
        hidden = hidden.view(-1, self.hc_count, self.hidden_size)
        hidden = self.fc_hidden.forward(hidden)
        return (hidden + embedded.unsqueeze(1)).flatten(1)

    def forward(
        self,
        expanded_hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.fuse_inputs(expanded_hidden, token_embeddings)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        return self.hyper_connection_mixer.mix(hidden)[0], hidden


def mtp_runtime_memory_bytes(config) -> int:
    """Pinned/GPU bytes the MTP runtime needs, reserved before cache auto-sizing."""
    model_config = config.model_config
    if not getattr(model_config.qwen4_args, "mtp_enabled", False):
        return 0
    from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    expert_bytes = _BANK_BYTES_PER_EXPERT[_mtp_expert_quant()](
        model_config.hidden_size,
        model_config.moe_intermediate_size,
    )
    linear_group = model_config.linear_attention_group()
    checkpoint_bytes = 0
    if linear_group is not None:
        # One accepted-prefix checkpoint per draft: GDN conv+recurrent plus the
        # declared slot_states (PLE conv and the n-gram context window).
        checkpoint_bytes = _mtp_max_drafts() * linear_state_bytes_per_req(
            linear_group,
            config.tp_info.size,
            config.dtype,
            getattr(model_config, "slot_states", ()),
        )
    return _mtp_moe_cache_size() * expert_bytes + checkpoint_bytes + (32 << 20)


def mtp_pinned_memory_bytes(config) -> int:
    """Pinned host bytes of the predictor expert bank (one layer, all experts)."""
    model_config = config.model_config
    if not getattr(model_config.qwen4_args, "mtp_enabled", False):
        return 0
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    return model_config.num_experts * _BANK_BYTES_PER_EXPERT[_mtp_expert_quant()](
        model_config.hidden_size,
        model_config.moe_intermediate_size,
    )


def setup_mtp_runtime(model, config, device: torch.device):
    """Load the predictor expert banks and attach their private offload cache."""
    if model.mtp is None:
        return None
    from freetoken.moe import is_offload_moe_backend

    if not is_offload_moe_backend(config.moe_backend):
        raise NotImplementedError(
            "Qwen4-Exp MTP currently requires an offload-family MoE backend"
        )
    if config.tp_info.size != 1:
        raise NotImplementedError(
            "Qwen4-Exp MTP currently requires tensor parallel size 1"
        )
    if getattr(config.model_config, "lm_head_quant", "none") != "none":
        # project_lm_head_all_positions reads lm_head.weight as a dense matrix
        raise NotImplementedError("Qwen4-Exp MTP requires an unquantized lm_head")
    from freetoken.moe.offload_cache import OffloadMoeCache

    from .weight import load_mtp_expert_banks

    banks = load_mtp_expert_banks(
        config.model_path,
        config.model_config,
        dummy=config.use_dummy_weight,
        expert_quant=_mtp_expert_quant(),
        device=device,
    )
    cache = OffloadMoeCache(
        num_layers=config.model_config.qwen4_args.mtp_num_hidden_layers,
        num_experts=config.model_config.num_experts,
        cache_size=_mtp_moe_cache_size(),
        device=device,
        cache_policy=config.moe_cache_policy,
        prefill_overlap=False,
        quant_format=banks.quant_format,
        decode_target="gpu",
        allow_partial_cache=True,
    )
    cache.collect_stats = config.moe_collect_stats
    cache.collect_decode_freq = config.moe_collect_stats
    cache.set_bank_sources(banks.sources)
    for layer in model.mtp.layers.op_list:
        layer.mlp.experts.offload_cache = cache
    model.mtp_offload_cache = cache
    return cache


__all__ = [
    "MTPModule",
    "mtp_pinned_memory_bytes",
    "mtp_runtime_memory_bytes",
    "setup_mtp_runtime",
]
