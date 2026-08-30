"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        attn_quant=config.attn_quant,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)
        # MTP accepted-prefix checkpoints of the shared ple_ngram_ctx window
        self._mtp_prefix_ngram_ctx: torch.Tensor | None = None

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(
        self, input_ids: torch.Tensor, batch: Batch, *, return_expanded: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            if batch.mtp_verify:
                self._commit_mtp_ngram_context(meta, batch.mtp_checkpoint_capacity)
            else:
                commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        mixed = self.hyper_connection_mixer.mix(hidden)[0]
        return (mixed, hidden) if return_expanded else mixed

    def _commit_mtp_ngram_context(self, meta, checkpoint_capacity: int) -> None:
        """Verify-forward sibling of ``commit_ngram_context``: roll the window past all
        rows, and save the window after each non-final row so a rejected suffix can
        restore the accepted prefix's context."""
        from .ple import _ngram_context_pool

        context_pool = _ngram_context_pool()
        ids = meta.input_ids.to(meta.ngram_context.dtype)
        total = ids.numel()
        full = torch.cat([meta.ngram_context[0], ids])
        ctx_len = meta.ngram_context.shape[1]
        expected = (checkpoint_capacity, ctx_len)
        if self._mtp_prefix_ngram_ctx is None:
            self._mtp_prefix_ngram_ctx = torch.empty(
                expected, dtype=full.dtype, device=full.device
            )
        elif self._mtp_prefix_ngram_ctx.shape != expected:
            # A captured verify graph holds this buffer's address; a silent
            # reallocation would detach it from the graph.
            raise RuntimeError("MTP ngram checkpoint geometry changed after capture")
        for row in range(total - 1):
            self._mtp_prefix_ngram_ctx[row].copy_(full[row + 1 : row + 1 + ctx_len])
        context_pool.index_copy_(
            0, meta.state_slots, full[total : total + ctx_len].unsqueeze(0).to(context_pool.dtype)
        )

    def commit_mtp_prefix_state(self, pool, slot: int, checkpoint_index: int) -> None:
        for layer in self.layers.op_list:
            mixer = getattr(layer, "linear_attn", None)
            if mixer is not None:
                mixer.commit_mtp_prefix_state(pool, slot, checkpoint_index)
            if layer.ple is not None:
                layer.ple.commit_mtp_prefix_state(slot, checkpoint_index)
        if self._ple:
            if self._mtp_prefix_ngram_ctx is None:
                raise RuntimeError("Qwen4-Exp MTP ngram context is not initialized")
            from .ple import _ngram_context_pool

            _ngram_context_pool()[slot].copy_(
                self._mtp_prefix_ngram_ctx[checkpoint_index]
            )


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        if config.qwen4_args.mtp_enabled:
            from .mtp import MTPModule

            self.mtp = MTPModule(config)
        else:
            self.mtp = None
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def _iter_offload_moe_layers(self):
        # The MTP predictor's MoE layer owns a private cache (setup_auxiliary_runtime);
        # keep it out of the main offload-cache attach walk.
        for layer in self.model.layers.op_list:
            yield layer.mlp.experts

    def auxiliary_runtime_memory_bytes(self, config) -> int:
        if self.mtp is None:
            return 0
        from .mtp import mtp_runtime_memory_bytes

        return mtp_runtime_memory_bytes(config)

    def auxiliary_pinned_memory_bytes(self, config) -> int:
        if self.mtp is None:
            return 0
        from .mtp import mtp_pinned_memory_bytes

        return mtp_pinned_memory_bytes(config)

    def setup_auxiliary_runtime(self, config, device: torch.device):
        from .mtp import setup_mtp_runtime

        return setup_mtp_runtime(self, config, device)

    def forward(
        self, *, return_expanded: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = get_global_ctx().batch
        output = self.model.forward(
            batch.input_ids, batch, return_expanded=return_expanded
        )
        if return_expanded:
            mixed, expanded = output
            return self.lm_head.forward(mixed), expanded
        return self.lm_head.forward(output)

    def commit_mtp_prefix_state(self, req, linear_pool, checkpoint_index: int) -> None:
        slot = (
            req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        )
        self.model.commit_mtp_prefix_state(linear_pool, slot, checkpoint_index)

    def mtp_forward(
        self, expanded_hidden: torch.Tensor, next_token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mtp is None:
            raise RuntimeError("Qwen4-Exp MTP is disabled")
        batch = get_global_ctx().batch
        token_embeddings = self.model.embed_tokens.forward(next_token_ids)
        mixed, expanded = self.mtp.forward(expanded_hidden, token_embeddings, batch)
        return self.lm_head.forward(mixed), expanded


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
