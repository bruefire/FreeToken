"""Single-request MTP speculative decode for Qwen3.8-Flash-Next.

One scheduler cycle: the MTP predictor has already proposed draft token(s) from the
target's previous expanded hidden state; the target verifies ``[current, draft...]``
in one multi-row forward; the longest accepted draft prefix plus the target's
replacement/bonus token is emitted; the accepted-prefix GDN/PLE/ngram checkpoints
commit the recurrent state; the predictor then re-runs over the emitted rows to
propose the next draft.

Current restrictions: one running request, greedy sampling, TP=1, offload-family
MoE backend, naive cache. The verify and predictor forwards replay fixed-row CUDA
graphs (_TargetVerifyGraph / _PredictorGraph below); the eager paths remain as
the fallback when graph capture is unavailable and for chunked prefill.
"""

from __future__ import annotations

import contextlib
import os
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata
from freetoken.core import Batch, Req
from freetoken.utils import init_logger

from .graph import project_lm_head_all_positions

if TYPE_CHECKING:
    from .engine import Engine
    from .sample import BatchSamplingArgs


logger = init_logger(__name__)


@dataclass
class MTPMetrics:
    cycles: int = 0
    proposed_drafts: int = 0
    accepted_drafts: int = 0
    emitted_tokens: int = 0
    cycle_trace: list[dict[str, int]] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float | None:
        if not self.proposed_drafts:
            return None
        return self.accepted_drafts / self.proposed_drafts


def _mtp_max_drafts() -> int:
    # Capped at 3: a 4-row verify writes 4 consecutive index-ring rows, exactly
    # the ring capacity (= index_ratio); a deeper chain would alias the ring.
    raw = os.getenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "1").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1, 2, or 3") from error
    if value not in (1, 2, 3):
        raise ValueError("FREETOKEN_QWEN4_MTP_MAX_DRAFTS must be 1, 2, or 3")
    return value


def _mtp_adaptive_enabled() -> bool:
    value = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE", "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("FREETOKEN_QWEN4_MTP_ADAPTIVE must be a boolean")


def _mtp_adaptive_cycles() -> int:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES", "64").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES must be positive"
        ) from error
    if value <= 0:
        raise ValueError("FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES must be positive")
    return value


def _mtp_adaptive_min_yield() -> float:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE", "0.75").strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE must be non-negative"
        ) from error
    if value < 0.0:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE must be non-negative"
        )
    return value


def _mtp_draft_p_min() -> float:
    raw = os.getenv("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN", "0").strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "FREETOKEN_QWEN4_MTP_DRAFT_P_MIN must be between 0 and 1"
        ) from error
    if not 0.0 <= value <= 1.0:
        raise ValueError("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN must be between 0 and 1")
    return value


@dataclass
class _TimingEvent:
    phase: str
    started: torch.cuda.Event
    ended: torch.cuda.Event


@dataclass
class _MutableForwardState:
    req_input_ids: torch.Tensor
    req_cached_len: int
    req_device_len: int
    batch_phase: str
    batch_input_ids: torch.Tensor
    batch_positions: torch.Tensor
    batch_out_loc: torch.Tensor | None
    batch_padded_reqs: list[Req]
    batch_fla_metadata: Any
    batch_attn_metadata: Any
    batch_moe_decode_cache: bool


@dataclass
class _TargetRuntimeState:
    linear_slot: int | None
    conv_state: torch.Tensor | None
    recurrent_state: torch.Tensor | None
    slot_states: dict[str, torch.Tensor] | None


def _snapshot_mutable_forward_state(batch: Batch, req: Req) -> _MutableForwardState:
    return _MutableForwardState(
        req_input_ids=req.input_ids,
        req_cached_len=req.cached_len,
        req_device_len=req.device_len,
        batch_phase=batch.phase,
        batch_input_ids=batch.input_ids,
        batch_positions=batch.positions,
        batch_out_loc=batch.out_loc,
        batch_padded_reqs=batch.padded_reqs,
        batch_fla_metadata=batch.fla_metadata,
        batch_attn_metadata=getattr(batch, "attn_metadata", None),
        batch_moe_decode_cache=batch.moe_decode_cache,
    )


def _restore_mutable_forward_state(
    batch: Batch, req: Req, state: _MutableForwardState
) -> None:
    req.input_ids = state.req_input_ids
    req.cached_len = state.req_cached_len
    req.device_len = state.req_device_len
    batch.phase = state.batch_phase
    batch.input_ids = state.batch_input_ids
    batch.positions = state.batch_positions
    batch.out_loc = state.batch_out_loc
    batch.padded_reqs = state.batch_padded_reqs
    batch.fla_metadata = state.batch_fla_metadata
    batch.attn_metadata = state.batch_attn_metadata
    batch.moe_decode_cache = state.batch_moe_decode_cache


def _linear_slot(req: Req) -> int:
    return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx


class _FixedRowGraphBase:
    """Shared staging of a captured fixed-row single-request QSA forward.

    All position-dependent values flow through persistent device buffers that
    replay() refreshes: the captured kernels recompute the per-token slab/ring
    plan (cmp_rows / ring_rows) from them, mirroring the restage-per-replay
    pattern of the ordinary decode graph.
    """

    def __init__(self, engine: Engine, token_count: int, uid: int):
        if not isinstance(engine.attn_backend, QSASparseAttnBackend):
            raise NotImplementedError(
                "Qwen4-Exp MTP graphs require the QSA sparse backend"
            )
        if engine.graph_runner.max_graph_bs < 1:
            raise NotImplementedError("Qwen4-Exp MTP graphs require CUDA graphs")
        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        self.token_count = token_count
        self.page_size = engine.config.page_size

        self.input_ids = torch.zeros(token_count, dtype=torch.int32, device=self.device)
        self.positions = torch.zeros_like(self.input_ids)
        # Capture on the persistent dummy row so KV/slab writes hit scratch.
        self.out_loc = engine.page_table[
            engine.dummy_req.table_idx, :token_count
        ].clone()
        self.linear_table_idx = torch.full(
            (1,),
            engine.linear_state_pool.padding_slot
            if engine.linear_state_pool is not None
            else engine.dummy_req.table_idx,
            dtype=torch.int32,
            device=self.device,
        )
        self.ring_slots = torch.full(
            (1,), engine.dummy_req.table_idx, dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.full(
            (1,), token_count, dtype=torch.int32, device=self.device
        )
        self.block_table = torch.zeros(
            (1, engine.page_table[:, :: self.page_size].shape[1]),
            dtype=torch.int32,
            device=self.device,
        )
        vocab_size = engine.config.model_config.vocab_size
        hidden_width = self.model.mtp.hc_count * self.model.mtp.hidden_size
        hidden_dtype = self.model.model.embed_tokens.weight.dtype
        self.logits = torch.empty(
            token_count, vocab_size, dtype=torch.float32, device=self.device
        )
        self.expanded = torch.empty(
            token_count, hidden_width, dtype=hidden_dtype, device=self.device
        )

        dummy_req = Req(
            input_ids=torch.zeros(token_count, dtype=torch.int32),
            table_idx=engine.dummy_req.table_idx,
            cached_len=0,
            output_len=1,
            uid=uid,
            sampling_params=engine.dummy_req.sampling_params,
            cache_handle=engine.dummy_req.cache_handle,
        )
        dummy_req.linear_slot_idx = (
            engine.linear_state_pool.padding_slot
            if engine.linear_state_pool is not None
            else None
        )
        self.batch = Batch(reqs=[dummy_req], phase="prefill")
        self.batch.padded_reqs = self.batch.reqs
        self.batch.input_ids = self.input_ids
        self.batch.positions = self.positions
        self.batch.out_loc = self.out_loc
        self.batch.linear_table_idx = self.linear_table_idx
        self.batch.moe_decode_cache = True
        self.batch.attn_metadata = self._attn_metadata()
        self.graph = torch.cuda.CUDAGraph()

    def _attn_metadata(self) -> QSASparseMetadata:
        return QSASparseMetadata(
            is_decode=False,
            last_indices=torch.tensor(
                [self.token_count - 1], dtype=torch.int32, device=self.device
            ),
            qo_indptr_cpu=torch.tensor(
                [0, self.token_count], dtype=torch.int32, pin_memory=True
            ),
            kv_len_cpu=torch.tensor(
                [self.token_count], dtype=torch.int32, pin_memory=True
            ),
            token_to_req=torch.zeros(
                self.token_count, dtype=torch.int32, device=self.device
            ),
            cu_seqlens=torch.tensor(
                [0, self.token_count], dtype=torch.int32, device=self.device
            ),
            seq_lens=self.seq_lens,
            ring_slots=self.ring_slots,
            block_table=self.block_table,
        )

    def _stage(self, req: Req, start: int) -> None:
        end = start + self.token_count
        page_table = self.engine.page_table
        torch.arange(
            start, end, dtype=torch.int32, device=self.device, out=self.positions
        )
        self.out_loc.copy_(page_table[req.table_idx, start:end])
        self.linear_table_idx.fill_(_linear_slot(req))
        self.ring_slots.fill_(req.table_idx)
        self.seq_lens.fill_(end)
        self.block_table[0].copy_(
            (page_table[req.table_idx, :: self.page_size] // self.page_size).to(
                torch.int32
            )
        )
        self.batch.reqs = [req]
        self.batch.padded_reqs = self.batch.reqs


class _TargetVerifyGraph(_FixedRowGraphBase):
    """Captured fixed-row target extension for one Qwen4-Exp request."""

    def __init__(
        self,
        engine: Engine,
        token_count: int,
        checkpoint_capacity: int,
        graph_pool=None,
    ):
        if token_count not in (2, 3, 4):
            raise ValueError("MTP target graph supports two to four tokens")
        if checkpoint_capacity < token_count - 1:
            raise ValueError("MTP target checkpoint capacity is too small")
        super().__init__(engine, token_count, uid=-2)
        self.graph_pool = graph_pool
        self.batch.mtp_verify = True
        self.batch.mtp_checkpoint_capacity = checkpoint_capacity
        from freetoken.attention.linear import FLAMetadata

        self.batch.fla_metadata = FLAMetadata(
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            cache_indices=self.linear_table_idx,
            has_initial_state=torch.ones(1, dtype=torch.bool, device=self.device),
        )
        self._capture()

    def _forward(self) -> None:
        hidden, expanded = self.model.model.forward(
            self.batch.input_ids, self.batch, return_expanded=True
        )
        self.logits.copy_(project_lm_head_all_positions(self.model.lm_head, hidden))
        self.expanded.copy_(expanded)

    def _capture(self) -> None:
        cache = self.engine.moe_offload_cache
        if cache is not None:
            cache.reset()
        with self.engine.ctx.forward_batch(self.batch):
            self._forward()
            # A cached warmup plan would bake the warmup's slab/ring addresses
            # into the graph; force the capture pass to re-plan.
            self.batch.attn_metadata.cmp_rows = None
            with torch.cuda.graph(
                self.graph, pool=self.graph_pool, stream=self.engine.stream
            ):
                self._forward()
        if cache is not None:
            cache.reset()

    def replay(
        self, req: Req, token_ids: torch.Tensor, start: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.numel() != self.token_count:
            raise ValueError(
                f"MTP target graph requires exactly {self.token_count} tokens"
            )
        self.input_ids.copy_(token_ids)
        self._stage(req, start)
        with self.engine.ctx.forward_batch(self.batch):
            self.graph.replay()
        return self.logits, self.expanded

    def commit_prefix(self, req: Req, checkpoint_index: int) -> None:
        self.model.commit_mtp_prefix_state(
            req, self.engine.linear_state_pool, checkpoint_index
        )


class _PredictorGraph(_FixedRowGraphBase):
    """Captured one- to three-token MTP predictor extension."""

    def __init__(self, engine: Engine, token_count: int, graph_pool=None):
        if token_count not in (1, 2, 3, 4):
            raise ValueError("MTP predictor graph supports one to four tokens")
        super().__init__(engine, token_count, uid=-3 - token_count)
        self.graph_pool = graph_pool
        hidden_width = self.model.mtp.hc_count * self.model.mtp.hidden_size
        hidden_dtype = self.model.model.embed_tokens.weight.dtype
        self.hidden = torch.zeros(
            token_count, hidden_width, dtype=hidden_dtype, device=self.device
        )
        self._capture()

    def _forward(self) -> None:
        logits, expanded = self.model.mtp_forward(self.hidden, self.input_ids)
        self.logits.copy_(logits)
        self.expanded.copy_(expanded)

    def _capture(self) -> None:
        cache = self.model.mtp_offload_cache
        cache.reset()
        with self.engine.ctx.forward_batch(self.batch):
            self._forward()
            # The predictor layer is not slot 0 of the QSA group, so qsa_forward
            # would reuse the warmup's plan; force the capture pass to re-plan.
            self.batch.attn_metadata.cmp_rows = None
            with torch.cuda.graph(
                self.graph, pool=self.graph_pool, stream=self.engine.stream
            ):
                self._forward()
        cache.reset()

    def replay(
        self,
        req: Req,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[0] != self.token_count or token_ids.numel() != self.token_count:
            raise ValueError(
                f"MTP predictor graph requires exactly {self.token_count} tokens"
            )
        self.hidden.copy_(hidden)
        self.input_ids.copy_(token_ids)
        self._stage(req, start)
        with self.engine.ctx.forward_batch(self.batch):
            self.graph.replay()
        return self.logits, self.expanded


def _snapshot_target_runtime(engine: Engine, req: Req) -> _TargetRuntimeState:
    """Whole-request recurrent-state snapshot: GDN conv + recurrent plus every
    declared slot state (PLE conv history and the ngram context window)."""
    pool = engine.linear_state_pool
    if pool is None:
        return _TargetRuntimeState(None, None, None, None)
    slot = _linear_slot(req)
    return _TargetRuntimeState(
        linear_slot=slot,
        conv_state=pool.conv_states[:, slot].clone(),
        recurrent_state=pool.recurrent_states[:, slot].clone(),
        slot_states={
            name: tensor[:, slot].clone() for name, tensor in pool.slot_states.items()
        },
    )


def _restore_target_runtime(
    engine: Engine, req: Req, state: _TargetRuntimeState
) -> None:
    if state.linear_slot is None:
        return
    pool = engine.linear_state_pool
    pool.conv_states[:, state.linear_slot].copy_(state.conv_state)
    pool.recurrent_states[:, state.linear_slot].copy_(state.recurrent_state)
    for name, saved in state.slot_states.items():
        pool.slot_states[name][:, state.linear_slot].copy_(saved)


class MTPWorker:
    """Single-request Qwen4-Exp MTP predictor and greedy verifier."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.model = engine.model
        cache = getattr(self.model, "mtp_offload_cache", None)
        if cache is None:
            raise RuntimeError("Qwen4-Exp MTP expert cache is not initialized")
        top_k = engine.config.model_config.num_experts_per_tok
        self.max_predictor_chunk = cache.cache_size // top_k
        if self.max_predictor_chunk < 1:
            raise ValueError(
                f"MTP expert cache has {cache.cache_size} slots but top_k={top_k}"
            )
        self.uid: int | None = None
        self.predictor_cached_len = 0
        self.pending_hidden: torch.Tensor | None = None
        self.pending_draft: torch.Tensor | None = None
        self.pending_predictor_hidden: torch.Tensor | None = None
        self.pending_draft_confidence = 1.0
        self.max_supported_drafts = _mtp_max_drafts()
        self.max_drafts = self.max_supported_drafts
        self.draft_p_min = _mtp_draft_p_min()
        # Request-local fallback: after an observation period, a rolling accepted
        # yield below the threshold reverts the request to ordinary decode. The
        # switch is one-way because ordinary decode does not advance the
        # recursive predictor state.
        self.fallback_enabled = _mtp_adaptive_enabled()
        self.fallback_cycles = _mtp_adaptive_cycles()
        self.fallback_min_yield = _mtp_adaptive_min_yield()
        self.fallback_uid: int | None = None
        self._accept_history: deque[int] = deque(maxlen=self.fallback_cycles)
        self._request_cycles = 0
        self.metrics = MTPMetrics()
        self.log_interval = max(
            0, int(os.getenv("FREETOKEN_QWEN4_MTP_LOG_INTERVAL", "40"))
        )
        self.timing_enabled = os.getenv(
            "FREETOKEN_QWEN4_MTP_TIMING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.timing_events: list[_TimingEvent] = []
        self.target_verify_graphs: dict[int, _TargetVerifyGraph] = {}
        try:
            graph_pool = None
            for token_count in range(2, self.max_supported_drafts + 2):
                graph = _TargetVerifyGraph(
                    engine,
                    token_count,
                    self.max_supported_drafts,
                    graph_pool=graph_pool,
                )
                self.target_verify_graphs[token_count] = graph
                graph_pool = graph.graph.pool()
        except NotImplementedError as error:
            logger.warning_rank0(f"Qwen4-Exp MTP target graph disabled: {error}")
        self.predictor_graphs: dict[int, _PredictorGraph] = {}
        if self.target_verify_graphs:
            for token_count in range(1, self.max_supported_drafts + 2):
                graph = _PredictorGraph(engine, token_count, graph_pool=graph_pool)
                self.predictor_graphs[token_count] = graph
                graph_pool = graph.graph.pool()

    def reset(self, uid: int | None = None) -> None:
        self.uid = uid
        self.predictor_cached_len = 0
        self.pending_hidden = None
        self.pending_draft = None
        self.pending_predictor_hidden = None
        self.pending_draft_confidence = 1.0
        self.fallback_uid = None
        self._accept_history.clear()
        self._request_cycles = 0

    def reset_metrics(self) -> None:
        self.metrics = MTPMetrics()
        self.timing_events.clear()

    def timing(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[float]] = {}
        for event in self.timing_events:
            grouped.setdefault(event.phase, []).append(
                event.started.elapsed_time(event.ended)
            )
        return {
            phase: {
                "calls": len(milliseconds),
                "total_ms": sum(milliseconds),
                "mean_ms": sum(milliseconds) / len(milliseconds),
            }
            for phase, milliseconds in grouped.items()
        }

    @contextlib.contextmanager
    def _profile_phase(self, phase: str):
        if not self.timing_enabled:
            yield
            return
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.engine.stream)
        try:
            yield
        finally:
            ended.record(self.engine.stream)
            self.timing_events.append(_TimingEvent(phase, started, ended))

    def _activate(self, uid: int) -> None:
        if self.uid != uid:
            self.reset(uid)

    def can_speculate(self, batch: Batch) -> bool:
        if not batch.is_decode or batch.size != 1:
            return False
        req = batch.reqs[0]
        return (
            req.sampling_params.is_greedy
            and req.remain_len >= 2
            and self.fallback_uid != req.uid
            and self.uid == req.uid
            and self.pending_hidden is None
            and self.pending_draft is not None
            and self.predictor_cached_len == req.cached_len
        )

    def ordinary_decode_selected(self, req: Req) -> bool:
        return self.fallback_uid == req.uid

    def update_prefill(
        self,
        batch: Batch,
        expanded_hidden: torch.Tensor,
        base_token: torch.Tensor,
        *,
        start: int,
        end: int,
        final: bool,
    ) -> None:
        req = batch.reqs[0]
        if start == 0:
            self.reset(req.uid)
        else:
            self._activate(req.uid)
        if expanded_hidden.shape[0] != end - start:
            raise ValueError(
                f"MTP target hidden rows {expanded_hidden.shape[0]} do not match "
                f"prefill range [{start}, {end})"
            )

        # The predictor pairs hidden row i with token i+1, so the last usable
        # source row of a non-final chunk is end-2; its own last row is carried
        # into the next chunk through pending_hidden.
        pair_end = end - 1 if final else end - 2
        source_start = self.predictor_cached_len
        hidden_parts = []
        if self.pending_hidden is not None:
            if source_start != start - 1:
                raise RuntimeError(
                    f"MTP pending source starts at {source_start}, expected {start - 1}"
                )
            hidden_parts.append(self.pending_hidden)
        elif source_start != start:
            raise RuntimeError(
                f"MTP predictor cache ends at {source_start}, target chunk starts at {start}"
            )

        current_first = max(source_start, start)
        if pair_end >= current_first:
            hidden_parts.append(
                expanded_hidden[current_first - start : pair_end - start + 1]
            )

        if pair_end >= source_start:
            hidden = torch.cat(hidden_parts, dim=0)
            token_start = source_start + 1
            prompt_token_end = min(pair_end + 2, end)
            token_ids = req.input_ids[token_start:prompt_token_end].to(
                device=self.engine.device, dtype=torch.int32
            )
            if final:
                token_ids = torch.cat([token_ids, base_token[:1].to(dtype=torch.int32)])
            expected = pair_end - source_start + 1
            if hidden.shape[0] != expected or token_ids.numel() != expected:
                raise RuntimeError(
                    f"MTP prefill alignment produced hidden={hidden.shape[0]}, "
                    f"tokens={token_ids.numel()}, expected={expected}"
                )
            logits, predictor_expanded = self._run_predictor(
                batch, req, hidden, token_ids, source_start
            )
            if final:
                (
                    self.pending_draft,
                    self.pending_draft_confidence,
                ) = self._draft_candidate(logits)
                if self.max_supported_drafts > 1:
                    self.pending_predictor_hidden = predictor_expanded[-1:].clone()

        self.pending_hidden = None if final else expanded_hidden[-1:].clone()
        expected_cached = end if final else max(0, end - 1)
        if self.predictor_cached_len != expected_cached:
            raise RuntimeError(
                f"MTP predictor cache ends at {self.predictor_cached_len}, "
                f"expected {expected_cached}"
            )

    def forward_decode(
        self, batch: Batch, _args: BatchSamplingArgs
    ) -> tuple[torch.Tensor, bool]:
        req = batch.reqs[0]
        self._activate(req.uid)
        source_position = req.cached_len
        if self.pending_hidden is not None:
            raise RuntimeError("MTP prompt initialization is incomplete")
        if self.pending_draft is None:
            raise RuntimeError("MTP draft is not initialized")
        if self.predictor_cached_len != source_position:
            raise RuntimeError(
                f"MTP predictor cache ends at {self.predictor_cached_len}, "
                f"target decode starts at {source_position}"
            )

        max_drafts = min(self.max_drafts, self.max_supported_drafts)
        max_drafts = min(max_drafts, max(req.remain_len - 1, 0))
        draft_parts = []
        predictor_speculated = False
        if max_drafts and self.pending_draft_confidence >= self.draft_p_min:
            draft_parts.append(self.pending_draft[:1])
            recursive_hidden = self.pending_predictor_hidden
            # Extend the chain recursively: each further draft embeds the previous
            # one on the predictor's own prior output hidden state. A failed
            # confidence gate shortens the chain instead of forcing a
            # low-confidence verification row.
            while len(draft_parts) < max_drafts:
                if recursive_hidden is None:
                    raise RuntimeError("MTP recursive draft hidden is not initialized")
                depth = len(draft_parts) + 1
                with self._profile_phase(f"PredictorDraft{depth}"):
                    next_logits, next_expanded = self._run_predictor(
                        batch,
                        req,
                        recursive_hidden,
                        draft_parts[-1][:1],
                        source_position + len(draft_parts) - 1,
                    )
                predictor_speculated = True
                next_draft, next_confidence = self._draft_candidate(next_logits)
                if next_confidence < self.draft_p_min:
                    break
                draft_parts.append(next_draft)
                recursive_hidden = next_expanded[-1:]

        draft_tokens = torch.cat(draft_parts) if draft_parts else batch.input_ids[:0]
        current_token = batch.input_ids[:1]
        verify_input = torch.cat([current_token, draft_tokens])
        target_graph = self.target_verify_graphs.get(verify_input.numel())
        target_state = None
        if draft_tokens.numel() and target_graph is None:
            with self._profile_phase("Snapshot"):
                target_state = _snapshot_target_runtime(self.engine, req)
        with self._profile_phase(f"TargetVerify{verify_input.numel()}"):
            verify_logits, verify_expanded = self._run_target_extension(
                batch, req, verify_input, source_position
            )
            verify_tokens = torch.argmax(verify_logits, dim=-1).to(torch.int32)
        with self._profile_phase("DecisionSync"):
            matches = (
                (verify_tokens[: draft_tokens.numel()] == draft_tokens)
                .to(device="cpu")
                .tolist()
            )
        accepted_count = 0
        for matches_draft in matches:
            if not matches_draft:
                break
            accepted_count += 1

        output = torch.cat(
            [
                draft_tokens[:accepted_count],
                verify_tokens[accepted_count : accepted_count + 1],
            ]
        )
        committed_expanded = verify_expanded[: output.numel()]
        if accepted_count < draft_tokens.numel():
            if target_graph is not None:
                with self._profile_phase("PrefixCommit"):
                    target_graph.commit_prefix(req, accepted_count)
            else:
                assert target_state is not None
                with self._profile_phase("Restore"):
                    _restore_target_runtime(self.engine, req, target_state)
                replay_count = accepted_count + 1
                with self._profile_phase(f"TargetReplay{replay_count}"):
                    _, committed_expanded = self._run_target_extension(
                        batch,
                        req,
                        verify_input[:replay_count],
                        source_position,
                    )

        if predictor_speculated:
            self.predictor_cached_len = source_position
        with self._profile_phase(f"PredictorCommit{output.numel()}"):
            draft_logits, predictor_expanded = self._run_predictor(
                batch,
                req,
                committed_expanded,
                output,
                source_position,
            )
        self.pending_draft, self.pending_draft_confidence = self._draft_candidate(
            draft_logits
        )
        if self.max_supported_drafts > 1:
            self.pending_predictor_hidden = predictor_expanded[-1:].clone()
        req.complete_one()

        self.metrics.cycles += 1
        self.metrics.proposed_drafts += draft_tokens.numel()
        self.metrics.accepted_drafts += accepted_count
        self.metrics.emitted_tokens += output.numel()
        if self.fallback_enabled:
            self._accept_history.append(accepted_count)
            self._request_cycles += 1
            if self._request_cycles >= self.fallback_cycles:
                accepted_yield = sum(self._accept_history) / len(self._accept_history)
                if accepted_yield < self.fallback_min_yield:
                    self.fallback_uid = req.uid
                    logger.info_rank0(
                        "Qwen4-Exp MTP fallback: "
                        f"cycle={self._request_cycles}, "
                        f"accepted_yield={accepted_yield:.3f}, "
                        "ordinary decode selected"
                    )
        if self.timing_enabled and len(self.metrics.cycle_trace) < 4096:
            self.metrics.cycle_trace.append(
                {
                    "cycle": self.metrics.cycles,
                    "width": draft_tokens.numel(),
                    "accepted": accepted_count,
                    "emitted": output.numel(),
                }
            )
        if self.log_interval and self.metrics.cycles % self.log_interval == 0:
            acceptance = self.metrics.acceptance_rate
            acceptance_text = f"{acceptance:.3f}" if acceptance is not None else "n/a"
            logger.info_rank0(
                "Qwen4-Exp MTP: "
                f"cycles={self.metrics.cycles}, "
                f"acceptance={acceptance_text}, "
                f"tokens/cycle={self.metrics.emitted_tokens / self.metrics.cycles:.3f}"
            )
        return output, accepted_count == draft_tokens.numel()

    def _draft_candidate(self, logits: torch.Tensor) -> tuple[torch.Tensor, float]:
        last = logits[-1]
        top_logit, token = torch.max(last, dim=-1, keepdim=True)
        token = token.to(torch.int32)
        if self.draft_p_min <= 0.0:
            return token, 1.0
        confidence = torch.exp(top_logit - torch.logsumexp(last, dim=-1))
        return token, float(confidence.item())

    def _run_predictor(
        self,
        batch: Batch,
        req: Req,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        source_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[0] != token_ids.numel():
            raise ValueError("MTP hidden/token row counts differ")
        if source_start != self.predictor_cached_len:
            raise RuntimeError(
                f"MTP predictor starts at {source_start}, "
                f"but cache ends at {self.predictor_cached_len}"
            )
        graph = (
            self.predictor_graphs.get(token_ids.numel()) if batch.is_decode else None
        )
        if graph is not None:
            logits, expanded = graph.replay(req, hidden, token_ids, source_start)
            self.predictor_cached_len += token_ids.numel()
            return logits, expanded
        state = _snapshot_mutable_forward_state(batch, req)
        last_logits = None
        last_expanded = None
        try:
            # Chunk so one predictor forward never routes more experts than the
            # predictor cache holds.
            for offset in range(0, token_ids.numel(), self.max_predictor_chunk):
                count = min(self.max_predictor_chunk, token_ids.numel() - offset)
                start = source_start + offset
                end = start + count
                batch.phase = "prefill"
                batch.padded_reqs = batch.reqs
                batch.input_ids = token_ids[offset : offset + count]
                batch.positions = torch.arange(
                    start, end, dtype=torch.int32, device=self.engine.device
                )
                batch.out_loc = self.engine.page_table[req.table_idx, start:end]
                batch.fla_metadata = None
                batch.moe_decode_cache = True
                req.cached_len = start
                req.device_len = end
                self.engine.attn_backend.prepare_metadata(batch)
                with self.engine.ctx.forward_batch(batch):
                    last_logits, last_expanded = self.model.mtp_forward(
                        hidden[offset : offset + count],
                        token_ids[offset : offset + count],
                    )
                self.predictor_cached_len = end
        finally:
            _restore_mutable_forward_state(batch, req, state)
        if last_logits is None or last_expanded is None:
            raise ValueError("MTP predictor received no tokens")
        return last_logits, last_expanded

    def _run_target_extension(
        self,
        batch: Batch,
        req: Req,
        token_ids: torch.Tensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = _snapshot_mutable_forward_state(batch, req)
        try:
            end = start + token_ids.numel()
            host_end = req.input_ids.numel()
            if host_end < start or host_end > end:
                raise RuntimeError(
                    f"MTP target extension host length {host_end} is outside "
                    f"forward range [{start}, {end}]"
                )
            if host_end < end:
                req.append_host(
                    token_ids[host_end - start :].to(
                        device="cpu", dtype=req.input_ids.dtype
                    )
                )
            req.cached_len = start
            req.device_len = end
            target_graph = self.target_verify_graphs.get(token_ids.numel())
            if target_graph is not None:
                return target_graph.replay(req, token_ids, start)
            if token_ids.numel() == 1 and self.engine.graph_runner.can_use_cuda_graph(
                batch
            ):
                batch.phase = "decode"
                batch.input_ids = token_ids
                batch.positions = torch.arange(
                    start, end, dtype=torch.int32, device=self.engine.device
                )
                batch.out_loc = self.engine.page_table[req.table_idx, start:end]
                with self.engine.ctx.forward_batch(batch):
                    return self.engine.graph_runner.replay(batch, return_expanded=True)
            batch.phase = "prefill"
            batch.padded_reqs = batch.reqs
            batch.input_ids = token_ids
            batch.positions = torch.arange(
                start, end, dtype=torch.int32, device=self.engine.device
            )
            batch.out_loc = self.engine.page_table[req.table_idx, start:end]
            batch.moe_decode_cache = True
            batch.fla_metadata = build_fla_metadata(batch, self.engine.device)
            self.engine.attn_backend.prepare_metadata(batch)
            with self.engine.ctx.forward_batch(batch):
                hidden, expanded = self.model.model.forward(
                    batch.input_ids, batch, return_expanded=True
                )
                logits = project_lm_head_all_positions(self.model.lm_head, hidden)
            return logits, expanded
        finally:
            _restore_mutable_forward_state(batch, req, state)


__all__ = ["MTPMetrics", "MTPWorker"]
