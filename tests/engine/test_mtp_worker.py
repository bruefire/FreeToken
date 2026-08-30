from __future__ import annotations

from collections import deque
from types import MethodType, SimpleNamespace

import pytest
import torch
from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.mtp_worker import (
    MTPMetrics,
    MTPWorker,
    _mtp_adaptive_cycles,
    _mtp_adaptive_enabled,
    _mtp_adaptive_min_yield,
    _mtp_draft_p_min,
    _mtp_max_drafts,
)


def _req(tokens: list[int]) -> Req:
    return Req(
        input_ids=torch.tensor(tokens, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=7,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(),
    )


def test_mtp_draft_environment_is_validated(monkeypatch):
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "2")
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN", "0.6")
    assert _mtp_max_drafts() == 2
    assert _mtp_draft_p_min() == 0.6

    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "3")
    assert _mtp_max_drafts() == 3
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_MAX_DRAFTS", "4")
    with pytest.raises(ValueError, match="must be 1, 2, or 3"):
        _mtp_max_drafts()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_DRAFT_P_MIN", "1.1")
    with pytest.raises(ValueError, match="between 0 and 1"):
        _mtp_draft_p_min()

    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE", "off")
    assert not _mtp_adaptive_enabled()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE", "maybe")
    with pytest.raises(ValueError, match="boolean"):
        _mtp_adaptive_enabled()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_CYCLES", "0")
    with pytest.raises(ValueError, match="positive"):
        _mtp_adaptive_cycles()
    monkeypatch.setenv("FREETOKEN_QWEN4_MTP_ADAPTIVE_MIN_ACCEPTANCE", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _mtp_adaptive_min_yield()


def test_mtp_can_speculate_only_for_initialized_single_request():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    worker = object.__new__(MTPWorker)
    worker.uid = req.uid
    worker.predictor_cached_len = req.cached_len
    worker.pending_hidden = None
    worker.pending_draft = torch.tensor([7], dtype=torch.int32)
    worker.fallback_uid = None

    assert worker.can_speculate(batch)

    worker.fallback_uid = req.uid
    assert not worker.can_speculate(batch)
    worker.fallback_uid = None

    other = _req([10, 11, 12])
    other.uid = req.uid + 1
    assert not worker.can_speculate(Batch(reqs=[other], phase="decode"))
    assert not worker.can_speculate(Batch(reqs=[req, other], phase="decode"))

    worker.predictor_cached_len += 1
    assert not worker.can_speculate(batch)


def test_mtp_prefill_keeps_hidden_and_next_token_alignment_across_chunks():
    worker = object.__new__(MTPWorker)
    worker.engine = SimpleNamespace(device=torch.device("cpu"))
    # Offline generation reuses request UIDs. A new position-zero prefill must
    # still reset all per-request predictor state.
    worker.uid = 7
    worker.predictor_cached_len = 99
    worker.pending_hidden = torch.tensor([[-1.0]])
    worker.pending_draft = torch.tensor([3], dtype=torch.int32)
    worker.pending_predictor_hidden = None
    worker.max_supported_drafts = 1
    worker.max_drafts = 1
    worker.draft_p_min = 0.0
    worker.fallback_uid = None
    worker._accept_history = deque(maxlen=64)
    worker._request_cycles = 0
    calls = []

    def run_predictor(self, batch, req, hidden, token_ids, source_start):
        calls.append((source_start, hidden.clone(), token_ids.clone()))
        self.predictor_cached_len += token_ids.numel()
        output = torch.zeros(token_ids.numel(), 4)
        return output, hidden + 10

    worker._run_predictor = MethodType(run_predictor, worker)

    first = _req([10, 11, 12])
    first_batch = Batch(reqs=[first], phase="prefill")
    worker.update_prefill(
        first_batch,
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([99], dtype=torch.int32),
        start=0,
        end=3,
        final=False,
    )

    second = Req(
        input_ids=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
        table_idx=0,
        cached_len=3,
        output_len=8,
        uid=7,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(),
    )
    second_batch = Batch(reqs=[second], phase="prefill")
    worker.update_prefill(
        second_batch,
        torch.tensor([[3.0], [4.0]]),
        torch.tensor([15], dtype=torch.int32),
        start=3,
        end=5,
        final=True,
    )

    assert calls[0][0] == 0
    torch.testing.assert_close(calls[0][1].flatten(), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(calls[0][2], torch.tensor([11, 12], dtype=torch.int32))
    assert calls[1][0] == 2
    torch.testing.assert_close(calls[1][1].flatten(), torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(
        calls[1][2], torch.tensor([13, 14, 15], dtype=torch.int32)
    )
    assert worker.predictor_cached_len == 5
    assert worker.pending_hidden is None
    assert worker.pending_draft.tolist() == [0]


def _decode_worker(
    req: Req,
    verify_tokens: list[int],
    *,
    prefix_checkpoint=True,
    max_drafts=1,
    predictor_outputs: list[int] | None = None,
):
    worker = object.__new__(MTPWorker)
    worker.engine = SimpleNamespace(
        device=torch.device("cpu"),
        linear_state_pool=None,
        model=SimpleNamespace(),
    )
    worker.model = worker.engine.model
    worker.uid = req.uid
    worker.predictor_cached_len = req.cached_len
    worker.pending_hidden = None
    worker.pending_draft = torch.tensor([7], dtype=torch.int32)
    worker.pending_predictor_hidden = torch.tensor([[50.0]])
    worker.pending_draft_confidence = 1.0
    worker.max_supported_drafts = max_drafts
    worker.max_drafts = max_drafts
    worker.draft_p_min = 0.0
    worker.metrics = MTPMetrics()
    worker.log_interval = 0
    worker.timing_enabled = False
    worker.timing_events = []
    worker.fallback_enabled = False
    worker.fallback_cycles = 64
    worker.fallback_min_yield = 0.75
    worker.fallback_uid = None
    worker._accept_history = deque(maxlen=64)
    worker._request_cycles = 0
    extensions = []
    predictors = []
    prefix_commits = []
    predictor_outputs = list(predictor_outputs or [4])

    class TargetGraph:
        def commit_prefix(self, target_req, checkpoint_index):
            prefix_commits.append((target_req, checkpoint_index))

    worker.target_verify_graphs = (
        {count: TargetGraph() for count in range(2, max_drafts + 2)}
        if prefix_checkpoint
        else {}
    )
    worker.predictor_graphs = {}

    def target_extension(self, batch, target_req, token_ids, start):
        extensions.append((start, token_ids.clone()))
        rows = token_ids.numel()
        logits = torch.full((rows, 16), -1.0)
        for row, token in enumerate(verify_tokens[:rows]):
            logits[row, token] = 1.0
        return logits, token_ids.to(torch.float32).unsqueeze(1)

    def predictor(self, batch, target_req, hidden, token_ids, source_start):
        predictors.append((source_start, hidden.clone(), token_ids.clone()))
        self.predictor_cached_len += token_ids.numel()
        logits = torch.full((token_ids.numel(), 16), -1.0)
        logits[-1, predictor_outputs.pop(0)] = 1.0
        return logits, hidden + 100

    worker._run_target_extension = MethodType(target_extension, worker)
    worker._run_predictor = MethodType(predictor, worker)
    return worker, extensions, predictors, prefix_commits


def test_mtp_decode_accepts_draft_and_carries_next_draft():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(req, [7, 9])

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 9]
    assert len(extensions) == 1
    assert extensions[0][0] == 2
    assert extensions[0][1].tolist() == [12, 7]
    assert predictors[0][0] == 2
    assert predictors[0][1].flatten().tolist() == [12.0, 7.0]
    assert predictors[0][2].tolist() == [7, 9]
    assert not prefix_commits
    assert worker.predictor_cached_len == 4
    assert worker.pending_draft.tolist() == [4]
    assert (req.cached_len, req.device_len) == (3, 4)


def test_mtp_decode_commits_verified_prefix_after_rejection():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(req, [8, 9])

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [8]
    assert len(extensions) == 1
    assert extensions[0][0] == 2
    assert extensions[0][1].tolist() == [12, 7]
    assert prefix_commits == [(req, 0)]
    assert predictors[0][0] == 2
    assert predictors[0][1].flatten().tolist() == [12.0]
    assert predictors[0][2].tolist() == [8]
    assert worker.predictor_cached_len == 3
    assert worker.pending_draft.tolist() == [4]
    assert (req.cached_len, req.device_len) == (3, 4)


def test_mtp_decode_replays_rejection_without_target_graph():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(
        req, [8, 9], prefix_checkpoint=False
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [8]
    assert [tokens.tolist() for _, tokens in extensions] == [[12, 7], [12]]
    assert not prefix_commits
    assert predictors[0][1].flatten().tolist() == [12.0]
    assert predictors[0][2].tolist() == [8]


def test_mtp_confidence_gate_can_skip_target_speculation():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(req, [8])
    worker.draft_p_min = 0.6
    worker.pending_draft_confidence = 0.5

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [8]
    assert extensions[0][1].tolist() == [12]
    assert predictors[0][2].tolist() == [8]
    assert not prefix_commits
    assert worker.metrics.proposed_drafts == 0


def test_mtp_two_drafts_commit_three_rows_when_both_are_accepted():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(
        req,
        [7, 4, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 4, 9]
    assert extensions[0][1].tolist() == [12, 7, 4]
    assert predictors[0][0] == 2
    assert predictors[0][1].flatten().tolist() == [50.0]
    assert predictors[0][2].tolist() == [7]
    assert predictors[1][0] == 2
    assert predictors[1][1].flatten().tolist() == [12.0, 7.0, 4.0]
    assert predictors[1][2].tolist() == [7, 4, 9]
    assert not prefix_commits
    assert worker.predictor_cached_len == 5
    assert worker.pending_draft.tolist() == [5]
    assert worker.metrics.proposed_drafts == 2
    assert worker.metrics.accepted_drafts == 2


def test_mtp_two_drafts_commit_second_prefix_after_partial_acceptance():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, _extensions, predictors, prefix_commits = _decode_worker(
        req,
        [7, 8, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [7, 8]
    assert prefix_commits == [(req, 1)]
    assert predictors[1][1].flatten().tolist() == [12.0, 7.0]
    assert predictors[1][2].tolist() == [7, 8]
    assert worker.predictor_cached_len == 4
    assert worker.metrics.accepted_drafts == 1


def test_mtp_two_drafts_commit_first_prefix_when_first_is_rejected():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, _extensions, predictors, prefix_commits = _decode_worker(
        req,
        [8, 4, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert not accepted
    assert output.tolist() == [8]
    assert prefix_commits == [(req, 0)]
    assert predictors[1][1].flatten().tolist() == [12.0]
    assert predictors[1][2].tolist() == [8]
    assert worker.predictor_cached_len == 3
    assert worker.metrics.accepted_drafts == 0


def test_mtp_three_drafts_commit_four_rows_when_all_are_accepted():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(
        req,
        [7, 4, 5, 9],
        max_drafts=3,
        predictor_outputs=[4, 5, 6],
    )

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 4, 5, 9]
    assert extensions[0][1].tolist() == [12, 7, 4, 5]
    # draft2 embeds draft1 on the carried hidden; draft3 embeds draft2 on the
    # previous predictor call's own output hidden.
    assert predictors[0][1].flatten().tolist() == [50.0]
    assert predictors[0][2].tolist() == [7]
    assert predictors[1][1].flatten().tolist() == [150.0]
    assert predictors[1][2].tolist() == [4]
    assert predictors[2][2].tolist() == [7, 4, 5, 9]
    assert not prefix_commits
    assert worker.predictor_cached_len == 6
    assert worker.metrics.proposed_drafts == 3
    assert worker.metrics.accepted_drafts == 3


def test_mtp_fallback_selects_ordinary_after_low_accepted_yield():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, _extensions, _predictors, _commits = _decode_worker(
        req, [8], predictor_outputs=[4, 4]
    )
    worker.fallback_enabled = True
    worker.fallback_cycles = 2
    worker._accept_history = deque(maxlen=2)

    worker.forward_decode(batch, SimpleNamespace())
    assert worker.fallback_uid is None

    worker.forward_decode(batch, SimpleNamespace())
    assert worker.fallback_uid == req.uid
    assert worker.ordinary_decode_selected(req)
    assert not worker.can_speculate(batch)

    # a new request resets the window and re-enables speculation
    worker.reset(req.uid + 1)
    assert worker.fallback_uid is None
    assert worker._request_cycles == 0


def test_mtp_fallback_keeps_speculating_at_high_yield():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, _extensions, _predictors, _commits = _decode_worker(
        req, [7, 9], predictor_outputs=[7, 7]
    )
    worker.fallback_enabled = True
    worker.fallback_cycles = 2
    worker._accept_history = deque(maxlen=2)

    worker.forward_decode(batch, SimpleNamespace())
    # emulate the scheduler drain of the 2-token output before the next cycle
    req.cached_len = worker.predictor_cached_len
    req.device_len = req.cached_len + 1
    worker.forward_decode(batch, SimpleNamespace())
    assert worker.fallback_uid is None
    assert not worker.ordinary_decode_selected(req)


def test_mtp_second_draft_confidence_gate_verifies_only_first_draft():
    req = _req([10, 11, 12])
    req.cached_len = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = torch.tensor([12], dtype=torch.int32)
    worker, extensions, predictors, prefix_commits = _decode_worker(
        req,
        [7, 9],
        max_drafts=2,
        predictor_outputs=[4, 5],
    )
    worker.draft_p_min = 0.6
    candidates = iter(
        [
            (torch.tensor([4], dtype=torch.int32), 0.5),
            (torch.tensor([5], dtype=torch.int32), 1.0),
        ]
    )

    def draft_candidate(self, logits):
        return next(candidates)

    worker._draft_candidate = MethodType(draft_candidate, worker)

    output, accepted = worker.forward_decode(batch, SimpleNamespace())

    assert accepted
    assert output.tolist() == [7, 9]
    assert extensions[0][1].tolist() == [12, 7]
    assert predictors[0][2].tolist() == [7]
    assert predictors[1][2].tolist() == [7, 9]
    assert not prefix_commits
    assert worker.metrics.proposed_drafts == 1
