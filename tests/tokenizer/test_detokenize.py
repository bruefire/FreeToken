from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _WordTokenizer:
    """id -> word lookup; joins with spaces like a plain word-level vocabulary."""

    eos_token_id = 0

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def batch_decode(self, batches):
        return [" ".join(self.vocab[i] for i in ids) for ids in batches]


def _msg(uid: int, token: int, finished: bool = False) -> DetokenizeMsg:
    return DetokenizeMsg(uid=uid, next_token=token, finished=finished)


def test_multiple_tokens_for_one_uid_in_one_batch_stay_incremental():
    # The worker drains its queue in batches, so several tokens of one request
    # can arrive in one detokenize() call; each delta must contain only its own
    # token, not the batch prefix again.
    tokenizer = _WordTokenizer({1: "how", 2: "can", 3: "i", 4: "help"})
    manager = DetokenizeManager(tokenizer, eos_token_ids=frozenset({0}))

    first = manager.detokenize([_msg(7, 1)])
    rest = manager.detokenize([_msg(7, 2), _msg(7, 3), _msg(7, 4, finished=True)])

    assert "".join(first + rest) == "how can i help"
    assert rest == [" can", " i", " help"]


def test_single_token_batches_match_the_multi_token_batch():
    tokenizer = _WordTokenizer({1: "a", 2: "b", 3: "c"})
    one = DetokenizeManager(tokenizer, eos_token_ids=frozenset({0}))
    parts = [one.detokenize([_msg(1, t, finished=t == 3)])[0] for t in (1, 2, 3)]

    many = DetokenizeManager(tokenizer, eos_token_ids=frozenset({0}))
    batch = many.detokenize([_msg(2, 1), _msg(2, 2), _msg(2, 3, finished=True)])

    assert parts == batch
