"""source/dpo.py: pair encoding, collation, answer log-probabilities and the loss, with a tiny model."""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
import dpo  # noqa: E402


class Tok:
    eos_token = "|"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "".join(m["content"] for m in messages) + ">"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) % 50 for c in text]}


def test_encode_pair_masks_the_prompt_and_drops_long_pairs():
    pair = dpo.encode_pair({"system": "s", "user": "u", "chosen": "ab", "rejected": "xyz"}, Tok(), 64)
    assert pair["chosen"]["labels"] == [-100] * 3 + pair["chosen"]["input_ids"][3:]
    assert len(pair["rejected"]["input_ids"]) == 3 + 4 and pair["chosen"]["input_ids"][:3] == pair["rejected"]["input_ids"][:3]
    assert dpo.encode_pair({"system": "s", "user": "u", "chosen": "ab", "rejected": "x" * 10}, Tok(), 8) is None


def test_collate_puts_chosen_first_and_carries_the_reference():
    a = dpo.encode_pair({"system": "s", "user": "u", "chosen": "a", "rejected": "bb"}, Tok(), 64)
    b = dpo.encode_pair({"system": "s", "user": "uu", "chosen": "ccc", "rejected": "d"}, Tok(), 64)
    a["ref"], b["ref"] = [-1.0, -2.0], [-3.0, -4.0]
    out = dpo.collate_pairs([a, b], pad_id=0)
    assert out["input_ids"].shape[0] == 4
    assert out["input_ids"][1, :len(b["chosen"]["input_ids"])].tolist() == b["chosen"]["input_ids"]
    assert out["input_ids"][2, :len(a["rejected"]["input_ids"])].tolist() == a["rejected"]["input_ids"]
    assert out["ref"].tolist() == [-1.0, -3.0, -2.0, -4.0]
    assert "ref" not in dpo.collate_pairs([dpo.encode_pair({"system": "s", "user": "u", "chosen": "a", "rejected": "b"}, Tok(), 64)], 0)


def test_sequence_logps_matches_a_full_log_softmax():
    torch.manual_seed(0)
    hidden = torch.randn(3, 7, 5)
    head = torch.nn.Linear(5, 11, bias=False)
    labels = torch.full((3, 7), -100)
    labels[0, 4:] = torch.tensor([1, 2, 3])
    labels[1, 2:5] = torch.tensor([4, 5, 6])
    labels[2, 6] = 7
    sums, counts = dpo.sequence_logps(hidden, labels, head)
    full = torch.log_softmax(head(hidden), dim=-1)
    for s in range(3):
        want = sum(full[s, t - 1, labels[s, t]].item() for t in range(1, 7) if labels[s, t] != -100)
        assert sums[s].item() == pytest.approx(want, abs=1e-5)
    assert counts.tolist() == [3, 3, 1]


def test_dpo_loss_at_the_reference_and_its_gradient():
    policy = torch.tensor([-4.0, -6.0], requires_grad=True)  # one pair: chosen, rejected
    ref = torch.tensor([-4.0, -6.0])
    counts = torch.tensor([2, 3])
    loss, stats = dpo.dpo_loss(policy, ref, counts, beta=0.1, sft_weight=0.0)
    assert loss.item() == pytest.approx(math.log(2)) and stats["margin"] == 0
    loss.backward()
    assert policy.grad[0] < 0 < policy.grad[1]  # descending raises the chosen answer and lowers the rejected
    loss, stats = dpo.dpo_loss(policy.detach(), ref, counts, beta=0.1, sft_weight=1.0)
    assert loss.item() == pytest.approx(math.log(2) + 4.0 / 2) and stats["chosen_nll"] == pytest.approx(2.0)
