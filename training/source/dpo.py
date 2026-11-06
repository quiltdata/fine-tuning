"""Preference training (DPO) pieces for train.py --dpo_beta: pairs, log-probabilities, loss.

A pair is one JSONL line {"system", "user", "chosen", "rejected"}: two answers to the same prompt.
The loss is DPO's, -log sigmoid(beta * ((policy_c - ref_c) - (policy_r - ref_r))), plus sft_weight
times the chosen answer's mean token NLL, so the model keeps writing the chosen answers while it
learns to prefer them (without it, DPO can lower both answers' likelihood and lose recall).
Log-probabilities are summed over answer tokens only. The reference is the model before the first
step, computed once and saved, so a Spot restart reuses it rather than scoring with a trained model.
Only torch here, so tests can run without the training image.
"""
import torch
import torch.nn.functional as F


def encode_pair(example, tokenizer, max_len):
    """Prompt and both answers as token ids, rendered as train.encode renders an SFT example."""
    messages = [{"role": "system", "content": example["system"]}, {"role": "user", "content": example["user"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    out = {}
    for side in ("chosen", "rejected"):
        target = tokenizer(example[side] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(target) > max_len:
            return None
        out[side] = {"input_ids": prompt_ids + target, "labels": [-100] * len(prompt_ids) + target}
    return out


def collate_pairs(batch, pad_id):
    """B pairs -> 2B padded sequences: the chosen answers first, then the rejected, in pair order.
    Carries the reference log-probabilities when the pairs have them."""
    seqs = [p["chosen"] for p in batch] + [p["rejected"] for p in batch]
    width = max(len(s["input_ids"]) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s["input_ids"])
        ids[i, :n] = torch.tensor(s["input_ids"])
        labels[i, :n] = torch.tensor(s["labels"])
        mask[i, :n] = 1
    out = {"input_ids": ids, "labels": labels, "attention_mask": mask}
    if all("ref" in p for p in batch):
        out["ref"] = torch.tensor([p["ref"][0] for p in batch] + [p["ref"][1] for p in batch], dtype=torch.float32)
    return out


def token_logps(logits, targets):
    """Log-probability of each target token, in float32."""
    return -F.cross_entropy(logits.float(), targets, reduction="none")


def sequence_logps(hidden, labels, head):
    """Summed answer-token log-probabilities and answer-token counts per sequence.

    hidden: (S, T, H) backbone outputs; labels: (S, T) with -100 off the answer. Like
    train.AnswerOnlyTrainer, projects only answer positions through the output layer."""
    labels = labels[:, 1:].to(hidden.device)
    keep = labels != -100
    h = hidden[:, :-1][keep]
    logits = head(h.to(device=head.weight.device, dtype=head.weight.dtype))
    lp = token_logps(logits, labels[keep].to(logits.device))
    rows = keep.nonzero()[:, 0].to(lp.device)
    sums = torch.zeros(labels.shape[0], device=lp.device, dtype=lp.dtype).index_add(0, rows, lp)
    counts = keep.sum(dim=1).to(lp.device)
    return sums, counts


def dpo_loss(policy, ref, counts, beta, sft_weight):
    """policy, ref, counts: (2B,) tensors, chosen first. Returns the mean loss over the B pairs and a
    dict of batch statistics (preference accuracy, margin, chosen NLL)."""
    b = policy.shape[0] // 2
    pc, pr = policy[:b], policy[b:]
    rc, rr = ref[:b].to(policy.device), ref[b:].to(policy.device)
    margin = beta * ((pc - rc) - (pr - rr))
    nll = -pc / counts[:b].to(policy.device).clamp(min=1)
    loss = -F.logsigmoid(margin) + sft_weight * nll
    stats = {"pref_acc": (margin > 0).float().mean().item(), "margin": margin.mean().item(), "chosen_nll": nll.mean().item()}
    return loss.mean(), stats
