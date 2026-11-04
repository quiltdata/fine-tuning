"""LoRA SFT of Nemotron 3.5 Lightning (30B-A3B, BF16) as the ingest summarizer, thinking off.

SageMaker training-job entry point. Channels and paths:
  SM_CHANNEL_MODEL   BF16 base checkpoint (JumpStart cache prefix, mounted as files)
  SM_CHANNEL_TRAIN   train.jsonl: one {"system", "user", "assistant"} object per line
  SM_MODEL_DIR       the LoRA adapter (becomes the job's model.tar.gz)
  /opt/ml/checkpoints  with --merge, the merged BF16 model, synced to S3 uncompressed so
                       a vLLM endpoint can load it directly

Each example is rendered with the model's own chat template, thinking off, exactly as
the summarizer is served: system and user turns, then "<|im_start|>assistant\\n<think></think>"
and the assistant's JSON rows, ending "<|im_end|>". Loss is on the assistant's tokens only.

LoRA targets attention (q/k/v/o_proj), the Mamba mixers (in_proj, out_proj) and the
shared expert's MLP; the 128 routed experts and the router stay frozen.

Hyperparameters (SageMaker passes them as --name value): epochs, lr, rank, alpha, max_len,
max_examples, max_steps, batch, grad_accum, save_steps, merge, diagnose. Trainer checkpoints go
to /opt/ml/checkpoints/trainer and a restarted (Spot) job resumes from the latest one.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

# Prebuilt wheels matching the image (CUDA 13, torch 2.9, Python 3.12, C++11 ABI). Source builds of
# older pins (mamba-ssm 2.2.4, causal-conv1d 1.5.0) fail to compile on CUDA 13 (smoke-02).
KERNELS = [
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
    "causal_conv1d-1.7.0%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/"
    "mamba_ssm-2.3.2.post1%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
]


def install_kernels():
    """Install NemotronH's Mamba2 and causal-conv1d kernels. The modeling code raises at import without
    mamba_ssm (there is no slow fallback), so a failed install stops the job here, with the reason."""
    started = time.time()
    for url in KERNELS:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", url],
                           capture_output=True, text=True, timeout=1800)
        print(json.dumps({"kernel": url.rsplit("/", 1)[-1], "ok": r.returncode == 0, "seconds": round(time.time() - started),
                          "tail": (r.stdout + r.stderr)[-1500:] if r.returncode else ""}), flush=True)
    import causal_conv1d  # noqa: F401
    import mamba_ssm  # noqa: F401
    print(json.dumps({"kernels_import": "ok", "mamba_ssm": mamba_ssm.__version__}), flush=True)


install_kernels()  # before transformers imports the modeling code, which checks for the kernels at import

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

MODEL_DIR = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")
TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CHECKPOINT_DIR = "/opt/ml/checkpoints"
TARGETS = r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj)$|.*\.shared_experts\.(up_proj|down_proj)$"


def encode(example, tokenizer, max_len):
    messages = [{"role": "system", "content": example["system"]}, {"role": "user", "content": example["user"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(example["assistant"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None  # dropped rather than truncated: a cut-off target teaches a cut-off answer
    return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}


def restore_checkpoint(model):
    """Copy every checkpoint tensor back into the model after from_pretrained.

    transformers 5 re-runs this remote model's _init_weights after loading, which redraws the
    Mamba dt_bias and rescales out_proj (diag-01: dt_bias off by up to 27, out_proj by 0.59;
    probe loss 11.3, near uniform). Copying the checkpoint back makes the weights exact
    whatever the loader did. Prints how many tensors it had to fix."""
    from safetensors import safe_open
    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    fixed, checked, absent = [], 0, 0
    with torch.no_grad():
        for shard in sorted(set(index.values())):
            with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
                for key in f.keys():
                    if key.startswith("mtp."):
                        continue
                    t = targets.get(key)
                    if t is None:
                        absent += 1
                        continue
                    ref = f.get_tensor(key).to(device=t.device, dtype=t.dtype)
                    checked += 1
                    if not torch.equal(t, ref):
                        fixed.append(key)
                        t.copy_(ref)
    kinds = sorted({re.sub(r"\.\d+\.", ".N.", k) for k in fixed})
    print(json.dumps({"restored_tensors": len(fixed), "checked": checked, "absent": absent, "kinds": kinds}), flush=True)


def diagnose(model, tokenizer, probe_ids):
    """Is the loaded model the checkpoint, and does it predict sensibly? Prints one JSON line per check."""
    from safetensors import safe_open
    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    params = dict(model.named_parameters())
    sample = ["backbone.embeddings.weight", "lm_head.weight", "backbone.norm_f.weight"]
    sample += [k for k in index if k.startswith("backbone.layers.0.") or k.startswith("backbone.layers.5.")][:12]
    sample += [k for k in index if ".experts.7.up_proj" in k][:2]
    report = {}
    for key in sample:
        with safe_open(os.path.join(MODEL_DIR, index[key]), framework="pt") as f:
            ref = f.get_tensor(key)
        p = params.get(key)
        report[key] = "not in model" if p is None else \
            ("shape " + str(tuple(p.shape)) + " vs " + str(tuple(ref.shape)) if p.shape != ref.shape else
             round((p.detach().float().cpu() - ref.float()).abs().max().item(), 6))
    print(json.dumps({"weights_max_abs_diff": report}), flush=True)
    emb, head = params["backbone.embeddings.weight"], params["lm_head.weight"]
    print(json.dumps({"lm_head_is_embeddings": emb.data_ptr() == head.data_ptr(),
                      "missing_in_model": sum(1 for k in index if not k.startswith("mtp.") and k not in params),
                      "model_not_in_ckpt": [k for k in params if k not in index][:10]}), flush=True)
    mod = sys.modules[type(model.backbone.layers[0].mixer).__module__]
    with torch.no_grad():
        for fast in (True, False):
            mod.is_fast_path_available = fast and mod.is_fast_path_available
            loss = model(input_ids=probe_ids, labels=probe_ids).loss.item()
            print(json.dumps({"probe_loss": round(loss, 3), "mamba_fast_path": mod.is_fast_path_available}), flush=True)
        mod.is_fast_path_available = True
        ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(model.device)
        for _ in range(12):
            nxt = model(input_ids=ids).logits[0, -1].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        print(json.dumps({"greedy": tokenizer.decode(ids[0])}), flush=True)


class AnswerOnlyTrainer(Trainer):
    """Loss from the output layer applied only at answer positions.

    The remote model computes float32 logits over the whole 131K vocabulary at every position;
    at batch 4 x ~4.6K tokens that is a 9.7 GB allocation on the GPU holding lm_head (v1-spot OOM).
    Only answer tokens carry loss, so run the backbone (LoRA lives there) and project just those."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
        hidden = causal_lm.backbone(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                    use_cache=False)[0]
        labels = inputs["labels"][:, 1:].to(hidden.device)  # inputs sit on GPU 0; the last layers on another GPU
        keep = labels != -100
        h = hidden[:, :-1][keep]
        head = causal_lm.lm_head
        logits = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).float()
        loss = torch.nn.functional.cross_entropy(logits, labels[keep].to(logits.device), reduction="sum")
        # back to the inputs' GPU, where the Trainer keeps num_items_in_batch and accumulates the loss
        device = inputs["input_ids"].device
        denominator = num_items_in_batch if num_items_in_batch is not None else keep.sum()
        loss = loss.to(device) / torch.as_tensor(denominator, device=device, dtype=torch.float32)
        return (loss, None) if return_outputs else loss


def fix_embedding_name(merged):
    """transformers 5 saves this remote model's embeddings as backbone.embedding.weight; the checkpoint
    and vLLM's loader use backbone.embeddings.weight (vLLM failed to start on merge-v1). Rename it in the
    shard header and the index. Tensor bytes are untouched; the header is rewritten in place."""
    import struct
    old, new = b'"backbone.embedding.weight"', b'"backbone.embeddings.weight"'
    index_path = os.path.join(merged, "model.safetensors.index.json")
    index = json.load(open(index_path))
    if "backbone.embedding.weight" not in index["weight_map"]:
        return
    shard = os.path.join(merged, index["weight_map"]["backbone.embedding.weight"])
    with open(shard, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = f.read(hlen)
        assert header.count(old) == 1
        new_header = header.replace(old, new).rstrip(b" ")
        new_header += b" " * ((-len(new_header)) % 8)
        if len(new_header) == hlen:  # padding absorbed the extra byte: rewrite the header in place
            data = None
        else:
            data = f.read()
    if data is None:
        with open(shard, "r+b") as f:
            f.seek(8)
            f.write(new_header)
    else:
        with open(shard, "wb") as f:
            f.write(struct.pack("<Q", len(new_header)) + new_header + data)
    index["weight_map"]["backbone.embeddings.weight"] = index["weight_map"].pop("backbone.embedding.weight")
    json.dump(index, open(index_path, "w"), indent=2)
    print(json.dumps({"renamed": "backbone.embedding.weight -> backbone.embeddings.weight", "shard": os.path.basename(shard)}), flush=True)


def save_merged(model, tokenizer, summary):
    """Merge LoRA into the base weights and save a servable folder to /opt/ml/checkpoints/merged."""
    merged = os.path.join(CHECKPOINT_DIR, "merged")
    model = model.merge_and_unload()
    # The remote class declares _tied_weights_keys as a list (transformers 4.x style); transformers 5
    # expects a dict and fails in save_pretrained (v1c). Nothing is tied: tie_word_embeddings is false.
    for module in model.modules():
        if isinstance(getattr(module, "_tied_weights_keys", None), list):
            module._tied_weights_keys = {}
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
    fix_embedding_name(merged)
    tokenizer.save_pretrained(merged)
    for name in os.listdir(MODEL_DIR):  # remote modeling code, chat template, parsers; the base config.json
        src = os.path.join(MODEL_DIR, name)
        if os.path.isfile(src) and not name.endswith(".safetensors") and name != "model.safetensors.index.json" \
                and (name == "config.json" or not os.path.exists(os.path.join(merged, name))):
            shutil.copy(src, merged)  # config.json: transformers 5 rewrites it; the architecture is unchanged
    json.dump(summary, open(os.path.join(merged, "training_summary.json"), "w"), indent=1)
    print(json.dumps({"merged_files": sorted(os.listdir(merged))[:40]}), flush=True)
    return model


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=12288)
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all; >0 for a smoke run")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4, help="examples per device step")
    ap.add_argument("--save_steps", type=int, default=50, help="optimizer steps between checkpoints (Spot resume)")
    ap.add_argument("--merge", type=int, default=0)
    ap.add_argument("--diagnose", type=int, default=0, help="check weights, paths and generation, then exit")
    ap.add_argument("--merge_only", type=int, default=0, help="merge the adapter in the 'adapter' channel, then exit")
    ap.add_argument("--init_adapter", type=int, default=0, help="continue training the adapter in the 'adapter' channel")
    args, _ = ap.parse_known_args()
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    rows = [json.loads(l) for l in open(os.path.join(TRAIN_DIR, "train.jsonl"))]
    if args.max_examples:
        rows = rows[: args.max_examples]
    data = [e for e in (encode(r, tokenizer, args.max_len) for r in rows) if e]
    lengths = sorted(len(e["input_ids"]) for e in data)
    print(json.dumps({"examples": len(rows), "kept": len(data), "dropped_too_long": len(rows) - len(data),
                      "tokens": sum(lengths), "target_tokens": sum(sum(1 for t in e["labels"] if t != -100) for e in data),
                      "len_p50": lengths[len(lengths) // 2], "len_max": lengths[-1]}), flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map="auto")
    restore_checkpoint(model)
    print(json.dumps({"loaded_seconds": round(time.time() - started),
                      "gpu_gb": [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]}),
          flush=True)
    probe = torch.tensor([data[0]["input_ids"][:256]], device=model.device)
    if args.merge_only:
        from peft import PeftModel
        adapter = os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter")
        name = "backbone.layers.5.mixer.q_proj.weight"
        before = dict(model.named_parameters())[name].detach().float().cpu().clone()
        model = PeftModel.from_pretrained(model, adapter)
        model = save_merged(model, tokenizer, {"merged_from": adapter})
        after = dict(model.named_parameters())[name].detach().float().cpu()
        with torch.no_grad():
            loss = model(input_ids=probe, labels=probe).loss.item()
        print(json.dumps({"merge_changed_q_proj_max_abs": round((after - before).abs().max().item(), 6),
                          "merged_probe_loss": round(loss, 3)}), flush=True)
        return
    if args.diagnose:
        diagnose(model, tokenizer, probe)
        return
    with torch.no_grad():  # one short forward pass, so a kernel or placement problem shows in minutes
        loss = model(input_ids=probe, labels=probe).loss
    print(json.dumps({"probe_loss": round(loss.item(), 3)}), flush=True)
    if loss.item() > 6:  # near-uniform (ln 131072 = 11.8): the model is broken, so don't pay to train it
        sys.exit(f"probe loss {loss.item():.2f} is too high; the loaded model is not working")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init_adapter:  # continue from an earlier adapter (rejection-sampling round)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.environ.get("SM_CHANNEL_ADAPTER", "/opt/ml/input/data/adapter"),
                                          is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                                                 task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()

    steps = args.max_steps if args.max_steps > 0 else math.ceil(len(data) * args.epochs / (args.batch * args.grad_accum))
    trainer_dir = os.path.join(CHECKPOINT_DIR, "trainer")  # synced to S3, so a Spot restart resumes here
    resume = None
    if os.path.isdir(trainer_dir):
        done = sorted((d for d in os.listdir(trainer_dir) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
        resume = os.path.join(trainer_dir, done[-1]) if done else None
    trainer = AnswerOnlyTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=trainer_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0,
            bf16=True, logging_steps=1, save_strategy="steps", save_steps=args.save_steps, save_total_limit=1,
            report_to=[],
            remove_unused_columns=False, dataloader_num_workers=2),
        train_dataset=data,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    print(json.dumps({"optimizer_steps": steps, "resume_from": resume}), flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    summary = {"train_loss": result.training_loss, "steps": result.global_step, "args": vars(args),
               "log": trainer.state.log_history, "seconds": round(time.time() - started)}
    json.dump(summary, open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w"), indent=1)

    if args.merge:
        save_merged(model, tokenizer, summary)
    print(json.dumps({"done_seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
