"""Export a merged BF16 Lightning fine-tune to NVFP4 with NVIDIA ModelOpt (METHODOLOGY.md §6).

SageMaker training-job entry point, run from the same image and source bundle as train.py.
Channels and paths:
  SM_CHANNEL_MODEL   the merged BF16 model (train.py --merge output: a vLLM-loadable folder)
  SM_CHANNEL_CALIB   train.jsonl: calibration prompts, the same {"system","user","assistant"} rows
  /opt/ml/checkpoints/nvfp4   the export, synced to S3 uncompressed so vLLM can load it

The recipe is the one NVIDIA ships for this model (modelopt_recipes/models/nvidia/
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/ptq/w4a16_nvfp4_4o6), the scheme of the JumpStart
NVFP4 artifact that production serves: routed experts, shared experts and lm_head weight-only
NVFP4 (block 16, four-over-six scales chosen by MSE), Mamba in_proj/out_proj FP8 with static
input scales, KV cache FP8, attention and the MTP tail BF16. Weight-only NVFP4 is why the
base runs on an L40S (no FP4 tensor cores needed).

Hyperparameters (--name value): calib_size, calib_seq, recipe, modelopt (pip spec).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

MODEL_DIR = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")
CALIB_DIR = os.environ.get("SM_CHANNEL_CALIB", "/opt/ml/input/data/calib")
EXPORT_DIR = "/opt/ml/checkpoints/nvfp4"
OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
RECIPE = "models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/ptq/w4a16_nvfp4_4o6"
# The JumpStart NVFP4 artifact's tensor families, for the structural check after export
REF_QUANTIZED = {"backbone.layers.N.mixer.experts.N.down_proj", "backbone.layers.N.mixer.experts.N.up_proj",
                 "backbone.layers.N.mixer.in_proj", "backbone.layers.N.mixer.out_proj",
                 "backbone.layers.N.mixer.shared_experts.down_proj", "backbone.layers.N.mixer.shared_experts.up_proj",
                 "lm_head"}


def log(**kw):
    print(json.dumps(kw), flush=True)


def pip(spec):
    started = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", spec], capture_output=True, text=True,
                       timeout=1800)
    log(pip=spec, ok=r.returncode == 0, seconds=round(time.time() - started), tail=(r.stdout + r.stderr)[-1200:] if r.returncode else "")
    if r.returncode:
        sys.exit(f"pip install {spec} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_size", type=int, default=64)
    ap.add_argument("--calib_seq", type=int, default=4096)
    ap.add_argument("--recipe", default=RECIPE)
    ap.add_argument("--modelopt", default="nvidia-modelopt==0.47.0")
    ap.add_argument("--max_shard_size", default="5GB")
    args, _ = ap.parse_known_args()
    started = time.time()

    pip(args.modelopt)
    import train  # installs the Mamba kernels at import, before the modeling code is loaded
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_recipe
    from modelopt.torch.export import export_hf_checkpoint
    log(modelopt=modelopt.__version__, torch=torch.__version__)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    train.MODEL_DIR = MODEL_DIR
    train.restore_checkpoint(model)  # transformers 5 re-initializes some remote-model tensors after loading (train.py)
    model.eval()
    log(loaded_seconds=round(time.time() - started),
        gpu_gb=[round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())],
        has_mtp=hasattr(model, "mtp"))

    # Calibration prompts: the summarizer's own system and user turns, thinking off, plus the answer,
    # truncated to calib_seq tokens (long inputs are the common case, so truncating keeps variety).
    rows = [json.loads(l) for l in open(os.path.join(CALIB_DIR, "train.jsonl"))]
    step = max(1, len(rows) // args.calib_size)
    samples = []
    for r in rows[::step][: args.calib_size]:
        messages = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] + \
            tokenizer(r["assistant"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        samples.append(ids[: args.calib_seq])
    log(calib_samples=len(samples), calib_tokens=sum(map(len, samples)), calib_seq=args.calib_seq)

    probe = torch.tensor([samples[0][:256]], device=model.device)
    with torch.no_grad():
        loss = model(input_ids=probe, labels=probe).loss.item()
    log(probe_loss_bf16=round(loss, 3))
    if loss > 6:
        sys.exit(f"probe loss {loss:.2f} is too high; the loaded model is not working")

    def forward_loop(m):
        t0 = time.time()
        with torch.no_grad():
            for i, ids in enumerate(samples):
                m(input_ids=torch.tensor([ids], device=m.device))
                if (i + 1) % 16 == 0:
                    log(calibrated=i + 1, seconds=round(time.time() - t0))

    recipe = load_recipe(args.recipe)
    cfg = recipe.quantize.model_dump()  # mtq.quantize takes the plain dict, as examples/hf_ptq does
    log(recipe=args.recipe, algorithm=str(cfg.get("algorithm"))[:300], quant_cfg_entries=len(cfg.get("quant_cfg", [])))
    t0 = time.time()
    model = mtq.quantize(model, cfg, forward_loop)
    log(quantized_seconds=round(time.time() - t0))
    os.makedirs(EXPORT_DIR, exist_ok=True)
    mtq.print_quant_summary(model, output_dir=os.path.join(OUTPUT_DIR))  # the full per-quantizer summary, in model.tar.gz
    kv = [n for n, _ in model.named_modules() if n.endswith("k_bmm_quantizer")]
    log(kv_cache_quantizers=len(kv), example=kv[:1])
    with torch.no_grad():
        loss = model(input_ids=probe, labels=probe).loss.item()
    log(probe_loss_quantized=round(loss, 3))

    if hasattr(model, "mtp"):
        model._mtp_layer_prefixes = ["mtp"]  # export adds "mtp*" to exclude_modules, as the JumpStart artifact has
    # The remote class declares _tied_weights_keys as a list (transformers 4.x style); transformers 5's
    # save_pretrained expects a dict and fails (first export attempt, 2026-10-27; train.py save_merged has the
    # same fix). Nothing is tied: tie_word_embeddings is false.
    for module in model.modules():
        if isinstance(getattr(module, "_tied_weights_keys", None), list):
            module._tied_weights_keys = {}
    t0 = time.time()
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=EXPORT_DIR, max_shard_size=args.max_shard_size)
    log(exported_seconds=round(time.time() - t0))

    # The serving extras the export doesn't carry: remote modeling code, chat template, reasoning parser,
    # tokenizer files. config.json comes from the export (it holds quantization_config); make sure the
    # remote-code keys survived, since vLLM and transformers load the model through auto_map.
    for name in os.listdir(MODEL_DIR):
        src = os.path.join(MODEL_DIR, name)
        if os.path.isfile(src) and not name.endswith(".safetensors") and name != "model.safetensors.index.json" \
                and name != "config.json" and not os.path.exists(os.path.join(EXPORT_DIR, name)):
            shutil.copy(src, EXPORT_DIR)
    base_cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    cfg_path = os.path.join(EXPORT_DIR, "config.json")
    exp_cfg = json.load(open(cfg_path))
    added = [k for k in base_cfg if k not in exp_cfg]
    for k in added:
        exp_cfg[k] = base_cfg[k]
    json.dump(exp_cfg, open(cfg_path, "w"), indent=2)
    train.fix_embedding_name(EXPORT_DIR)
    log(config_keys_added=added, quantization_config=str(exp_cfg.get("quantization_config"))[:400])

    # Structural check against the JumpStart NVFP4 artifact
    import re
    from safetensors import safe_open
    index = json.load(open(os.path.join(EXPORT_DIR, "model.safetensors.index.json")))["weight_map"]
    fam = lambda k: re.sub(r"\.\d+\.", ".N.", k).rsplit(".", 1)[0]
    scaled = {fam(k) for k in index if k.endswith("weight_scale")}
    weights = {fam(k) for k in index if k.endswith(".weight")}
    hq = json.load(open(os.path.join(EXPORT_DIR, "hf_quant_config.json")))["quantization"]
    sample = {}
    for key in ["backbone.layers.1.mixer.experts.0.up_proj.weight", "backbone.layers.1.mixer.experts.0.up_proj.weight_scale",
                "backbone.layers.1.mixer.experts.0.up_proj.weight_scale_2", "backbone.layers.0.mixer.in_proj.weight",
                "backbone.layers.0.mixer.in_proj.input_scale", "backbone.layers.5.mixer.q_proj.weight", "lm_head.weight",
                "backbone.layers.5.mixer.k_proj.k_scale", "backbone.embeddings.weight"]:
        if key in index:
            with safe_open(os.path.join(EXPORT_DIR, index[key]), framework="pt") as f:
                t = f.get_slice(key)
                sample[key] = [str(t.get_dtype()), list(t.get_shape())]
        else:
            sample[key] = "absent"
    log(export_check={"quantized_families": sorted(scaled), "matches_reference": scaled == REF_QUANTIZED,
                      "missing_vs_reference": sorted(REF_QUANTIZED - scaled), "extra_vs_reference": sorted(scaled - REF_QUANTIZED),
                      "unquantized_weight_families": sorted(weights - scaled)[:40], "tensors": len(index),
                      "mtp_tensors": sum(k.startswith("mtp.") for k in index),
                      "quant_algo": hq.get("quant_algo"), "kv_cache_quant_algo": hq.get("kv_cache_quant_algo"),
                      "exclude_modules": hq.get("exclude_modules", [])[:12], "n_quantized_layers": len(hq.get("quantized_layers", {})),
                      "samples": sample})
    size = sum(os.path.getsize(os.path.join(EXPORT_DIR, f)) for f in os.listdir(EXPORT_DIR))
    log(export_files=sorted(os.listdir(EXPORT_DIR))[:40], export_gb=round(size / 2**30, 1), done_seconds=round(time.time() - started))
    json.dump({"recipe": args.recipe, "modelopt": modelopt.__version__, "calib_size": len(samples), "calib_seq": args.calib_seq,
               "source_model": MODEL_DIR, "export_gb": round(size / 2**30, 1)}, open(os.path.join(EXPORT_DIR, "export_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
