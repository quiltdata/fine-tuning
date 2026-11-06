#!/usr/bin/env python3
"""Apply the two post-export config fixes (README.md, "After export: two config fixes") to an NVFP4
export's config.json and hf_quant_config.json, and check the result against NVIDIA's NVFP4 artifact.

1. Remove ModelOpt's leaked name-mapper placeholder (".\\x00backbone.pt_name_sentinel") from every
   module name.
2. Replace the unquantized-module list (config.json `ignore`, hf_quant_config.json
   `exclude_modules`) with the reference artifact's.

Then every other part of both quantization sections must equal the reference's, except the
producer version; the script exits 1 if not. Fixed files are written to out_dir.

Usage: python3 fix_nvfp4_config.py <export_dir> <reference_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

SENTINEL = ".\\u0000backbone.pt_name_sentinel"  # as it appears in the JSON text


def load_fixed(path):
    return json.loads(path.read_text().replace(SENTINEL, ""))


def main():
    exp, ref, out = (Path(p) for p in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    problems = []

    c, rc = load_fixed(exp / "config.json"), json.loads((ref / "config.json").read_text())
    q, rq = c["quantization_config"], rc["quantization_config"]
    q["ignore"] = rq["ignore"]
    for k in rq:
        if k != "producer" and q.get(k) != rq[k]:
            problems.append(f"config.json quantization_config.{k} differs from the reference")

    h, rh = load_fixed(exp / "hf_quant_config.json"), json.loads((ref / "hf_quant_config.json").read_text())
    h["quantization"]["exclude_modules"] = rh["quantization"]["exclude_modules"]
    if h["quantization"] != rh["quantization"]:
        problems.append("hf_quant_config.json quantization differs from the reference")

    for name, obj in (("config.json", c), ("hf_quant_config.json", h)):
        text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
        if "pt_name_sentinel" in text:
            problems.append(f"{name}: placeholder still present")
        (out / name).write_text(text)
    print("\n".join(problems) or "both quantization sections match the reference (producer aside)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
