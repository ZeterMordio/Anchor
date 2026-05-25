"""
Qwen3.5 fastpath smoke canary.

This is intentionally not a training run. It proves:
- causal-conv1d and flash-linear-attention fastpath imports are present.
- Qwen/Qwen3.5-9B loads text-only.
- one short generate works.
- one tiny full-model backward works.

HF Jobs install note:
- Use a CUDA devel image, e.g. pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel.
- Install build deps + flash-linear-attention first.
- Install causal-conv1d with --no-build-isolation so it compiles against the
  image torch/CUDA stack. Build isolation pulled torch 2.12.0+cu130 during the
  2026-05-23 canary and failed against the CUDA 12.8 image.
"""

import importlib.metadata
import os
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B")
DTYPE = torch.bfloat16
FASTPATH_IMPORTS = (
    ("causal_conv1d", "causal_conv1d_fn"),
    ("causal_conv1d", "causal_conv1d_update"),
    ("fla.modules", "FusedRMSNormGated"),
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
)
PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "causal-conv1d",
    "flash-linear-attention",
    "triton",
)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def run(cmd):
    try:
        out = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except Exception as exc:
        print(f"$ {' '.join(cmd)} -> {type(exc).__name__}: {exc}")
        return
    print(f"$ {' '.join(cmd)}")
    print(out.stdout.strip())
    if out.stderr.strip():
        print(out.stderr.strip())


def probe_fastpath():
    missing = []
    for module_name, attr_name in FASTPATH_IMPORTS:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            obj = getattr(module, attr_name, None)
            if obj is None:
                missing.append(f"{module_name}.{attr_name}=None")
        except Exception as exc:
            missing.append(f"{module_name}.{attr_name}: {type(exc).__name__}: {exc}")
    return missing


def cuda_stats(prefix):
    if not torch.cuda.is_available():
        print(f"{prefix}: cuda unavailable")
        return
    torch.cuda.synchronize()
    print(
        f"{prefix}: alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB "
        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB "
        f"peak_alloc={torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
        f"peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.2f}GB"
    )


def first_param_device(model):
    return next(model.parameters()).device


def move_batch(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"python={sys.version}")
    print(f"model={MODEL_NAME}")
    for package in PACKAGES:
        print(f"{package}={package_version(package)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
    run(["nvidia-smi"])

    missing = probe_fastpath()
    if missing:
        print("[FASTPATH] MISSING")
        for item in missing:
            print(f"[FASTPATH] {item}")
        raise SystemExit(20)
    print("[FASTPATH] OK")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"load_s={time.time() - t0:.1f}")
    print(f"input_device={first_param_device(model)}")
    cuda_stats("after_load")

    prompt = "Buyer goal: negotiate a fair price.\nTalk: Hello, I can make an opening offer.\nAction:"
    batch = move_batch(tokenizer(prompt, return_tensors="pt"), first_param_device(model))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generated = tokenizer.decode(output_ids[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"generate_s={time.time() - t0:.2f}")
    print(f"generated={generated!r}")
    cuda_stats("after_generate")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_text = "Buyer: Talk: I can pay $25. Action: [BUY] $25.00 (1x item)"
    train_batch = move_batch(tokenizer(train_text, return_tensors="pt"), first_param_device(model))
    train_batch["labels"] = train_batch["input_ids"].clone()
    t0 = time.time()
    out = model(**train_batch, use_cache=False)
    loss = out.loss
    print(f"loss={float(loss.detach().cpu()):.6f}")
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"backward_s={time.time() - t0:.2f}")
    cuda_stats("after_backward")
    print("[FASTPATH_CANARY] PASS")


if __name__ == "__main__":
    main()
