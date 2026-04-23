"""Toy Run 1: Self-contained GRPO. Auto-installs deps. Qwen3-1.7B + LoRA."""
import os, sys, subprocess, json, time

def ensure_deps():
    deps = ["transformers", "torch", "datasets", "accelerate", "peft"]
    try:
        import transformers, torch, datasets, accelerate, peft
        print("[deps] All packages already installed.")
        return
    except ImportError:
        pass
    print("[deps] Installing packages...")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + deps
    # Actually install with deps
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + deps
    subprocess.check_call(cmd)
    print("[deps] Done.")

ensure_deps()

# Now run the actual training
exec(open("/tmp/run.py").read())
