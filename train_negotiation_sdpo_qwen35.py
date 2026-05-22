"""
Negotiation SDPO+GRPO training for bilateral price negotiation.

This script is a separate experimental sibling of train_negotiation_pure.py.
It keeps the negotiation paper's buyer-only RLVR setup but augments the buyer
update with Self-Distillation Policy Optimization (SDPO):

- Trainable buyer policy.
- Frozen regulated seller model as the environment counterparty; no reference-policy model.
- Buyer always starts; only buyer turns receive updates.
- Verifiable reward remains the paper's economic-surplus scalar.
- SDPO adds feedback-conditioned self-teacher log-probs for dense token credit.

Default run policy:
- Use Qwen/Qwen3.5-9B by default. Qwen3.5 is a newer multimodal
  ImageTextToText/ForConditionalGeneration family, so this script keeps text-only
  wrappers around processor/model calls and runs a startup canary by default.
- Start GRPO-heavy and gradually hand off to SDPO shaping: by default
  A_total decays from 0.9 * A_GRPO + 0.1 * A_SDPO to a balanced 0.5/0.5 mix by iter 20.
- Use strict feedback by default: no exact seller cost or private floor is placed
  into the teacher prompt. Oracle feedback is an explicit ablation only.
- Keep the HF Jobs shape analogous to train_negotiation_pure.py: one standalone
  file, env-var config, W&B logging, and periodic Hub checkpoints.
"""

import gc
import importlib.metadata
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# HF Jobs log streaming: avoid buffered multi-minute stalls.
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # Older Transformers fallback; Qwen3.5 canary will fail loudly.
    AutoModelForImageTextToText = None


# ─── Liger Kernel: optional fused Triton kernels for Qwen3 ────────────────────
# Liger's Triton kernels are safe on a single fully-GPU-resident model, but they
# are not safe with Accelerate device_map CPU/offload. Explicit memory caps may
# force CPU/disk offload, so default Liger off when those caps are present; users
# can still force USE_LIGER=1 for a known fully GPU-resident sharded setup.
_EXPLICIT_MEMORY_CAP_REQUESTED = bool(os.environ.get("MAX_MEMORY_PER_GPU_GIB", "").strip())
_DEFAULT_USE_LIGER = "0" if _EXPLICIT_MEMORY_CAP_REQUESTED else "1"
USE_LIGER = os.environ.get("USE_LIGER", _DEFAULT_USE_LIGER) == "1"
if _EXPLICIT_MEMORY_CAP_REQUESTED and USE_LIGER:
    print("[LIGER] WARNING: USE_LIGER=1 with explicit max_memory caps; disable it if device_map offloads to CPU")
if USE_LIGER:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3()
        print("[LIGER] Qwen3 kernels patched (SwiGLU, RMSNorm, RoPE, FusedLinearCE)")
    except ImportError:
        print("[LIGER] liger-kernel not installed, skipping")
        USE_LIGER = False
    except Exception as e:
        print(f"[LIGER] Patch failed (non-fatal): {e}")
        USE_LIGER = False


_IMAGE_TEXT_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}
_TEXT_BATCH_KEYS = {"input_ids", "attention_mask", "position_ids", "labels"}
_QWEN35_FASTPATH_IMPORTS = (
    ("causal_conv1d", "causal_conv1d_fn"),
    ("causal_conv1d", "causal_conv1d_update"),
    ("fla.modules", "FusedRMSNormGated"),
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
)
# Transformers checks the import names `causal_conv1d` and `fla`. On CUDA jobs,
# the PyPI packages below provide those modules and enable the optimized Qwen3.5
# gated-delta/conv path. They are CUDA-only, so local macOS/CPU installs may fail.
_QWEN35_FASTPATH_PACKAGES = ("causal-conv1d", "flash-linear-attention")
_QWEN35_FASTPATH_INSTALL_HINT = " ".join(_QWEN35_FASTPATH_PACKAGES)


# ─── Config ──────────────────────────────────────────────────────────────────
# SDPO is scale-sensitive. This Qwen3.5 variant defaults to Qwen/Qwen3.5-9B,
# a newer ImageTextToText/ForConditionalGeneration checkpoint.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B")
SELLER_MODEL_NAME = os.environ.get("SELLER_MODEL_NAME", MODEL_NAME)
NUM_ITERS = int(os.environ.get("NUM_ITERS", "60"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "6"))
# Dense Qwen RLVR collapsed at 3e-5 in this repo. Qwen3.5 9B's 2-iter
# production-shape smoke only exercised warmup up to 1e-6, so use a cautious
# real-run default while keeping warmup, clipping, and weight decay.
LR = float(os.environ.get("LR", "3e-6"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
EPSILON = float(os.environ.get("EPSILON", "0.2"))
# True ref-free/on-policy SDPO+GRPO: no frozen reference-policy forward is
# used in the update. Keep the env var for run metadata/backward-compatible
# launch commands, but default to the paper-aligned no-KL setting.
KL_COEF = float(os.environ.get("KL_COEF", "0.0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "300"))
BUYER_TEMP = float(os.environ.get("BUYER_TEMP", "1.0"))
SELLER_TEMP = float(os.environ.get("SELLER_TEMP", "0.7"))  # paper Table 5
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/model")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1"
MODEL_DEVICE_MAP = os.environ.get("MODEL_DEVICE_MAP", "auto")
MAX_MEMORY_PER_GPU_GIB = os.environ.get("MAX_MEMORY_PER_GPU_GIB", "").strip()
GEN_BATCH_LIMIT = int(os.environ.get("GEN_BATCH_LIMIT", "128"))
NUM_INNER_EPOCHS = int(os.environ.get("NUM_INNER_EPOCHS", "1"))
NORMALIZE_ADVANTAGES = os.environ.get("NORMALIZE_ADVANTAGES", "1") == "1"
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "10"))
SEED = int(os.environ.get("SEED", "42"))
TRAIN_SPLIT_SIZE = int(os.environ.get("TRAIN_SPLIT_SIZE", "802"))
TEST_SPLIT_SIZE = int(os.environ.get("TEST_SPLIT_SIZE", "128"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "anchor-negotiation-sdpo")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "chalk") or None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_TAGS = [t.strip() for t in os.environ.get("WANDB_TAGS", "sdpo,negotiation,rlvr").split(",") if t.strip()]
WANDB_JOB_TYPE = os.environ.get("WANDB_JOB_TYPE", "train")
RUN_NAME = os.environ.get("RUN_NAME", "")
PUSH_TRAINING_SCRIPT = os.environ.get("PUSH_TRAINING_SCRIPT", "1") == "1"
QWEN35_TEXT_CANARY = os.environ.get("QWEN35_TEXT_CANARY", "1") == "1"
# Public/private reasoning control:
# - option_a: disable native Qwen thinking in the chat template and rely on the
#   explicit private Thought/Talk/Action protocol. Public text is canonicalized to
#   Talk + the first Action line.
# - option_b: enable native Qwen thinking in the chat template for Qwen3.5-style
#   models. Generated <think>...</think> blocks are retained in raw same-role
#   assistant history, but stripped before opponent visibility/action parsing.
# - off/0/false: compatibility alias for option_a behavior without extra labeling.
REASONING_MODE = os.environ.get("REASONING_MODE", "option_a").strip().lower().replace("-", "_")
ENABLE_NATIVE_THINKING = REASONING_MODE in {"option_b", "native", "native_thinking", "thinking", "qwen", "qwen_thinking"}
CHAT_TEMPLATE_ENABLE_THINKING = os.environ.get("CHAT_TEMPLATE_ENABLE_THINKING", "1" if ENABLE_NATIVE_THINKING else "0") == "1"
STRIP_NATIVE_THINKING_FROM_HISTORY = os.environ.get("STRIP_NATIVE_THINKING_FROM_HISTORY", "0" if ENABLE_NATIVE_THINKING else "1") == "1"
NATIVE_REASONING_PROTOCOL = ENABLE_NATIVE_THINKING or CHAT_TEMPLATE_ENABLE_THINKING
DEBUG_SAMPLE_BUYER_OUTPUTS = int(os.environ.get("DEBUG_SAMPLE_BUYER_OUTPUTS", "0"))
NATIVE_PUBLIC_FINALIZER = os.environ.get("NATIVE_PUBLIC_FINALIZER", "1" if NATIVE_REASONING_PROTOCOL else "0") == "1"
NATIVE_THINK_TOKENS = int(os.environ.get("NATIVE_THINK_TOKENS", str(MAX_NEW_TOKENS)))
NATIVE_FINAL_TOKENS = int(os.environ.get("NATIVE_FINAL_TOKENS", "96"))
SDPO_LAMBDA = float(os.environ.get("SDPO_LAMBDA", "0.9"))  # 1.0 = pure GRPO, 0.0 = pure SDPO
# For Qwen3.5 runs, start GRPO-heavy while the self-teacher is noisy, then
# gradually hand off to SDPO shaping. Default: 0.9 -> 0.5 by iter 20.
SDPO_LAMBDA_FINAL = float(os.environ.get("SDPO_LAMBDA_FINAL", "0.5"))
SDPO_LAMBDA_DECAY_ITERS = int(os.environ.get("SDPO_LAMBDA_DECAY_ITERS", "20"))
SDPO_FEEDBACK_MODE = os.environ.get("SDPO_FEEDBACK_MODE", "strict").lower()
SDPO_ADV_CLIP = float(os.environ.get("SDPO_ADV_CLIP", "5.0"))
SDPO_MAX_DEMO_CHARS = int(os.environ.get("SDPO_MAX_DEMO_CHARS", "1400"))
SDPO_MAX_FEEDBACK_CHARS = int(os.environ.get("SDPO_MAX_FEEDBACK_CHARS", "1800"))
# On 8B full fine-tuning, foreach AdamW briefly materializes extra tensor lists
# at optimizer.step(); disabling foreach preserves the objective and lowers peak VRAM.
ADAMW_FOREACH = os.environ.get("ADAMW_FOREACH", "0") == "1"
# Optimizer choice is an implementation detail, not a training-method change. The
# default remains exact full-parameter AdamW updates, but stores optimizer state
# on CPU to avoid the 8B A100 optimizer.step OOM seen in job 6a05acb... .
# Set OPTIMIZER=adamw_cuda for a fully CUDA AdamW attempt; keep it only if VRAM is enough.
OPTIMIZER = os.environ.get("OPTIMIZER", "adamw_cpu").lower()
# Update-path performance controls. The old implementation processed one buyer
# turn at a time and stepped CPU AdamW once per GRPO group. These defaults batch
# buyer turns into microbatches and step once per 16 groups (= once per 16-product
# production-shape iteration) while preserving full-parameter training.
UPDATE_MICROBATCH_SIZE = int(os.environ.get("UPDATE_MICROBATCH_SIZE", "4"))
OPTIM_STEP_EVERY_GROUPS = int(os.environ.get("OPTIM_STEP_EVERY_GROUPS", "16"))
UPDATE_PAD_TO_MULTIPLE_OF = int(os.environ.get("UPDATE_PAD_TO_MULTIPLE_OF", "8"))
# Same objective, lower padding waste: sort buyer-turn update examples by padded
# student/teacher sequence length before forming microbatches within each GRPO group.
UPDATE_LENGTH_BUCKETING = os.environ.get("UPDATE_LENGTH_BUCKETING", "1") == "1"
ROLLOUT_MAX_LENGTH = int(os.environ.get("ROLLOUT_MAX_LENGTH", "3072"))
UPDATE_MAX_LENGTH = int(os.environ.get("UPDATE_MAX_LENGTH", "3072"))
# Experimental SDPO/SDRO design metadata for run naming and W&B config. These do
# not alter the current token-level SDPO objective unless corresponding code is
# added/enabled in a future ablation.
DISTILLATION_LEVEL = os.environ.get("DISTILLATION_LEVEL", "token")  # token | topk-logit
TOP_K_DISTILLATION = int(os.environ.get("TOP_K_DISTILLATION", "0"))
DISTILLATION_DIVERGENCE = os.environ.get("DISTILLATION_DIVERGENCE", "token-logprob-gap")
TRUST_REGION_INTERPOLATION = os.environ.get("TRUST_REGION_INTERPOLATION", "0") == "1"
TEACHER_EMA_DECAY = os.environ.get("TEACHER_EMA_DECAY", "")
FORMAT_WARN_THRESHOLD = int(os.environ.get("FORMAT_WARN_THRESHOLD", "5"))
FORMAT_STOP_THRESHOLD = int(os.environ.get("FORMAT_STOP_THRESHOLD", "10"))
FORMAT_STOP_PATIENCE = int(os.environ.get("FORMAT_STOP_PATIENCE", "2"))
BUDGET_WARN_THRESHOLD = int(os.environ.get("BUDGET_WARN_THRESHOLD", "32"))
BUDGET_STOP_THRESHOLD = int(os.environ.get("BUDGET_STOP_THRESHOLD", "40"))
BUDGET_STOP_PATIENCE = int(os.environ.get("BUDGET_STOP_PATIENCE", "2"))
REWARD_STOP_THRESHOLD = float(os.environ.get("REWARD_STOP_THRESHOLD", "-0.25"))
REWARD_STOP_PATIENCE = int(os.environ.get("REWARD_STOP_PATIENCE", "2"))
EARLY_STOP_SAVE_CHECKPOINT = os.environ.get("EARLY_STOP_SAVE_CHECKPOINT", "1") == "1"


def _fmt_run_value(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("e-0", "e-").replace("e+0", "e+").replace(".", "p")


def _model_slug(model_name):
    slug = model_name.split("/")[-1].lower()
    slug = slug.replace("qwen3", "q3").replace("instruct", "i")
    slug = slug.replace("-2507", "2507")
    slug = re.sub(r"[^a-z0-9.]+", "-", slug).strip("-")
    slug = slug.replace("-i-", "-i")
    return slug


def _distill_slug():
    if DISTILLATION_LEVEL == "topk-logit":
        div = DISTILLATION_DIVERGENCE.lower().replace("jensen-shannon", "js").replace("jshannon", "js")
        div = re.sub(r"[^a-z0-9]+", "", div) or "kl"
        interp = "tri" if TRUST_REGION_INTERPOLATION else "notri"
        ema = f"ema{_fmt_run_value(TEACHER_EMA_DECAY)}" if TEACHER_EMA_DECAY else "noema"
        return f"topk{TOP_K_DISTILLATION}-{div}-{interp}-{ema}"
    return "tokgap"


def default_run_name():
    return (
        f"sdpo__{_model_slug(MODEL_NAME)}__l{_fmt_run_value(SDPO_LAMBDA)}__{_distill_slug()}"
        f"__i{NUM_ITERS}_b{BATCH_SIZE}xg{GROUP_SIZE}"
        f"__fb{SDPO_FEEDBACK_MODE}_clip{_fmt_run_value(SDPO_ADV_CLIP)}"
        f"__lr{_fmt_run_value(LR)}_kl{_fmt_run_value(KL_COEF)}__s{SEED}"
    )


def default_wandb_group():
    return f"sdpo__{_model_slug(MODEL_NAME)}__{_distill_slug()}__fb{SDPO_FEEDBACK_MODE}"


def active_sdpo_lambda(iteration):
    """GRPO-heavy -> SDPO-balanced schedule for Qwen3.5 experiments."""
    if SDPO_LAMBDA_DECAY_ITERS <= 0:
        return SDPO_LAMBDA_FINAL
    frac = min(max(float(iteration) / float(SDPO_LAMBDA_DECAY_ITERS), 0.0), 1.0)
    return SDPO_LAMBDA + frac * (SDPO_LAMBDA_FINAL - SDPO_LAMBDA)


def _consecutive_count(history, predicate):
    count = 0
    for row in reversed(history):
        if predicate(row):
            count += 1
        else:
            break
    return count


def evaluate_early_stop(metrics, n_episodes):
    """Return (should_stop, reasons) from recent rollout health metrics."""
    if not metrics:
        return False, []
    latest = metrics[-1]
    outcomes = latest.get("outcomes", {})
    format_errors = outcomes.get("BUYER_FORMAT_ERROR", 0)
    budget_violations = outcomes.get("BUYER_BUDGET_VIOLATION", 0)
    mean_reward = latest.get("mean_reward", 0.0)
    reasons = []

    format_bad = _consecutive_count(
        metrics,
        lambda row: row.get("outcomes", {}).get("BUYER_FORMAT_ERROR", 0) >= FORMAT_STOP_THRESHOLD,
    )
    budget_bad = _consecutive_count(
        metrics,
        lambda row: row.get("outcomes", {}).get("BUYER_BUDGET_VIOLATION", 0) >= BUDGET_STOP_THRESHOLD,
    )
    reward_bad = _consecutive_count(
        metrics,
        lambda row: row.get("mean_reward", 0.0) <= REWARD_STOP_THRESHOLD,
    )

    if FORMAT_STOP_PATIENCE > 0 and format_bad >= FORMAT_STOP_PATIENCE:
        reasons.append(
            f"format_errors={format_errors}/{n_episodes} for {format_bad} consecutive iters "
            f"(threshold={FORMAT_STOP_THRESHOLD}; next run: lower LR or slow SDPO handoff)"
        )
    if BUDGET_STOP_PATIENCE > 0 and budget_bad >= BUDGET_STOP_PATIENCE:
        reasons.append(
            f"budget_violations={budget_violations}/{n_episodes} for {budget_bad} consecutive iters "
            f"(threshold={BUDGET_STOP_THRESHOLD}; next run: strengthen budget prompt or lower LR)"
        )
    if REWARD_STOP_PATIENCE > 0 and reward_bad >= REWARD_STOP_PATIENCE:
        reasons.append(
            f"mean_reward={mean_reward:.4f} for {reward_bad} consecutive iters "
            f"(threshold={REWARD_STOP_THRESHOLD}; next run: lower LR or inspect product mix)"
        )
    return bool(reasons), reasons


random.seed(SEED)
torch.manual_seed(SEED)


def _model_load_kwargs():
    """Shared model loading kwargs."""
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": MODEL_DEVICE_MAP,
        "trust_remote_code": True,
    }
    if MAX_MEMORY_PER_GPU_GIB:
        if not torch.cuda.is_available():
            return kwargs
        n_gpu = torch.cuda.device_count()
        # Explicit memory caps are primarily for manual sharding experiments. They
        # can cause CPU/disk offload; _assert_no_cpu_offload() fails fast because
        # this full-parameter training script requires all layers on GPU.
        kwargs["max_memory"] = {i: f"{MAX_MEMORY_PER_GPU_GIB}GiB" for i in range(n_gpu)}
        kwargs["max_memory"]["cpu"] = "240GiB"
    return kwargs


def _dependency_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return "unknown"


def _probe_qwen35_fastpath():
    missing = []
    for module_name, attr_name in _QWEN35_FASTPATH_IMPORTS:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            obj = getattr(module, attr_name, None)
            if obj is None:
                missing.append(f"{module_name}.{attr_name}=None")
        except Exception as exc:
            missing.append(f"{module_name}.{attr_name}: {type(exc).__name__}: {exc}")
    versions = {pkg: _dependency_version(pkg) for pkg in _QWEN35_FASTPATH_PACKAGES}
    ok = not missing
    return ok, missing, versions


def _log_qwen35_fastpath_status():
    ok, missing, versions = _probe_qwen35_fastpath()
    version_text = ", ".join(f"{k}={v or 'missing'}" for k, v in versions.items())
    if ok:
        print(f"[QWEN3.5 FASTPATH] OK: optimized gated-delta/conv kernels import. Versions: {version_text}")
    else:
        print("[QWEN3.5 FASTPATH] MISSING: transformers will fall back to slower torch gated-delta path.")
        print(f"[QWEN3.5 FASTPATH] Versions: {version_text}")
        print(f"[QWEN3.5 FASTPATH] Install with: pip install {_QWEN35_FASTPATH_INSTALL_HINT}")
        for item in missing[:8]:
            print(f"[QWEN3.5 FASTPATH] Missing detail: {item}")
    return ok


def _config_model_type(config):
    return str(getattr(config, "model_type", "") or "")


def _is_image_text_config(config):
    return _config_model_type(config) in _IMAGE_TEXT_MODEL_TYPES and hasattr(config, "vision_config")


def _is_image_text_model(model):
    return _is_image_text_config(getattr(model, "config", None))


def _first_model_device(model):
    return next(model.parameters()).device


def _model_input_device(model):
    """Device where tokenized inputs should be placed.

    Accelerate-dispatched models can span multiple GPUs. For those, inputs must
    start on the first layer's device, not necessarily on the lm_head device.
    If the map starts on CPU, fail early: generation with CPU-dispatched input
    layers is incompatible with the CUDA/Triton path we use for training.
    """
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for key in ("model.embed_tokens", "transformer.wte", "model"):  # Qwen, GPT-like, fallback
            if key in hf_map:
                dev = torch.device(hf_map[key])
                if dev.type == "cpu":
                    raise RuntimeError(f"Input layer {key} is on CPU in hf_device_map; use larger GPU memory or lower MAX_MEMORY_PER_GPU_GIB")
                return dev
        first = next(iter(hf_map.values()))
        dev = torch.device(first)
        if dev.type == "cpu":
            raise RuntimeError("First model shard is on CPU; use larger GPU memory or lower MAX_MEMORY_PER_GPU_GIB")
        return dev
    return _first_model_device(model)


def _load_text_or_image_text_stack(model_name):
    """Load Qwen3/Qwen3.5 text-compatible model plus processor/tokenizer.

    Qwen3.5/3.6 official checkpoints are ImageTextToText wrappers with a text
    backbone. For text-only negotiation, use AutoProcessor +
    AutoModelForImageTextToText when available, while exposing the underlying
    tokenizer to the rest of the script. Older CausalLM checkpoints keep the
    original AutoTokenizer + AutoModelForCausalLM path.
    """
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if _is_image_text_config(cfg):
        if AutoModelForImageTextToText is None:
            raise ImportError(
                f"{model_name} is {_config_model_type(cfg)} and requires AutoModelForImageTextToText. "
                "Install a recent transformers version (Qwen3.5 model card uses >=4.57)."
            )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"AutoProcessor for {model_name} does not expose .tokenizer")
        model = AutoModelForImageTextToText.from_pretrained(model_name, **_model_load_kwargs())
        return model, processor, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **_model_load_kwargs())
    return model, tokenizer, tokenizer


def _text_only_batch(batch):
    return {k: v for k, v in batch.items() if k in _TEXT_BATCH_KEYS and v is not None}


def _move_batch_to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def _apply_chat_template_text(processor_or_tokenizer, messages, *, tokenize, add_generation_prompt, **kwargs):
    try:
        return processor_or_tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    except TypeError:
        # Some processor/tokenizer variants do not support enable_thinking.
        kwargs.pop("enable_thinking", None)
        return processor_or_tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def _qwen35_text_canary(model, processor_or_tokenizer, tokenizer, device, label):
    if not QWEN35_TEXT_CANARY or not _is_image_text_model(model):
        return
    print(f"[QWEN3.5 CANARY] Text-only compatibility check for {label}...")
    msgs = [{"role": "user", "content": "Reply with exactly: OK"}]
    try:
        inputs = _apply_chat_template_text(
            processor_or_tokenizer,
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        prompt = _apply_chat_template_text(
            tokenizer,
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, "items"):
        inputs = dict(inputs.items())
    elif not isinstance(inputs, dict):
        inputs = {"input_ids": inputs}
    else:
        inputs = dict(inputs)
    if "input_ids" in inputs and not torch.is_tensor(inputs["input_ids"]):
        inputs["input_ids"] = torch.as_tensor(inputs["input_ids"])
    if "attention_mask" in inputs and not torch.is_tensor(inputs["attention_mask"]):
        inputs["attention_mask"] = torch.as_tensor(inputs["attention_mask"])
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    if any(k.startswith("pixel") or "image" in k or "video" in k for k in inputs):
        raise AssertionError(f"{label} text-only canary unexpectedly produced multimodal tensors: {sorted(inputs)}")
    bad_token_ids = [
        getattr(processor_or_tokenizer, "image_token_id", None),
        getattr(processor_or_tokenizer, "video_token_id", None),
        getattr(getattr(model, "config", None), "image_token_id", None),
        getattr(getattr(model, "config", None), "video_token_id", None),
    ]
    ids = inputs["input_ids"]
    for bad in {x for x in bad_token_ids if x is not None}:
        if bool((ids == int(bad)).any().item()):
            raise AssertionError(f"{label} text-only canary contains multimodal token id {bad}")
    inputs = _move_batch_to_device(_text_only_batch(inputs), device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    if not hasattr(out, "logits") or out.logits.shape[:2] != inputs["input_ids"].shape:
        raise AssertionError(f"{label} text-only forward did not return expected logits shape")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"[QWEN3.5 CANARY] {label} OK; sample={text!r}")


def _assert_no_cpu_offload(model, name):
    """Fail fast if Accelerate placed any module on CPU/disk.

    Full fine-tuning and rollout generation require all modules on GPU. CPU/disk
    offload is especially unsafe with Triton/Liger kernels and would make dense
    optimizer updates invalid or unusably slow.
    """
    hf_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_map, dict):
        return
    bad = {k: v for k, v in hf_map.items() if str(v).startswith("cpu") or str(v).startswith("disk")}
    if bad:
        sample = list(bad.items())[:8]
        raise RuntimeError(
            f"{name} has CPU/disk-offloaded modules in hf_device_map (sample={sample}). "
            "This script does full-parameter training/generation and requires all modules on GPU. "
            "Use A100/L40S-class memory or reduce per-GPU memory caps; do not use Liger with CPU offload."
        )


# ─── Dataset: AmazonHistoryPrice ─────────────────────────────────────────────
DATASET_URL_BASE = (
    "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory"
    "/main/data/AmazonHistoryPrice/"
)
# Important: include all 18 categories. Older scripts omitted toys-games and
# video-games, producing 901 examples. The paper/JOURNAL dataset has 930.
CATEGORIES = [
    "automotive",
    "baby-products",
    "beauty",
    "books",
    "electronics",
    "health-personal-care",
    "home-kitchen",
    "industrial-scientific",
    "movies-tv",
    "music",
    "other",
    "patio-lawn-garden",
    "pet-supplies",
    "software",
    "sports-outdoors",
    "tools-home-improvement",
    "toys-games",
    "video-games",
]


def parse_price(s):
    return float(str(s).replace("$", "").replace(",", "").strip())


def load_products(seed=SEED):
    import urllib.request

    all_items = []
    category_counts = {}
    for cat in CATEGORIES:
        url = DATASET_URL_BASE + f"{cat}.json"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                items = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  [WARN] Skip {cat}: {e}")
            continue
        n_valid = 0
        for idx, it in enumerate(items):
            try:
                lp = parse_price(it.get("list_price", "0"))
                cost = parse_price(it.get("lowest_price", "0"))
                if lp <= 0 or cost <= 0:
                    continue
                budget = round(lp * 0.8, 2)
                all_items.append(
                    {
                        "codename": f"{cat}_{idx}",
                        "title": it.get("title", "")[:120],
                        "description": it.get("description", "")[:200],
                        "features": it.get("features", "")[:300],
                        "current_price": parse_price(it.get("current_price", lp)),
                        "average_price": parse_price(it.get("average_price", lp)),
                        "highest_price": parse_price(it.get("highest_price", lp)),
                        "category": cat,
                        "list_price": lp,
                        "cost": cost,
                        "budget": budget,
                        "mi": budget > cost,
                    }
                )
                n_valid += 1
            except Exception:
                continue
        category_counts[cat] = n_valid

    random.seed(seed)
    random.shuffle(all_items)
    if len(all_items) >= TRAIN_SPLIT_SIZE + TEST_SPLIT_SIZE:
        train = all_items[:TRAIN_SPLIT_SIZE]
        test = all_items[TRAIN_SPLIT_SIZE : TRAIN_SPLIT_SIZE + TEST_SPLIT_SIZE]
    else:
        split = int(len(all_items) * 0.8623655913978494)  # 802/930
        train, test = all_items[:split], all_items[split:]

    mi_total = sum(1 for p in all_items if p["mi"])
    mi_train = sum(1 for p in train if p["mi"])
    mi_test = sum(1 for p in test if p["mi"])
    print(
        f"[DATA] Products={len(all_items)} train={len(train)} test={len(test)} "
        f"MI={mi_total} CI={len(all_items)-mi_total}",
        flush=True,
    )
    print(f"[DATA] Train MI={mi_train} CI={len(train)-mi_train}; Test MI={mi_test} CI={len(test)-mi_test}")
    print(f"[DATA] Category counts: {category_counts}")
    return train, test


# ─── Prompts (paper Appendix C, adapted to Thought/Talk/Action format) ───────
# Qwen3.5 native-thinking mode already creates a private <think>...</think>
# scratchpad in the assistant completion. In that mode, do not ask the model to
# emit a public `Thought:` field as well; it wastes tokens and confused the tiny
# smoke into spending all 120 tokens inside private reasoning. Instead, use the
# full native thinking budget privately and require only public Talk + Action
# after </think>. The legacy explicit protocol remains the default for Option A.
BUYER_SYSTEM_EXPLICIT = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product, instead of the title.
You can only buy things that cost less than your budget. Never offer above your private budget. Never accept or DEAL above your private budget; if no legal deal is possible, choose [QUIT].
Again, try to make deal with a price as low as possible. That is, your goal is to spend as little money as possible, not just reaching your budget.

Protocol compliance is mandatory. Every reply must contain exactly one Thought line, exactly one Talk line, and exactly one Action line. The final Action line must be parseable and must use one of the allowed action tags exactly.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the seller. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of "[BUY],[REJECT],[DEAL],[QUIT]".
1. '[BUY] $M (N codename_1)' if you wish to offer the seller $M to purchase all N items of the product with the codename "codename_1".
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M (N codename_1)' if you finally accept on a former offer proposed by the seller. $M (N codename_1) is an exact copy of seller's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL] $M' before seller's action '[SELL] $M'. Your first action should be '[BUY] $M (N codename_1)' or '[REJECT]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the seller's previous offer '[SELL] $M (N codename_1)' and only if $M is within your budget. Otherwise, you always choose from '[BUY]', '[REJECT]' and '[QUIT]'.
Do not output any extra text before Thought or after the Action line. Do not output multiple Action lines.

Your reply should STRICTLY follow this format (not following the format will directly lead to failure), for example:
Thought: I'm a buyer and I want to bargain. The listing price of codename "apple_1" is $15, which is too expensive, so I try to buy an apple for $10.
Talk: Hello, I'm tight on budget. can you sell it for 10$?
Action: [BUY] $10 (1x apple_1)"""

BUYER_SYSTEM_NATIVE = """You are a buyer looking forward to buying things on your Shopping List from me, the seller.
You have access to the seller's Inventory List and you can bargain about the prices.
Your task is to bargain with the seller and reach a deal with the price as low as possible in limited turns.
You can only buy things on the Shopping List in the limited quantity. Use the codename of the product, instead of the title.
You can only buy things that cost less than your budget. Never offer above your private budget. Never accept or DEAL above your private budget; if no legal deal is possible, choose [QUIT].
Again, try to make deal with a price as low as possible. That is, your goal is to spend as little money as possible, not just reaching your budget.

Use your native private thinking block to reason carefully about budget, seller offers, strategy, and exact action legality. That private thinking is hidden from the seller.
After the private thinking block, output ONLY the public message below. Do not include a public Thought line. Do not reveal your budget. Do not output extra text after the Action line.

After </think>, the final public answer must be exactly two lines:
Talk: one short seller-facing sentence in your own words
Action: one concrete legal buyer action

Do not write labels, markdown, explanations, examples, placeholders, or quoted actions in the public answer. Never output strings like "short talk", "Talk content here", "One of", "[Action]", "[ACTION]", or "must be". For BUY or DEAL, the Action line must include a concrete price and the current product codename.

Legal buyer actions:
1. '[BUY] $M (N codename_1)' if you wish to offer the seller $M to purchase all N items of the product with the codename "codename_1".
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the seller.
3. '[DEAL] $M (N codename_1)' if you finally accept on a former offer proposed by the seller. $M (N codename_1) is an exact copy of seller's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL] $M' before seller's action '[SELL] $M'. Your first action should be '[BUY] $M (N codename_1)' or '[REJECT]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the seller's previous offer '[SELL] $M (N codename_1)' and only if $M is within your budget. Otherwise, you always choose from '[BUY]', '[REJECT]' and '[QUIT]'.

Example final public reply after your hidden thinking:
Talk: Hello, I'm tight on budget. can you sell it for 10$?
Action: [BUY] $10 (1x apple_1)"""

SELLER_SYSTEM_EXPLICIT = """You are a seller looking forward to selling things on your Inventory List to me, the buyer.
Your task is to bargain with the buyer and reach a deal with the price as high as possible in limited turns.
You can only sell things that are on the Inventory List. Use the codename of the product, instead of the title.
You have access to private information: the cost price of each product in the Inventory List, and do not disclose the real cost to the buyer.
You should only agree on a deal when the selling price is higher than the cost, otherwise, you should quit negotiating.

Your Reply should include 3 parts: Thought, Talk, and Action.
Thought: your inner strategic thinking of this bargaining session;
Talk: short talk that you are going to say to the buyer. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation;
Action: one of the limited actions that define the real intention of your Talk. The type of your Action must be one of "[SELL],[REJECT],[DEAL],[QUIT]".
1. '[SELL] $M (N codename_1)' if you want to propose selling N items of the product with the codename "codename_1" to the buyer for the total price of $M.
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the buyer.
3. '[DEAL] $M (N codename_1)' if you finally agree on a former offer proposed by the buyer, and sell N items of the product with the codename "codename_1" to the buyer for the total price of $M. $M (N codename_1) is an exact copy of buyer's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL]' before buyer's action '[BUY]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the buyer's previous offer '[BUY] $M (N codename_1)'. Otherwise, you always choose from '[SELL]', '[REJECT]' and '[QUIT]'.

Your reply should strictly follow this format, for example:
Thought: I'm a seller, so I must sell the product with codename "apple_1" higher than its cost.
Talk: blah, blah...
Action: [SELL] $15 (1x apple_1)"""

SELLER_SYSTEM_NATIVE = """You are a seller looking forward to selling things on your Inventory List to me, the buyer.
Your task is to bargain with the buyer and reach a deal with the price as high as possible in limited turns.
You can only sell things that are on the Inventory List. Use the codename of the product, instead of the title.
You have access to private information: the cost price of each product in the Inventory List, and do not disclose the real cost to the buyer.
You should only agree on a deal when the selling price is higher than the cost, otherwise, you should quit negotiating.

Use your native private thinking block to reason carefully about your cost, the buyer offer, strategy, and exact action legality. That private thinking is hidden from the buyer.
After the private thinking block, output ONLY the public message below. Do not include a public Thought line. Do not reveal the cost. Do not output extra text after the Action line.

After </think>, the final public answer must be exactly two lines:
Talk: one short buyer-facing sentence in your own words
Action: one concrete legal seller action

Do not write labels, markdown, explanations, examples, placeholders, or quoted actions in the public answer. Never output strings like "short talk", "Talk content here", "One of", "[Action]", "[ACTION]", or "must be". For SELL or DEAL, the Action line must include a concrete price and the current product codename.

Legal seller actions:
1. '[SELL] $M (N codename_1)' if you want to propose selling N items of the product with the codename "codename_1" to the buyer for the total price of $M.
2. '[REJECT]' if you choose to reject the other side's offer and await a new offer from the buyer.
3. '[DEAL] $M (N codename_1)' if you finally agree on a former offer proposed by the buyer, and sell N items of the product with the codename "codename_1" to the buyer for the total price of $M. $M (N codename_1) is an exact copy of buyer's previous offer. You should not use this action to propose a new price. This action will immediately end the conversation and close the deal.
4. '[QUIT]' if you believe that a mutually acceptable deal cannot be reached in limited turns. This action will immediately end the conversation.
You shouldn't choose action '[DEAL]' before buyer's action '[BUY]'.
'[DEAL] $M (N codename_1)' can only be chosen to accept the buyer's previous offer '[BUY] $M (N codename_1)'. Otherwise, you always choose from '[SELL]', '[REJECT]' and '[QUIT]'.

Example final public reply after your hidden thinking:
Talk: I can offer it for $15.
Action: [SELL] $15 (1x apple_1)"""

BUYER_SYSTEM = BUYER_SYSTEM_NATIVE if NATIVE_REASONING_PROTOCOL else BUYER_SYSTEM_EXPLICIT
SELLER_SYSTEM = SELLER_SYSTEM_NATIVE if NATIVE_REASONING_PROTOCOL else SELLER_SYSTEM_EXPLICIT


def build_buyer_prompt(product):
    inv = (
        f"Inventory List\n"
        f"- codename: {product['codename']}\n"
        f"  title: {product['title']}\n"
        f"  description: {product['description']}\n"
        f"  features: {product.get('features', '')}\n"
        f"  category: {product['category']}\n"
        f"  list_price: ${product['list_price']:.2f}\n"
        f"  current_price: ${product.get('current_price', product['list_price']):.2f}\n"
        f"  average_price: ${product.get('average_price', product['list_price']):.2f}\n"
        f"  highest_price: ${product.get('highest_price', product['list_price']):.2f}"
    )
    need = (
        f"Shopping List\n"
        f"- codename: {product['codename']}\n"
        f"  quantity: 1\n"
        f"  budget_limit: ${product['budget']:.2f}"
    )
    user = (
        f"{inv}\n\n{need}\n\n"
        f"Now, I play the role of seller and you play the role of buyer. "
        f"We are going to negotiate based on the Inventory List in {MAX_TURNS} turns."
    )
    return [
        {"role": "system", "content": BUYER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_seller_prompt(product):
    inv = (
        f"Inventory List\n"
        f"- codename: {product['codename']}\n"
        f"  title: {product['title']}\n"
        f"  description: {product['description']}\n"
        f"  features: {product.get('features', '')}\n"
        f"  category: {product['category']}\n"
        f"  list_price: ${product['list_price']:.2f}\n"
        f"  current_price: ${product.get('current_price', product['list_price']):.2f}\n"
        f"  average_price: ${product.get('average_price', product['list_price']):.2f}\n"
        f"  highest_price: ${product.get('highest_price', product['list_price']):.2f}\n"
        f"  cost_price (private): ${product['cost']:.2f}"
    )
    user = (
        f"{inv}\n\n"
        f"Now, I play the role of buyer and you play the role of seller. "
        f"We are going to negotiate based on the Inventory List in {MAX_TURNS} turns."
    )
    return [
        {"role": "system", "content": SELLER_SYSTEM},
        {"role": "user", "content": user},
    ]


# ─── Action extraction + hidden scratchpad stripping ─────────────────────────
# Capture numeric prices without swallowing trailing sentence punctuation such as
# "$25.00.". The old [\d,.]+ pattern could include the final period and crash
# float() during long rollouts.
PRICE_PATTERN = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
ACTION_PATTERN = r"\[(BUY|SELL|DEAL|REJECT|QUIT)\](?:\s*\$" + PRICE_PATTERN + r")?(?:\s*\(([^)]*)\))?"
ACTION_RE = re.compile(ACTION_PATTERN, re.IGNORECASE)
ACTION_LINE_RE = re.compile(r"(?:^|\n)\s*Action\s*:\s*" + ACTION_PATTERN, re.IGNORECASE)
QWEN_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
QWEN_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
QWEN_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_qwen_native_thinking(text):
    """Remove Qwen3 native <think>...</think> content from visible text.

    Qwen3 thinking mode emits private reasoning wrapped in native think tags. That
    content must never be passed to the opponent or used as the parsed economic
    action. If an opening tag is unterminated, keep only content after the first
    public protocol marker (Thought/Talk/Action), otherwise drop the tail.
    """
    text = text or ""
    had_native_marker = bool(QWEN_THINK_OPEN_RE.search(text) or QWEN_THINK_CLOSE_RE.search(text))
    text = QWEN_THINK_BLOCK_RE.sub("", text)

    closes = list(QWEN_THINK_CLOSE_RE.finditer(text))
    if closes and not QWEN_THINK_OPEN_RE.search(text):
        # Decoded completions from an open native-thinking prompt often contain the
        # private body followed by </think> but not the opening <think>. Preserve
        # only the public suffix after the final close tag.
        text = text[closes[-1].end() :]

    m = QWEN_THINK_OPEN_RE.search(text)
    if m:
        tail = text[m.end() :]
        public = re.search(r"(?:^|\n)\s*(?:Thought|Talk|Action)\s*:", tail, re.IGNORECASE)
        text = text[: m.start()] + (tail[public.start() :] if public else "")
    return text.strip()


def _parse_action_match(m):
    ps = m.group(2)
    price = float(ps.replace(",", "")) if ps else None
    return {"type": m.group(1).upper(), "price": price, "objects": m.group(3)}


def extract_action(text):
    # Parse only the canonical public surface form. Raw generations may contain
    # useful private Thought/self-correction after the first public Action; those
    # tails are kept in the acting model's private history but must not affect the
    # economic action or leak to the counterparty.
    public_text = strip_thought(text or "")

    line_matches = list(ACTION_LINE_RE.finditer(public_text))
    if line_matches:
        return _parse_action_match(line_matches[-1])

    if not re.search(r"(?:^|\n)\s*Action\s*:", public_text or "", re.IGNORECASE):
        matches = list(ACTION_RE.finditer(public_text or ""))
        m = matches[-1] if matches else None
        if m:
            return _parse_action_match(m)
    return {"type": "UNKNOWN", "price": None, "objects": None}


def replace_final_action(text, action_type, price, product):
    """Replace the first public structured action after environment regulation.

    Keep any raw private Thought tail for the acting model, but modify only the
    canonical public Action that downstream parsing/visibility will use.
    """
    replacement = f"[{action_type}] ${price:.2f} (1x {product['codename']})"
    text = strip_qwen_native_thinking(text or "")

    talk_match = re.search(r"(?:^|\n)\s*Talk\s*:", text, re.IGNORECASE)
    action_match = re.search(r"(?:^|\n)\s*Action\s*:", text, re.IGNORECASE)
    thought_match = re.search(r"(?:^|\n)\s*Thought\s*:", text, re.IGNORECASE)
    if talk_match:
        public_start = talk_match.start()
    elif action_match and thought_match and thought_match.start() < action_match.start():
        public_start = action_match.start()
    else:
        public_start = 0

    public_tail = text[public_start:]
    line_match = ACTION_LINE_RE.search(public_tail)
    if line_match:
        old_match = list(ACTION_RE.finditer(line_match.group(0)))[-1]
        start = public_start + line_match.start() + old_match.start()
        end = public_start + line_match.start() + old_match.end()
        return text[:start] + replacement + text[end:]

    visible = strip_thought(text)
    matches = list(ACTION_RE.finditer(visible))
    if not matches:
        return text.rstrip() + f"\nAction: {replacement}"
    old = matches[0].group(0)
    idx = text.find(old, public_start)
    if idx < 0:
        return text.rstrip() + f"\nAction: {replacement}"
    return text[:idx] + replacement + text[idx + len(old) :]


def canonical_public_message(text):
    """Return the counterparty-visible Talk/Action surface form.

    Option A: preserve the raw generation in the acting model's private history,
    but canonicalize anything shown to the opponent. The visible message starts at
    the first public Talk: marker when available, drops private Thought: content,
    and hard-truncates immediately after the first complete Action: line. This
    handles reasoning models that produce a valid public action and then continue
    with a second private Thought/self-correction block.
    """
    raw_text = text or ""
    had_native_marker = bool(QWEN_THINK_OPEN_RE.search(raw_text) or QWEN_THINK_CLOSE_RE.search(raw_text))
    text = strip_qwen_native_thinking(raw_text)

    talk_match = re.search(r"(?:^|\n)\s*Talk\s*:", text, re.IGNORECASE)
    action_match = re.search(r"(?:^|\n)\s*Action\s*:", text, re.IGNORECASE)
    thought_match = re.search(r"(?:^|\n)\s*Thought\s*:", text, re.IGNORECASE)

    if talk_match:
        public = text[talk_match.start() :]
    elif action_match and thought_match and thought_match.start() < action_match.start():
        public = text[action_match.start() :]
    elif thought_match:
        return ""
    elif NATIVE_REASONING_PROTOCOL and CHAT_TEMPLATE_ENABLE_THINKING and not had_native_marker and "<|im_start|>" not in text:
        return ""
    else:
        public = text

    public = re.sub(r"\[(Action|ACTION)\]", "", public)
    public = re.sub(r"\[ACTION\]", "", public)

    action_line = ACTION_LINE_RE.search(public)
    if action_line:
        # Keep the whole public action line (including any harmless trailing text
        # before its newline), but discard any subsequent hidden/self-correction
        # content. This prevents private Thought tails after a valid Action from
        # leaking to the counterparty while preserving the raw text elsewhere.
        line_end = public.find("\n", action_line.end())
        public = public[: line_end if line_end >= 0 else len(public)]
    else:
        # No parseable structured Action line: still prevent post-public hidden
        # scratchpad leaks if the model starts another Thought block later.
        leaked_thought = re.search(r"\n\s*Thought\s*:", public, re.IGNORECASE)
        if leaked_thought:
            public = public[: leaked_thought.start()]

    result = public.strip()
    _assert_strip_thought_complete(result, text)
    return result


def strip_thought(text):
    """Remove hidden scratchpads, keeping only canonical Talk + Action text."""
    return canonical_public_message(text)


def _assert_strip_thought_complete(stripped_text, original_text):
    has_structured_thought = bool(re.search(r"(?:^|\n)\s*Thought\s*:", original_text or ""))
    leaked_thought = bool(re.search(r"(?:^|\n)\s*Thought\s*:", stripped_text or ""))
    leaked_qwen_think = bool(QWEN_THINK_OPEN_RE.search(stripped_text or "") or QWEN_THINK_CLOSE_RE.search(stripped_text or ""))
    if (has_structured_thought and leaked_thought) or leaked_qwen_think:
        raise AssertionError(
            "strip_thought() INCOMPLETE: hidden reasoning block survived. "
            f"Original={original_text[:200]!r}; Stripped={stripped_text[:200]!r}"
        )


def _assert_no_private_info_leak(prompt_text, product, role):
    """Crash on clear counterparty-private-info leakage.

    Avoids regex heuristics that caused false positives in JOURNAL v10.4.
    """
    budget_str = f"${product['budget']:.2f}"
    if role == "buyer":
        if "cost_price" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: buyer prompt contains seller cost field. Product={product['codename']}"
            )
    elif role == "seller":
        if "Shopping List" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: seller prompt contains buyer Shopping List. Product={product['codename']}"
            )
        if f"budget_limit: {budget_str}" in prompt_text:
            raise AssertionError(
                f"INFORMATION LEAK: seller prompt contains buyer budget_limit={budget_str}. "
                f"Product={product['codename']}"
            )


# ─── Reward + seller regulation ──────────────────────────────────────────────
def regulate_seller(seller_action, buyer_price, product):
    """Regulate seller per paper: prevent below-cost accepts/proposals."""
    cost = product["cost"]
    at = seller_action["type"]
    price = seller_action["price"]

    if at == "UNKNOWN":
        return None, True, "SELLER_FORMAT_ERROR"
    if at == "QUIT":
        return None, True, "SELLER_QUIT"
    if at == "DEAL":
        if buyer_price is None:
            return None, True, "NO_PRIOR_BUYER_OFFER"
        if buyer_price < cost:
            return None, True, "SELLER_CANNOT_ACCEPT_BELOW_COST"
        return buyer_price, True, "DEAL_SELLER_ACCEPTS"
    if at == "SELL":
        if price is None:
            return None, True, "NO_PRICE_IN_SELL"
        if price < cost:
            price = round(cost * 1.05, 2)
        return price, False, "SELL"
    if at == "REJECT":
        return None, False, "REJECT"
    return None, True, f"UNEXPECTED_{at}"


def compute_buyer_reward(final_price, budget, cost, outcome):
    """Buyer reward per paper Eq. 1, with terminal penalties."""
    if "FORMAT_ERROR" in outcome or "UNEXPECTED" in outcome:
        return -1.0
    if outcome in {"BUYER_BUDGET_VIOLATION", "BUYER_DEAL_INVALID_SELLER_OFFER", "BUYER_DEAL_PRICE_MISMATCH"}:
        return -1.0
    if final_price is None:
        return 0.0
    if final_price > budget:
        return -1.0
    denom = abs(budget - cost)
    if denom < 1e-6:
        return 0.0
    r = (budget - final_price) / denom
    return max(-1.0, min(1.0, r))


# ─── Batched generation ──────────────────────────────────────────────────────
def _append_native_public_finalizer(prompt_text, decoded_thinking):
    """Close the native thinking block and ask for the public Talk/Action only.

    Qwen3.5 often uses the entire native-thinking budget for useful private
    reasoning and does not reach </think> by itself. Rather than truncating that
    reasoning or training on format errors, use a second short decode after the
    same hidden scratchpad that explicitly starts the public answer. This mirrors
    how an interactive user would let the model finish thinking, then require the
    final answer in the task protocol.
    """
    body = (decoded_thinking or "").strip()
    if QWEN_THINK_CLOSE_RE.search(body):
        prefix = prompt_text + body
        if not prefix.rstrip().endswith("\n"):
            prefix += "\n"
    else:
        prefix = prompt_text + body.rstrip() + "\n</think>\n\n"
    # Important: this is still the same assistant turn, not a new user/system
    # instruction. Only close the hidden scratchpad and prefill the public marker;
    # the task prompt already tells the model which actions are legal. Adding more
    # prose here made Qwen copy schema text such as "Action: [Action]".
    prefix += "Talk:"
    return prefix


@torch.no_grad()
def generate_batched(model, tokenizer, prompts_text_list, max_new, temp, device):
    """Generate completions for a list of prompts using sub-batched HF generate."""
    if not prompts_text_list:
        return []

    all_results = []
    for batch_start in range(0, len(prompts_text_list), GEN_BATCH_LIMIT):
        batch_prompts = prompts_text_list[batch_start : batch_start + GEN_BATCH_LIMIT]
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ROLLOUT_MAX_LENGTH,
        )
        inputs = _move_batch_to_device(_text_only_batch(inputs), device)
        tokenizer.padding_side = orig_side

        think_budget = max_new
        if NATIVE_PUBLIC_FINALIZER and NATIVE_REASONING_PROTOCOL and CHAT_TEMPLATE_ENABLE_THINKING:
            think_budget = max(1, min(max_new, NATIVE_THINK_TOKENS))

        output_ids = model.generate(
            **inputs,
            max_new_tokens=think_budget,
            do_sample=True,
            temperature=max(temp, 0.01),
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        decoded_batch = []
        for i in range(len(batch_prompts)):
            gen_tokens = output_ids[i][prompt_len:]
            gen_tokens = gen_tokens[gen_tokens != tokenizer.pad_token_id]
            # Option A stores explicit Thought/Talk/Action only and strips any
            # unexpected native Qwen <think> blocks from private history. Option B
            # intentionally enables native thinking; keep those raw blocks in the
            # acting model's same-role assistant history while strip_thought() still
            # removes them before opponent visibility and action parsing.
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            decoded_batch.append(decoded)

        if NATIVE_PUBLIC_FINALIZER and NATIVE_REASONING_PROTOCOL and CHAT_TEMPLATE_ENABLE_THINKING:
            unfinished = [i for i, decoded in enumerate(decoded_batch) if not ACTION_LINE_RE.search(strip_qwen_native_thinking(decoded))]
            if unfinished:
                finalizer_prompts = [_append_native_public_finalizer(batch_prompts[i], decoded_batch[i]) for i in unfinished]
                orig_side = tokenizer.padding_side
                tokenizer.padding_side = "left"
                fin_inputs = tokenizer(
                    finalizer_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=ROLLOUT_MAX_LENGTH,
                )
                fin_inputs = _move_batch_to_device(_text_only_batch(fin_inputs), device)
                tokenizer.padding_side = orig_side
                fin_output_ids = model.generate(
                    **fin_inputs,
                    max_new_tokens=max(1, NATIVE_FINAL_TOKENS),
                    do_sample=True,
                    temperature=max(temp, 0.01),
                    top_p=1.0,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                fin_prompt_len = fin_inputs["input_ids"].shape[1]
                for local_i, orig_i in enumerate(unfinished):
                    fin_tokens = fin_output_ids[local_i][fin_prompt_len:]
                    fin_tokens = fin_tokens[fin_tokens != tokenizer.pad_token_id]
                    fin_decoded = tokenizer.decode(fin_tokens, skip_special_tokens=True)
                    decoded_batch[orig_i] = (
                        (decoded_batch[orig_i] or "").rstrip()
                        + "\n</think>\n\nTalk:"
                        + fin_decoded.lstrip()
                    )

        for decoded in decoded_batch:
            # The native-thinking prompt pre-fills an opening '<think>\n' inside
            # the prompt. Decoded new tokens may therefore contain only the body
            # plus a closing tag. Reattach the opening tag so private histories and
            # metrics faithfully record a complete native-think block.
            if CHAT_TEMPLATE_ENABLE_THINKING and QWEN_THINK_CLOSE_RE.search(decoded) and not QWEN_THINK_OPEN_RE.search(decoded):
                decoded = "<think>\n" + decoded
            if STRIP_NATIVE_THINKING_FROM_HISTORY:
                decoded = strip_qwen_native_thinking(decoded)
            all_results.append(decoded)
    return all_results


# ─── Episode state/data ──────────────────────────────────────────────────────
@dataclass
class EpisodeState:
    product: dict
    idx: int
    buyer_texts: List[str] = field(default_factory=list)
    seller_texts: List[str] = field(default_factory=list)
    all_turns: List[Tuple[str, str]] = field(default_factory=list)
    final_price: Optional[float] = None
    outcome: str = "TIMEOUT"
    done: bool = False
    last_buyer_price: Optional[float] = None


@dataclass
class Episode:
    product: dict
    turns: List[Tuple[str, str]]  # [("buyer"|"seller", text), ...]
    final_price: Optional[float]
    reward: float
    num_turns: int
    outcome: str
    first_offer_price: Optional[float]
    budget_violations: int


def run_episodes_batched(buyer_model, seller_model, tokenizer, products_expanded, device, seller_tokenizer=None):
    """Run buyer-only negotiation episodes with frozen seller, batched per turn."""
    seller_tokenizer = seller_tokenizer or tokenizer
    states = [EpisodeState(product=p, idx=i) for i, p in enumerate(products_expanded)]

    for turn_round in range(MAX_TURNS):
        active_buyer = [s for s in states if not s.done]
        if not active_buyer:
            break

        buyer_prompts = []
        for s in active_buyer:
            msgs = build_buyer_prompt(s.product)
            for bt, st in zip(s.buyer_texts, s.seller_texts):
                msgs.append({"role": "assistant", "content": bt})  # own Thought kept
                msgs.append({"role": "user", "content": strip_thought(st)})
            prompt_text = _apply_chat_template_text(
                tokenizer, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=CHAT_TEMPLATE_ENABLE_THINKING
            )
            _assert_no_private_info_leak(prompt_text, s.product, "buyer")
            buyer_prompts.append(prompt_text)

        buyer_texts = generate_batched(buyer_model, tokenizer, buyer_prompts, MAX_NEW_TOKENS, BUYER_TEMP, device)

        still_active_for_seller = []
        for idx, (s, b_text) in enumerate(zip(active_buyer, buyer_texts)):
            if DEBUG_SAMPLE_BUYER_OUTPUTS > 0 and idx < DEBUG_SAMPLE_BUYER_OUTPUTS:
                print(f"  [DEBUG buyer raw turn={turn_round} ep={s.idx}] {b_text[:1600]!r}", flush=True)
                print(f"  [DEBUG buyer public turn={turn_round} ep={s.idx}] {strip_thought(b_text)[:500]!r}", flush=True)
            b_act = extract_action(b_text)
            s.buyer_texts.append(b_text)
            s.all_turns.append(("buyer", b_text))

            if b_act["type"] == "QUIT":
                s.outcome = "BUYER_QUIT"
                s.done = True
            elif b_act["type"] == "UNKNOWN":
                s.outcome = "BUYER_FORMAT_ERROR"
                s.done = True
            elif b_act["type"] == "DEAL":
                if not s.seller_texts:
                    s.outcome = "BUYER_DEAL_NO_SELLER_OFFER"
                    s.done = True
                else:
                    last_s_act = extract_action(s.seller_texts[-1])
                    if last_s_act["type"] != "SELL" or last_s_act.get("price") is None:
                        s.outcome = "BUYER_DEAL_INVALID_SELLER_OFFER"
                        s.done = True
                    elif b_act.get("price") is not None and abs(b_act["price"] - last_s_act["price"]) > 0.01:
                        s.outcome = "BUYER_DEAL_PRICE_MISMATCH"
                        s.done = True
                    else:
                        s.final_price = last_s_act["price"]
                        s.outcome = "DEAL_BUYER_ACCEPTS"
                        s.done = True
            elif b_act["type"] == "BUY":
                b_price = b_act["price"]
                if b_price is not None and b_price > s.product["budget"]:
                    s.outcome = "BUYER_BUDGET_VIOLATION"
                    s.done = True
                else:
                    s.last_buyer_price = b_price
                    still_active_for_seller.append(s)
            elif b_act["type"] == "REJECT":
                s.last_buyer_price = None
                still_active_for_seller.append(s)
            else:
                s.outcome = f"UNEXPECTED_{b_act['type']}"
                s.done = True

        if not still_active_for_seller:
            continue

        seller_prompts = []
        for s in still_active_for_seller:
            msgs = build_seller_prompt(s.product)
            for bt, st in zip(s.buyer_texts, s.seller_texts):
                msgs.append({"role": "user", "content": strip_thought(bt)})
                msgs.append({"role": "assistant", "content": st})  # seller own Thought kept
            if len(s.buyer_texts) > len(s.seller_texts):
                msgs.append({"role": "user", "content": strip_thought(s.buyer_texts[-1])})
            prompt_text = _apply_chat_template_text(
                seller_tokenizer, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=CHAT_TEMPLATE_ENABLE_THINKING
            )
            _assert_no_private_info_leak(prompt_text, s.product, "seller")
            seller_prompts.append(prompt_text)

        seller_texts = generate_batched(seller_model, seller_tokenizer, seller_prompts, MAX_NEW_TOKENS, SELLER_TEMP, device)

        for s, s_text in zip(still_active_for_seller, seller_texts):
            s_act = extract_action(s_text)
            r_price, done, reason = regulate_seller(s_act, s.last_buyer_price, s.product)
            if reason == "SELL" and r_price is not None and s_act.get("price") != r_price:
                # The regulated seller environment intercepts below-cost proposals.
                # Update the visible Action so future buyer context and reward parsing
                # use the valid regulated price, not the hallucinated below-cost one.
                s_text = replace_final_action(s_text, "SELL", r_price, s.product)
                s_act = extract_action(s_text)

            s.seller_texts.append(s_text)
            s.all_turns.append(("seller", s_text))

            if done:
                if reason == "DEAL_SELLER_ACCEPTS":
                    s.final_price = r_price
                s.outcome = reason
                s.done = True

    episodes = []
    for s in states:
        first_offer = None
        budget_violations = 0
        for role, text in s.all_turns:
            if role != "buyer":
                continue
            act = extract_action(text)
            if act["type"] == "BUY" and act["price"] is not None:
                if first_offer is None:
                    first_offer = act["price"]
                if act["price"] > s.product["budget"]:
                    budget_violations += 1
        reward = compute_buyer_reward(s.final_price, s.product["budget"], s.product["cost"], s.outcome)
        episodes.append(
            Episode(
                product=s.product,
                turns=s.all_turns,
                final_price=s.final_price,
                reward=reward,
                num_turns=len(s.all_turns),
                outcome=s.outcome,
                first_offer_price=first_offer,
                budget_violations=budget_violations,
            )
        )
    return episodes


# ─── Log-probs and SDPO+GRPO buyer update ────────────────────────────────────
def _token_logprobs(model, input_ids, attention_mask):
    """Per-token log-probs using gather + logsumexp, avoiding full softmax tensor."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if not hasattr(out, "logits"):
        raise RuntimeError(f"Model forward returned {type(out)} without .logits")
    logits = out.logits[:, :-1, :]
    target = input_ids[:, 1:].unsqueeze(-1)
    target_logit = torch.gather(logits, 2, target).squeeze(-1)
    log_z = torch.logsumexp(logits, dim=-1)
    return target_logit - log_z


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_memory_stats_all_gpus():
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)


def _peak_memory_allocated_all_gpus_gb():
    if not torch.cuda.is_available():
        return 0.0, []
    per_gpu = [torch.cuda.max_memory_allocated(idx) / 1e9 for idx in range(torch.cuda.device_count())]
    return max(per_gpu) if per_gpu else 0.0, per_gpu


def _peak_memory_reserved_all_gpus_gb():
    if not torch.cuda.is_available():
        return 0.0, []
    per_gpu = [torch.cuda.max_memory_reserved(idx) / 1e9 for idx in range(torch.cuda.device_count())]
    return max(per_gpu) if per_gpu else 0.0, per_gpu


def _memory_allocated_all_gpus_gb():
    if not torch.cuda.is_available():
        return 0.0, []
    per_gpu = [torch.cuda.memory_allocated(idx) / 1e9 for idx in range(torch.cuda.device_count())]
    return sum(per_gpu), per_gpu


def _memory_reserved_all_gpus_gb():
    if not torch.cuda.is_available():
        return 0.0, []
    per_gpu = [torch.cuda.memory_reserved(idx) / 1e9 for idx in range(torch.cuda.device_count())]
    return sum(per_gpu), per_gpu


def _fmt_gpu_gb(values):
    return "[" + ", ".join(f"{v:.1f}" for v in values) + "]"


def _timer_start():
    _sync_cuda()
    return time.perf_counter()


def _timer_add(timers, key, start):
    _sync_cuda()
    timers[key] = timers.get(key, 0.0) + (time.perf_counter() - start)


def _chunked(seq, size):
    size = max(int(size), 1)
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _norm_advantages(t):
    if t.numel() < 2 or not NORMALIZE_ADVANTAGES:
        return t
    return (t - t.mean()) / (t.std() + 1e-8)


def build_buyer_turn_prompt(ep, turn_idx):
    """Reconstruct buyer prompt for ep.turns[turn_idx]."""
    role, _ = ep.turns[turn_idx]
    assert role == "buyer"
    prompt_msgs = build_buyer_prompt(ep.product)
    for j in range(turn_idx):
        prev_role, prev_text = ep.turns[j]
        if prev_role == "buyer":
            prompt_msgs.append({"role": "assistant", "content": prev_text})
        else:
            prompt_msgs.append({"role": "user", "content": strip_thought(prev_text)})
    return prompt_msgs


def _public_transcript(ep, max_chars=SDPO_MAX_DEMO_CHARS):
    """Public Talk/Action-only transcript for feedback demos."""
    lines = []
    for role, text in ep.turns:
        speaker = "Buyer" if role == "buyer" else "Seller"
        lines.append(f"{speaker}: {strip_thought(text)}")
    transcript = "\n".join(lines).strip()
    if len(transcript) <= max_chars:
        return transcript
    return transcript[:max_chars].rstrip() + "\n[truncated]"


def _quality_label(reward):
    if reward >= 0.75:
        return "strong surplus"
    if reward >= 0.35:
        return "moderate surplus"
    if reward > 0:
        return "weak surplus"
    if reward == 0:
        return "no positive surplus"
    return "negative outcome"


def _best_demo_for(ep, group_eps):
    """Pick an on-policy same-product demo without using any external teacher."""
    better = [
        other
        for other in group_eps
        if other.final_price is not None and other.reward > max(ep.reward + 1e-6, 0.0)
    ]
    if better:
        return max(better, key=lambda other: other.reward), "sibling"
    if ep.final_price is not None and ep.reward > 0:
        return ep, "self"
    return None, ""


def _format_outcome_feedback(ep):
    product = ep.product
    budget = product["budget"]
    lines = [
        "Verifier feedback for the previous negotiation rollout:",
        f"- Outcome: {ep.outcome}.",
    ]

    if "FORMAT_ERROR" in ep.outcome:
        lines.append("- Diagnosis: the buyer output was not parseable as Thought/Talk/Action with a valid Action tag.")
        lines.append("- Fix: keep the exact format and end with one explicit Action line.")
    elif ep.outcome in {"BUYER_BUDGET_VIOLATION", "BUYER_DEAL_PRICE_MISMATCH", "BUYER_DEAL_INVALID_SELLER_OFFER"}:
        lines.append("- Diagnosis: the buyer violated a hard protocol or budget constraint.")
        lines.append("- Fix: never offer above the private budget; only DEAL an exact prior seller SELL offer.")
    elif ep.final_price is None:
        lines.append("- Diagnosis: no deal was reached, so the buyer captured no surplus.")
        lines.append("- Fix: use an opening anchor and concessions that keep the seller engaged without revealing the budget.")
    else:
        price_ratio = ep.final_price / max(budget, 1e-6)
        lines.append(f"- Final price: ${ep.final_price:.2f} against buyer budget ${budget:.2f}.")
        lines.append(f"- Verifier label: {_quality_label(ep.reward)}.")
        if price_ratio > 0.85:
            lines.append("- Fix: the deal was too close to the budget; anchor lower and concede more slowly.")
        elif price_ratio > 0.60:
            lines.append("- Fix: valid deal, but there may be room for stronger anchoring or persuasion.")
        else:
            lines.append("- Keep: the price was meaningfully below budget; preserve the pressure-and-concession pattern.")

    if SDPO_FEEDBACK_MODE == "oracle":
        lines.extend(
            [
                "",
                "Oracle-only ablation details:",
                f"- Seller private cost: ${product['cost']:.2f}.",
                f"- Mutual-interest instance: {product['mi']}.",
                f"- Numeric reward: {ep.reward:.4f}.",
            ]
        )

    return "\n".join(lines)


def build_sdpo_feedback(ep, group_eps):
    """Build concise feedback for the self-teacher prompt.

    Strict mode intentionally avoids exact seller cost/private floor text. It may
    use qualitative verifier labels and on-policy public sibling demos.
    """
    if SDPO_FEEDBACK_MODE not in {"strict", "oracle"}:
        raise ValueError(f"Unsupported SDPO_FEEDBACK_MODE={SDPO_FEEDBACK_MODE!r}")

    feedback_parts = [_format_outcome_feedback(ep)]
    demo, demo_kind = _best_demo_for(ep, group_eps)
    has_demo = demo is not None
    if demo is not None:
        if demo_kind == "sibling":
            feedback_parts.append(
                "\nA better rollout sampled by the current policy for this same product is shown below. "
                "It is a public Talk/Action transcript, not an external teacher answer."
            )
        else:
            feedback_parts.append(
                "\nThis rollout itself reached a positive deal. Use the public transcript below to reinforce "
                "the useful parts without overfitting to wording."
            )
        feedback_parts.append(_public_transcript(demo))

    feedback_parts.append(
        "\nCorrectly continue the original buyer turn. Prefer valid format, budget discipline, "
        "low but plausible anchoring, and concise persuasion."
    )
    feedback = "\n".join(feedback_parts).strip()
    if len(feedback) > SDPO_MAX_FEEDBACK_CHARS:
        feedback = feedback[:SDPO_MAX_FEEDBACK_CHARS].rstrip() + "\n[feedback truncated]"

    if SDPO_FEEDBACK_MODE == "strict" and "cost_price" in feedback:
        raise AssertionError("Strict SDPO feedback leaked cost_price field.")
    return feedback, has_demo


def build_sdpo_teacher_turn_prompt(ep, turn_idx, feedback):
    """Prompt the same buyer model as a hindsight self-teacher."""
    prompt_msgs = build_buyer_turn_prompt(ep, turn_idx)
    prompt_msgs.append(
        {
            "role": "user",
            "content": (
                "Hindsight training feedback is available for your previous negotiation attempt. "
                "Use it only to judge what the next buyer message should make more or less likely.\n\n"
                f"{feedback}"
            ),
        }
    )
    return prompt_msgs


def _current_lr(step_idx):
    """Linear warmup only; no decay for the short 42-step RLVR run.

    ``step_idx`` is the zero-based optimizer-step index. With WARMUP_STEPS=10,
    the first step uses 10% of LR and the 10th step reaches the configured LR.
    """
    if WARMUP_STEPS <= 0:
        return LR
    return LR * min(1.0, float(step_idx + 1) / float(WARMUP_STEPS))


def _cpu_adamw_step(params, state, lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=WEIGHT_DECAY):
    """Exact AdamW step with optimizer state kept on CPU.

    This preserves full fine-tuning and AdamW semantics while removing ~2x
    parameter-size optimizer state from CUDA memory. Gradients are copied to CPU
    one parameter at a time, so peak VRAM stays close to forward/backward memory.
    """
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            sid = id(p)
            st = state.get(sid)
            if st is None:
                st = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(p.detach(), device="cpu", dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(p.detach(), device="cpu", dtype=torch.float32),
                }
                state[sid] = st
            st["step"] += 1
            grad = p.grad.detach().to(device="cpu", dtype=torch.float32)
            exp_avg = st["exp_avg"]
            exp_avg_sq = st["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** st["step"]
            bias_correction2 = 1.0 - beta2 ** st["step"]
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
            update = exp_avg / bias_correction1 / denom
            update = update.to(device=p.device, dtype=p.dtype)
            if weight_decay:
                # Decoupled AdamW decay: p <- p * (1 - lr * wd), independent of
                # the adaptive gradient update. This mirrors torch.optim.AdamW.
                p.add_(p, alpha=-lr * weight_decay)
            p.add_(update, alpha=-lr)
            p.grad = None
            del grad, update


def _optimizer_step(buyer_model, optimizer, cpu_adamw_state, step_idx):
    grad_norm = torch.nn.utils.clip_grad_norm_(buyer_model.parameters(), GRAD_CLIP_NORM)
    step_lr = _current_lr(step_idx)
    if OPTIMIZER == "adamw_cpu":
        _cpu_adamw_step([p for p in buyer_model.parameters() if p.requires_grad], cpu_adamw_state, lr=step_lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm), step_lr


def _normalize_completion_for_update(completion_text):
    """Return completion tokens consistent with the prompt's chat-template state.

    In Qwen3.5 native-thinking mode, add_generation_prompt=True puts an open
    ``<think>\n`` in the prompt. If we reattach that marker to the decoded private
    history for auditing, do not train the model to emit the marker again.
    """
    text = completion_text or ""
    if CHAT_TEMPLATE_ENABLE_THINKING:
        text = re.sub(r"^\s*<think\b[^>]*>\s*", "", text, count=1, flags=re.IGNORECASE)
        if NATIVE_REASONING_PROTOCOL:
            # The supervised target should preserve the hidden native-thinking
            # context and reinforce the public Talk/Action that follows it. The
            # two-stage finalizer may inject a close tag plus public suffix when
            # Qwen spends the full thinking budget before answering; keep that
            # suffix trainable instead of collapsing the sample to EOS.
            if QWEN_THINK_CLOSE_RE.search(text):
                public_suffix = QWEN_THINK_CLOSE_RE.split(text, maxsplit=1)[-1]
                text = "</think>" + public_suffix
            else:
                text = strip_qwen_native_thinking(text)
    return text


def _encode_prompt_completion(tokenizer, prompt_text, completion_text):
    """Pre-tokenize prompt and completion once and keep completion-mask metadata.

    Prompt and completion are tokenized separately, then concatenated. That avoids
    BPE boundary drift from tokenizing ``prompt`` and ``prompt + completion`` in
    separate calls while still matching the causal-LM shifted-logprob objective.
    If the pair exceeds UPDATE_MAX_LENGTH, left-truncate the prompt first so at
    least the generated completion remains trainable.
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(_normalize_completion_for_update(completion_text), add_special_tokens=False)["input_ids"]
    if not completion_ids:
        completion_ids = [tokenizer.eos_token_id]

    max_len = max(int(UPDATE_MAX_LENGTH), 2)
    if len(completion_ids) >= max_len:
        completion_ids = completion_ids[: max_len - 1]
        prompt_ids = prompt_ids[-1:]
    else:
        prompt_budget = max_len - len(completion_ids)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + completion_ids
    if len(input_ids) < 2:
        input_ids = [tokenizer.eos_token_id] + input_ids
    prompt_len = min(len(prompt_ids), len(input_ids) - 1)
    completion_shift_start = max(prompt_len - 1, 0)
    completion_shift_end = max(len(input_ids) - 1, completion_shift_start)
    return {
        "input_ids": input_ids,
        "prompt_len": prompt_len,
        "completion_shift_start": completion_shift_start,
        "completion_shift_end": completion_shift_end,
    }


def _encoded_padded_len(item):
    seq_len = len(item["input_ids"])
    if UPDATE_PAD_TO_MULTIPLE_OF > 1:
        rem = seq_len % UPDATE_PAD_TO_MULTIPLE_OF
        if rem:
            seq_len += UPDATE_PAD_TO_MULTIPLE_OF - rem
    return seq_len


def _update_example_bucket_key(example):
    # Sort by the larger of student/teacher padded lengths because both forwards
    # are executed for each microbatch. This preserves per-turn loss weights; only
    # the grouping into forward/backward calls changes.
    return max(_encoded_padded_len(example["student"]), _encoded_padded_len(example["teacher"]))


def _maybe_length_bucket_examples(examples):
    if not UPDATE_LENGTH_BUCKETING or len(examples) <= UPDATE_MICROBATCH_SIZE:
        return examples
    return sorted(examples, key=_update_example_bucket_key)


def _pad_encoded_batch(encoded_items, tokenizer, device):
    if not encoded_items:
        raise ValueError("Cannot collate an empty update microbatch")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(item["input_ids"]) for item in encoded_items)
    if UPDATE_PAD_TO_MULTIPLE_OF > 1:
        rem = max_len % UPDATE_PAD_TO_MULTIPLE_OF
        if rem:
            max_len += UPDATE_PAD_TO_MULTIPLE_OF - rem
    input_rows, attn_rows, mask_rows = [], [], []
    for item in encoded_items:
        ids = item["input_ids"]
        seq_len = len(ids)
        pad_len = max_len - seq_len
        input_rows.append(ids + [pad_id] * pad_len)
        attn_rows.append([1] * seq_len + [0] * pad_len)
        # Mask shape matches shifted log-probs: [seq_len - 1]. Completion tokens
        # occupy the precomputed shifted interval, clamped to the unpadded length.
        shifted_len = max(max_len - 1, 0)
        mask = [0] * shifted_len
        start = max(min(item.get("completion_shift_start", item["prompt_len"] - 1), shifted_len), 0)
        end = max(min(item.get("completion_shift_end", seq_len - 1), shifted_len), start)
        for j in range(start, end):
            mask[j] = 1
        mask_rows.append(mask)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_rows, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(mask_rows, dtype=torch.float32, device=device),
    }


def _build_group_update_examples(tokenizer, group_eps, advantages, timers):
    """Flatten a GRPO group into buyer-turn update examples and pre-tokenize."""
    start = _timer_start()
    examples = []
    sdpo_demo_count = 0
    for i, ep in enumerate(group_eps):
        feedback, has_demo = build_sdpo_feedback(ep, group_eps)
        sdpo_demo_count += int(has_demo)
        for turn_idx, (role, text) in enumerate(ep.turns):
            if role != "buyer":
                continue
            prompt_msgs = build_buyer_turn_prompt(ep, turn_idx)
            prompt_text = _apply_chat_template_text(
                tokenizer, prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=CHAT_TEMPLATE_ENABLE_THINKING
            )
            _assert_no_private_info_leak(prompt_text, ep.product, "buyer")

            teacher_prompt_msgs = build_sdpo_teacher_turn_prompt(ep, turn_idx, feedback)
            teacher_prompt_text = _apply_chat_template_text(
                tokenizer, teacher_prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=CHAT_TEMPLATE_ENABLE_THINKING
            )
            _assert_no_private_info_leak(teacher_prompt_text, ep.product, "buyer")

            student_enc = _encode_prompt_completion(tokenizer, prompt_text, text)
            teacher_enc = _encode_prompt_completion(tokenizer, teacher_prompt_text, text)
            if len(student_enc["input_ids"]) < 2 or len(teacher_enc["input_ids"]) < 2:
                continue
            examples.append(
                {
                    "student": student_enc,
                    "teacher": teacher_enc,
                    "grpo_adv": float(advantages[i].detach().cpu().item()),
                }
            )
    _timer_add(timers, "update_pretokenize_s", start)
    return examples, sdpo_demo_count


def _maybe_optimizer_step(buyer_model, optimizer, cpu_adamw_state, timers, step_idx, force=False):
    start = _timer_start()
    grad_present = any(p.grad is not None for p in buyer_model.parameters() if p.requires_grad)
    _timer_add(timers, "update_grad_check_s", start)
    if not grad_present:
        return False, 0.0, _current_lr(step_idx)
    start = _timer_start()
    grad_norm, step_lr = _optimizer_step(buyer_model, optimizer, cpu_adamw_state, step_idx)
    _timer_add(timers, "update_optimizer_s", start)
    return True, grad_norm, step_lr


def _rowwise_sdpo_adv(pol_lp, teacher_lp, student_mask, teacher_mask):
    """Align teacher/student completion log-prob gaps independently per row."""
    sdpo_adv = torch.zeros_like(pol_lp)
    total_tokens = 0
    abs_adv = 0.0
    for row in range(pol_lp.shape[0]):
        student_idx = student_mask[row].bool().nonzero(as_tuple=False).flatten()
        teacher_idx = teacher_mask[row].bool().nonzero(as_tuple=False).flatten()
        n_align = min(student_idx.numel(), teacher_idx.numel())
        if not n_align:
            continue
        s_idx = student_idx[:n_align]
        t_idx = teacher_idx[:n_align]
        sdpo_values = (teacher_lp[row, t_idx] - pol_lp.detach()[row, s_idx]).clamp(-SDPO_ADV_CLIP, SDPO_ADV_CLIP)
        sdpo_adv[row, s_idx] = sdpo_values.to(sdpo_adv.dtype)
        total_tokens += int(n_align)
        abs_adv += float(sdpo_values.abs().sum().item())
    return sdpo_adv, total_tokens, abs_adv


def sdpo_grpo_update(
    buyer_model,
    tokenizer,
    episodes,
    optimizer,
    device,
    cpu_adamw_state=None,
    optimizer_step_start=0,
    sdpo_lambda_value=None,
):
    """Buyer-only ref-free/on-policy GRPO update plus feedback-conditioned SDPO.

    Performance notes:
    - Buyer turns are flattened to pre-tokenized update examples per GRPO group.
    - Forward/backward runs over UPDATE_MICROBATCH_SIZE examples rather than one
      buyer turn at a time, improving A100 occupancy.
    - CPU AdamW steps every OPTIM_STEP_EVERY_GROUPS groups by default. Losses are
      scaled by the active accumulation window to preserve update magnitude.

    Objective note:
    - This is true ref-free/on-policy training: no frozen reference-policy model,
      no reference forward, and no KL penalty. The loss is the sampled-token policy
      gradient ``-A * log πθ(token)`` over buyer completion tokens. The SDPO
      self-teacher is still the current buyer model under hindsight feedback.
    """
    buyer_model.train()
    sdpo_lambda_value = SDPO_LAMBDA if sdpo_lambda_value is None else float(sdpo_lambda_value)
    cpu_adamw_state = {} if cpu_adamw_state is None else cpu_adamw_state
    G = GROUP_SIZE
    num_groups = len(episodes) // G
    total_loss = 0.0
    turn_count = 0
    sdpo_tokens = 0
    sdpo_abs_adv = 0.0
    sdpo_demo_count = 0
    optimizer_steps = 0
    grad_norm_last = 0.0
    global_step = int(optimizer_step_start)
    lr_last = _current_lr(global_step)
    timers: Dict[str, float] = {
        "update_pretokenize_s": 0.0,
        "update_collate_s": 0.0,
        "update_policy_forward_s": 0.0,
        # Retained as a zero-valued schema-compatible metric for old dashboards.
        "update_ref_forward_s": 0.0,
        "update_teacher_forward_s": 0.0,
        "update_loss_backward_s": 0.0,
        "update_optimizer_s": 0.0,
        "update_grad_check_s": 0.0,
    }

    optim_every = max(1, OPTIM_STEP_EVERY_GROUPS)
    accum_groups = 0
    loss_count = 0

    for g in range(num_groups):
        group_eps = episodes[g * G : (g + 1) * G]
        rewards = torch.tensor([ep.reward for ep in group_eps], dtype=torch.float32, device=device)
        advantages = _norm_advantages(rewards - rewards.mean())
        group_examples, demo_count = _build_group_update_examples(tokenizer, group_eps, advantages, timers)
        sdpo_demo_count += demo_count
        if not group_examples:
            continue
        group_examples = _maybe_length_bucket_examples(group_examples)

        group_turn_count = len(group_examples)
        window_start_group = (g // optim_every) * optim_every
        loss_scale_groups = max(1, min(optim_every, num_groups - window_start_group))
        for _inner in range(NUM_INNER_EPOCHS):
            for mb_examples in _chunked(group_examples, UPDATE_MICROBATCH_SIZE):
                start = _timer_start()
                student_batch = _pad_encoded_batch([ex["student"] for ex in mb_examples], tokenizer, device)
                teacher_batch = _pad_encoded_batch([ex["teacher"] for ex in mb_examples], tokenizer, device)
                grpo_adv = torch.tensor([ex["grpo_adv"] for ex in mb_examples], dtype=torch.float32, device=device)
                _timer_add(timers, "update_collate_s", start)

                start = _timer_start()
                pol_lp = _token_logprobs(buyer_model, student_batch["input_ids"], student_batch["attention_mask"])
                _timer_add(timers, "update_policy_forward_s", start)

                start = _timer_start()
                with torch.inference_mode():
                    teacher_lp = _token_logprobs(
                        buyer_model, teacher_batch["input_ids"], teacher_batch["attention_mask"]
                    )
                _timer_add(timers, "update_teacher_forward_s", start)

                student_mask = student_batch["completion_mask"]
                teacher_mask = teacher_batch["completion_mask"]
                sdpo_adv, n_align, mb_abs_adv = _rowwise_sdpo_adv(pol_lp, teacher_lp, student_mask, teacher_mask)
                sdpo_abs_adv += mb_abs_adv
                sdpo_tokens += n_align

                adv = (sdpo_lambda_value * grpo_adv.view(-1, 1) + (1.0 - sdpo_lambda_value) * sdpo_adv).detach()
                policy_loss = -adv * pol_lp

                token_counts = student_mask.sum(dim=1).clamp_min(1.0)
                row_losses = (policy_loss * student_mask).sum(dim=1) / token_counts
                # Preserve the old per-turn averaged objective: each buyer turn
                # contributes one mean-over-completion loss, and microbatching only
                # changes how those turn losses are grouped into forward/backward calls.
                unscaled_loss = row_losses.sum()
                loss = unscaled_loss / loss_scale_groups

                start = _timer_start()
                loss.backward()
                _timer_add(timers, "update_loss_backward_s", start)

                total_loss += float(row_losses.detach().sum().item())
                loss_count += int(row_losses.numel())

        turn_count += group_turn_count
        accum_groups += 1

        if accum_groups >= optim_every:
            stepped, grad_norm_last, lr_last = _maybe_optimizer_step(
                buyer_model, optimizer, cpu_adamw_state, timers, global_step
            )
            if stepped:
                optimizer_steps += 1
                global_step += 1
            accum_groups = 0

    if accum_groups > 0:
        # If NUM_ITERS/BATCH_SIZE creates a remainder, take the final accumulated step.
        stepped, grad_norm_last, lr_last = _maybe_optimizer_step(
            buyer_model, optimizer, cpu_adamw_state, timers, global_step, force=True
        )
        if stepped:
            optimizer_steps += 1
            global_step += 1

    return {
        "loss": total_loss / max(loss_count, 1),
        "sdpo_tokens": sdpo_tokens,
        "sdpo_mean_abs_adv": sdpo_abs_adv / max(sdpo_tokens, 1),
        "sdpo_demo_count": sdpo_demo_count,
        "update_examples": turn_count,
        "optimizer_steps": optimizer_steps,
        "optimizer_global_step": global_step,
        "grad_norm_last": grad_norm_last,
        "lr_last": lr_last,
        "sdpo_lambda_active": sdpo_lambda_value,
        **timers,
    }


# ─── Metrics / checkpoint helpers ────────────────────────────────────────────
def compute_product_mix_metrics(products):
    if not products:
        return {
            "sample_product_count": 0,
            "sample_unique_product_count": 0,
            "sample_product_hash": "",
            "sample_mean_budget": 0.0,
            "sample_mean_cost": 0.0,
            "sample_mean_list_price": 0.0,
            "sample_mean_current_price": 0.0,
            "sample_mean_average_price": 0.0,
            "sample_mean_highest_price": 0.0,
            "sample_mean_budget_cost_spread": 0.0,
            "sample_mean_budget_cost_spread_ratio": 0.0,
            "sample_mi_rate": 0.0,
            "sample_category_counts": {},
            "sample_product_ids": [],
        }

    import hashlib

    ids = [str(p.get("codename", "")) for p in products]
    categories: Dict[str, int] = {}
    for p in products:
        cat = str(p.get("category", "unknown"))
        categories[cat] = categories.get(cat, 0) + 1

    def mean_key(key):
        return sum(float(p.get(key, 0.0) or 0.0) for p in products) / len(products)

    spreads = [float(p.get("budget", 0.0) or 0.0) - float(p.get("cost", 0.0) or 0.0) for p in products]
    budgets = [max(float(p.get("budget", 0.0) or 0.0), 1e-6) for p in products]
    digest = hashlib.sha1(",".join(ids).encode("utf-8")).hexdigest()[:12]
    return {
        "sample_product_count": len(products),
        "sample_unique_product_count": len(set(ids)),
        "sample_product_hash": digest,
        "sample_mean_budget": mean_key("budget"),
        "sample_mean_cost": mean_key("cost"),
        "sample_mean_list_price": mean_key("list_price"),
        "sample_mean_current_price": mean_key("current_price"),
        "sample_mean_average_price": mean_key("average_price"),
        "sample_mean_highest_price": mean_key("highest_price"),
        "sample_mean_budget_cost_spread": sum(spreads) / len(spreads),
        "sample_mean_budget_cost_spread_ratio": sum(s / b for s, b in zip(spreads, budgets)) / len(spreads),
        "sample_mi_rate": sum(1 for p in products if p.get("mi")) / len(products),
        "sample_category_counts": dict(sorted(categories.items())),
        "sample_product_ids": ids,
    }


def compute_iter_metrics(episodes):
    rewards = [ep.reward for ep in episodes]
    mean_r = sum(rewards) / max(len(rewards), 1)
    deals = [ep for ep in episodes if ep.final_price is not None]
    deal_rate = len(deals) / max(len(episodes), 1)
    mean_price = sum(ep.final_price for ep in deals) / max(len(deals), 1) if deals else 0.0
    mean_turns = sum(ep.num_turns for ep in episodes) / max(len(episodes), 1)
    outcomes: Dict[str, int] = {}
    for ep in episodes:
        outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1
    role_confusions = 0
    for ep in episodes:
        for role, text in ep.turns:
            act = extract_action(text)
            if role == "buyer" and act["type"] == "SELL":
                role_confusions += 1
            elif role == "seller" and act["type"] == "BUY":
                role_confusions += 1
    budget_violations = sum(1 for ep in episodes if ep.budget_violations > 0)
    total_turns = sum(len(ep.turns) for ep in episodes)
    native_think_marker_turns = sum(
        1
        for ep in episodes
        for _, text in ep.turns
        if QWEN_THINK_OPEN_RE.search(text or "") or QWEN_THINK_CLOSE_RE.search(text or "")
    )
    structured_thought_turns = sum(
        1
        for ep in episodes
        for _, text in ep.turns
        if re.search(r"(?:^|\n)\s*Thought\s*:", text or "", re.IGNORECASE)
    )
    first_offer_ratios = [ep.first_offer_price / ep.product["budget"] for ep in episodes if ep.first_offer_price]
    mean_first_offer_ratio = sum(first_offer_ratios) / max(len(first_offer_ratios), 1) if first_offer_ratios else None

    return {
        "mean_reward": mean_r,
        "deal_rate": deal_rate,
        "mean_price": mean_price,
        "mean_turns": mean_turns,
        "outcomes": dict(sorted(outcomes.items(), key=lambda x: -x[1])),
        "role_confusions": role_confusions,
        "native_think_marker_turns": native_think_marker_turns,
        "native_think_marker_rate": native_think_marker_turns / max(total_turns, 1),
        "structured_thought_turns": structured_thought_turns,
        "structured_thought_rate": structured_thought_turns / max(total_turns, 1),
        "price_overshoot_rate": budget_violations / max(len(episodes), 1),
        "first_offer_ratio": mean_first_offer_ratio,
    }


def save_and_push_checkpoint(buyer_model, tokenizer, metrics, iteration, final=False, processor=None):
    if not HUB_MODEL_ID:
        return
    branch = "main" if final else f"iter-{iteration + 1}"
    label = "FINAL" if final else f"iter {iteration + 1}"
    path = Path(OUTPUT_DIR if final else f"/tmp/sdpo-ckpt-{iteration+1}")
    path.mkdir(parents=True, exist_ok=True)
    buyer_model.save_pretrained(path)
    (processor or tokenizer).save_pretrained(path)
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if PUSH_TRAINING_SCRIPT:
        try:
            shutil.copyfile(__file__, path / "train_negotiation_sdpo.py")
        except Exception:
            pass

    try:
        from huggingface_hub import HfApi, create_repo

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        create_repo(HUB_MODEL_ID, exist_ok=True, token=token)
        if not final:
            try:
                api.create_branch(HUB_MODEL_ID, branch=branch, repo_type="model")
            except Exception:
                pass
        api.upload_folder(
            folder_path=path,
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            revision=branch,
            commit_message=f"SDPO negotiation {label}",
        )
        print(f"  [CHECKPOINT] ✅ Pushed {label} to {HUB_MODEL_ID}@{branch}")
    except Exception as e:
        print(f"  [CHECKPOINT] ⚠️ Push failed (non-fatal): {e}")
    finally:
        if not final:
            shutil.rmtree(path, ignore_errors=True)


# ─── Main ────────────────────────────────────────────────────────────────────
def check_cuda():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    _log_qwen35_fastpath_status()
    if not torch.cuda.is_available():
        print("FATAL: No CUDA")
        sys.exit(1)
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"GPU[{idx}]: {torch.cuda.get_device_name(idx)} VRAM={torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB")
    x = torch.randn(2, 2).cuda() @ torch.randn(2, 2).cuda()
    print(f"Compute test: {x.device} OK")
    print("=" * 70, flush=True)


def main():
    check_cuda()

    print("[CONFIG] Negotiation SDPO+GRPO buyer-only training")
    print(f"[CONFIG] BuyerModel={MODEL_NAME} SellerModel={SELLER_MODEL_NAME}")
    print(f"[CONFIG] Iters={NUM_ITERS} Batch={BATCH_SIZE} Group={GROUP_SIZE} Episodes/iter={BATCH_SIZE*GROUP_SIZE}")
    print(f"[CONFIG] Turns={MAX_TURNS} LR={LR} WarmupSteps={WARMUP_STEPS} WD={WEIGHT_DECAY} GradClip={GRAD_CLIP_NORM}")
    print(f"[CONFIG] RefFree=True KL={KL_COEF} Eps={EPSILON} (unused ref-free)")
    print(f"[CONFIG] BuyerTemp={BUYER_TEMP} SellerTemp={SELLER_TEMP} MaxNew={MAX_NEW_TOKENS}")
    print(
        f"[CONFIG] ReasoningMode={REASONING_MODE} NativeThinking={ENABLE_NATIVE_THINKING} "
        f"NativeProtocol={NATIVE_REASONING_PROTOCOL} ChatTemplateThinking={CHAT_TEMPLATE_ENABLE_THINKING} "
        f"NativeFinalizer={NATIVE_PUBLIC_FINALIZER} ThinkTokens={NATIVE_THINK_TOKENS} FinalTokens={NATIVE_FINAL_TOKENS} "
        f"StripNativeFromHistory={STRIP_NATIVE_THINKING_FROM_HISTORY} DebugBuyerOutputs={DEBUG_SAMPLE_BUYER_OUTPUTS}"
    )
    print(f"[CONFIG] GradCheckpoint={GRADIENT_CHECKPOINTING} GenBatchLimit={GEN_BATCH_LIMIT} RolloutMaxLength={ROLLOUT_MAX_LENGTH}")
    print(f"[CONFIG] DeviceMap={MODEL_DEVICE_MAP} MaxMemoryPerGPUGiB={MAX_MEMORY_PER_GPU_GIB or '(unset)'}")
    print(f"[CONFIG] InnerEpochs={NUM_INNER_EPOCHS} NormAdvantages={NORMALIZE_ADVANTAGES}")
    print(
        f"[CONFIG] SDPO_LambdaStart={SDPO_LAMBDA} Final={SDPO_LAMBDA_FINAL} "
        f"DecayIters={SDPO_LAMBDA_DECAY_ITERS} FeedbackMode={SDPO_FEEDBACK_MODE} "
        f"AdvClip={SDPO_ADV_CLIP} MaxFeedbackChars={SDPO_MAX_FEEDBACK_CHARS} AdamWForeach={ADAMW_FOREACH}"
    )
    print(
        f"[CONFIG] UpdateMicrobatch={UPDATE_MICROBATCH_SIZE} "
        f"OptimStepEveryGroups={OPTIM_STEP_EVERY_GROUPS} "
        f"UpdatePadMultiple={UPDATE_PAD_TO_MULTIPLE_OF} UpdateLengthBucketing={UPDATE_LENGTH_BUCKETING} "
        f"UpdateMaxLength={UPDATE_MAX_LENGTH}"
    )
    print(
        f"[CONFIG] EarlyStop format>={FORMAT_STOP_THRESHOLD}x{FORMAT_STOP_PATIENCE} "
        f"budget>={BUDGET_STOP_THRESHOLD}x{BUDGET_STOP_PATIENCE} "
        f"reward<={REWARD_STOP_THRESHOLD}x{REWARD_STOP_PATIENCE} "
        f"save={EARLY_STOP_SAVE_CHECKPOINT}"
    )
    print(f"[CONFIG] CheckpointEvery={CHECKPOINT_EVERY} Hub={HUB_MODEL_ID or '(disabled)'}")
    print("=" * 70, flush=True)

    try:
        import wandb

        run_name = RUN_NAME or default_run_name()
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            group=os.environ.get("WANDB_GROUP", default_wandb_group()),
            job_type=WANDB_JOB_TYPE,
            mode=WANDB_MODE,
            tags=WANDB_TAGS,
            save_code=False,
            config={
                "method": "negotiation_sdpo_grpo_qwen35_ref_free",
                "buyer_model": MODEL_NAME,
                "seller_model": SELLER_MODEL_NAME,
                "num_iters": NUM_ITERS,
                "batch_size": BATCH_SIZE,
                "group_size": GROUP_SIZE,
                "max_turns": MAX_TURNS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "epsilon": EPSILON,
                "kl_coef": KL_COEF,
                "ref_free_objective": True,
                "reference_model_used": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "buyer_temp": BUYER_TEMP,
                "seller_temp": SELLER_TEMP,
                "reasoning_mode": REASONING_MODE,
                "enable_native_thinking": ENABLE_NATIVE_THINKING,
                "native_reasoning_protocol": NATIVE_REASONING_PROTOCOL,
                "chat_template_enable_thinking": CHAT_TEMPLATE_ENABLE_THINKING,
                "native_public_finalizer": NATIVE_PUBLIC_FINALIZER,
                "native_think_tokens": NATIVE_THINK_TOKENS,
                "native_final_tokens": NATIVE_FINAL_TOKENS,
                "strip_native_thinking_from_history": STRIP_NATIVE_THINKING_FROM_HISTORY,
                "debug_sample_buyer_outputs": DEBUG_SAMPLE_BUYER_OUTPUTS,
                "normalize_advantages": NORMALIZE_ADVANTAGES,
                "num_inner_epochs": NUM_INNER_EPOCHS,
                "sdpo_lambda": SDPO_LAMBDA,
                "sdpo_lambda_final": SDPO_LAMBDA_FINAL,
                "sdpo_lambda_decay_iters": SDPO_LAMBDA_DECAY_ITERS,
                "sdpo_feedback_mode": SDPO_FEEDBACK_MODE,
                "sdpo_adv_clip": SDPO_ADV_CLIP,
                "distillation_level": DISTILLATION_LEVEL,
                "top_k_distillation": TOP_K_DISTILLATION,
                "distillation_divergence": DISTILLATION_DIVERGENCE,
                "trust_region_interpolation": TRUST_REGION_INTERPOLATION,
                "teacher_ema_decay": TEACHER_EMA_DECAY,
                "sdpo_max_demo_chars": SDPO_MAX_DEMO_CHARS,
                "sdpo_max_feedback_chars": SDPO_MAX_FEEDBACK_CHARS,
                "format_warn_threshold": FORMAT_WARN_THRESHOLD,
                "format_stop_threshold": FORMAT_STOP_THRESHOLD,
                "format_stop_patience": FORMAT_STOP_PATIENCE,
                "budget_warn_threshold": BUDGET_WARN_THRESHOLD,
                "budget_stop_threshold": BUDGET_STOP_THRESHOLD,
                "budget_stop_patience": BUDGET_STOP_PATIENCE,
                "reward_stop_threshold": REWARD_STOP_THRESHOLD,
                "reward_stop_patience": REWARD_STOP_PATIENCE,
                "early_stop_save_checkpoint": EARLY_STOP_SAVE_CHECKPOINT,
                "optimizer": OPTIMIZER,
                "adamw_foreach": ADAMW_FOREACH,
                "update_microbatch_size": UPDATE_MICROBATCH_SIZE,
                "optim_step_every_groups": OPTIM_STEP_EVERY_GROUPS,
                "update_pad_to_multiple_of": UPDATE_PAD_TO_MULTIPLE_OF,
                "update_length_bucketing": UPDATE_LENGTH_BUCKETING,
                "rollout_max_length": ROLLOUT_MAX_LENGTH,
                "update_max_length": UPDATE_MAX_LENGTH,
                "model_device_map": MODEL_DEVICE_MAP,
                "max_memory_per_gpu_gib": MAX_MEMORY_PER_GPU_GIB,
                "liger_kernel": USE_LIGER,
                "dataset_categories": CATEGORIES,
            },
        )
        WANDB_OK = True
        print(f"[WANDB] Run: {wandb_run.url}")
    except Exception as e:
        print(f"[WANDB] Init failed (non-fatal): {e}")
        WANDB_OK = False
        wandb = None
        wandb_run = None

    print("\n[1/5] Loading dataset...")
    train_products, _ = load_products(seed=SEED)

    print(f"\n[2/5] Loading tokenizer/processor ({MODEL_NAME})...")
    buyer_processor = None
    seller_processor = None
    print("  [OK] Deferred to model loader so Qwen3.5 can use AutoProcessor")

    print("\n[3/5] Loading trainable buyer model...")
    buyer_model, buyer_processor, tokenizer = _load_text_or_image_text_stack(MODEL_NAME)
    _assert_no_cpu_offload(buyer_model, "buyer_model")
    if GRADIENT_CHECKPOINTING:
        buyer_model.gradient_checkpointing_enable()
        if hasattr(buyer_model, "config"):
            buyer_model.config.use_cache = False
    dev = _model_input_device(buyer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _qwen35_text_canary(buyer_model, buyer_processor, tokenizer, dev, "buyer_model")
    print(f"  [OK] InputDevice={dev} FirstParamDevice={_first_model_device(buyer_model)} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    print("\n[4/5] Loading frozen seller/environment model (no reference-policy model)...")
    seller_model, seller_processor, seller_tokenizer = _load_text_or_image_text_stack(SELLER_MODEL_NAME)
    _assert_no_cpu_offload(seller_model, "seller_model")
    if seller_tokenizer.pad_token is None:
        seller_tokenizer.pad_token = seller_tokenizer.eos_token
    if SELLER_MODEL_NAME != MODEL_NAME:
        print("  [WARN] Separate seller tokenizer loaded; rollouts use buyer tokenizer for shared prompt formatting")
    _qwen35_text_canary(seller_model, seller_processor, seller_tokenizer, _model_input_device(seller_model), "seller_model")
    seller_model.eval()
    for p in seller_model.parameters():
        p.requires_grad = False
    seller_trainable_params = sum(p.numel() for p in seller_model.parameters() if p.requires_grad)
    if seller_trainable_params != 0:
        raise RuntimeError(f"seller_model unexpectedly has {seller_trainable_params:,} trainable params")
    print(f"  [OK] Frozen seller trainable_params={seller_trainable_params:,} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    print(
        f"\n[5/5] Optimizer (AdamW, lr={LR}, warmup_steps={WARMUP_STEPS}, "
        f"betas=(0.9,0.95), wd={WEIGHT_DECAY}, grad_clip={GRAD_CLIP_NORM}, "
        f"optimizer={OPTIMIZER}, foreach={ADAMW_FOREACH})..."
    )
    if OPTIMIZER == "adamw_cpu":
        optimizer = None
        cpu_adamw_state = {}
        print("  [OK] AdamW optimizer state will be stored on CPU to avoid CUDA optimizer-step OOM")
    elif OPTIMIZER == "adamw_cuda":
        optimizer = torch.optim.AdamW(
            buyer_model.parameters(),
            lr=LR,
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
            foreach=ADAMW_FOREACH,
        )
        cpu_adamw_state = None
    else:
        raise ValueError(f"Unsupported OPTIMIZER={OPTIMIZER}; use adamw_cpu or adamw_cuda")
    n_params = sum(p.numel() for p in buyer_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    print(f"\n{'=' * 70}\nNEGOTIATION SDPO+GRPO TRAINING\n{'=' * 70}")
    metrics = []
    optimizer_global_step = 0
    t0 = time.time()

    for iteration in range(NUM_ITERS):
        t_iter = time.time()
        print(f"\n--- Iteration {iteration} ---")
        current_vram_total, current_vram_per_gpu = _memory_allocated_all_gpus_gb()
        print(f"  VRAM: total={current_vram_total:.1f}GB per_gpu={_fmt_gpu_gb(current_vram_per_gpu)}")

        products = random.sample(train_products, min(BATCH_SIZE, len(train_products)))
        product_mix = compute_product_mix_metrics(products)
        products_expanded = [p for p in products for _ in range(GROUP_SIZE)]
        n_episodes = len(products_expanded)
        print(f"  Sampling {len(products)} products × {GROUP_SIZE} rollouts = {n_episodes} episodes...")
        print(
            f"  Product mix: hash={product_mix['sample_product_hash']} "
            f"budget=${product_mix['sample_mean_budget']:.2f} cost=${product_mix['sample_mean_cost']:.2f} "
            f"list=${product_mix['sample_mean_list_price']:.2f} spread=${product_mix['sample_mean_budget_cost_spread']:.2f} "
            f"MI={product_mix['sample_mi_rate']:.1%} cats={product_mix['sample_category_counts']}"
        )

        lambda_active = active_sdpo_lambda(iteration)
        print(f"  Active SDPO_LAMBDA={lambda_active:.4f} (GRPO weight; SDPO weight={1.0 - lambda_active:.4f})")
        buyer_model.eval()
        seller_model.eval()
        _reset_peak_memory_stats_all_gpus()
        rollout_t0 = time.time()
        episodes = run_episodes_batched(buyer_model, seller_model, tokenizer, products_expanded, dev, seller_tokenizer=seller_tokenizer)
        rollout_time = time.time() - rollout_t0
        rollout_peak_vram, rollout_peak_vram_per_gpu = _peak_memory_allocated_all_gpus_gb()
        rollout_reserved_peak_vram, rollout_reserved_peak_vram_per_gpu = _peak_memory_reserved_all_gpus_gb()
        print(f"  Rollout: {n_episodes} episodes in {rollout_time:.0f}s ({rollout_time/n_episodes:.1f}s/ep)")
        torch.cuda.empty_cache()
        gc.collect()

        buyer_model.train()
        print("  SDPO+GRPO update on buyer turns only...")
        _reset_peak_memory_stats_all_gpus()
        update_stats = sdpo_grpo_update(
            buyer_model,
            tokenizer,
            episodes,
            optimizer,
            dev,
            cpu_adamw_state,
            optimizer_global_step,
            sdpo_lambda_value=lambda_active,
        )
        update_peak_vram, update_peak_vram_per_gpu = _peak_memory_allocated_all_gpus_gb()
        update_reserved_peak_vram, update_reserved_peak_vram_per_gpu = _peak_memory_reserved_all_gpus_gb()
        optimizer_global_step = update_stats["optimizer_global_step"]
        loss = update_stats["loss"]
        torch.cuda.empty_cache()
        gc.collect()

        iter_metrics = compute_iter_metrics(episodes)
        fmt_errors = iter_metrics["outcomes"].get("BUYER_FORMAT_ERROR", 0)
        budget_violations = iter_metrics["outcomes"].get("BUYER_BUDGET_VIOLATION", 0)
        elapsed = time.time() - t_iter
        update_time = elapsed - rollout_time
        current_vram, current_vram_per_gpu = _memory_allocated_all_gpus_gb()
        current_reserved_vram, current_reserved_vram_per_gpu = _memory_reserved_all_gpus_gb()
        peak_vram = max(rollout_peak_vram, update_peak_vram)
        peak_vram_per_gpu = [
            max(r, u) for r, u in zip(rollout_peak_vram_per_gpu, update_peak_vram_per_gpu)
        ]
        reserved_peak_vram = max(rollout_reserved_peak_vram, update_reserved_peak_vram)
        reserved_peak_vram_per_gpu = [
            max(r, u) for r, u in zip(rollout_reserved_peak_vram_per_gpu, update_reserved_peak_vram_per_gpu)
        ]

        print(
            f"  Loss={loss:.4f} Reward={iter_metrics['mean_reward']:.4f} "
            f"Deal={iter_metrics['deal_rate']:.1%} Price=${iter_metrics['mean_price']:.2f} "
            f"Turns={iter_metrics['mean_turns']:.1f}"
        )
        print(
            f"  SDPO tokens={update_stats['sdpo_tokens']} "
            f"mean|A|={update_stats['sdpo_mean_abs_adv']:.4f} "
            f"demos={update_stats['sdpo_demo_count']}"
        )
        print(f"  FirstOfferRatio={iter_metrics['first_offer_ratio']} Overshoot={iter_metrics['price_overshoot_rate']:.1%}")
        print(
            f"  Update: examples={update_stats['update_examples']} "
            f"optimizer_steps={update_stats['optimizer_steps']} lr={update_stats['lr_last']:.2e} "
            f"grad_norm={update_stats['grad_norm_last']:.4f}"
        )
        print(f"  Time={elapsed:.0f}s (rollout={rollout_time:.0f}s update={update_time:.0f}s)")
        print(
            f"  Phase VRAM allocated peaks: rollout={rollout_peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(rollout_peak_vram_per_gpu)} "
            f"update={update_peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(update_peak_vram_per_gpu)} "
            f"overall={peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(peak_vram_per_gpu)}"
        )
        print(
            f"  Phase VRAM reserved peaks: rollout={rollout_reserved_peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(rollout_reserved_peak_vram_per_gpu)} "
            f"update={update_reserved_peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(update_reserved_peak_vram_per_gpu)} "
            f"overall={reserved_peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(reserved_peak_vram_per_gpu)}"
        )
        print(
            "  Update timers: "
            f"pretokenize={update_stats['update_pretokenize_s']:.1f}s "
            f"collate={update_stats['update_collate_s']:.1f}s "
            f"policy_fwd={update_stats['update_policy_forward_s']:.1f}s "
            f"ref_fwd={update_stats['update_ref_forward_s']:.1f}s(ref-free) "
            f"teacher_fwd={update_stats['update_teacher_forward_s']:.1f}s "
            f"backward={update_stats['update_loss_backward_s']:.1f}s "
            f"optimizer={update_stats['update_optimizer_s']:.1f}s "
            f"grad_check={update_stats['update_grad_check_s']:.1f}s"
        )
        print(f"  Outcomes: {dict(list(iter_metrics['outcomes'].items())[:6])}")
        if fmt_errors >= FORMAT_WARN_THRESHOLD:
            print(
                f"  ⚠️ BUYER FORMAT ERRORS: {fmt_errors}/{n_episodes} "
                f"(stop if >= {FORMAT_STOP_THRESHOLD} for {FORMAT_STOP_PATIENCE} consecutive iters)"
            )
        if budget_violations >= BUDGET_WARN_THRESHOLD:
            print(
                f"  ⚠️ BUYER BUDGET VIOLATIONS: {budget_violations}/{n_episodes} "
                f"(stop if >= {BUDGET_STOP_THRESHOLD} for {BUDGET_STOP_PATIENCE} consecutive iters)"
            )
        if iter_metrics["role_confusions"]:
            print(f"  ⚠️ ROLE CONFUSIONS: {iter_metrics['role_confusions']}")
        print(
            f"  Reasoning markers: native_think={iter_metrics['native_think_marker_turns']} "
            f"({iter_metrics['native_think_marker_rate']:.1%} of turns), "
            f"explicit Thought={iter_metrics['structured_thought_turns']} "
            f"({iter_metrics['structured_thought_rate']:.1%} of turns)"
        )
        print(
            f"  VRAM allocated: total_current={current_vram:.1f}GB per_gpu={_fmt_gpu_gb(current_vram_per_gpu)} "
            f"peak={peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(peak_vram_per_gpu)}",
            flush=True,
        )
        print(
            f"  VRAM reserved: total_current={current_reserved_vram:.1f}GB per_gpu={_fmt_gpu_gb(current_reserved_vram_per_gpu)} "
            f"peak={reserved_peak_vram:.1f}GB per_gpu={_fmt_gpu_gb(reserved_peak_vram_per_gpu)}",
            flush=True,
        )

        row = {
            "iteration": iteration,
            "loss": loss,
            **iter_metrics,
            **product_mix,
            "sdpo_tokens": update_stats["sdpo_tokens"],
            "sdpo_mean_abs_adv": update_stats["sdpo_mean_abs_adv"],
            "sdpo_demo_count": update_stats["sdpo_demo_count"],
            "update_examples": update_stats["update_examples"],
            "optimizer_steps": update_stats["optimizer_steps"],
            "optimizer_global_step": update_stats["optimizer_global_step"],
            "lr_last": update_stats["lr_last"],
            "grad_norm_last": update_stats["grad_norm_last"],
            "sdpo_lambda_active": update_stats["sdpo_lambda_active"],
            "time": elapsed,
            "rollout_time": rollout_time,
            "update_time": update_time,
            "update_pretokenize_s": update_stats["update_pretokenize_s"],
            "update_collate_s": update_stats["update_collate_s"],
            "update_policy_forward_s": update_stats["update_policy_forward_s"],
            "update_ref_forward_s": update_stats["update_ref_forward_s"],
            "update_teacher_forward_s": update_stats["update_teacher_forward_s"],
            "update_loss_backward_s": update_stats["update_loss_backward_s"],
            "update_optimizer_s": update_stats["update_optimizer_s"],
            "update_grad_check_s": update_stats["update_grad_check_s"],
            "vram_current_gb": current_vram,
            "vram_current_per_gpu_gb": current_vram_per_gpu,
            "vram_reserved_current_gb": current_reserved_vram,
            "vram_reserved_current_per_gpu_gb": current_reserved_vram_per_gpu,
            "vram_peak_gb": peak_vram,
            "vram_peak_per_gpu_gb": peak_vram_per_gpu,
            "vram_reserved_peak_gb": reserved_peak_vram,
            "vram_reserved_peak_per_gpu_gb": reserved_peak_vram_per_gpu,
            "rollout_vram_peak_gb": rollout_peak_vram,
            "rollout_vram_peak_per_gpu_gb": rollout_peak_vram_per_gpu,
            "rollout_vram_reserved_peak_gb": rollout_reserved_peak_vram,
            "rollout_vram_reserved_peak_per_gpu_gb": rollout_reserved_peak_vram_per_gpu,
            "update_vram_peak_gb": update_peak_vram,
            "update_vram_peak_per_gpu_gb": update_peak_vram_per_gpu,
            "update_vram_reserved_peak_gb": update_reserved_peak_vram,
            "update_vram_reserved_peak_per_gpu_gb": update_reserved_peak_vram_per_gpu,
        }
        metrics.append(row)

        if WANDB_OK:
            try:
                wandb.log(
                    {
                        "train/loss": loss,
                        "reward/buyer": iter_metrics["mean_reward"],
                        "negotiation/deal_rate": iter_metrics["deal_rate"],
                        "negotiation/mean_price": iter_metrics["mean_price"],
                        "negotiation/mean_turns": iter_metrics["mean_turns"],
                        "negotiation/first_offer_ratio": iter_metrics["first_offer_ratio"] or 0.0,
                        "negotiation/price_overshoot_rate": iter_metrics["price_overshoot_rate"],
                        "sample/product_count": product_mix["sample_product_count"],
                        "sample/unique_product_count": product_mix["sample_unique_product_count"],
                        "sample/mean_budget": product_mix["sample_mean_budget"],
                        "sample/mean_cost": product_mix["sample_mean_cost"],
                        "sample/mean_list_price": product_mix["sample_mean_list_price"],
                        "sample/mean_current_price": product_mix["sample_mean_current_price"],
                        "sample/mean_average_price": product_mix["sample_mean_average_price"],
                        "sample/mean_highest_price": product_mix["sample_mean_highest_price"],
                        "sample/mean_budget_cost_spread": product_mix["sample_mean_budget_cost_spread"],
                        "sample/mean_budget_cost_spread_ratio": product_mix["sample_mean_budget_cost_spread_ratio"],
                        "sample/mi_rate": product_mix["sample_mi_rate"],
                        **{f"sample/category_count/{k}": v for k, v in product_mix["sample_category_counts"].items()},
                        "sdpo/tokens": update_stats["sdpo_tokens"],
                        "sdpo/mean_abs_adv": update_stats["sdpo_mean_abs_adv"],
                        "sdpo/demo_count": update_stats["sdpo_demo_count"],
                        "perf/iter_time_s": elapsed,
                        "perf/rollout_time_s": rollout_time,
                        "perf/update_time_s": update_time,
                        "objective/ref_free": 1,
                        "objective/reference_model_used": 0,
                        "objective/kl_coef": KL_COEF,
                        "objective/sdpo_lambda_active": update_stats["sdpo_lambda_active"],
                        "perf/update_pretokenize_s": update_stats["update_pretokenize_s"],
                        "perf/update_collate_s": update_stats["update_collate_s"],
                        "perf/update_policy_forward_s": update_stats["update_policy_forward_s"],
                        "perf/update_ref_forward_s": update_stats["update_ref_forward_s"],
                        "perf/update_teacher_forward_s": update_stats["update_teacher_forward_s"],
                        "perf/update_loss_backward_s": update_stats["update_loss_backward_s"],
                        "perf/update_optimizer_s": update_stats["update_optimizer_s"],
                        "perf/update_grad_check_s": update_stats["update_grad_check_s"],
                        "perf/update_examples": update_stats["update_examples"],
                        "perf/optimizer_steps": update_stats["optimizer_steps"],
                        "train/optimizer_global_step": update_stats["optimizer_global_step"],
                        "train/lr": update_stats["lr_last"],
                        "train/grad_norm_last": update_stats["grad_norm_last"],
                        "perf/vram_gb": current_vram,
                        "perf/vram_peak_gb": peak_vram,
                        "perf/vram_reserved_gb": current_reserved_vram,
                        "perf/vram_reserved_peak_gb": reserved_peak_vram,
                        "perf/rollout_vram_peak_gb": rollout_peak_vram,
                        "perf/rollout_vram_reserved_peak_gb": rollout_reserved_peak_vram,
                        "perf/update_vram_peak_gb": update_peak_vram,
                        "perf/update_vram_reserved_peak_gb": update_reserved_peak_vram,
                        **{f"perf/vram_current_gpu{i}_gb": v for i, v in enumerate(current_vram_per_gpu)},
                        **{f"perf/vram_peak_gpu{i}_gb": v for i, v in enumerate(peak_vram_per_gpu)},
                        **{f"perf/vram_reserved_current_gpu{i}_gb": v for i, v in enumerate(current_reserved_vram_per_gpu)},
                        **{f"perf/vram_reserved_peak_gpu{i}_gb": v for i, v in enumerate(reserved_peak_vram_per_gpu)},
                        **{f"perf/rollout_vram_peak_gpu{i}_gb": v for i, v in enumerate(rollout_peak_vram_per_gpu)},
                        **{f"perf/rollout_vram_reserved_peak_gpu{i}_gb": v for i, v in enumerate(rollout_reserved_peak_vram_per_gpu)},
                        **{f"perf/update_vram_peak_gpu{i}_gb": v for i, v in enumerate(update_peak_vram_per_gpu)},
                        **{f"perf/update_vram_reserved_peak_gpu{i}_gb": v for i, v in enumerate(update_reserved_peak_vram_per_gpu)},
                        "sanity/role_confusions": iter_metrics["role_confusions"],
                    },
                    step=iteration,
                )
                if iteration == 0:
                    wandb.alert(
                        "sdpo_negotiation_started",
                        f"iter=0 reward={iter_metrics['mean_reward']:.4f} deal_rate={iter_metrics['deal_rate']:.3f}; continue 42-iter run if format errors stay low",
                        level=wandb.AlertLevel.INFO,
                    )
                if iter_metrics["mean_reward"] < -0.5:
                    wandb.alert(
                        "low_reward_warning",
                        f"reward={iter_metrics['mean_reward']:.4f} at iter={iteration}; if persistent, reduce LR or lower SDPO weight",
                        level=wandb.AlertLevel.WARN,
                    )
                if fmt_errors >= FORMAT_WARN_THRESHOLD:
                    wandb.alert(
                        "format_error_warning",
                        f"buyer_format_errors={fmt_errors}/{n_episodes} at iter={iteration}; if repeated, stop and reduce LR or slow SDPO handoff",
                        level=wandb.AlertLevel.WARN,
                    )
                if fmt_errors >= FORMAT_STOP_THRESHOLD:
                    wandb.alert(
                        "format_collapse_error",
                        f"buyer_format_errors={fmt_errors}/{n_episodes} at iter={iteration}; early-stop threshold hit, checkpoint will be saved before exit if patience is met",
                        level=wandb.AlertLevel.ERROR,
                    )
                if budget_violations >= BUDGET_WARN_THRESHOLD:
                    wandb.alert(
                        "budget_violation_warning",
                        f"buyer_budget_violations={budget_violations}/{n_episodes} at iter={iteration}; if repeated, strengthen budget prompt or reduce LR",
                        level=wandb.AlertLevel.WARN,
                    )
                if budget_violations >= BUDGET_STOP_THRESHOLD:
                    wandb.alert(
                        "budget_violation_error",
                        f"buyer_budget_violations={budget_violations}/{n_episodes} at iter={iteration}; early-stop threshold hit, checkpoint will be saved before exit if patience is met",
                        level=wandb.AlertLevel.ERROR,
                    )
            except Exception as e:
                print(f"  [WANDB] Log/alert failed (non-fatal): {e}")

        should_stop, stop_reasons = evaluate_early_stop(metrics, n_episodes)
        if should_stop:
            reason_text = "; ".join(stop_reasons)
            print(f"  [EARLY STOP] {reason_text}", flush=True)
            if WANDB_OK:
                try:
                    wandb.alert(
                        "sdpo_early_stop",
                        f"iter={iteration}; {reason_text}; recommended next run: lower LR and/or slow SDPO handoff",
                        level=wandb.AlertLevel.ERROR,
                    )
                except Exception as e:
                    print(f"  [WANDB] Early-stop alert failed (non-fatal): {e}")
            if EARLY_STOP_SAVE_CHECKPOINT and HUB_MODEL_ID:
                print(f"  [CHECKPOINT] Saving early-stop iter {iteration+1}...")
                save_and_push_checkpoint(buyer_model, tokenizer, metrics, iteration, final=False, processor=buyer_processor)
            break

        should_ckpt = (
            CHECKPOINT_EVERY > 0
            and (iteration + 1) % CHECKPOINT_EVERY == 0
            and iteration < NUM_ITERS - 1
        )
        if should_ckpt:
            print(f"  [CHECKPOINT] Saving iter {iteration+1}...")
            save_and_push_checkpoint(buyer_model, tokenizer, metrics, iteration, final=False, processor=buyer_processor)

    print(f"\n{'=' * 70}\nSAVING FINAL\n{'=' * 70}")
    save_path = Path(OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    buyer_model.save_pretrained(save_path)
    (buyer_processor or tokenizer).save_pretrained(save_path)
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if PUSH_TRAINING_SCRIPT:
        try:
            shutil.copyfile(__file__, save_path / "train_negotiation_sdpo.py")
        except Exception:
            pass
    print(f"  Saved to {save_path}")

    if HUB_MODEL_ID:
        save_and_push_checkpoint(buyer_model, tokenizer, metrics, NUM_ITERS - 1, final=True, processor=buyer_processor)

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"COMPLETE  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'=' * 70}")

    if WANDB_OK:
        try:
            wandb.alert("sdpo_negotiation_complete", f"iters={NUM_ITERS}; final_reward={metrics[-1]['mean_reward']:.4f}; final_deal_rate={metrics[-1]['deal_rate']:.3f}", level=wandb.AlertLevel.INFO)
            wandb.finish()
            print(f"[WANDB] Finished. Run: {wandb_run.url if wandb_run else '(unavailable)'}")
        except Exception as e:
            print(f"[WANDB] Finish failed (non-fatal): {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
