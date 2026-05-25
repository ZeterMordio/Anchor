"""
Launch dense Qwen3.5 SDPO on the prebuilt fastpath HF Jobs image.

Default policy is cost-safe but objective-preserving:
- a100-large only.
- CUDA fastpath dependency image supplied by --image.
- dense full-parameter path: USE_LORA=0.
- CHECKPOINT_EVERY=10; W&B still logs every iteration inside the training script.
- finite timeout, default 22h for a 60-iteration run.

The script uploads the current training file to a small Hub model repo, then the
remote job downloads that exact snapshot and runs it. It dry-runs by default.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPO_ROOT / "train_negotiation_sdpo_qwen35.py"
DEFAULT_OWNER = "ZeterMordio"
DEFAULT_IMAGE = "hf.co/spaces/ZeterMordio/anchor-qwen35-fastpath"
LOCAL_IMAGE_PREFIXES = ("localhost/", "127.0.0.1/", "0.0.0.0/")


def utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run(cmd: list[str], *, execute: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd))
    if execute:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def validate_remote_image_ref(image: str) -> None:
    """Reject local-only image refs that HF Jobs cannot pull."""
    if image.startswith(LOCAL_IMAGE_PREFIXES):
        raise SystemExit(f"Image must be pullable by HF Jobs, not local-only: {image}")
    if "/" not in image:
        raise SystemExit(
            "Image must include a remote namespace/registry, e.g. "
            "zetermordio/anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0 "
            "or hf.co/spaces/ZeterMordio/anchor-qwen35-fastpath."
        )


def build_env(args: argparse.Namespace, run_repo: str, script_repo: str) -> dict[str, str]:
    return {
        "SCRIPT_REPO": script_repo,
        "MODEL_NAME": args.model,
        "SELLER_MODEL_NAME": args.seller_model,
        "HUB_MODEL_ID": run_repo,
        "OUTPUT_DIR": "/tmp/model",
        "NUM_ITERS": str(args.num_iters),
        "BATCH_SIZE": str(args.batch_size),
        "GROUP_SIZE": str(args.group_size),
        "MAX_TURNS": "6",
        "REASONING_MODE": "option_b",
        "NATIVE_THINK_TOKENS": str(args.native_think_tokens),
        "NATIVE_FINAL_TOKENS": str(args.native_final_tokens),
        "ROLLOUT_MAX_LENGTH": "3072",
        "UPDATE_MAX_LENGTH": "3072",
        "LR": args.lr,
        "WARMUP_STEPS": "10",
        "SDPO_LAMBDA": "0.9",
        "SDPO_LAMBDA_FINAL": "0.5",
        "SDPO_LAMBDA_DECAY_ITERS": "20",
        "SDPO_FEEDBACK_MODE": "strict",
        "SDPO_ADV_CLIP": "5.0",
        "OPTIMIZER": "adamw_cpu",
        "UPDATE_LENGTH_BUCKETING": "1",
        "UPDATE_MICROBATCH_SIZE": "4",
        "OPTIM_STEP_EVERY_GROUPS": "16",
        "CHECKPOINT_EVERY": str(args.checkpoint_every),
        "EARLY_STOP_SAVE_CHECKPOINT": "1",
        "ROLLOUT_TOKEN_TELEMETRY": "1",
        "USE_LIGER": "0",
        "USE_LORA": "0",
        "GEN_BATCH_LIMIT": str(args.gen_batch_limit),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_TAGS": "sdpo,negotiation,rlvr,qwen35,fastpath,dense,cost-safe",
        "WANDB_GROUP": "sdpo__q3.5-9b__fastpath__tokgap__fbstrict",
        "RUN_NAME": args.run_name
        or f"sdpo__q3.5-9b__fastpath__i{args.num_iters}_b{args.batch_size}xg{args.group_size}__lr{args.lr}__s42",
    }


def build_remote_command() -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            "export PYTHONUNBUFFERED=1",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "PY=/usr/bin/python",
            '$PY - <<\'PY\'',
            "import importlib.metadata as md, torch",
            "for pkg in ['torch', 'transformers', 'accelerate', 'causal-conv1d', 'flash-linear-attention', 'triton', 'wandb', 'huggingface_hub']:",
            "    try:",
            "        print(f'{pkg}={md.version(pkg)}')",
            "    except Exception as exc:",
            "        print(f'{pkg}=missing ({exc})')",
            "print('cuda', torch.version.cuda, 'devices', torch.cuda.device_count())",
            "PY",
            'SCRIPT_PATH=$($PY -c "import os; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=os.environ[\\"SCRIPT_REPO\\"], filename=\\"train_negotiation_sdpo_qwen35.py\\"))")',
            'echo "script=$SCRIPT_PATH"',
            '$PY "$SCRIPT_PATH"',
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Public Docker image containing the Qwen3.5 fastpath deps.",
    )
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--num-iters", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--timeout", default="22h")
    parser.add_argument("--flavor", default="a100-large")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--seller-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--native-think-tokens", type=int, default=300)
    parser.add_argument("--native-final-tokens", type=int, default=96)
    parser.add_argument("--gen-batch-limit", type=int, default=128)
    parser.add_argument("--lr", default="3e-6")
    parser.add_argument("--wandb-entity", default="chalk")
    parser.add_argument("--wandb-project", default="anchor-negotiation-sdpo")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--slug", default=utc_slug())
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting. This is the default.")
    parser.add_argument("--execute", action="store_true", help="Actually upload script and submit the HF Job.")
    args = parser.parse_args()

    if args.flavor != "a100-large":
        raise SystemExit("Refusing non-a100-large flavor for dense Qwen3.5 cost-safe policy.")
    if args.checkpoint_every != 10:
        raise SystemExit("Refusing CHECKPOINT_EVERY other than 10 for agreed cost-safe policy.")
    if not TRAINING_SCRIPT.exists():
        raise SystemExit(f"Missing training script: {TRAINING_SCRIPT}")
    validate_remote_image_ref(args.image)

    script_repo = f"{args.owner}/anchor-negotiation-sdpo-qwen35-fastpath-script-{args.slug}"
    run_repo = f"{args.owner}/anchor-negotiation-sdpo-qwen35-fastpath-{args.num_iters}iter-{args.slug}"

    upload_cmd = [
        "hf",
        "upload",
        script_repo,
        str(TRAINING_SCRIPT.relative_to(REPO_ROOT)),
        "train_negotiation_sdpo_qwen35.py",
        "--repo-type",
        "model",
        "--commit-message",
        f"Qwen3.5 fastpath training script {args.slug}",
    ]

    job_cmd = ["hf", "jobs", "run", "--flavor", args.flavor, "--timeout", args.timeout, "--detach"]
    for key, value in build_env(args, run_repo, script_repo).items():
        job_cmd.extend(["--env", f"{key}={value}"])
    job_cmd.extend(
        [
            "--secrets",
            "HF_TOKEN",
            "--secrets",
            "WANDB_API_KEY",
            "--label",
            "project=anchor",
            "--label",
            "purpose=qwen35-fastpath-dense",
            "--label",
            "cost_policy=a100-checkpoint10-timeout",
            "--",
            args.image,
            "bash",
            "-lc",
            build_remote_command(),
        ]
    )

    if not args.execute:
        print("[DRY-RUN] Add --execute to upload script and submit job.")
    run(upload_cmd, execute=args.execute)
    run(job_cmd, execute=args.execute)
    print(f"run_repo={run_repo}")
    print(f"script_repo={script_repo}")
    print("monitor: hf jobs logs -f <job-id>")
    print("recent logs: hf jobs logs --tail 100 <job-id>")
    print("cancel:  hf jobs cancel <job-id>")


if __name__ == "__main__":
    main()
