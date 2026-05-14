# Anchor: Bilateral Negotiation via RLVR

Replication and extension of ["Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards"](https://huggingface.co/papers/2604.09855) (paper `2604.09855`).

**Goal:** train a small open model (Qwen3-4B/8B family) to negotiate effectively under incomplete information on a tight compute budget.

## Active scripts

| File | Purpose |
|---|---|
| `train_negotiation_pure.py` | **Current pure Negotiation-RLVR training script.** Buyer-only GRPO against a frozen regulated seller, starting from the optimized SPIRAL/RLVR infrastructure but with all self-play/RAE/seller-training pieces removed. |
| `train_negotiation_sdpo.py` | **SDPO+GRPO buyer-only experiment.** Forks the pure script, defaults to Qwen3-8B, and adds strict feedback-conditioned self-teacher credit assignment while preserving the frozen regulated seller setup. |
| `train_negotiation_dual_role.py` | SPIRAL/RLVR hybrid self-play script. Shared policy plays buyer + seller with RAE. Kept for comparison and future dual-role experiments. |
| `eval_negotiation.py` | Evaluation script for trained checkpoints against a frozen seller under the paper-style protocol. |
| `JOURNAL.md` | Engineering log: papers, bugs, runs, hyperparameters, job IDs, and rationale. |

`train_negotiation_clean.py` was the old pure buyer-only prototype and has been deleted. It predated Thought stripping, batched generation, Trackio, checkpoint branches, and later memory/stability fixes.

## Pure Negotiation-RLVR setup

`train_negotiation_pure.py` follows the negotiation paper architecture:

- Trainable **buyer** policy.
- Frozen **seller/reference** model as the environment counterparty.
- Buyer starts every episode; only buyer turns receive policy-gradient updates.
- Seller is regulated: it cannot accept/propose below its private cost.
- Opponent-visible dialogue strips `Thought:` blocks, preserving the paper's hidden-scratchpad / incomplete-information assumption.
- Reward: `R = (budget - P_final) / |budget - cost|`, clipped to `[-1, 1]`; no-deal/quit/timeout = `0`; buyer format/budget/protocol errors = `-1`.

Non-conflicting improvements retained from the SPIRAL/RLVR script:

- Batched turn-parallel generation.
- Runtime private-info leak guards.
- Memory-efficient token log-prob computation.
- Clamped log-ratio for GRPO stability.
- Group-level advantage normalization.
- Optional Liger kernels.
- Trackio logging + alerts.
- Periodic HF Hub branch checkpoints.

Explicitly removed from the SPIRAL setup:

- Shared-policy self-play.
- Seller training.
- Zero-sum seller reward.
- RAE / per-role baselines.
- Dual-role ratio.

## Default initial pure run

The 42-iteration pure run has completed and pushed to [`ZeterMordio/anchor-negotiation-pure`](https://huggingface.co/ZeterMordio/anchor-negotiation-pure). Its launch defaults were:

| Setting | Default |
|---|---:|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Iterations | `42` |
| Batch size | `16` products |
| Group size | `8` rollouts/product |
| Episodes / iter | `128` |
| Max turns | `6` |
| Max tokens / response | `300` |
| LR | `1e-6` |
| KL anchor | `0.01` |
| Buyer temp | `1.0` |
| Seller temp | `0.7` |
| Checkpoint every | `10` iters |
| Recommended hardware | `a100-large` |

## Local syntax check

```bash
python3 -m py_compile train_negotiation_pure.py train_negotiation_sdpo.py train_negotiation_dual_role.py eval_negotiation.py
```

## SDPO+GRPO 8B run

`train_negotiation_sdpo.py` is the intended first self-distillation experiment. It keeps the pure buyer-only environment but uses hindsight verifier feedback and same-product on-policy rollout demos to compute dense SDPO token advantages. Defaults are conservative: `SDPO_LAMBDA=0.9` keeps 90% GRPO scalar advantage and 10% SDPO token advantage; `SDPO_FEEDBACK_MODE=strict` avoids exact seller-cost leakage.

Smoke validation completed on A100 with Qwen3-4B-Instruct-2507 and full format settings (`MAX_TURNS=6`, `MAX_NEW_TOKENS=300`): job [`6a05a28a3308d79117b8f560`](https://huggingface.co/jobs/ZeterMordio/6a05a28a3308d79117b8f560), model repo [`ZeterMordio/anchor-negotiation-sdpo-smoke-fullfmt`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-smoke-fullfmt). It completed 1 iteration, pushed metrics/model, and confirmed Trackio alerts after switching from string alert levels to `trackio.AlertLevel.*`.

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 12h \
  --with torch==2.6.0 \
  --with transformers \
  --with accelerate \
  --with huggingface_hub \
  --with trackio \
  --with liger-kernel \
  --secrets HF_TOKEN \
  --env MODEL_NAME=Qwen/Qwen3-8B \
  --env SELLER_MODEL_NAME=Qwen/Qwen3-8B \
  --env NUM_ITERS=42 \
  --env BATCH_SIZE=16 \
  --env GROUP_SIZE=8 \
  --env MAX_TURNS=6 \
  --env MAX_NEW_TOKENS=300 \
  --env LR=1e-6 \
  --env KL_COEF=0.01 \
  --env SDPO_LAMBDA=0.9 \
  --env SDPO_FEEDBACK_MODE=strict \
  --env SDPO_ADV_CLIP=5.0 \
  --env BUYER_TEMP=1.0 \
  --env SELLER_TEMP=0.7 \
  --env CHECKPOINT_EVERY=10 \
  --env GEN_BATCH_LIMIT=128 \
  --env GRADIENT_CHECKPOINTING=1 \
  --env HUB_MODEL_ID=ZeterMordio/anchor-negotiation-sdpo \
  --env TRACKIO_SPACE=ZeterMordio/anchor-dashboard \
  --env TRACKIO_PROJECT=anchor-negotiation-sdpo \
  --env RUN_NAME=sdpo-qwen3-8b-42it \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  train_negotiation_sdpo.py
```

## HF Jobs launch command (do not run unless ready)

Use the lightweight `uv` flow and exact `torch==2.6.0` pin. The `<2.7` form caused shell-redirection failures via the Jobs API.

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 8h \
  --with torch==2.6.0 \
  --with transformers \
  --with accelerate \
  --with huggingface_hub \
  --with trackio \
  --with liger-kernel \
  --secrets HF_TOKEN \
  --env MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  --env SELLER_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  --env NUM_ITERS=42 \
  --env BATCH_SIZE=16 \
  --env GROUP_SIZE=8 \
  --env MAX_TURNS=6 \
  --env MAX_NEW_TOKENS=300 \
  --env LR=1e-6 \
  --env KL_COEF=0.01 \
  --env BUYER_TEMP=1.0 \
  --env SELLER_TEMP=0.7 \
  --env CHECKPOINT_EVERY=10 \
  --env GEN_BATCH_LIMIT=128 \
  --env GRADIENT_CHECKPOINTING=1 \
  --env HUB_MODEL_ID=ZeterMordio/anchor-negotiation-pure \
  --env TRACKIO_SPACE=ZeterMordio/anchor-dashboard \
  --env TRACKIO_PROJECT=anchor-negotiation-pure \
  --env RUN_NAME=pure-qwen3-4b-42it \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  train_negotiation_pure.py
```
