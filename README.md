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

`train_negotiation_clean.py` was the old pure buyer-only prototype and has been deleted. It predated Thought stripping, batched generation, monitoring, checkpoint branches, and later memory/stability fixes.

## Pure Negotiation-RLVR setup

`train_negotiation_pure.py` follows the negotiation paper architecture:

- Trainable **buyer** policy.
- Frozen **seller** model as the environment counterparty.
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
- W&B logging + alerts.
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
| KL anchor | `0.01` (pure script only; SDPO is now ref-free) |
| Buyer temp | `1.0` |
| Seller temp | `0.7` |
| Checkpoint every | `10` iters |
| Recommended hardware | `a100-large` |

## Local syntax check

```bash
python3 -m py_compile train_negotiation_pure.py train_negotiation_sdpo.py train_negotiation_dual_role.py eval_negotiation.py
```

`eval_negotiation.py` uses the same 18-category AmazonHistoryPrice loader and the same 802/128 seed split as the training scripts. Override `TRAIN_SPLIT_SIZE` / `TEST_SPLIT_SIZE` only for controlled split-parity experiments.

## SDPO+GRPO 8B run

`train_negotiation_sdpo.py` is the intended first self-distillation experiment. It keeps the pure buyer-only environment but uses hindsight verifier feedback and same-product on-policy rollout demos to compute dense SDPO token advantages. Current real-run defaults are deliberately bolder than the initial canaries: `SDPO_LAMBDA=0.5` gives equal weight to the GRPO scalar reward and SDPO token shaping; `SDPO_FEEDBACK_MODE=strict` avoids exact seller-cost leakage.

Earlier smoke validation completed on A100 with Qwen3-4B-Instruct-2507 and full format settings (`MAX_TURNS=6`, `MAX_NEW_TOKENS=300`): job [`6a05a28a3308d79117b8f560`](https://huggingface.co/jobs/ZeterMordio/6a05a28a3308d79117b8f560), model repo [`ZeterMordio/anchor-negotiation-sdpo-smoke-fullfmt`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-smoke-fullfmt). It completed 1 iteration and pushed metrics/model.

Qwen3.5 reasoning ablations use `REASONING_MODE`: `option_a` (default) disables native chat-template thinking and keeps the explicit private `Thought:` field, while hard-canonicalizing public text to `Talk:` plus the first `Action:` line. `option_b` enables native Qwen thinking (`enable_thinking=True` in chat-template calls) and switches the role prompts to a native protocol: the model reasons inside private `<think>...</think>`, then outputs only public `Talk:` + `Action:`. `NATIVE_PUBLIC_FINALIZER=1` (default for option_b) lets Qwen spend `NATIVE_THINK_TOKENS` on hidden reasoning, forcibly closes the private block if needed, and runs a short `NATIVE_FINAL_TOKENS` decode for the final public Talk/Action so longer reasoning budgets do not become format errors. Generated native-think blocks are kept in same-role private assistant history and stripped before opponent visibility/action parsing. The script logs native/explicit reasoning marker counts per iteration and can print raw/public buyer samples with `DEBUG_SAMPLE_BUYER_OUTPUTS`.

The SDPO update path is optimized for the 8B run: buyer turns are flattened into pre-tokenized microbatch examples, teacher/student completion masks are aligned row-wise, and CPU-state AdamW steps once per production-shape iteration by default (`UPDATE_MICROBATCH_SIZE=4`, `OPTIM_STEP_EVERY_GROUPS=16`). It is now a true ref-free/on-policy objective (`KL_COEF=0.0` by default): the update does not load a frozen reference-policy model or run a reference forward; it uses sampled-token policy-gradient loss `-A * logπ` over buyer completion tokens, while the SDPO self-teacher remains the current buyer model under hindsight feedback. Current serious-run optimizer defaults are `NUM_ITERS=60`, `LR=5e-6` with `WARMUP_STEPS=10`, `WEIGHT_DECAY=0.01`, `GRAD_CLIP_NORM=1.0`, and `ROLLOUT_MAX_LENGTH=UPDATE_MAX_LENGTH=3072`. W&B logs per-phase update timers under `perf/update_*_s` plus `perf/update_examples`, `perf/optimizer_steps`, `train/optimizer_global_step`, `train/lr`, and `train/grad_norm_last`; `perf/update_ref_forward_s` is retained as a zero-valued compatibility metric. The ref-free smoke completed at [`hf job 6a0a6998e7940de6ee6cdfa3`](https://huggingface.co/jobs/ZeterMordio/6a0a6998e7940de6ee6cdfa3), W&B run [`itnu5od5`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/itnu5od5), and model/metrics repo [`ZeterMordio/anchor-negotiation-sdpo-ref-free-smoke`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-ref-free-smoke). The previous pre-ref-free GPU smoke completed at [`hf job 6a0a5fd2a5e509f1a8413e2b`](https://huggingface.co/jobs/ZeterMordio/6a0a5fd2a5e509f1a8413e2b), W&B run [`fnscexas`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/fnscexas), and model/metrics repo [`ZeterMordio/anchor-negotiation-sdpo-perf-smoke`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-perf-smoke).

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 12h \
  --with torch==2.6.0 \
  --with transformers \
  --with accelerate \
  --with huggingface_hub \
  --with wandb \
  --with liger-kernel \
  --secrets HF_TOKEN \
  --secrets WANDB_API_KEY \
  --env MODEL_NAME=Qwen/Qwen3-8B \
  --env SELLER_MODEL_NAME=Qwen/Qwen3-8B \
  --env NUM_ITERS=60 \
  --env BATCH_SIZE=16 \
  --env GROUP_SIZE=8 \
  --env MAX_TURNS=6 \
  --env MAX_NEW_TOKENS=300 \
  --env LR=5e-6 \
  --env WEIGHT_DECAY=0.01 \
  --env WARMUP_STEPS=10 \
  --env GRAD_CLIP_NORM=1.0 \
  --env KL_COEF=0.0 \
  --env SDPO_LAMBDA=0.5 \
  --env SDPO_FEEDBACK_MODE=strict \
  --env SDPO_ADV_CLIP=5.0 \
  --env BUYER_TEMP=1.0 \
  --env SELLER_TEMP=0.7 \
  --env CHECKPOINT_EVERY=10 \
  --env GEN_BATCH_LIMIT=128 \
  --env UPDATE_MICROBATCH_SIZE=4 \
  --env OPTIM_STEP_EVERY_GROUPS=16 \
  --env UPDATE_PAD_TO_MULTIPLE_OF=8 \
  --env ROLLOUT_MAX_LENGTH=3072 \
  --env UPDATE_MAX_LENGTH=3072 \
  --env GRADIENT_CHECKPOINTING=1 \
  --env HUB_MODEL_ID=ZeterMordio/anchor-negotiation-sdpo \
  --env WANDB_ENTITY=chalk \
  --env WANDB_PROJECT=anchor-negotiation-sdpo \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  train_negotiation_sdpo.py
```

## W&B monitoring

The scripts now use [Weights & Biases](https://wandb.ai/) instead of Trackio.

- `WANDB_ENTITY` is the W&B account or team namespace that owns the run. For the available API key, the default entity is `chalk`.
- `WANDB_PROJECT` is the project bucket/group inside that entity. Projects are created automatically on first run; no web-page setup is required.
- `RUN_NAME` is the human-readable run name. Leave unset for the standardized auto-name; set it only for one-off manual labels.
- `WANDB_GROUP` groups related reruns/ablations. Leave unset for the standardized auto-group.
- `WANDB_MODE=online` logs to the W&B web app; use `WANDB_MODE=offline` for local dry runs.
- HF Jobs must receive `--secrets WANDB_API_KEY` in addition to `--secrets HF_TOKEN`.

Best practice for this repo: keep run names short enough to scan, but put full detail in `wandb.config`.

Run-name schema:

| Script | Default name shape |
|---|---|
| Pure | `pure__<model>__i<iters>_b<batch>xg<group>__lr<lr>_kl<kl>__s<seed>` |
| SDPO/SDRO | `sdpo__<model>__l<lambda>__<distill>__i<iters>_b<batch>xg<group>__fb<mode>_clip<clip>__lr<lr>_kl<kl>__s<seed>` |
| Dual-role | `dual__<model>__i<iters>_b<batch>xg<group>__dr<ratio>_rae<decay>_<ref>_<advnorm>__lr<lr>_kl<kl>` |

For SDPO/SDRO, `<distill>` is currently `tokgap`. Future logit-level variants are named like `topk64-js-tri-ema0p99`.

Put in the title: method, model, lambda, distillation family, iters, batch×group, feedback mode, clip, LR, KL, seed. Put everything else in W&B config/tags: max turns/tokens, temps, optimizer, CPU/CUDA AdamW, Liger, memory caps, demo length, exact divergence, EMA/trust-region flags, Hub repo, git commit, hardware, etc.

Example run URLs will look like `https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/<run-id>`.

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
  --with wandb \
  --with liger-kernel \
  --secrets HF_TOKEN \
  --secrets WANDB_API_KEY \
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
  --env WANDB_ENTITY=chalk \
  --env WANDB_PROJECT=anchor-negotiation-pure \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  train_negotiation_pure.py
```
