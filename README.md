# Anchor: Bilateral Negotiation via RLVR

Replication and extension of ["Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards"](https://huggingface.co/papers/2604.09855) (paper `2604.09855`).

**Goal:** train a small open model (currently: Qwen3.5/Qwen3 8B–9B family) to negotiate effectively under incomplete information on a tight compute budget.

**Current direction:** SDPO+GRPO buyer-only training against a frozen regulated seller is the main path for improving negotiation. Pure GRPO (analogous to the paper) is the baseline. The older dual-role/SPIRAL path is archived for comparison, not active development.

## Active scripts

| File | Purpose |
|---|---|
| `train_negotiation_sdpo_qwen35.py` | **Main current training script.** Qwen3.5/ImageTextToText SDPO variant with native-thinking/finalizer support, Qwen3.5 loader compatibility, LoRA adapter canaries, and production-shape smoke instrumentation. |
| `train_negotiation_sdpo.py` | Qwen3-8B fallback/control script. SDPO+GRPO buyer-only training with strict feedback-conditioned self-teacher credit assignment and the same optional LoRA adapter path. |
| `train_negotiation_pure.py` | Baseline pure Negotiation-RLVR script. Buyer-only GRPO against a frozen regulated seller, with older dual-role/SPIRAL-like self-play/RAE/seller-training pieces removed. |
| `eval_negotiation.py` | Evaluation script for trained checkpoints against a frozen seller under the paper-style protocol; now supports both CausalLM checkpoints and Qwen3.5 ImageTextToText wrappers. |
| `JOURNAL.md` | Engineering log: papers, bugs, runs, hyperparameters, job IDs, and rationale. |
| `deprecated/` | Historical notes and old approaches, including the dual-role/SPIRAL script. |
| `tools/` | One-off support utilities such as W&B metric backfills. |

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
python3 -m py_compile train_negotiation_pure.py train_negotiation_sdpo.py train_negotiation_sdpo_qwen35.py eval_negotiation.py
```

`eval_negotiation.py` uses the same 18-category AmazonHistoryPrice loader and the same 802/128 seed split as the training scripts. Override `TRAIN_SPLIT_SIZE` / `TEST_SPLIT_SIZE` only for controlled split-parity experiments.

## SDPO+GRPO Qwen3.5 main path

`train_negotiation_sdpo_qwen35.py` is now the main self-distillation experiment. It keeps the pure buyer-only environment but uses hindsight verifier feedback and same-product on-policy rollout demos to compute dense SDPO token advantages. Qwen3-8B remains the fallback/control path in `train_negotiation_sdpo.py`.

Earlier smoke validation completed on A100 with Qwen3-4B-Instruct-2507 and full format settings (`MAX_TURNS=6`, `MAX_NEW_TOKENS=300`): job [`6a05a28a3308d79117b8f560`](https://huggingface.co/jobs/ZeterMordio/6a05a28a3308d79117b8f560), model repo [`ZeterMordio/anchor-negotiation-sdpo-smoke-fullfmt`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-smoke-fullfmt). It completed 1 iteration and pushed metrics/model.

Qwen3.5 reasoning ablations use `REASONING_MODE`: `option_a` (default) disables native chat-template thinking and keeps the explicit private `Thought:` field, while hard-canonicalizing public text to `Talk:` plus the first `Action:` line. `option_b` enables native Qwen thinking (`enable_thinking=True` in chat-template calls) and switches the role prompts to a native protocol: the model reasons inside private `<think>...</think>`, then outputs only public `Talk:` + `Action:`. `NATIVE_PUBLIC_FINALIZER=1` (default for option_b) lets Qwen spend `NATIVE_THINK_TOKENS` on hidden reasoning, forcibly closes the private block if needed, and runs a short `NATIVE_FINAL_TOKENS` same-assistant continuation prefilled only with `Talk:` for the final public Talk/Action so longer reasoning budgets do not become format errors. Generated native-think blocks are kept in same-role private assistant history and stripped before opponent visibility/action parsing. The script logs native/explicit reasoning marker counts per iteration and can print raw/public buyer samples with `DEBUG_SAMPLE_BUYER_OUTPUTS`. The Qwen3.5 parser now handles trailing action punctuation such as `$25.00.`, and the privacy guard still fails on structured buyer `Shopping List`/`budget_limit` leakage while allowing harmless public mentions of the phrase “shopping list”.

Latest Qwen3.5 2-iteration production-shape smoke completed: HF job [`6a0fbffbb33ece92698bfe69`](https://huggingface.co/jobs/ZeterMordio/6a0fbffbb33ece92698bfe69), model/metrics repo [`ZeterMordio/anchor-negotiation-sdpo-qwen35-2iter-gen96`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-2iter-gen96). Config: `Qwen/Qwen3.5-9B`, `NUM_ITERS=2`, `BATCH_SIZE=16`, `GROUP_SIZE=8`, option-B native thinking, `NATIVE_FINAL_TOKENS=96`, `LR=3e-6` with warmup, `UPDATE_LENGTH_BUCKETING=1`, `torch.inference_mode()` teacher forward, CPU-state AdamW, and no ABI-fragile `causal-conv1d`/`flash-linear-attention` wheels. It pushed `iter-1`, `iter-2`, and final `main`. Metrics: iter0 reward `-0.0744`, deal `46.9%`, buyer-format errors `12/128`; iter1 reward `-0.1196`, deal `38.3%`, buyer-format errors `19/128`; early-stopped after 2 consecutive format-warning iterations. Runtime averaged `~20.9 min/iter`: rollout `~661s`, update `~595s`; update bottlenecks were backward `~56%`, teacher+policy forwards `~29%`, and CPU AdamW optimizer `~13%`. Peak reserved VRAM was `~80.3GB`, so A100 80GB is effectively saturated.

The SDPO update path is already optimized around flattened buyer-turn examples: teacher/student completion masks are aligned row-wise, and dense full-parameter runs use CPU-state AdamW once per production-shape iteration by default (`UPDATE_MICROBATCH_SIZE=4`, `OPTIM_STEP_EVERY_GROUPS=16`). It is now a true ref-free/on-policy objective (`KL_COEF=0.0` by default): the update does not load a frozen reference-policy model or run a reference forward; it uses sampled-token policy-gradient loss `-A * logπ` over buyer completion tokens, while the SDPO self-teacher remains the current buyer model under hindsight feedback. W&B logs per-phase update timers under `perf/update_*_s` plus `perf/update_examples`, `perf/optimizer_steps`, `train/optimizer_global_step`, `train/lr`, and `train/grad_norm_last`; `perf/update_ref_forward_s` is retained as a zero-valued compatibility metric. The ref-free Qwen3-8B smoke completed at [`hf job 6a0a6998e7940de6ee6cdfa3`](https://huggingface.co/jobs/ZeterMordio/6a0a6998e7940de6ee6cdfa3), W&B run [`itnu5od5`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/itnu5od5), and model/metrics repo [`ZeterMordio/anchor-negotiation-sdpo-ref-free-smoke`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-ref-free-smoke). The previous pre-ref-free GPU smoke completed at [`hf job 6a0a5fd2a5e509f1a8413e2b`](https://huggingface.co/jobs/ZeterMordio/6a0a5fd2a5e509f1a8413e2b), W&B run [`fnscexas`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/fnscexas), and model/metrics repo [`ZeterMordio/anchor-negotiation-sdpo-perf-smoke`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-perf-smoke).

## LoRA adapter canary

The next efficiency path is an opt-in LoRA adapter ablation, not a replacement for the dense SDPO control. Enable it with `USE_LORA=1`; both active SDPO scripts then freeze base buyer weights, inject LoRA into exact text-transformer linear modules only, default to CUDA AdamW over adapter params, keep the frozen seller unadapted, and push adapter checkpoints plus processor/tokenizer files. Qwen3.5 target inference intentionally avoids broad `all-linear` targeting so vision/MTP/head modules are not adapted.

Completed Qwen3.5 LoRA canary: HF job [`6a111358b33ece92698c0fb4`](https://huggingface.co/jobs/ZeterMordio/6a111358b33ece92698c0fb4), W&B run [`1c7zz7e9`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/1c7zz7e9), adapter repo [`ZeterMordio/anchor-negotiation-sdpo-qwen35-lora-r64-canary-20260523`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-lora-r64-canary-20260523). Config: `LORA_R=64`, `LORA_ALPHA=64`, `LORA_DROPOUT=0.0`, `LR=1e-5`, `REASONING_MODE=option_b`, `NATIVE_FINAL_TOKENS=96`, `GEN_BATCH_LIMIT=96`, `UPDATE_MICROBATCH_SIZE=4`, and `OPTIM_STEP_EVERY_GROUPS=16`. It injected `248` exact text targets and trained `173,113,344 / 9,582,927,088` params (`1.806%`).

Result: the path is technically sound but `LR=1e-5` is too aggressive for the serious run. Iter0 matched the dense control before any update: reward `-0.0744`, deal `46.9%`, buyer-format errors `12/128`. Iter1 after one LoRA update degraded below dense: reward `-0.1890`, deal `32.8%`, buyer-format errors `18/128`; it early-stopped after 2 consecutive format-warning iterations. The dense Qwen3.5 smoke at `LR=3e-6` had iter1 reward `-0.1196`, deal `38.3%`, buyer-format errors `19/128`.

Efficiency read: LoRA eliminated optimizer-state pressure and pushed compact adapter checkpoints (`adapter_model.safetensors` ~`693MB`), but it did not materially reduce production-shape wall clock. Iter update times were `602s` and `574s`, roughly dense-like (`~595s`), because policy/teacher forwards plus backward still dominate. Update peak reserved VRAM fell to `54.6GB`/`57.6GB`, but rollout still peaked at `81.0GB`/`83.0GB`, so Qwen3.5 still needs A100-80GB-class hardware until generation/fastpath work changes.

Recommendation: do not spend more A100 time on LoRA for wall-clock speed. Keep it as an opt-in memory/checkpoint ablation only. Lower-LR LoRA (`5e-6`/`3e-6`) remains a possible quality ablation, but it is not the next lever if the goal is shorter full Qwen3.5 jobs; the next speed work should target rollout generation, Qwen3.5 fastpath kernels, or avoiding the second live seller model during update/rollout.

Dense compatibility check after adding LoRA: with `USE_LORA` unset, both active SDPO scripts keep the pre-LoRA dense defaults for `OPTIMIZER`, default W&B run name, and default W&B group. The current LoRA code is inert unless `USE_LORA=1`.

Rollout-token telemetry is now enabled by default with `ROLLOUT_TOKEN_TELEMETRY=1`. Each Qwen3.5 iteration logs per-role and total rollout token counters: prompt mean, first-pass generated mean, finalizer rate/tokens, total generated mean, first-pass parseable-action rate, final parseable-action rate, public tokens to first action, public tail tokens after the first action, tail share, and max prompt/generated lengths. W&B receives the same values under `perf/rollout_token_*`. This is instrumentation only; it does not stop generation early or change the objective.

Qwen3.5 fastpath kernel canary: HF job [`6a112e00e3c0b51e1ca5d9a4`](https://huggingface.co/jobs/ZeterMordio/6a112e00e3c0b51e1ca5d9a4) completed on `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` with `torch=2.10.0+cu128`, `transformers=5.9.0`, `causal-conv1d=1.6.2.post1`, `flash-linear-attention=0.5.0`, and `triton=3.6.0`. The key install detail is `pip install --no-build-isolation causal-conv1d` after installing build deps, so the extension compiles against the image torch/CUDA instead of pulling an isolated torch/cu130 build env. The canary proved all fastpath imports, Qwen3.5 load, one short generate, and one tiny full-model backward. It is not a throughput benchmark: cold model download/load and one prompt dominate. Earlier uv-image job [`6a112b6ee3c0b51e1ca5d98a`](https://huggingface.co/jobs/ZeterMordio/6a112b6ee3c0b51e1ca5d98a) failed because the standard HF uv image lacked `nvcc` for `causal-conv1d` source build.

Same-shape fastpath 2-iteration run completed: HF job [`6a11339eb33ece92698c11b9`](https://huggingface.co/jobs/ZeterMordio/6a11339eb33ece92698c11b9), W&B run [`fat0mgqc`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/fat0mgqc), model/metrics repo [`ZeterMordio/anchor-negotiation-sdpo-qwen35-fastpath-2iter-20260523-045622`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-fastpath-2iter-20260523-045622). Config matched the prior dense Qwen3.5 2-iter smoke shape (`BATCH_SIZE=16`, `GROUP_SIZE=8`, option-B native thinking, `NATIVE_THINK_TOKENS=300`, `NATIVE_FINAL_TOKENS=96`, `ROLLOUT_MAX_LENGTH=UPDATE_MAX_LENGTH=3072`, `LR=3e-6`, CPU-state AdamW, `USE_LORA=0`) on the CUDA-devel fastpath stack. Result: fastpath is a real update-path improvement, not a rollout fix. Printed compute time was `1098s` then `798s` (`15.8 min/iter` average), versus the old dense fallback-stack average `~20.9 min/iter`; update fell to `423s`/`276s` from `~595s`, while rollout stayed large at `675s`/`523s`. Full script wall time was `39.2 min` because dense checkpoint writes/uploads still dominate canary overhead. Peak reserved VRAM improved to `73.3GB`/`74.6GB` versus the old `~80.3GB`. Token telemetry explains the remaining wall-clock ceiling: first-pass generation averaged `~299` tokens, finalizer ran `~99%` of sequences, and total generated length stayed `~349` tokens/turn, so parseable-action stopping or shorter native-thinking/finalizer behavior is a bigger remaining lever than more kernel work. Quality was not improved on this stack: iter0/iter1 reward `-0.1019`/`-0.1905`, deal `46.1%`/`35.2%`, buyer-format errors `21`/`17`, and the run early-stopped after two format-warning iterations.

## Cost-safe Qwen3.5 launch policy

For dense Qwen3.5 SDPO, keep the current hardware and paper-aligned training shape: `a100-large`, full-parameter buyer update, frozen full-size seller, `BATCH_SIZE=16`, `GROUP_SIZE=8`, `MAX_TURNS=6`, option-B native thinking, and strict feedback. Cheaper single-GPU flavors do not have enough VRAM for buyer+seller dense rollout, and multi-GPU flavors are not cost-effective without a separate sharding plan. Hugging Face bills Jobs by the minute while Starting/Running, and `a100-large` is currently `$2.50/hr`; always set a finite timeout and cancel irrelevant jobs. See [HF Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing).

Use the prebuilt fastpath image path instead of rebuilding dependencies inside each job. The image definition is in [`docker/qwen35-fastpath/Dockerfile`](docker/qwen35-fastpath/Dockerfile); build/push notes are in [`docker/qwen35-fastpath/README.md`](docker/qwen35-fastpath/README.md). The training script is still uploaded per run, so code changes are not baked into the dependency image.

The `--image` value must be pullable by HF Jobs, such as a Docker Hub image or Hugging Face Space image (`hf.co/spaces/<owner>/<space>`). Do not submit a local-only tag like `anchor-qwen35-fastpath:...`; the launcher rejects those refs to avoid paid failed jobs.

Normal workflow has one launcher and two modes:

- `--dry-run` prints the exact upload/job commands and costs nothing.
- `--execute` uploads the current training-script snapshot and starts the HF Job.

Preview first:

```bash
uv run --with huggingface_hub python tools/launch_qwen35_fastpath_sdpo.py \
  --dry-run
```

Submit only after reviewing the printed command:

```bash
uv run --with huggingface_hub python tools/launch_qwen35_fastpath_sdpo.py \
  --execute
```

The launcher default image is [`hf.co/spaces/ZeterMordio/anchor-qwen35-fastpath`](https://huggingface.co/spaces/ZeterMordio/anchor-qwen35-fastpath), which was built from the pinned Dockerfile and verified by cheap CPU Jobs import smoke [`6a14d2e4404eb93b204f1c99`](https://huggingface.co/jobs/ZeterMordio/6a14d2e4404eb93b204f1c99).

Launcher invariants:

- `--flavor a100-large`
- `--timeout 22h` for the 60-iteration Qwen3.5 dense run
- `CHECKPOINT_EVERY=10`; W&B still logs every iteration
- `USE_LORA=0`
- `OPTIMIZER=adamw_cpu`
- `ROLLOUT_TOKEN_TELEMETRY=1`
- labels include `cost_policy=a100-checkpoint10-timeout`

Monitor/cancel discipline:

```bash
hf jobs ps
hf jobs logs -f <job-id>
hf jobs logs --tail 100 <job-id>
hf jobs cancel <job-id>
```

## Qwen3-8B fallback/control

`train_negotiation_sdpo.py` remains the Qwen3-8B fallback/control script. Its dense defaults are deliberately bolder than the initial canaries: `SDPO_LAMBDA=0.5` gives equal weight to the GRPO scalar reward and SDPO token shaping; `SDPO_FEEDBACK_MODE=strict` avoids exact seller-cost leakage.

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
| Dual-role (deprecated) | `dual__<model>__i<iters>_b<batch>xg<group>__dr<ratio>_rae<decay>_<ref>_<advnorm>__lr<lr>_kl<kl>` |

For SDPO/SDRO, `<distill>` is currently `tokgap`. Future logit-level variants are named like `topk64-js-tri-ema0p99`.

Put in the title: method, model, lambda, distillation family, iters, batch×group, feedback mode, clip, LR, KL, seed. Put everything else in W&B config/tags: max turns/tokens, temps, optimizer, CPU/CUDA AdamW, Liger, memory caps, demo length, exact divergence, EMA/trust-region flags, Hub repo, git commit, hardware, etc.

Example run URLs will look like `https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/<run-id>`.

## HF Jobs launch command (do not run unless ready)

The SDPO command above is the current production template. The pure-GRPO command below is retained as a baseline/reproduction template.

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
