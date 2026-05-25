# Anchor Negotiation — Engineering Journal

> Authored by: Anton Künzi
> Last updated: 2026-05-25 22:47 UTC
> Session with: ZeterMordio

This document is the single source of truth for all design decisions,
bug fixes, hyperparameters, memory budgets, and experimental results.
It replaces the old `engineering_notebook.md` which is archived for reference.

---

## Quick Reference

### Current direction / canonical active plan

SDPO+GRPO buyer-only training is now the main approach for improving negotiation. It preserves the negotiation RLVR paper's buyer-vs-frozen-regulated-seller setup and adds feedback-conditioned self-teacher token credit. Pure GRPO remains the baseline; the dual-role/SPIRAL+RAE script is archived under `deprecated/` for provenance/comparison only.

| Parameter | Active SDPO plan | Qwen3.5 smoke lesson |
|-----------|------------------|----------------------|
| Main script | `train_negotiation_sdpo_qwen35.py` | `train_negotiation_sdpo.py` remains Qwen3-8B fallback/control |
| Model | `Qwen/Qwen3.5-9B` | Qwen3-8B if Qwen3.5 adapter path or native-thinking stability blocks progress |
| Environment | Buyer-only vs frozen regulated seller | Same, with ImageTextToText loader/native thinking support |
| Iters | 60 | 2-iter production-shape smoke first |
| Batch × group | 16 × 8 = 128 episodes/iter | 16 × 8 = 128 episodes/iter |
| LR | dense Qwen3.5 remains the serious path; LoRA `LR=1e-5` was too aggressive | lower-LR LoRA is not worth more A100 time for wall-clock speed unless specifically testing quality |
| Max turns / tokens | 6 / 300 | 6 / 300 hidden-think + short public finalizer |
| Buyer / seller temp | 1.0 / 0.7 | 1.0 / 0.7 |
| KL/reference | `KL_COEF=0.0`, no frozen reference-policy forward | same |
| SDPO lambda | GRPO-heavy handoff `0.9 -> 0.5` for Qwen3.5 LoRA canary | immediate balanced/dense updates looked less safe |
| Adapter path | dense full-parameter by default; `USE_LORA=1` is opt-in only | LoRA helps memory/checkpoint size, not current end-to-end wall clock |
| Hardware | `a100-large` minimum; Qwen3.5 rollout saturates 80GB | LoRA reduced update peak but rollout still needs A100-class VRAM |

### Historical dual-role hyperparameters (archived)

| Parameter | v10 A100 ✅ | v10 L40S ❌ |
|-----------|------------|------------|
| Model | Qwen3-4B-Instruct-2507 | Qwen3-4B-Instruct-2507 |
| Iters | 15 ✅ | 15 (OOM at iter 1) |
| Batch (products) | 16 | 16 |
| Group (rollouts/product) | 8 | 8 |
| LR | 1e-6 | 1e-6 |
| Max turns | 6 | 6 |
| Max tokens/turn | 300 | 300 |
| Buyer temp | 1.0 | 1.0 |
| Seller temp | 1.0 | 1.0 (self-play) |
| KL penalty | 0.01 | 0.0 (ref-free) |
| RAE decay | 0.95 | 0.95 |
| Dual-role ratio | 0.5 | 0.5 |
| Hardware | a100-large (80GB) | l40sx1 (44GB) |

### Verified Model IDs

| Model | Exists | Size | Notes |
|-------|--------|------|-------|
| `Qwen/Qwen3-4B` | ✅ | ~7.5GB bf16 | Instruct-merged. Toy Runs 1-3 v1-v6. |
| `Qwen/Qwen3-4B-Instruct-2507` | ✅ | ~7.5GB bf16 | **Toy Run 3 v7+.** Aug 2025, better IFEval. |
| `Qwen/Qwen3-8B` | ✅ | ~15GB bf16 | Current SDPO real-run target. |
| `Qwen/Qwen3.5-9B` | ✅ | ~18.8GB bf16 | Qwen3.5/ImageTextToText SDPO smoke target; requires processor-aware loader. |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ✅ | ~56GB | Paper's exact model. |
| `Qwen/Qwen3-1.7B` | ✅ | ~3.4GB | Toy Run 1 successful. |

### Verified Dataset

- **Name:** AmazonHistoryPrice (NOT on HF Hub direct)
- **Source:** https://github.com/TianXiaSJTU/AmazonPriceHistory
- **Products:** 930 total
- **Split:** 802 train / 128 test  
- **MI / CI:** 886 MI / 44 CI
- **Per product:** title, description, list_price, cost, budget(=list×0.8)

### Memory Budget

Qwen3-4B dual-role on A100 (80GB):
- Policy model: ~8GB bf16 parameters
- Reference model: ~8GB (frozen, no grad)
- AdamW states: ~16GB (2× fp32 per bf16 param)
- Activations per turn: ~2GB (gradient checkpointing halves this)
- **Total operational: ~32GB** ✓ fits with headroom

Qwen3-8B would need ~64GB operational — fits on A100x1 but tight.
Real Run needs A100x4 (320GB) safe.

---

## Session 2026-05-23 — LoRA SDPO Update Canary

### Direction change

The active path is now Qwen3.5-first. `train_negotiation_sdpo_qwen35.py` is the main training entrypoint; `train_negotiation_sdpo.py` remains the Qwen3-8B fallback/control. This reflects the newer Qwen3.5/ImageTextToText stack and the recent native-thinking/finalizer work.

### Why LoRA now

The latest Qwen3.5 dense production-shape smoke proved the full-parameter update is feasible but expensive and format-fragile: A100 80GB is effectively saturated, update time is still dominated by backward plus policy/teacher forwards, and CPU-state AdamW remains non-trivial. The LoRA ablation is intended to preserve the same ref-free SDPO+GRPO objective while reducing trainable parameters and optimizer state.

### Implementation choices

- Added opt-in `USE_LORA=1` to both active SDPO scripts.
- Default LoRA config for canaries: `LORA_R=64`, `LORA_ALPHA=64`, `LORA_DROPOUT=0.0`.
- LoRA target selection is dynamic and exact-name based. It adapts only text-transformer `torch.nn.Linear` modules under `model.language_model.layers.*` or `model.layers.*`, including attention/linear-attention/MLP projections.
- Broad `LORA_TARGET_MODULES=all-linear` is rejected because Qwen3.5 has adjacent vision/MTP/head modules that should not be adapted for this text-only negotiation run.
- LoRA defaults to `OPTIMIZER=adamw_cuda`; dense full-parameter training keeps `adamw_cpu`.
- Seller/environment model remains frozen and unadapted.
- Pre-LoRA scripts were preserved under `working_docs/script_snapshots/2026-05-23-pre-lora/`.

### Planned evidence gates before full Qwen3.5 run

1. Local LoRA target/config tests pass.
2. Local compile/ruff checks pass.
3. Tiny HF GPU smoke imports PEFT, wraps Qwen3.5, runs dataset/model load, reaches first rollout/update, and pushes a durable artifact.
4. Production-shape 2-iteration Qwen3.5 LoRA canary completes or early-stops with interpretable metrics.
5. Compare against dense Qwen3.5 smoke on reward, deal rate, format errors, budget violations, rollout/update timers, optimizer time, trainable params, and peak VRAM.
6. Stop before full 60-iteration Qwen3.5 launch; launch only after explicit review of the canary evidence.

### Tiny smoke results

- First tiny HF job `6a111227b33ece92698c0fa4` failed before model load because `AutoProcessor` pulled Qwen3.5 image-processing backends and the uv environment lacked `pillow`/`torchvision`.
- Fix: Qwen3.5 text-only loader now falls back to `AutoTokenizer` if `AutoProcessor` raises an image-backend `ImportError`; canary commands also include `pillow` and `torchvision`.
- Second tiny HF job `6a1112aae3c0b51e1ca5d8b1` completed. W&B run: `https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/9fe34g0k`. Hub artifact: `ZeterMordio/anchor-negotiation-sdpo-qwen35-lora-r64-tiny-smoke2-20260523`.
- LoRA injected `248` exact text targets, `173,113,344` trainable params out of `9,582,927,088` total (`1.806%`). Seller trainable params remained `0`.
- Tiny run was intentionally too small for quality: 2 episodes, reward `-0.5000`, deal rate `0%`, outcomes `TIMEOUT=1`, `BUYER_FORMAT_ERROR=1`. It proved the load/wrap/update/save path, not training quality.
- Tiny update timing: rollout `44s`, update `8s`, policy forward `1.7s`, teacher forward `0.9s`, backward `4.1s`, optimizer `0.2s`; peak reserved VRAM `44.0GB`.

### Production-shape 2-iteration LoRA canary

- HF job `6a111358b33ece92698c0fb4` completed and early-stopped after two format-warning iterations. W&B run: `https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/1c7zz7e9`. Hub artifact: `ZeterMordio/anchor-negotiation-sdpo-qwen35-lora-r64-canary-20260523`.
- Config: `Qwen/Qwen3.5-9B`, `NUM_ITERS=2`, `BATCH_SIZE=16`, `GROUP_SIZE=8`, option-B native thinking, `NATIVE_FINAL_TOKENS=96`, `USE_LORA=1`, `LORA_R=64`, `LORA_ALPHA=64`, `LR=1e-5`, CUDA AdamW, ref-free SDPO+GRPO, `CHECKPOINT_EVERY=1`.
- Adapter scope remained correct: `248` exact text-transformer targets, `173,113,344 / 9,582,927,088` trainable params (`1.806%`), seller trainable params `0`.
- Iter0 matched the dense control before any learned update: reward `-0.0744`, deal `46.9%`, buyer-format errors `12/128`, rollout `778s`, update `602s`.
- Iter1 after one LoRA update degraded below the dense `LR=3e-6` Qwen3.5 smoke: reward `-0.1890` vs dense `-0.1196`, deal `32.8%` vs dense `38.3%`, buyer-format errors `18/128` vs dense `19/128`, overshoot `7.0%`.
- Update timers were dense-like despite trainable params dropping to `1.806%`: iter0 policy forward `104.3s`, teacher forward `122.9s`, backward `367.3s`, optimizer `0.2s`; iter1 policy forward `100.2s`, teacher forward `116.5s`, backward `349.0s`, optimizer `0.1s`.
- Memory result is the real win: update reserved peak dropped to `54.6GB`/`57.6GB`, and adapter checkpoints were ~`693MB`. Overall peak still reached `81.0GB`/`83.0GB` during rollout, so LoRA does not lower the current Qwen3.5 hardware class.
- Decision: do not launch a 60-iteration Qwen3.5 LoRA run at `LR=1e-5`. Keep LoRA as opt-in; if continuing, run only another 2-iteration canary at `5e-6` or `3e-6` and compare against the dense control before promotion.

### Follow-up sweep cancellation / dense-path guardrail

- Two lower-LR LoRA canaries were briefly launched, then canceled during iteration-0 rollout before useful metrics because the observed bottleneck is not LoRA-sensitive enough for wall-clock optimization.
- Canceled jobs: `6a1121b6e3c0b51e1ca5d928` (`LR=5e-6`) and `6a1121b6e3c0b51e1ca5d92a` (`LR=3e-6`). Both had passed startup, injected `248` targets, and reached rollout; neither produced training metrics before cancellation.
- Dense-path guardrail added: when `USE_LORA` is unset, `train_negotiation_sdpo_qwen35.py` and `train_negotiation_sdpo.py` keep the pre-LoRA dense defaults for `OPTIMIZER`, default run name, and default W&B group. LoRA-specific W&B config/logging and model wrapping now run only under `USE_LORA=1`.
- Side-by-side import check against `working_docs/script_snapshots/2026-05-23-pre-lora/` confirmed dense defaults match the snapshots for both Qwen3.5 and Qwen3 scripts.
- Current recommendation: stop spending compute on LoRA as a speed lever. If wall-clock is the target, work next on rollout generation cost, Qwen3.5 fastpath kernels (`causal-conv1d`/`flash-linear-attention`), or architecture changes that avoid keeping buyer+seller live at rollout peak.

### Rollout token telemetry / Qwen3.5 fastpath canary

- Added `ROLLOUT_TOKEN_TELEMETRY=1` to `train_negotiation_sdpo_qwen35.py`. This is instrumentation only. It logs per-role and total rollout token counters: sequence count, prompt mean, first-pass generated mean, finalizer rate/tokens, total generated mean, first-pass parseable-action rate, final parseable-action rate, public tokens to first action, public tail tokens after first action, tail share, and max prompt/generated lengths.
- The telemetry is printed after rollout and included in metrics rows plus W&B under `perf/rollout_token_*`. It is meant to decide whether parseable-action early stopping is worth implementing; it does not alter generation or training.
- Added `tools/qwen35_fastpath_canary.py` as a reusable import/load/generate/backward smoke for the Qwen3.5 fastpath stack.
- Fastpath uv-image canary [`6a112b6ee3c0b51e1ca5d98a`](https://huggingface.co/jobs/ZeterMordio/6a112b6ee3c0b51e1ca5d98a) failed before script execution. `causal-conv1d==1.6.2.post1` has only an sdist on PyPI; the standard HF uv image lacks `nvcc`, so the source build failed.
- Two devel-image command-shape attempts were canceled/failed before useful model work: `6a112c13b33ece92698c113c` / `6a112c28b33ece92698c1144` misparsed `bash -lc` until the HF CLI `--` separator was used, and `6a112c4bb33ece92698c114c` hit PEP-668/system-pip plus empty mounted-script lookup.
- Fastpath devel-image canary [`6a112d34e3c0b51e1ca5d99a`](https://huggingface.co/jobs/ZeterMordio/6a112d34e3c0b51e1ca5d99a) failed usefully: build isolation pulled torch `2.12.0+cu130` while the image CUDA was 12.8, causing a CUDA version mismatch during `causal-conv1d` build.
- Final no-build-isolation devel-image canary [`6a112e00e3c0b51e1ca5d9a4`](https://huggingface.co/jobs/ZeterMordio/6a112e00e3c0b51e1ca5d9a4) completed. Environment: `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel`, Python `3.12.3`, `torch=2.10.0+cu128`, `transformers=5.9.0`, `accelerate=1.13.0`, `causal-conv1d=1.6.2.post1`, `flash-linear-attention=0.5.0`, `triton=3.6.0`, A100 80GB with driver `575.57.08`.
- Passing install recipe: install `numpy packaging ninja wheel setuptools transformers>=4.57.0 accelerate flash-linear-attention safetensors`, then install `causal-conv1d` with `--no-build-isolation` so it compiles against the image's torch/cu128 stack rather than an isolated torch/cu130 build env.
- Result: all Qwen3.5 fastpath imports passed (`causal_conv1d_fn`, `causal_conv1d_update`, `FusedRMSNormGated`, `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`), `Qwen/Qwen3.5-9B` loaded, a 24-token generate completed, and a tiny full-model backward completed. Canary timings/VRAM: load `15.2s`, generate `44.72s` cold, backward `71.08s`, after-load `17.91GB`, after-generate peak `18.19GB`, after-backward peak `35.85GB`.
- Interpretation: the fastpath stack is now technically viable on HF Jobs, but only with a CUDA-devel image or prebuilt/wheel artifact strategy. The canary is not a production throughput benchmark because it includes cold dependency work, cold Hub download, and one tiny prompt.

### Same-shape Qwen3.5 fastpath 2-iteration run

- HF job [`6a11339eb33ece92698c11b9`](https://huggingface.co/jobs/ZeterMordio/6a11339eb33ece92698c11b9) completed on `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel`; W&B run [`fat0mgqc`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/fat0mgqc); model/metrics repo [`ZeterMordio/anchor-negotiation-sdpo-qwen35-fastpath-2iter-20260523-045622`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-fastpath-2iter-20260523-045622) has `main`, `iter-1`, and `iter-2` branches.
- Config matched the prior dense Qwen3.5 2-iter smoke shape: `Qwen/Qwen3.5-9B` buyer + seller, `NUM_ITERS=2`, `BATCH_SIZE=16`, `GROUP_SIZE=8`, `MAX_TURNS=6`, option-B native thinking, `NATIVE_THINK_TOKENS=300`, `NATIVE_FINAL_TOKENS=96`, `ROLLOUT_MAX_LENGTH=UPDATE_MAX_LENGTH=3072`, `LR=3e-6`, `WARMUP_STEPS=10`, `SDPO_LAMBDA=0.9 -> 0.88`, strict feedback, CPU-state AdamW, `UPDATE_LENGTH_BUCKETING=1`, `UPDATE_MICROBATCH_SIZE=4`, `OPTIM_STEP_EVERY_GROUPS=16`, `USE_LORA=0`, and `ROLLOUT_TOKEN_TELEMETRY=1`.
- Iter0: reward `-0.1019`, deal `46.1%`, buyer-format errors `21/128`, budget violations `3/128`, printed time `1098s` (`rollout=675s`, `update=423s`), reserved peak `73.3GB`. Update timers: policy forward `54.0s`, teacher forward `53.4s`, backward `224.7s`, CPU optimizer `85.9s`.
- Iter1: reward `-0.1905`, deal `35.2%`, buyer-format errors `17/128`, budget violations `11/128`, printed time `798s` (`rollout=523s`, `update=276s`), reserved peak `74.6GB`. Update timers: policy forward `37.1s`, teacher forward `49.9s`, backward `107.1s`, CPU optimizer `76.5s`.
- Speed read: compared with the old dense fallback-stack run (`~661s` rollout, `~595s` update, `~20.9 min/iter`), the fastpath stack materially improves update compute (`423s`/`276s`) and lowers reserved VRAM (`73-75GB` vs `~80GB`). It does not fix rollout: rollout remained `675s`/`523s`, and full script wall time was still `39.2 min` because dense checkpoint writes/uploads plus generation dominate a 2-iter canary.
- Token telemetry explains the remaining bottleneck: total sequences `652`/`613`, prompt mean `1067`/`1040`, first-pass generated mean `299.9`/`298.8`, finalizer rate `99.2%`/`98.9%`, total generated mean `349.3`/`348.6`, first-pass action rate only `0.8%`/`1.1%`, parseable action rate `95.9%`/`96.6%`.
- Decision: keep the CUDA-devel fastpath recipe as viable for dense Qwen3.5 if running more Qwen3.5 work, especially for memory and update time. Do not spend more effort on fastpath sweeps as the main wall-clock lever. Next meaningful speed work is rollout-token reduction: parseable-action stop criteria, native-thinking/finalizer budget changes, or a design that avoids buyer+seller live generation pressure.

### Cost-safe Qwen3.5 launch policy

- User accepted the objective-preserving cost policy: keep `a100-large`, use the fastpath stack, set `CHECKPOINT_EVERY=10` while keeping W&B per-iteration logging, enforce finite timeouts/cancel discipline, and move dependency install/build work into a prebuilt image. Pre-update health gates and rollout-only canary mode are intentionally not part of this pass.
- Added `docker/qwen35-fastpath/Dockerfile` for a pinned dependency image based on `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` with `transformers==5.9.0`, `accelerate==1.13.0`, `flash-linear-attention==0.5.0`, `causal-conv1d==1.6.2.post1`, `wandb==0.27.0`, and `huggingface_hub==1.16.1`.
- Added `docker/qwen35-fastpath/README.md` with build/push instructions and the `causal-conv1d --no-build-isolation` invariant.
- Added `tools/launch_qwen35_fastpath_sdpo.py`. It uploads the current `train_negotiation_sdpo_qwen35.py` to a script repo, launches the prebuilt image with `hf jobs run`, and dry-runs unless `--execute` is passed. It refuses non-`a100-large` flavors and refuses `CHECKPOINT_EVERY` values other than `10`.
- Launcher default for the serious dense Qwen3.5 run is `NUM_ITERS=60`, `BATCH_SIZE=16`, `GROUP_SIZE=8`, `LR=3e-6`, `USE_LORA=0`, `OPTIMIZER=adamw_cpu`, `ROLLOUT_TOKEN_TELEMETRY=1`, `--timeout 22h`, and labels `project=anchor`, `purpose=qwen35-fastpath-dense`, `cost_policy=a100-checkpoint10-timeout`.
- Rationale: HF Jobs currently bills per minute while Starting/Running; `a100-large` is still the cheapest shape that fits dense Qwen3.5 buyer+seller without changing model/protocol. The prebuilt image reduces repeated setup/build overhead across canaries/runs, while checkpoint cadence avoids dense model uploads every iteration.
- Local Docker build succeeded for `anchor-qwen35-fastpath:torch2.10-cu128-fla0.5.0` with image ID `sha256:73e9d2e4553e9ca20b3edf2d985e31de76d72581d81f4c75ce6edc545c3fcea4` (`linux/amd64`, `17.6GB`). Build-time and runtime import checks printed `torch=2.10.0+cu128`, `transformers=5.9.0`, `accelerate=1.13.0`, `flash-linear-attention=0.5.0`, `causal-conv1d=1.6.2.post1`, `wandb=0.27.0`, and `huggingface_hub=1.16.1`, plus all required Qwen3.5 fastpath symbols. The Apple-local build warned that the base image is `linux/amd64` while the local host is `arm64`, and runtime warned that no NVIDIA driver was present; both are expected locally and fine for the HF GPU target.
- Follow-up guardrail: HF docs describe Jobs images as Docker Hub or Hugging Face Spaces images (`hf.co/spaces/<owner>/<space>`). The launcher now rejects local-only image refs such as `anchor-qwen35-fastpath:...` or `localhost/...` before submission, so an accidental `--execute` cannot burn paid A100 startup time on an image HF Jobs cannot pull.
- User approved either Docker Hub or HF Space image publication; HF Space was chosen because Jobs/artifacts/auth are already Hugging Face-centered. Created [`ZeterMordio/anchor-qwen35-fastpath`](https://huggingface.co/spaces/ZeterMordio/anchor-qwen35-fastpath) as a public Docker Space and uploaded the pinned Dockerfile at commit `b9ae2ebc32b548e9fa4f5a251aaae85c47872dc8`.
- HF Space build succeeded: it installed the pinned stack, built `causal-conv1d` with `--no-build-isolation`, printed the expected package versions, pushed the image, and exported cache. A later metadata/default-command update landed at commit `c0c67b6db1492b9944823dd15405f959e8669ec0`, and the Space README was clarified at commit `d74870b3ef6e5f3dc2a68f801d74b21d13d04417`. The canonical prebuilt image ref is now `hf.co/spaces/ZeterMordio/anchor-qwen35-fastpath`, and the launcher defaults to this ref.
- Cheap CPU HF Jobs import smokes [`6a14d140404eb93b204f1c82`](https://huggingface.co/jobs/ZeterMordio/6a14d140404eb93b204f1c82) and final [`6a14d2e4404eb93b204f1c99`](https://huggingface.co/jobs/ZeterMordio/6a14d2e4404eb93b204f1c99) completed using `space_id=ZeterMordio/anchor-qwen35-fastpath`. Logs confirmed HF Jobs can pull the Space image and import `causal_conv1d_fn`, `causal_conv1d_update`, `FusedRMSNormGated`, `chunk_gated_delta_rule`, and `fused_recurrent_gated_delta_rule`, with package versions `torch=2.10.0+cu128`, `transformers=5.9.0`, `accelerate=1.13.0`, `flash-linear-attention=0.5.0`, `causal-conv1d=1.6.2.post1`, `wandb=0.27.0`, and `huggingface_hub=1.16.1`.

---

## Session 2026-04-26/27 — Toy Run 3 Dual-Role + RAE

### Context
Continuing from prior work: Toy Runs 1 & 2 (buyer-only, Qwen3-4B, pipeline validated
and convergence signal observed) on A100-large. Toy Run 3 was repeatedly failing
on A100 with silent crashes / no clear logs. User wants to combine RLVR paper
(2604.09855) negotiation framework with SPIRAL paper (2506.24119) self-play and
Role-Conditioned Advantage Estimation (RAE) to train a model that can negotiate
robustly REGARDLESS of who goes first, and eventually beat GPT-5.4 on a fraction
of the compute budget.

### Major Issues Found & Fixed

#### 1. Silent A100 Crash — GPU Memory Leak in `generate()`

- **Root cause:** `model.generate(output_scores=True, return_dict_in_generate=True)`
  creates a float32 tensor of shape [batch, gen_len, 151936] = ~55MB PER generation.
  In dual-role, each episode has ~6 turns × 2 roles. With full batch (16×8=128 episodes):
  that accumulates to GBs of score tensors per iteration, eventually OOMing on the
  80GB A100. The OOM happened silently (no Python traceback — just kernel kill).
- **Fix:** Removed `output_scores=True` entirely. `generate_turn()` now returns only
  the decoded text. Log-probs are recomputed later during GRPO backward pass by
  re-tokenising and doing a forward pass on the saved text.

#### 2. Loss Explosion (→10^17 in iteration 2)

- **Root cause 1:** Raw log-ratio `ratio = exp(tok_logprobs - ref_logprobs)` has
  no upper bound. After a GRPO step shifts weights, the policy can drift far from
  ref, making ratios -> infinity. The `torch.clamp(ratio, 0.8, 1.2)` only bounds
  `clipped` — `ratio` in `surr1` is still unbounded.
- **Fix 1:** Clamp the *log* ratio first: `(pol_lp - ref_lp).clamp(-5.0, 5.0)`.
  This caps ratio at ~148 and floor at ~0.007. Prevents NaN/inf completely.
- **Root cause 2:** Advantages were raw `reward - mean`, un-normalised. Big reward
  spread = large step sizes = destabilisation.
- **Fix 2:** Zero-mean, unit-std normalisation per role per group:
  `(t - mean) / (std + 1e-8)`. Standard GRPO trick, essential for stability.

#### 3. Seller Prompt Bug — Role Mapping Was Backwards

- **Root cause:** Original `build_seller_prompt(product, buyer_history_texts)` did
  `even=assistant, odd=user` on a flat list of all messages, but this is from the
  *buyer's* POV. From the seller's perspective, buyer messages should be "user"
  (the counterparty speaking TO the seller) and seller's own prior responses are
  "assistant".
- **Fix:** For self-play, restructured `run_dual_episode()` to build seller
  messages explicitly from `buyer_texts` (→ `"user"`) and `seller_texts` (→ `"assistant"`).

#### 4. `_build_turn_prompt` Double-Added Last Seller Message

- **Root cause:** The function had a loop adding all prior turns (j < turn_idx),
  then an *extra* if-block that re-added the last seller message for buyer turns.
  This caused the final seller message to appear TWICE in buyer prompts, confusing
  the model.
- **Fix:** Removed the redundant extra append. The loop handles all history.

#### 5. Per-Episode Optimizer Step Breaking GRPO

- **Root cause:** Stepping the optimizer after every single episode meant that
  within a GRPO group of 8 episodes, the first episode's update changed weights
  before processing episodes 2-8, invalidating their reference log-probs and group
  advantage computation.
- **Fix:** Moved `optimizer.step()` to after the FULL group of G episodes is
  processed. Gradient accumulation per-turn backward is valid; the *step* must be
  group-level.

#### 6. Full Vocab Softmax Memory Waste

- **Root cause:** `F.log_softmax(logits[:, :-1, :], dim=-1)` materialises a
  [batch, seq, 151936] tensor — ~1.1GB for 2048 tokens — EVERY forward pass
  (both policy and ref = 2.2GB per turn).
- **Fix:** `_token_logprobs()` gathers the target logits first, then computes
  log-prob via `logsumexp` on the fly. No full vocab tensor. Cuts ~1GB per call.

### Key Design Choices

- **Shared policy vs separate:** One model plays both buyer and seller. Role
  conditioning is purely via system prompt. This is SPIRAL's approach and is
  essential for self-play: as the buyer improves, the seller improves too,
  maintaining challenge level.
- **Zero-sum rewards:** `seller_reward = -buyer_reward`. This ensures symmetric
  competitive pressure. The paper's regulated seller (cannot accept below cost)
  is a leftover from their frozen-seller design; we keep it as a safety net but
  both sides get verifiable reward signals.
- **RAE (Role-Conditioned Advantage Estimation):** Per SPIRAL Eq 2, we maintain
  separate EMA baselines for buyer and seller. This prevents RAE from being
  biased by role asymmetry (e.g. first-mover advantage in negotiations).
- **Seller temperature = 1.0:** Originally matched paper's 0.7, but that was for
  a FROZEN seller (consistency). In self-play, BOTH roles need equal exploration
  (SPIRAL uses 1.0 for both). Reverted to 1.0.
- **DUAL_ROLE_RATIO = 0.5:** Fraction of seller turns that get trained on.
  Empirically: buyer always trains (it's the primary), seller trains 50% of
  the time. Prevents seller gradients from dominating early when buyer is more
  important for baseline metrics.

### Files

| File | Role |
|------|------|
| `train_negotiation_clean.py` | Original buyer-only script (Toy Runs 1&2) |
| `train_negotiation_dual_role.py` | Dual-role + RAE script (Toy Run 3+) |
| `JOURNAL.md` | This file — living log of decisions |

### Toy Run 3 Status — ✅ COMPLETE (v10 on A100, 15 iters)

**Job IDs attempted:**
- `69eec939d70108f37ace0516` — STUCK at SCHEDULING 7+ hours, cancelled (Docker pytorch image)
- `69eeb7f6d70108f37ace04d5` — CANCELED (Docker pytorch image)
- `69ee4d86d70108f37ace02ba` — CANCELED (Docker pytorch image)
- `69ee4d61d2c8bd8662bd0109` — CANCELED (Docker pytorch image)
- `69ef3557d70108f37ace074c` — SCHEDULING forever (Docker pytorch image, REST API)
- `69ef5164d70108f37ace07e4` — SCHEDULING forever (Docker pytorch image, REST API)
- `69ef53d6d70108f37ace07f1` — SCHEDULING forever (Docker pytorch image, REST API)
- `69ef556bd70108f37ace07f5` — ✅ **RAN** 5.8 min, ERROR (`hf jobs uv run`, uv image)
- `69ef56f8d70108f37ace0803` — ⏳ SUBMITTED (`hf jobs uv run`, torch pinned to CUDA 12)

**Diagnosis (2026-04-27, updated ~13:00 UTC):**

#### ROOT CAUSE 1: The `pytorch/pytorch` Docker Image Causes Infinite SCHEDULING

**This was the primary culprit for the 7-hour stall and all "stuck in scheduling" jobs.**

All jobs using `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` as `dockerImage`
appeared stuck in SCHEDULING indefinitely. Analysis of 100 historical jobs shows:

- Jobs using the pytorch Docker image with large payloads (>5KB env) → stuck forever
- Jobs using lightweight images (uv, python:3.11, nvidia/cuda) → run promptly
- The one successful training job (`69eab541`, 1142s) used pytorch image but was
  submitted during a period of low demand

The `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` image is ~8GB. On HF Jobs infra,
pulling this image competes with GPU scheduling. When A100 demand is even moderate,
the combination of large image pull + GPU allocation creates a deadlock-like state
where the job shows "Job started" (image pull began) but never progresses to running.

**Evidence:** Switching to `ghcr.io/astral-sh/uv:python3.12-bookworm` (~200MB)
immediately resolved the scheduling hang. Job `69ef556b` went from SCHEDULING to
RUNNING within minutes and actually executed.

**Fix:** Use `hf jobs uv run` which uses the lightweight uv image. Dependencies
(torch, transformers, etc.) are installed via `uv` at runtime, which is fast (~60s)
and avoids the massive Docker image pull.

#### ROOT CAUSE 2: PyTorch CUDA Version Mismatch (uv auto-resolution)

When using `hf jobs uv run --with torch`, uv resolves the latest PyTorch (2.11.0+cu130,
CUDA 13.0). But HF Jobs A100 machines have NVIDIA driver 12090 (CUDA 12.x only).

Error: `CUDA initialization: The NVIDIA driver on your system is too old (found version 12090)`

**Fix:** Pin torch version: `--with 'torch>=2.6.0,<2.7'` to get a CUDA 12.4 build.

#### Additional Script Issues Fixed (from earlier analysis)

1. **`torch_dtype` deprecated** → changed to `dtype=torch.bfloat16`
2. **`torch.inference_mode()` risky with shared tensors** → changed to `torch.no_grad()`
3. **No output buffering control** → added `PYTHONUNBUFFERED=1` + `flush=True`
4. **No rollout progress** → added logging every 4 products
5. **No generation safety** → added `repetition_penalty=1.1`
6. **Inaccurate `_token_logprobs` docstring** → corrected

### Toy Run 3 — Correct Submission Method

The official way to submit HF Jobs is via the `hf` CLI (installed via `curl -LsSf https://hf.co/cli/install.sh | bash`):

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 4h \
    --with 'torch>=2.6.0,<2.7' --with transformers --with accelerate --with trackio --with huggingface_hub \
    --secrets HF_TOKEN \
    --env BATCH_SIZE=16 \
    --env GROUP_SIZE=8 \
    --env MODEL_NAME=Qwen/Qwen3-4B \
    --env NUM_ITERS=15 \
    --env HUB_MODEL_ID=ZeterMordio/anchor-negotiation-dual-role \
    --env PYTHONUNBUFFERED=1 \
    --env RUN_NAME=toy3-qwen3-4b-15it \
    --env TRACKIO_SPACE=ZeterMordio/anchor-dashboard \
    --detach \
    train_negotiation_dual_role.py
```

Key points:
- Uses `uv` (not pip) for dependency resolution — much faster, lightweight image
- `ghcr.io/astral-sh/uv:python3.12-bookworm` image (~200MB) — **NOT** pytorch (~8GB)
- **Pin torch to <2.7** to avoid CUDA 13 builds (HF A100 driver is CUDA 12.x)
- `--secrets HF_TOKEN` properly injects the token from HF's secret store
- Script is uploaded as `LOCAL_FILES_ENCODED` env var (base64, handled by CLI)
- `--detach` returns immediately; logs via `hf jobs logs <ID> --follow`

### Real Run 4 Status

- **Status:** READY TO LAUNCH (pending user approval)
- **Prerequisite:** ✅ Toy Run 3 v10 complete — format stable, rewards improving
- **Option A:** Extended v10 (same 4B model, 40-60 iters) — $18, 4.6h, validates full convergence
- **Option B:** Qwen3-8B, 40-60 iters, a100-large — $40-60, 10-15h, the "real" experiment
- **Recommended:** Option A first (cheap validation), then Option B

### Next Steps

1. ✅ ~~Fix and re-launch Toy Run 3 with corrected job submission~~
2. ✅ Format stability confirmed, buyer reward trending up
3. Launch extended 40-60 iter run (Option A or B above)
4. Build evaluation script for structured benchmarking
5. Add evaluation script: benchmark vs GPT-5.4, adversarial personas (RLVR §5)

---

## Archived: Prior Session Notes (from engineering_notebook.md)

### Toy Run 1
- **Status:** COMPLETED (17.6 min)
- **Result:** Pipeline works! Iter 0 had 37.5% deals, then format collapsed
  (100% BUYER_FORMAT_ERROR)
- **Analysis:** Format collapse expected for 1.7B — too small to maintain
  structured output under RL pressure

### Past Job IDs
- Toy Run 2: `69eabb91e8e12c6f0a675661` (buyer-only, Qwen3-4B, 15 iters)
- Toy Run 3: `69eeb7f6d70108f37ace04d5` (dual-role + RAE, Qwen3-4B, 15 iters)

### Known HF Jobs Quirks
- **Use `hf jobs uv run`, NOT the REST API** — handles auth, script upload, log streaming
- Pin `torch>=2.6.0,<2.7` to avoid CUDA 13 builds (HF A100 driver is 12090)
- `--secrets HF_TOKEN` injects from HF secret store (NOT `--env HF_TOKEN=xxx`)
- REST API `secrets` must NOT be passed as array — causes "expected record"
- `create_repo()` and `HfApi()` need explicit `token=` param

---

## Toy Run 3 — FINALLY RUNNING (2026-04-27 ~12:45 UTC)

**Job ID:** `69ef56f8d70108f37ace0803`
**Status:** ✅ RUNNING on A100-SXM4-80GB

### What finally made it work

| Change | Impact |
|--------|--------|
| `hf jobs uv run` instead of Docker REST API | **Unblocked scheduling** — uv image (~200MB) vs pytorch (~8GB) |
| `--with 'torch>=2.6.0,<2.7'` | Fixed CUDA mismatch (torch 2.11 needs CUDA 13, HF has 12.x) |
| `--secrets HF_TOKEN` | Proper auth for model download + trackio + hub push |
| `PYTHONUNBUFFERED=1` + `flush=True` | Live log streaming from Python (uv install phase still batched) |

### Observed Performance (Iteration 0)

| Metric | Value | Notes |
|--------|-------|-------|
| Rollout time (128 eps) | ~34 min | Sequential model.generate(), major bottleneck |
| Time per episode | ~16s | 12 generate calls × ~1.3s each |
| Time per generate call | ~1.3s | Qwen3-4B, 300 max_new_tokens, A100 |
| GRPO update time | TBD | Awaiting iter 0 completion |
| **Est. time per iteration** | **~40 min** | Rollout + update + cache clearing |
| **Est. total (15 iters)** | **~10 hours** | ⚠️ Exceeds 4h timeout! |
| **Est. total cost** | **~$25** | At $2.50/hr A100 |

### Hyperparameter & Cost Tracking

| Config | Batch | Group | Eps/Iter | Est. Iter Time | Est. Total (15it) | Est. Cost | Hardware |
|--------|-------|-------|----------|----------------|-------------------|-----------|----------|
| **Current (v6)** | 16 | 8 | 128 | ~40 min | ~10 hr | ~$25 | A100 $2.50/hr |
| Reduced batch | 8 | 8 | 64 | ~20 min | ~5 hr | ~$12.50 | A100 $2.50/hr |
| Reduced group | 16 | 4 | 64 | ~20 min | ~5 hr | ~$12.50 | A100 $2.50/hr |
| Reduced both | 8 | 4 | 32 | ~10 min | ~2.5 hr | ~$6.25 | A100 $2.50/hr |
| **Batched gen** (opt) | 16 | 8 | 128 | ~5-8 min | ~1.5-2 hr | ~$4-5 | A100 $2.50/hr |
| Batched + L40S | 16 | 8 | 128 | ~8-12 min | ~2-3 hr | ~$4-5 | L40S $1.80/hr |
| Batched + no ref | 16 | 8 | 128 | ~4-6 min | ~1-1.5 hr | ~$2.50-4 | A100 $2.50/hr |

### Key for Qwen3-8B Real Run (planned)

| Config | Batch | Group | Eps/Iter | Est. Iter Time | Est. Total (40it) | Est. Cost | Hardware |
|--------|-------|-------|----------|----------------|-------------------|-----------|----------|
| Sequential (current arch) | 64 | 8 | 512 | ~160 min | ~107 hr | ~$267 | A100 $2.50/hr |
| **Batched gen** | 64 | 8 | 512 | ~15-25 min | ~10-17 hr | ~$25-42 | A100 $2.50/hr |
| Batched + vLLM | 64 | 8 | 512 | ~8-12 min | ~5-8 hr | ~$12-20 | A100 $2.50/hr |

**Conclusion:** Batched generation is mandatory for the Real Run. Without it, Qwen3-8B at paper scale (batch=64, group=8) would cost ~$267 and take 4+ days.

---

## Optimization Analysis (2026-04-27)

### The Bottleneck

Rollout is **>85% of iteration time**. Current approach calls `model.generate()` 
sequentially 1,536 times per iteration (128 episodes × 12 turns). Each call is 
~1.3s but the A100 is severely underutilized — it's doing single-sequence decoding
at batch_size=1 while 80GB of VRAM sits mostly idle.

### Optimization 1: Batched Turn-Parallel Generation (HIGH IMPACT, MODERATE EFFORT)

**Core insight:** Episodes are independent of each other — only turns within an 
episode are sequential. We can batch all 128 buyer-turn-1 prompts into a single 
`model.generate()` call, then all 128 seller-turn-1, then buyer-turn-2, etc.

- **Before:** 1,536 sequential generate calls (~34 min)
- **After:** 12 batched calls (~2-4 min, depending on padding overhead)
- **Speedup:** ~10-15×
- **Effort:** Moderate refactor of `run_dual_episode()` → `run_dual_episodes_batched()`
- **Complications:** Variable-length responses need left-padding; some episodes 
  terminate early and need masking in subsequent turns.

### Optimization 2: Drop the Reference Model (HIGH IMPACT, TRIVIAL EFFORT)

SPIRAL (our RAE source paper) uses **no KL penalty** (`kl_loss_coefficient=0.0`,
`kl_penalty_coefficient=0.0`). They validated this works for self-play. We're
already at `KL_COEF=0.0`, so the ref model does nothing except consume:
- ~8 GB VRAM (half the model memory)
- ~50% of the GRPO update forward pass time

**Fix:** Remove ref_model entirely. Use `ratio = 1.0` (no importance sampling) 
or simply `policy_loss = -advantage * log_prob`. The clipping still prevents 
catastrophic updates.

**Risk:** Without KL anchor, policy might drift faster. But RAE baselines 
provide variance reduction, and clip_epsilon=0.2 bounds step size. SPIRAL 
shows this is fine for competitive self-play.

### Optimization 3: vLLM / Prefix Caching (HIGH IMPACT, HIGHER EFFORT)

Replace `model.generate()` with vLLM inference engine:
- **Prefix caching:** System prompt (~200 tokens) computed once, reused for all 
  128 episodes. Saves ~25% of prefill compute per generation.
- **PagedAttention:** Handles variable-length batches efficiently, no padding waste.
- **Continuous batching:** Better GPU utilization during decoding phase.
- **Colocate mode:** Runs in same process, shares GPU with training. Use 
  `vllm_enable_sleep_mode=True` to offload during GRPO update.

### Optimization 4: L40S Instead of A100 (COST SAVING)

If batched generation brings rollout under control, L40S (48GB, $1.80/hr) fits:
- 2× Qwen3-4B bf16 = 16 GB
- Optimizer states = 16 GB  
- Activations + cache = ~8 GB
- **Total: ~40 GB** fits in 48 GB

L40S has lower memory bandwidth (864 GB/s vs 2 TB/s) so generation is ~30-50% 
slower per token. But with batching, total wall time is dominated by compute 
not bandwidth. **Estimated 28% cost savings** with ~20% time increase.

### Optimization 5: Liger Kernel for GRPO Update (MODERATE IMPACT, TRIVIAL)

Fuses cross-entropy + logprob computations in the update phase. Reported 
20% throughput boost and 60% memory reduction on the forward/backward pass.
Just needs `pip install liger-kernel` and a one-line config change.

### Recommended Priority Order

1. **Batched generation** — 10-15× rollout speedup, mandatory for Real Run
2. **Drop ref model** — free 8GB VRAM + 50% update speedup, follows SPIRAL
3. **L40S hardware** — 28% cost savings with minimal time increase
4. **vLLM colocate** — additional 2-3× on top of batching
5. **Liger kernel** — easy 20% update phase speedup

---

## Toy Run 3 v6 Results & Collapse Analysis (2026-04-27 ~14:40 UTC)

**Job ID:** `69ef56f8d70108f37ace0803` — CANCELED after 4 iterations (collapse confirmed)

### Iteration-by-Iteration Results

| Iter | Deal Rate | BuyerR | SellerR | Loss | Turns | Time | Top Outcome |
|------|-----------|--------|---------|------|-------|------|-------------|
| 0 | **53.9%** | +0.018 | -0.018 | 0.167 | 4.1 | 2305s | 51 seller accepts, 37 budget violations |
| 1 | **0.0%** | -1.000 | +1.000 | 0.000 | 1.3 | 1352s | 92 BUYER_FORMAT_ERROR, 36 UNEXPECTED_BUY |
| 2 | 0.0% | -0.992 | +0.992 | 0.240 | 1.1 | 1392s | 116 FORMAT_ERROR |
| 3 | 0.0% | -1.000 | +1.000 | 0.000 | 1.0 | 1671s | 128 FORMAT_ERROR (100%) |

### Root Cause: LR=3e-5 Too Aggressive for Dense 4B

The RLVR paper used `lr=3e-5` with `Qwen3-30B-A3B-Instruct-2507` — a **MoE with only 3.3B active params**. 
In MoE, only 8/128 experts receive gradient per token, so most parameters get zero gradient. 
On our **dense** Qwen3-4B, ALL 4B params get gradient. So 3e-5 is effectively ~30× more destructive.

SPIRAL used `lr=1e-6` on the exact same dense Qwen3-4B family. That's 30× lower.

### v7 Fix: Three Changes

1. **Model → `Qwen3-4B-Instruct-2507`**: Aug 2025 post-training with massively improved IFEval.
   GPQA +20, AIME +28 pts vs original. Same size (7.5GB bf16), drop-in replacement.
2. **LR → `1e-6`**: SPIRAL's proven stable value for dense Qwen3-4B. 30× lower than v6.
3. **KL_COEF → `0.01`**: Small anchor to prevent format collapse. Both papers used KL=0,
   but they had either (a) a much bigger MoE model or (b) games with much simpler output format.
   Negotiation has complex structured output (Thought/Talk/Action) — a small KL helps.
4. **AdamW betas → (0.9, 0.95)**: Match SPIRAL's optimizer config exactly.

### Qwen3.6 Investigation — NOT SUITABLE

Both `Qwen3.6-27B` and `Qwen3.6-35B-A3B` are **vision-language** models (`qwen3_5` architecture),
NOT text-generation. They use hybrid DeltaNet + full attention layers with a vision encoder built in.
Architecture type is `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`.

| Model | Architecture | Type | Params | Active | Notes |
|-------|-------------|------|--------|--------|-------|
| Qwen3.6-27B | qwen3_5 (DeltaNet hybrid) | Vision-Language | 27.8B | 27.8B dense | NOT suitable for text negotiation |
| Qwen3.6-35B-A3B | qwen3_5_moe (MoE + DeltaNet) | Vision-Language | 36.0B | ~3B active | NOT suitable — vision encoder + hybrid attention |

**Key insight:** "Qwen3.6" is a collection name, not a model series. The text-only models with improved
instruction following are the `-2507` suffix models: `Qwen3-4B-Instruct-2507` and `Qwen3-30B-A3B-Instruct-2507`.
No `Qwen3-8B-Instruct-2507` exists.

### GPU Sweep for Real Run (text-only models)

| Model | Params | bf16 | Policy+Ref VRAM | GPU | Cost/hr |
|-------|--------|------|-----------------|-----|---------|
| Qwen3-4B-Instruct-2507 | 4.0B | 7.5GB | ~16GB | a10g-large (24GB) | $2/hr |
| Qwen3-8B | 8.2B | 16.3GB | ~33GB | a100-large (80GB) | $4/hr |
| Qwen3-30B-A3B-Instruct-2507 | 30.5B | 56.9GB | ~115GB | a100x4 (320GB) | $16/hr |

### Real Run Recommendation

For the $37-70 budget:
- **Qwen3-8B** on **a100-large** ($4/hr) with batched generation
- 40-60 iterations at ~15 min/iter (batched) = 10-15 hours → $40-60
- LR=1e-6, same KL/RAE config as v7
---

## v8: Batched Generation + Reference-Free GRPO (2026-04-27)

### Changes

**Optimization 1: Batched turn-parallel generation**
- Replaced `run_dual_episode()` (sequential, 1 episode) with `run_dual_episodes_batched()` (all episodes in parallel)
- Each turn round: one batched `model.generate()` for all active buyers, then one for all active sellers
- Episodes that terminate early are masked out of subsequent turns
- Left-padding for variable-length prompts in batched calls
- Sub-batching via `GEN_BATCH_LIMIT=128` to cap peak VRAM
- **Expected speedup: 10-15× on rollout phase** (2-4 min vs 34 min)
- Bug fixed: buyer `[REJECT]` was falling through to `UNEXPECTED_REJECT` — now correctly routes to seller turn

**Optimization 2: Reference model removed**
- No separate frozen ref model → saves **8GB VRAM** (half of model memory)
- GRPO update does ONE forward pass per turn instead of TWO
- Within each group, `optimizer.step()` hasn't been called, so policy weights = rollout-time weights
- `ratio = π/π_old = 1.0` → clipped surrogate degenerates to weighted REINFORCE
- Loss = `-advantage * log_prob(completion)` — exactly what SPIRAL uses
- **Expected speedup: ~50% on GRPO update phase** (1 forward pass vs 2)
- KL_COEF kept as config but currently inert (KL between policy and itself = 0)

### Combined Expected Impact

| Phase | v6 (128 eps) | v8 (128 eps) | Speedup |
|-------|-------------|-------------|---------|
| Rollout | ~34 min | ~3-5 min | 7-10× |
| GRPO update | ~3 min (2 fwd/turn) | ~1.5 min (1 fwd/turn) | ~2× |
| **Total/iter** | **~38 min** | **~5-7 min** | **5-8×** |
| **15 iters** | **~9.5 hr** | **~1.2-1.7 hr** | - |
| **Est. cost** | ~$24 | **~$3-4** | 6-8× cheaper |

### VRAM Budget (v8, Qwen3-4B-Instruct-2507 on A100-80GB)

| Component | VRAM |
|-----------|------|
| Policy model (bf16) | ~8 GB |
| ~~Ref model (bf16)~~ | ~~8 GB~~ → **0 GB** |
| AdamW states (fp32) | ~16 GB |
| Batched generation KV cache | ~2-4 GB (128 seqs × ~2K tokens) |
| GRPO update activations | ~2 GB (grad checkpoint) |
| **Total** | **~28-30 GB** (was ~36 GB) |

**Headroom: ~50 GB free** — could increase GEN_BATCH_LIMIT or run larger models.
---

## v9: Paper-Faithful Implementation Audit (2026-04-27)

### Deep Paper Re-read: Three Discrepancies Found and Fixed

**Discrepancy 1: Double-normalization (FIXED)**
- SPIRAL Eq. 2: `A = R_p - b_EMA` — raw EMA-subtracted advantage, NO further normalization
- GRPO: `A = (R - mean) / std` — group-level normalization
- Our v8 code: `A = (R_p - b_EMA - mean) / std` — BOTH applied (hybrid from neither paper)
- **Fix:** Removed group-level normalization. Pure RAE advantages as in SPIRAL.

**Discrepancy 2: Single inner epoch (FIXED)**
- SPIRAL Table 6: `Inner proximal update epochs: 2`
- Our v8 code: 1 epoch per group
- **Fix:** Added `NUM_INNER_EPOCHS=2`. Each epoch: forward pass on same trajectories with current (updated) weights → backward → step. 2nd epoch uses shifted weights, so gradients differ.

**Discrepancy 3: KL=0.01 (FIXED)**
- RLVR paper Table 5: `KL penalty: 0` (explicit)
- SPIRAL Table 6: `KL loss coefficient: 0.0`, `KL penalty coefficient: 0.0` (explicit)
- Our v8 code: `KL_COEF=0.01` (was no-op anyway since ref-free)
- **Fix:** Set default to `0.0` to match both papers.

### What the Audit Confirmed as Correct

| Aspect | Paper Source | Our Implementation | Status |
|--------|-------------|-------------------|--------|
| Loss type | SPIRAL Eq. 3 | `-A * log_prob` (pure REINFORCE) | ✅ |
| EMA baselines | SPIRAL Eq. 2, α=0.95 | Per-role EMA, α=0.95 | ✅ |
| No IS ratio | SPIRAL (no π/π_old ratio) | No ratio | ✅ |
| No ref model | SPIRAL (KL=0, no ref) | Single model, no ref | ✅ |
| Optimizer | SPIRAL Table 6 | AdamW β=(0.9,0.95), wd=0.0 | ✅ |
| LR | SPIRAL 1e-6 for dense 4B | 1e-6 | ✅ |
| Grad clip | SPIRAL Table 6: 1.0 | 1.0 | ✅ |
---

## v10: Domain-Aware SPIRAL-RLVR Hybrid (2026-04-27)

### Why v9's Pure SPIRAL Settings Don't Transfer to Negotiation

SPIRAL trains on TicTacToe and Kuhn Poker — short outputs (~100 tokens), near-binary rewards (win/lose),
simple action format. Our negotiation has ~200 tokens/turn, continuous rewards [-1,1] with mass at
-1/0 boundaries, and complex Thought/Talk/Action structured output with price parsing.

| Setting | SPIRAL (simple games) | v9 (copied SPIRAL) | v10 (domain-aware) | Reasoning |
|---------|----------------------|--------------------|--------------------|-----------|
| **Advantage norm** | OFF (binary rewards) | OFF | **ON** | Continuous multi-modal rewards need comparative within-group signal |
| **KL penalty** | 0.0 (simple format) | 0.0 | **0.01** | Complex structured NL output needs format stability anchor |
| **Ref model** | None (KL=0) | None | **Frozen copy** | Needed for meaningful KL; we have VRAM headroom on A100 |
| **Inner epochs** | 2 (short episodes) | 2 | **1** | Long ~2K token episodes make 2nd epoch stale-gradient risky |
| **Loss** | REINFORCE | REINFORCE | **Clipped surrogate + KL** | Bounds per-step drift on long sequences |
| **RAE baselines** | ✅ | ✅ | ✅ | Core dual-role innovation, always keep |

### What We Keep From SPIRAL (the part that actually matters)
- RAE: per-role EMA baselines — this is THE contribution for dual-role training
- Self-play: shared policy plays both roles via system prompt conditioning
- Zero-sum rewards: R_seller = -R_buyer

### What We Keep From RLVR/GRPO (domain-appropriate for NL negotiation)
- Clipped IS ratio: bounds policy drift per step on long NL sequences
- Group advantage normalization: comparative signal for continuous reward distributions
- KL penalty (small): format stability for Thought/Talk/Action structure
- Frozen reference model: provides meaningful KL anchor

### All Configs Made Env-Var Overridable
- `NORMALIZE_ADVANTAGES=1` (ON by default, set 0 for pure SPIRAL)
- `KL_COEF=0.01` (set 0.0 for pure SPIRAL)
- `NUM_INNER_EPOCHS=1` (set 2 for SPIRAL's config)
- Everything else: same env vars as before

---

## v10 Results — FIRST SUCCESSFUL FULL RUN (2026-04-28)

### Job Summary

| Job | Hardware | Ref Model | KL | Status | Iters | Total Time | Cost |
|-----|----------|-----------|-----|--------|-------|-----------|------|
| `69efcf15d2c8bd8662bd1359` | A100-large (80GB) | ✅ frozen | 0.01 | ✅ **COMPLETE** | 15/15 | 106 min | ~$7 |
| `69efd58bd2c8bd8662bd139b` | L40Sx1 (44GB) | ❌ ref-free | 0.0 | ❌ **OOM iter 1** | 1/15 | 12 min | ~$0.35 |

**Model pushed to:** [ZeterMordio/anchor-negotiation-dual-role](https://huggingface.co/ZeterMordio/anchor-negotiation-dual-role)

### A100 Run — Iteration-by-Iteration Results

| Iter | Loss | BuyerR | SellerR | Deal% | Price | Turns | Top Outcomes |
|------|------|--------|---------|-------|-------|-------|-------------|
| 0 | 0.0315 | +0.0838 | -0.0838 | 45.3% | $142.77 | 4.0 | BQ39 DBA34 DSA27 SQ16 BV7 |
| 1 | 0.0591 | +0.0925 | -0.0925 | 50.8% | $112.07 | 4.0 | DSA35 DBA32 BQ26 SQ23 BV7 |
| 2 | 0.0191 | +0.0760 | -0.0760 | **54.7%** | $166.49 | 4.2 | DSA39 DBA34 BQ31 SQ12 BV7 |
| 3 | 0.0152 | +0.0411 | -0.0411 | 44.5% | $94.02 | 4.1 | DSA44 BQ39 SQ25 DBA13 BV5 |
| 4 | 0.0319 | +0.0421 | -0.0421 | 41.4% | $385.31 | 4.0 | BQ40 DSA29 DBA27 SQ18 BV9 |
| 5 | 0.0370 | +0.0827 | -0.0827 | 43.8% | $238.29 | 4.1 | BQ34 DSA30 DBA27 SQ21 BV9 |
| 6 | -0.0022 | +0.0664 | -0.0664 | 48.4% | $304.04 | 4.2 | DSA36 BQ33 DBA28 SQ22 NP6 |
| 7 | 0.0351 | +0.0556 | -0.0556 | 47.7% | $222.54 | 4.1 | DSA39 BQ28 SQ25 DBA22 BV9 |
| 8 | 0.0305 | +0.0539 | -0.0539 | 46.1% | $198.46 | 3.9 | BQ39 DSA32 DBA28 SQ21 BV6 |
| 9 | 0.0035 | +0.1004 | -0.1004 | 43.8% | $169.24 | 4.1 | DSA33 BQ31 SQ27 DBA26 BV7 |
| 10 | -0.0237 | +0.0781 | -0.0781 | 39.1% | $294.11 | 4.1 | BQ42 DSA31 SQ27 DBA19 BV5 |
| 11 | -0.0034 | +0.0897 | -0.0897 | 43.8% | $238.63 | 4.3 | BQ42 DSA31 DBA26 SQ19 BV5 |
| 12 | 0.0320 | +0.0751 | -0.0751 | 43.8% | $159.99 | 3.9 | DSA34 BQ32 DBA22 SQ18 **BV16** |
| 13 | 0.0282 | +0.1083 | -0.1083 | 46.9% | $260.32 | 4.0 | **DBA45** BQ36 DSA19 SQ15 BV7 |
| 14 | 0.0457 | **+0.1518** | -0.1518 | **50.8%** | $253.87 | 4.2 | BQ35 DBA35 DSA35 SQ18 NP3 |

Legend: DSA=DEAL_SELLER_ACCEPTS, DBA=DEAL_BUYER_ACCEPTS, BQ=BUYER_QUIT, SQ=SELLER_QUIT, BV=BUYER_BUDGET_VIOLATION, NP=NO_PRIOR_BUYER_OFFER

### RAE Baseline Evolution

| Iter | b_buyer | b_seller | n (total episodes) |
|------|---------|----------|--------------------|
| 0 | +0.031 | -0.031 | 128 |
| 2 | +0.065 | -0.065 | 384 |
| 5 | +0.096 | -0.096 | 768 |
| 9 | +0.081 | -0.081 | 1280 |
| 11 | +0.162 | -0.162 | 1536 |
| 14 | **+0.211** | **-0.211** | 1920 |

### Key Findings

#### 1. FORMAT STABILITY SOLVED ✅

The #1 failure mode from v6 (100% format collapse at iter 1) is **completely gone**.
Zero FORMAT_ERROR outcomes across 15 iterations (1,920 episodes total).

What fixed it:
- **LR=1e-6** (30× lower than v6's 3e-5) — prevents weight destruction
- **Qwen3-4B-Instruct-2507** — superior instruction following baseline
- **KL=0.01 + frozen ref model** — format stability anchor

#### 2. BUYER REWARD TRENDING UP ↑

| Period | Mean BuyerR | Mean Deal% |
|--------|-------------|-----------|
| Iter 0-4 | +0.0671 | 47.3% |
| Iter 5-9 | +0.0717 | 46.0% |
| Iter 10-14 | **+0.1006** | **44.9%** |

Buyer reward increased 50% from first 5 to last 5 iterations. The model IS learning
to negotiate better deals. Deal rate dipped slightly (47.3→44.9%) which is expected:
better buyer negotiation = harder for seller to accept = slightly fewer deals but
higher buyer surplus per deal.

#### 3. SELF-PLAY DYNAMICS WORKING ✅

Iter 13 is the smoking gun: **DBA=45** (buyer accepts seller's offer) surged to the
highest count of any iteration. The SELLER learned to make more attractive counter-offers
that the buyer accepts. Meanwhile iter 14 shows perfect balance: DBA=35, DSA=35 — both
roles contributing equally to deals.

This is exactly what SPIRAL's RAE is designed to do: prevent one role from dominating.

#### 4. PRICE DYNAMICS ARE NOISY BUT INFORMATIVE

Mean deal price oscillates wildly ($94-$385) because the product mix varies per iteration
(16 random products from 802). This is expected — the reward structure matters more than
absolute prices. The buyer's increasing reward despite price variation shows it's learning
to negotiate relative to budget, not just absolute prices.

#### 5. BUDGET VIOLATIONS LOW AND STABLE

Avg ~7/128 (5.5%) — the Qwen3-4B-Instruct-2507 model has strong enough base capabilities
to respect budget constraints. Exception: iter 12 had BV=16 (12.5%), likely a batch of
high-priced products with tight budgets.

#### 6. PERFORMANCE: ~7 MIN/ITER ON A100

| Phase | Time | % of Iteration |
|-------|------|---------------|
| Rollout (batched generation) | ~260s | 63% |
| GRPO update (with ref model) | ~155s | 37% |
| **Total per iteration** | **~415s (6.9 min)** | 100% |

Compare to v6: 2,305s/iter (38 min) → **5.5× faster** thanks to batched generation.

### L40S OOM Analysis

The ref-free L40S job (`69efd58b`) completed iter 0 successfully (VRAM peaked at 24.3GB)
but OOM'd during iter 1's GRPO update at 43.96GB/44.39GB. The issue:

- Iter 0 update allocates ~24.3GB (model 8GB + optimizer 16GB + activations)
- By iter 1, PyTorch memory fragmentation leaves only 429MB free
- The `empty_cache()` + `gc.collect()` between iter 0 and 1 wasn't enough

**Root cause:** L40S has 44.39GB usable (not 48GB as spec'd — ~3.6GB consumed by driver/OS).
With 8GB model + 16GB optimizer + generation KV cache + activations, it's just too tight.

**Fix for future:** Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and/or reduce
`GEN_BATCH_LIMIT` from 64 to 32 to lower peak KV cache. But honestly, A100 at $4/hr
for 106 min = $7 is cheap enough — L40S savings aren't worth the fragility.

### Comparison to Paper (RLVR 2604.09855)

| Metric | Paper (30B-A3B, buyer-only) | v10 (4B, dual-role) | Notes |
|--------|---------------------------|--------------------|----|
| Deal rate | ~40-55% (iter 0-20) | 39-55% (iter 0-14) | **Matching paper range** |
| Format errors | ~0% (30B MoE) | ~0% (4B Instruct-2507) | Both models maintain format |
| Convergence speed | ~20 iters to plateau | Still improving at iter 14 | Expected: 4B needs more iters |
| Reward sign | Buyer reward only | Buyer ↑, Seller ↓ (zero-sum) | Our dual-role adds seller signal |
| Rollout speed | vLLM-colocated | Batched HF generate | ~2s/ep comparable |

### Trackio Dashboard

Dashboard at https://huggingface.co/spaces/ZeterMordio/anchor-dashboard had server 500 errors
during the run (Space may have been sleeping). Logs show `trackio could not flush buffered
remote data` warnings. Non-fatal — all metrics saved to `metrics.json` in the model repo.

**TODO:** Wake the Space before next run, or switch to a persistent monitoring solution.

---

## v10 Post-Mortem & Next Steps (2026-04-28)

### What Worked
1. **LR=1e-6 + Instruct-2507** — format stability solved permanently
2. **Batched generation** — 5.5× faster than v6, makes iteration cheap (~$0.47/iter)
3. **RAE** — buyer/seller baselines diverging as expected, no gradient cancellation
4. **Clipped surrogate + KL** — smooth loss curve, no explosions
5. **A100 VRAM budget** — 32.3GB stable, tons of headroom

### What Needs Improvement
1. **Buyer reward plateau** — +0.15 at iter 14 is good but paper reaches +0.3-0.4 at iter 40+
2. **Need more iterations** — 15 iters is clearly not enough, model still improving
3. **No evaluation yet** — need structured eval against baseline + adversarial sellers
4. **Trackio flaky** — need to ensure Space is awake before launching runs

### Recommended Next Steps (priority order)

1. **Extended v10 run on A100** — 40-60 iterations with same config
   - Est. time: 40×7min = 4.6h, cost: ~$18
   - Should see buyer reward reach +0.3+ based on paper trajectory
   - Save intermediate checkpoints every 10 iters for ablation

2. **Build evaluation script** — `eval_negotiation.py` needs:
   - Greedy decode (temp=0) against frozen base model as seller
   - Greedy decode against GPT-5.4 API as seller (if budget allows)
   - Adversarial seller personas (RLVR paper §5 tests)
   - Metric: deal rate, buyer surplus, price efficiency ratio

3. **Scale to Qwen3-8B** — the "real" run
   - 8B on A100-large: ~33GB (model 16GB + ref 16GB + optimizer 32GB) — needs gradient checkpointing + careful VRAM management, or A100x2
   - 40-60 iters at ~15 min/iter = 10-15h → $40-60
   - With batched gen, this is within budget

4. **Hyperparameter sweep** — now that format is stable:
   - LR: {5e-7, 1e-6, 2e-6}
   - KL: {0.0, 0.005, 0.01, 0.02}
   - Group size: {4, 8, 16}
   - RAE decay: {0.9, 0.95, 0.99}

---

## v10.2: Liger Kernel Integration (2026-04-28)

### What Changed

Added Liger Kernel monkey-patching for Qwen3 before model loading. Two lines of core code:

```python
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
apply_liger_kernel_to_qwen3()
```

This replaces four Qwen3 module-level classes with fused Triton kernels:

| Original | Liger Replacement | What it fuses | Impact |
|----------|------------------|---------------|--------|
| `Qwen3MLP` | `LigerSwiGLUMLP` | gate_proj + up_proj + SiLU + down_proj | **60% memory savings** on backward (no intermediate tensor) |
| `Qwen3RMSNorm` | `LigerRMSNorm` | Norm + variance in single kernel | Minor speedup |
| `apply_rotary_pos_emb` | `liger_rotary_pos_emb` | RoPE sin/cos fused | Minor speedup |
| `CrossEntropyLoss` | `LigerFusedLinearCrossEntropy` | lm_head + CE in one pass | **Major**: 151K-vocab logit tensor never materialized |

### Design Decisions

- **Env-var toggle:** `USE_LIGER=1` (default ON). Set `USE_LIGER=0` to disable for debugging.
- **Graceful fallback:** If `liger-kernel` not installed, prints a warning and continues without it.
- **Global patch:** `apply_liger_kernel_to_qwen3()` patches the module-level classes, so BOTH
  policy and ref models automatically use fused kernels.
- **Dependency:** `pip install liger-kernel` (add `--with liger-kernel` to hf jobs command).

### Expected Impact

- **GRPO update phase:** ~20% speedup (fused SwiGLU backward + fused CE avoids 1.1GB logit tensor)
- **Rollout phase:** ~5-10% (only forward pass, no backward — fused RMSNorm/RoPE help slightly)
- **VRAM:** ~40-60% reduction in peak activation memory during backward. This may allow
  disabling gradient checkpointing for an additional speedup, or fitting larger batch sizes.
- **Quality:** No impact. Liger kernels are mathematically equivalent in bf16.

### Risks

- **Near zero.** Qwen3 is explicitly supported. Liger has dedicated `apply_liger_kernel_to_qwen3`.
  Verified: Qwen3-4B-Instruct-2507 has `model_type: qwen3` which matches Liger's dispatch table.
- Only risk is Liger version incompatibility with future transformers versions, handled by the
  try/except fallback.

---

## v10.3: Fix Thought Block Information Leakage in Self-Play (2026-04-28)

### THE BUG

**Severity: CRITICAL — invalidates all prior self-play training results.**

In the model's output format (Thought/Talk/Action), the Thought block contains private strategic
reasoning: the buyer's budget, the seller's cost, intended strategy, bottom-line prices. Example:

```
Thought: My budget is $278.40 and the list price is $348. I'll start low at $180 to leave room.
Talk: Hi, I'm interested in these headphones. Would you accept $180?
Action: [BUY] $180 (1x electronics_42)
```

This **entire text** — including the Thought block — was being injected verbatim into the
counterparty's conversation context. The seller saw the buyer's budget and strategy; the buyer
saw the seller's cost and reservation price.

### WHY IT WASN'T CAUGHT BEFORE

The RLVR paper (2604.09855) uses a **frozen seller** that doesn't see the buyer's output at all.
The seller is a fixed policy that only generates from its own system prompt + the buyer's Talk/Action
(implicitly, since the frozen seller doesn't process the buyer's Thought). So the paper's
implementation doesn't need Thought stripping — it's architecturally impossible for the seller
to see the buyer's Thought.

Our SPIRAL dual-role setup breaks this assumption: the SAME model plays both roles, and both
roles see the other's full output as conversation history. The Thought leak has been present
since v6 (the first dual-role version).

### IMPACT ON v10 RESULTS

The v10 A100 run's results are **partially invalidated**:
- The model was learning to negotiate, but also learning to read the counterparty's Thought
- This artificially deflated the difficulty: the seller knew the budget, the buyer knew the cost
- Deal rate (~46%) and buyer reward (+0.08 avg) were measured under "open cards" conditions
- True adversarial negotiation (with information asymmetry) will be harder → expect:
  - Lower deal rate initially (harder to find agreement zone without knowing limits)
  - Slower reward improvement (must learn actual negotiation tactics, not just Thought-reading)
  - But ultimately BETTER negotiation behavior (forced to develop real strategy)

The good news: format stability, training dynamics, RAE baseline evolution, and performance
benchmarks are all still valid. The algorithm works; it was just solving an easier problem.

### THE FIX

Added `strip_thought(text)` function that extracts only Talk+Action from model output:

```python
def strip_thought(text):
    m = re.search(r'(?:^|\n)\s*Talk\s*:', text, re.IGNORECASE)
    if m:
        return text[m.start():].strip()
    return text  # fallback: no Thought block found
```

Applied in THREE places:
1. **Buyer prompt construction** (rollout): seller texts → `strip_thought(st)`
2. **Seller prompt construction** (rollout): buyer texts → `strip_thought(bt)`
3. **`_build_turn_prompt()`** (GRPO update): counterparty texts → `strip_thought(prev_text)`

Design principle: **each role sees its OWN full text (Thought+Talk+Action) as `assistant`,
but sees the OTHER role's text stripped (Talk+Action only) as `user`.**

This preserves chain-of-thought continuity within each role while enforcing information asymmetry.

### EXPECTED IMPACT ON NEXT RUN

- Negotiation will be genuinely adversarial — the model must learn to infer the other side's
  limits from Talk behavior (anchoring, concession patterns) rather than reading Thought
- Initial deal rate will likely be lower (~35-40% vs 46%)
- Reward improvement will be slower but more meaningful
- The resulting negotiation strategies should transfer better to real-world settings
  where counterparties don't share their internal reasoning
  where counterparties don't share their internal reasoning

---

## v10.4: Periodic Checkpoint Saving + Extended 60-Iter Run (2026-04-28)

### What Changed

Added `CHECKPOINT_EVERY` env var (default: 10). Every N iterations, the script:
1. Saves model weights + tokenizer to a temp directory
2. Saves `metrics.json` (all iters so far) and `rae_state.json`
3. Pushes to HF Hub as a **named branch** `iter-N` (e.g. `iter-10`, `iter-20`, ...)
4. Cleans up local checkpoint to save disk space

This enables:
- **Phase transition analysis**: compare model behavior at iter 10 vs 20 vs 30 etc.
- **Rollback**: if training degrades, we can revert to a known-good checkpoint
- **Intermediate evaluation**: run `eval_negotiation.py` against any checkpoint branch

Final model is still pushed to `main` branch as before.

### Run 4 Plan — First Adversarial Self-Play (v10.3 + v10.4)

| Setting | Value |
|---------|-------|
| Model | Qwen3-4B-Instruct-2507 |
| Iterations | **60** |
| Hardware | A100-large (80GB) |
| Timeout | 8h |
| Checkpoint every | 10 iters |
| Est. time | 60 × 7 min = ~7h |
| Est. cost | ~$28 |
| Key change | v10.3 Thought stripping — first genuinely adversarial run |

**Expected trajectory** (based on RLVR paper 4-phase evolution):
- Iter 0-12: Aggressive anchoring, deal rate may dip as buyer learns to lowball
- Iter 12-20: Temporary deadlock — aggressive offers without persuasion skill
- Iter 20-40: Rational concession — learns moderate openers that sellers engage with
- Iter 40-60: Advanced persuasion — re-aggresses with linguistic skill

**Key metrics to watch:**
- Buyer reward: should reach +0.2 to +0.3 by iter 40-60 (paper got +0.77 at 60 with 30B MoE)
- Deal rate: initial dip then recovery to ~50-60%
- Budget violations: should stay <5% (Instruct-2507 has strong constraint following)
- Role confusions: should stay near 0

**Checkpoints pushed to:** `ZeterMordio/anchor-negotiation-dual-role` branches `iter-10` through `iter-60`
---

## Bugfix: Shell Redirect in `torch>=2.6.0,<2.7` Dependency (2026-04-28)

### THE BUG

Job `69f0c5d8d70108f37ace1021` (and our first v10.4 attempt `69f0d5e1d70108f37ace10cb`)
failed instantly with:

```
/bin/sh: 1: cannot open 2.7: No such file
```

**Root cause:** When submitting via the `hf_jobs` API tool (as opposed to the `hf` CLI),
dependency strings are interpolated into a `/bin/sh -lc "..."` command **without quoting**.
The `<2.7` in `torch>=2.6.0,<2.7` gets interpreted by the shell as an input redirect
from a file called `2.7`.

The successful v10 job (`69efcf15`) was submitted via the `hf` CLI which properly
single-quotes each `--with` argument: `'--with' 'torch>=2.6.0,<2.7'`.

### THE FIX

Changed dependency from `torch>=2.6.0,<2.7` to `torch==2.6.0` (exact pin).
No `<` character → no shell redirect ambiguity. This is the exact version that
ran successfully on the v10 A100 job (CUDA 12.4, matches HF driver 12090).

### Job IDs

| Job | Torch Spec | Status |
|-----|-----------|--------|
| `69f0c5d8` | `torch>=2.6.0,<2.7` (unquoted) | ❌ shell redirect error |
| `69f0d5e1` | `torch>=2.6.0,<2.7` (unquoted) | ❌ cancelled (same bug) |
| `69f0d81e` | `torch==2.6.0` | ✅ submitted, scheduling |
---

## Bugfix: False Positive in Thought-Leak Heuristic (2026-04-28)

### THE BUG

Job `69f0d81e` (v10.4, 60-iter run) crashed at iteration 1 with:

```
INFORMATION LEAK: buyer prompt contains seller's cost ($91.99) in a Thought block!
strip_thought() may have failed. Product=other_227
```

### ANALYSIS: FALSE POSITIVE — NOT A REAL LEAK

The assertion was in `_assert_no_private_info_leak()`, specifically the heuristic:
```python
if re.search(rf'Thought:.*(?:cost|private).*{re.escape(cost_str)}', prompt_text, ...):
    raise AssertionError(...)
```

This regex matches `Thought:...cost...$91.99` **anywhere** in the buyer's prompt. But the
buyer's prompt contains the buyer's OWN Thought blocks (in `assistant` messages, kept by
design for chain-of-thought continuity). The buyer can legitimately reason about prices
it has seen in the negotiation:

> "Thought: The seller seems firm at $91.99, which might be near their cost..."

This is the buyer **guessing** — not a leak from `strip_thought()` failing. The seller's
Thought blocks are correctly stripped before injection into buyer context. But the buyer's
own reasoning can reference dollar amounts that coincidentally equal the seller's cost.

### THE FIX

Removed the Thought-block value heuristic entirely (both buyer cost check and seller budget
check). The regex cannot distinguish between the buyer's OWN Thought (legitimate) and a
leaked seller Thought (bug) because the prompt is already chat-template-formatted text.

The **primary guards remain** and have zero false positive risk:
1. `"cost_price"` string check — catches seller's structured cost field leaking to buyer
2. `"Shopping List"` string check — catches buyer's structured budget section leaking to seller
3. `"budget_limit: {budget_str}"` check — catches exact budget value in seller prompt
4. `strip_thought()` + `_assert_strip_thought_complete()` — ensures no Thought: block survives stripping

### Job IDs

| Job | Issue | Status |
|-----|-------|--------|
| `69f0d81e` | False positive Thought heuristic | ❌ crashed iter 1 |
| next job | Heuristic removed | ✅ resubmitting |

---

## 2026-05-14: Pure Negotiation-RLVR Script Rebuild

### Request

After evaluating the SPIRAL/RLVR hybrid (`ZeterMordio/69f1dc2bd70108f37ace15b8`), we decided to try a pure implementation of the negotiation paper first. Hypothesis: the self-play run learned slowly because buyer and seller co-evolved at roughly equal strength, making progress harder to measure. The new experiment should use the negotiation paper's buyer-only structure, but keep non-conflicting engineering improvements from the SPIRAL script.

### Repository Cleanup

- Deleted stale `train_negotiation_clean.py`.
- Added `train_negotiation_pure.py`.
- Kept `train_negotiation_dual_role.py` for the SPIRAL/RLVR hybrid comparison.

### Pure Script Design

`train_negotiation_pure.py` implements the paper's structure:

- Trainable buyer policy only.
- Frozen seller/reference model as environment.
- Buyer always starts.
- Only buyer turns receive GRPO updates.
- Seller temperature restored to paper value `0.7`.
- Seller regulation retained: seller cannot accept/propose below private cost.
- Reward retained: `(budget - final_price) / |budget - cost|`, clipped to `[-1, 1]`.
- No-deal/quit/timeout reward is `0`; buyer format/budget/protocol errors are `-1`.

Explicitly removed SPIRAL components:

- Shared-policy self-play.
- Seller training and seller rewards.
- Zero-sum objective.
- RAE / role-conditioned baselines.
- `DUAL_ROLE_RATIO`.

### Non-Conflicting Improvements Kept

- Batched turn-parallel generation.
- Hidden `Thought:` stripping before cross-role context injection, matching paper §3.1.
- Runtime private-info guards (`cost_price` must not enter buyer prompt; `Shopping List` / exact `budget_limit` must not enter seller prompt).
- Memory-efficient token log-probs via target-gather + `logsumexp`; no full-vocab softmax tensor.
- Clamped log-ratio before exponentiation.
- Group-level advantage normalization for continuous rewards.
- Small KL/reference anchor (`KL_COEF=0.01`) and `LR=1e-6`, matching the stable dense-4B settings from v10 rather than the paper's 30B-MoE `3e-5`.
- AdamW betas `(0.9, 0.95)` and gradient clip `1.0`.
- Optional Liger kernel patching.
- Trackio metrics + alerts.
- Periodic checkpoint branches every `CHECKPOINT_EVERY` iterations.
- All 18 AmazonHistoryPrice categories included (930 valid products). Older scripts omitted two categories and loaded 901.

### Initial Run Config (Prepared, Not Launched)

| Setting | Value |
|---------|-------|
| Script | `train_negotiation_pure.py` |
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Seller/reference | `Qwen/Qwen3-4B-Instruct-2507` frozen |
| Iterations | **42** |
| Batch | 16 products |
| Group | 8 rollouts/product |
| Episodes/iter | 128 |
| Max turns | 6 |
| Max tokens/turn | 300 |
| LR | 1e-6 |
| KL | 0.01 |
| Buyer temp | 1.0 |
| Seller temp | 0.7 |
| Hardware | a100-large |
| Timeout | 8h |
| Hub target | `ZeterMordio/anchor-negotiation-pure` |
| Trackio project | `anchor-negotiation-pure` |

### Launch Notes

Do not use dependency spec `torch>=2.6.0,<2.7` with `hf_jobs`; the `<` can be interpreted as shell redirection. Use exact `torch==2.6.0` or the quoted `hf jobs uv run --with 'torch>=2.6.0,<2.7'` CLI form.


---

## 2026-05-14: SDPO+GRPO Buyer-Only Training File

### Request

Add a separate self-distillation training script rather than mutating the pure GRPO baseline, and default the serious experiment to a larger Qwen3-8B model because SDPO relies on the model's retrospective in-context learning ability.

### File Added

- `train_negotiation_sdpo.py`

The script forks the pure buyer-only environment from `train_negotiation_pure.py` and preserves the HF Jobs-compatible shape: single-file entrypoint, env-var config, Trackio logging, periodic Hub checkpoints, and the same frozen regulated seller setup.

### Design Decisions

- `MODEL_NAME=Qwen/Qwen3-8B`: chosen over 4B because the SDPO paper's scaling study shows self-teacher retrospection improves with model scale. Use 4B only for cheap smoke tests.
- `SDPO_LAMBDA=0.5`: balanced hybrid for the coming real runs; GRPO scalar reward and SDPO token shaping contribute equally.
- `SDPO_FEEDBACK_MODE=strict`: default teacher feedback does not include exact seller cost/private floor. Oracle feedback is supported only as an explicit ablation.
- `SDPO_ADV_CLIP=5.0`: clips teacher-student token logprob gaps before mixing into the PPO-style surrogate, preventing a bad feedback prompt from dominating the update.
- Token-level SDPO first, not top-K logit SDPO: cheaper, easier to audit, and enough to prove signal before adding full logit-level memory complexity.
- Same-product on-policy rollout demos: failed or weak rollouts can learn from better sibling rollouts inside the same GRPO group, matching SDPO's no-external-teacher premise.
- `CHECKPOINT_EVERY=10`: keeps the existing phase-analysis cadence without adding much upload overhead.
- `TRACKIO_PROJECT=anchor-negotiation-sdpo`, Hub target in README: `ZeterMordio/anchor-negotiation-sdpo`.

### Expected First Run

Use A100-large, 12h timeout, exact `torch==2.6.0`, `NUM_ITERS=60`, `BATCH_SIZE=16`, `GROUP_SIZE=8`, and `GEN_BATCH_LIMIT=128`. Run a tiny smoke job first with `NUM_ITERS=1`, `BATCH_SIZE=1`, `GROUP_SIZE=2`, `MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507`, `SELLER_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507`, `CHECKPOINT_EVERY=0`, and a smoke Hub repo before the 8B run.

### 2026-05-14 Resume Note

The pure 42-iter run `6a0538b5e48bea4538b9c3cf` completed successfully and pushed `ZeterMordio/anchor-negotiation-pure`. Logs exposed a Trackio API issue: `trackio.alert()` now expects `trackio.AlertLevel.*`, not string levels (`"INFO"` / `"WARN"`). Updated both `train_negotiation_pure.py` and `train_negotiation_sdpo.py` to use enum levels before launching the SDPO smoke test.

### SDPO Smoke Tests

1. `6a05a1b73308d79117b8f55e` used tiny truncating settings (`MAX_TURNS=2`, `MAX_NEW_TOKENS=120`) to validate loading/update/push quickly. It completed and pushed `ZeterMordio/anchor-negotiation-sdpo-smoke`, but both buyer outputs were format errors; this was attributed to the artificial token/turn truncation, not to the SDPO update path.
2. `6a05a28a3308d79117b8f560` reran with full format settings (`MAX_TURNS=6`, `MAX_NEW_TOKENS=300`) while keeping tiny batch/group (`BATCH_SIZE=1`, `GROUP_SIZE=2`). It completed successfully and pushed `ZeterMordio/anchor-negotiation-sdpo-smoke-fullfmt`.

Full-format smoke metrics: loss `0.3836`, buyer reward `+0.4573`, deal rate `50.0%`, mean turns `4.0`, first-offer ratio `0.7344`, overshoot `0.0%`, outcomes `{DEAL_SELLER_ACCEPTS: 1, SELLER_QUIT: 1}`, SDPO tokens `523`, mean |SDPO advantage| `0.8184`, demos `2`, peak VRAM `48.5GB`. Trackio alerts printed cleanly with enum levels. The SDPO script is ready for the 8B run, subject to launching with a long enough timeout (`12h`) and A100-large or larger.

### SDPO / SDRO Implementation Details from `train_negotiation_sdpo.py` (2026-05-15)

_Note: the repository currently has `train_negotiation_sdpo.py`; there is no separate `sdro` file. This section documents the SDPO/SDRO implementation we have been referring to in discussion._

#### Objective and training topology

The SDPO script is still a **buyer-only negotiation-RLVR setup**, not SPIRAL self-play:

- The **buyer policy** is trainable full-parameter Qwen CausalLM.
- The **seller/environment model** is frozen and regulated by the environment; the current script is ref-free and does not load a separate reference-policy model.
- Buyer always starts; only buyer turns get gradient updates.
- Seller generation remains at paper-style `SELLER_TEMP=0.7`; buyer exploration stays `BUYER_TEMP=1.0`.
- Economic reward is unchanged from the pure script: `(budget - final_price) / abs(budget - cost)`, clipped to `[-1, 1]`; format/protocol/budget failures are `-1`; no-deal/quit/timeouts are `0`.

Default serious-run config in the script:

| Parameter | Default |
|-----------|---------|
| Buyer model | `Qwen/Qwen3-8B` |
| Seller/environment | same as buyer unless `SELLER_MODEL_NAME` overrides; no reference-policy model |
| Iters | `60` |
| Batch × group | `16 × 8 = 128` episodes/iter |
| Max turns | `6` |
| Max new tokens/turn | `300` |
| LR | `5e-6` with `WARMUP_STEPS=10` linear warmup |
| Weight decay | `0.01` AdamW |
| Gradient clip | `GRAD_CLIP_NORM=1.0` |
| Rollout/update context | `ROLLOUT_MAX_LENGTH=3072`, `UPDATE_MAX_LENGTH=3072` |
| PPO clip epsilon | `0.2` config retained, unused in the ref-free on-policy update |
| KL/reference coefficient | `0.0`; no reference-policy model or KL term |
| SDPO mix | `SDPO_LAMBDA=0.5` → 50% GRPO scalar advantage, 50% SDPO token advantage |
| Feedback mode | `strict` by default; `oracle` is explicit ablation only |
| SDPO advantage clip | `±5.0` token logprob gap |
| Checkpoints | branch every `CHECKPOINT_EVERY=10`, final to `main` |

#### Hindsight feedback / self-teacher signal

For each GRPO group (same product repeated `GROUP_SIZE` times), the script builds feedback per episode with no external teacher:

1. `_format_outcome_feedback(ep)` creates verifier text from the rollout outcome:
   - format error → “valid Thought/Talk/Action with one Action line” fix;
   - budget/protocol error → “never offer above budget; only DEAL exact prior SELL” fix;
   - no deal → “use opening anchor/concessions while keeping seller engaged” fix;
   - deal → qualitative quality label based on final price vs buyer budget.
2. `_best_demo_for(ep, group_eps)` optionally attaches a **same-product on-policy demo**:
   - prefers a sibling rollout with `final_price is not None` and strictly better positive reward;
   - otherwise uses the episode itself if it reached a positive deal;
   - transcript is public-only via `_public_transcript()`, which strips hidden Thought before demo insertion.
3. `build_sdpo_teacher_turn_prompt()` reconstructs the original buyer-turn prompt and appends one extra user message containing the hindsight feedback.
4. The same buyer model is evaluated under this feedback-conditioned prompt in `torch.no_grad()`; no external model or oracle answer is sampled.

Strict feedback mode intentionally avoids seller-private data. It can mention the buyer budget and outcome quality, but must not include `cost_price`. `oracle` mode adds seller private cost, MI flag, and numeric reward only for controlled ablations.

#### Token-level SDPO advantage construction

For each buyer turn in each episode:

1. Reconstruct the original buyer prompt with `build_buyer_turn_prompt()`.
2. Compute policy token logprobs on `prompt + actual_completion`.
3. Build a completion mask by zeroing all prompt tokens.
4. Build the feedback-conditioned teacher prompt and evaluate the **same completion** under that prompt.
5. Align student completion logprobs and teacher completion logprobs row-wise by token count.
6. Compute token SDPO values:

```python
sdpo_values = (teacher_completion_lp - student_completion_lp).clamp(-SDPO_ADV_CLIP, SDPO_ADV_CLIP)
```

These values are scattered back onto the original completion-token mask. The mixed advantage is:

```python
adv = SDPO_LAMBDA * grpo_adv + (1.0 - SDPO_LAMBDA) * sdpo_adv
```

where `grpo_adv` is the normalized group reward advantage and `sdpo_adv` is token-level. With the default `SDPO_LAMBDA=0.5`, SDPO acts as an equal-weight dense shaping term while retaining the scalar economic reward.

#### Ref-free on-policy update used after SDPO mixing

The current script uses true ref-free/on-policy sampled-token policy gradient around the mixed advantage:

```python
adv = (SDPO_LAMBDA * grpo_adv + (1.0 - SDPO_LAMBDA) * sdpo_adv).detach()
policy_loss = -adv * pol_lp
loss = mean_over_buyer_completion_tokens(policy_loss)
```

Important stability details:

- no frozen reference-policy model is loaded or evaluated during update;
- `KL_COEF` defaults to `0.0` and no KL/reference term is applied;
- loss is averaged over buyer completion tokens only;
- seller turns are never trained;
- optimizer step is accumulated according to `OPTIM_STEP_EVERY_GROUPS`, defaulting to once per production-shape iteration;
- gradient norm is clipped to `1.0`.

#### Privacy protocol and action parsing

The SDPO script now enforces a stricter public/private boundary than the early v10 runs:

- `strip_qwen_native_thinking()` removes native Qwen `<think>...</think>` blocks, including malformed open/close cases.
- `strip_thought()` removes our explicit `Thought:` scratchpad before a message is shown to the counterparty or inserted in public demos; only `Talk:` + `Action:` should cross roles.
- `extract_action()` first strips native Qwen thinking, then prefers explicit public `Action:` lines. Fallback parsing runs only after structured Thought is stripped, so actions mentioned only in private reasoning are ignored.
- `_assert_strip_thought_complete()` crashes if a Thought or native think marker survives stripping.
- `_assert_no_private_info_leak()` keeps zero-false-positive guards:
  - buyer prompt must not contain `cost_price`;
  - seller prompt must not contain `Shopping List`;
  - seller prompt must not contain exact `budget_limit: $X`.

Design invariant: a role may keep its **own** Thought as assistant history, but the opponent only sees public Talk/Action.

#### Regulated seller environment

Seller regulation remains environment-side, not learned:

- `DEAL` below cost is rejected as `SELLER_CANNOT_ACCEPT_BELOW_COST`.
- `SELL` below cost is rewritten to `round(cost * 1.05, 2)` via `replace_final_action()`, so future buyer context and reward parsing see the regulated public price.
- seller format errors, no-price sells, unexpected buyer/seller actions, invalid buyer DEALs, and budget violations all become explicit terminal outcomes for metrics and reward.

#### Batched rollout path

Rollouts are turn-parallel across active episodes:

- all active buyer prompts are generated in batches, then all active seller prompts;
- finished episodes are masked out of later turns;
- tokenizer uses left padding only inside generation and restores the previous padding side afterward;
- generation truncates prompts at `ROLLOUT_MAX_LENGTH=3072`, samples with `top_p=1.0`, `repetition_penalty=1.1`, and strips native Qwen think blocks before saving history;
- `enable_thinking=False` is passed to `apply_chat_template()` for both buyer and seller prompts.

This preserves the v8/v10 batched-generation speedup while adding SDPO bookkeeping only during the update phase.

#### 8B memory fixes now in the script

Two important implementation changes make the 8B full-finetune path more realistic on A100-class hardware:

1. **Liger default is conditional.** `USE_LIGER` defaults to off when `MAX_MEMORY_PER_GPU_GIB` is set because Triton/Liger kernels failed when Accelerate placed tensors on CPU/offload (`Pointer argument ... cannot be accessed from Triton`). It still defaults on for normal fully GPU-resident loads.
2. **CPU-state AdamW is the default.** `OPTIMIZER=adamw_cpu` keeps exact AdamW `exp_avg` / `exp_avg_sq` state on CPU and copies gradients parameter-by-parameter for the update. This preserves full-parameter training semantics while avoiding the ~2×-parameter CUDA optimizer-state spike that caused 8B A100 optimizer-step OOM. `OPTIMIZER=adamw_cuda` remains available for machines with enough VRAM. `ADAMW_FOREACH=0` is default to avoid extra foreach tensor-list allocations.

The script also fails fast if `device_map` leaves any module on CPU/disk via `_assert_no_cpu_offload()`. Full-parameter training plus generation requires all trainable buyer modules on GPU; silent CPU offload is treated as a configuration error, not a fallback.

#### Metrics and monitoring

Per iteration, metrics include:

- loss, mean buyer reward, deal rate, mean price, mean turns;
- first-offer ratio and budget/price overshoot rate;
- outcome histogram and role confusions;
- SDPO token count, mean absolute SDPO advantage, demo count;
- iteration/rollout/update times and current/peak VRAM.

Trackio logs the same metrics under `TRACKIO_PROJECT=anchor-negotiation-sdpo` and now uses `trackio.AlertLevel.INFO/WARN` enum values. Alerts fire on start, low reward, and format-collapse warning conditions. Final checkpoints include `metrics.json` and a copy of `train_negotiation_sdpo.py` for exact reproducibility.

---

## 2026-05-16: Monitoring Migration from Trackio to W&B

The training scripts were migrated from Trackio to Weights & Biases after the pure run showed Trackio alert fragility and the next SDPO/SDRO runs need reliable monitoring.

### W&B account discovery

Local `WANDB_API_KEY` is available and authenticated successfully. The API reports:

- logged-in W&B username: `akuenzi`
- default/team entity: `chalk`

No W&B webpage setup is required before training: W&B auto-creates a project on the first successful run.

### Code changes

Updated:

- `train_negotiation_pure.py`
- `train_negotiation_sdpo.py`
- `train_negotiation_dual_role.py`
- `README.md`

New env vars used by scripts:

| Variable | Default | Meaning |
|---|---|---|
| `WANDB_ENTITY` | `chalk` | W&B namespace/team/account where runs are stored. |
| `WANDB_PROJECT` | script-specific | Project bucket inside the entity, e.g. `anchor-negotiation-sdpo`. Auto-created if absent. |
| `WANDB_MODE` | `online` | Use `online` for normal cloud logging, `offline` for local dry runs. |
| `WANDB_TAGS` | script-specific | Comma-separated tags for filtering runs. |
| `WANDB_GROUP` | generated | Groups related runs/ablations. |
| `WANDB_JOB_TYPE` | `train` | W&B job type. |
| `RUN_NAME` | generated | Human-readable run name; leave unset for standard naming. |

HF Jobs launch commands must now use dependency `wandb` and pass `--secrets WANDB_API_KEY` in addition to `--secrets HF_TOKEN`.

### Run naming standard

Use short, scan-friendly names for the most important axes and rely on `wandb.config` for the full parameter record.

Default schemas:

| Script | Default run name |
|---|---|
| Pure | `pure__<model>__i<iters>_b<batch>xg<group>__lr<lr>_kl<kl>__s<seed>` |
| SDPO/SDRO | `sdpo__<model>__l<lambda>__<distill>__i<iters>_b<batch>xg<group>__fb<mode>_clip<clip>__lr<lr>_kl<kl>__s<seed>` |
| Dual-role | `dual__<model>__i<iters>_b<batch>xg<group>__dr<ratio>_rae<decay>_<ref>_<advnorm>__lr<lr>_kl<kl>` |

For current SDPO, `<distill>=tokgap` because the script uses token-level teacher-student logprob gaps. Future logit-level SDRO variants should encode the top-k/divergence/trust-region/EMA family compactly, e.g. `topk64-js-tri-ema0p99`.

Title should include method, model, SDPO lambda, distillation family, iters, batch×group, feedback mode, clip, LR, KL, seed. Full `wandb.config` remains the source of truth for max turns/tokens, temperatures, optimizer, Liger, memory caps, exact divergence, EMA/trust-region flags, Hub repo, hardware, etc.

### Validation

Local W&B smoke test succeeded and logged a metric + alert:

- project: `chalk/anchor-negotiation-setup-smoke`
- run: https://wandb.ai/chalk/anchor-negotiation-setup-smoke/runs/x5suqkt7

All patched scripts compile with `python3 -m py_compile`.

---

## 2026-05-18: SDPO 8B Update-Path Performance Implementation

Implemented the planned SDPO update-path speedups for `train_negotiation_sdpo.py` without changing the buyer-only SDPO+GRPO objective or model/dataset scope.

### Changes

- Added update controls:
  - `UPDATE_MICROBATCH_SIZE=4` default: batches flattened buyer-turn examples during the update forward/backward path.
  - `OPTIM_STEP_EVERY_GROUPS=16` default: accumulates across the 16 GRPO groups in the production `BATCH_SIZE=16, GROUP_SIZE=8` iteration and performs one CPU AdamW step per iteration.
  - `UPDATE_PAD_TO_MULTIPLE_OF=8` and `UPDATE_MAX_LENGTH=3072` for efficient update collation with less prompt truncation.
- Flattened each GRPO group into pre-tokenized buyer-turn examples:
  - original buyer prompt + sampled completion;
  - hindsight-feedback teacher prompt + same sampled completion;
  - scalar normalized GRPO advantage.
- Tokenized prompt and completion separately, then concatenated IDs to avoid prompt/completion BPE-boundary drift from separate `prompt` vs `prompt+completion` tokenization calls.
- Left-truncated prompt tokens on overlength update examples so generated completion tokens remain trainable under `UPDATE_MAX_LENGTH`.
- Batched update microbatches through policy, reference, and feedback-conditioned teacher forwards instead of processing one buyer turn at a time. This was superseded by the later ref-free update below, which removes the reference forward.
- Fixed SDPO token-gap alignment to be row-wise per microbatch example; the earlier flattened mask alignment could misalign later rows when teacher/student completion token counts differed.
- Kept CPU-state AdamW exact full-parameter semantics but moved stepping cadence from once per GRPO group to once per `OPTIM_STEP_EVERY_GROUPS` groups; gradients are scaled by the accumulation window and cleared after the CPU optimizer update.
- Added CUDA-synchronized phase timers and logs:
  - `perf/update_pretokenize_s`
  - `perf/update_collate_s`
  - `perf/update_policy_forward_s`
  - `perf/update_ref_forward_s` (schema compatibility; zero for ref-free runs)
  - `perf/update_teacher_forward_s`
  - `perf/update_loss_backward_s`
  - `perf/update_optimizer_s`
  - `perf/update_grad_check_s`
  - plus `perf/update_examples`, `perf/optimizer_steps`, and `train/grad_norm_last`.

### Documentation / eval parity

- Updated `README.md` SDPO launch command with the new update env vars and W&B timer notes.
- Updated `eval_negotiation.py` to match the current training dataset loader: all 18 categories, `features/current_price/average_price/highest_price`, and the exact 802/128 seeded split.
- Tightened eval parity with training for buyer DEAL validation, budget/protocol reward penalties, and regulated seller below-cost SELL rewrites before cross-role context insertion.

### Validation

Local syntax and helper/update-path validation passed:

```bash
python3 -m py_compile train_negotiation_sdpo.py eval_negotiation.py
uv run --with torch --with transformers --no-project python <tiny SDPO CPU update smoke>
```

A short GPU HF smoke also completed successfully:

- Job: [`6a0a5fd2a5e509f1a8413e2b`](https://huggingface.co/jobs/ZeterMordio/6a0a5fd2a5e509f1a8413e2b)
- W&B: [`chalk/anchor-negotiation-sdpo/fnscexas`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/fnscexas)
- Smoke model/metrics: [`ZeterMordio/anchor-negotiation-sdpo-perf-smoke`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-perf-smoke)
- Config: `Qwen/Qwen3-0.6B`, `NUM_ITERS=1`, `BATCH_SIZE=1`, `GROUP_SIZE=2`, `MAX_TURNS=1`, `MAX_NEW_TOKENS=32`, `UPDATE_MICROBATCH_SIZE=4`, `OPTIM_STEP_EVERY_GROUPS=16`, `UPDATE_MAX_LENGTH=512`, `OPTIMIZER=adamw_cpu`.
- Result: completed in 24.4s, pushed final checkpoint, logged W&B `perf/update_*` timers, and metrics showed `update_examples=2`, `optimizer_steps=1`, `sdpo_tokens=64`, peak VRAM `4.2GB`.

---

## 2026-05-18: True Ref-Free / On-Policy SDPO+GRPO Update

Implemented the requested removal of the frozen reference-policy model from `train_negotiation_sdpo.py`.

### Rationale

The previous SDPO script used the frozen seller copy as a PPO-style reference policy during update:

```python
ref_lp = logprob(ref_model, prompt + completion)
log_ratio = pol_lp - ref_lp
policy_loss = clipped_ratio_loss(log_ratio, advantage) + KL_COEF * log_ratio
```

That anchor was a conservative engineering carryover from earlier dense-model stability work, not part of core SDPO. The negotiation RLVR paper uses `KL penalty = 0`, and SDPO's self-teacher is the current model under feedback, not a frozen reference model.

### Changes

- `KL_COEF` now defaults to `0.0`.
- Removed reference-policy model loading. The script still loads the frozen regulated seller/environment model for rollouts, but there is no separate `ref_model` and no reuse of the seller as a reference policy.
- Changed `sdpo_grpo_update(...)` to accept only `buyer_model, tokenizer, episodes, optimizer, device, cpu_adamw_state`.
- Removed the no-grad reference forward from the update microbatch.
- Replaced fixed-reference PPO-style ratio loss with true on-policy sampled-token policy-gradient loss:

```python
adv = (SDPO_LAMBDA * grpo_adv + (1.0 - SDPO_LAMBDA) * sdpo_adv).detach()
policy_loss = -adv * pol_lp
row_loss = mean_over_buyer_completion_tokens(policy_loss)
```

- Kept the SDPO self-teacher forward: the current buyer model is still evaluated under hindsight feedback in `torch.no_grad()` to produce token-level SDPO advantages.
- Kept `perf/update_ref_forward_s` as a zero-valued compatibility metric for existing W&B dashboards, and added `objective/ref_free=1`, `objective/reference_model_used=0`, and `objective/kl_coef` to W&B logs.
- Updated W&B warning text so it no longer recommends increasing a KL anchor; ref-free recovery suggestions are now LR reduction or moving `SDPO_LAMBDA` closer to `1.0`.

### Notes

This makes the update closer to both paper defaults:

- Negotiation RLVR Table 5: `KL penalty = 0`.
- SDPO: self-teacher = current policy conditioned on feedback; no external/frozen reference teacher is required.

Loss values are no longer numerically comparable to the previous fixed-reference clipped-ratio loss. Reward, deal rate, overshoot, format errors, gradient norm, active learning rate, and W&B phase timers are the primary metrics to watch.

### Ref-free smoke validation

- Job: [`6a0a6998e7940de6ee6cdfa3`](https://huggingface.co/jobs/ZeterMordio/6a0a6998e7940de6ee6cdfa3)
- W&B: [`chalk/anchor-negotiation-sdpo/itnu5od5`](https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/itnu5od5)
- Smoke model/metrics: [`ZeterMordio/anchor-negotiation-sdpo-ref-free-smoke`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-ref-free-smoke)
- Config: `Qwen/Qwen3-0.6B`, `NUM_ITERS=1`, `BATCH_SIZE=1`, `GROUP_SIZE=2`, `MAX_TURNS=1`, `MAX_NEW_TOKENS=32`, `KL_COEF=0.0`, `UPDATE_MICROBATCH_SIZE=4`, `OPTIM_STEP_EVERY_GROUPS=16`, `UPDATE_MAX_LENGTH=512`, `OPTIMIZER=adamw_cpu`.
- Result: completed in 24.0s and pushed the final checkpoint. Logs confirm `RefFree=True`, no reference-policy model load, `ref_fwd=0.0s(ref-free)`, `objective/ref_free=1`, `objective/reference_model_used=0`, and one CPU AdamW optimizer step. The tiny truncating rollout produced buyer format errors as expected for smoke-only settings, but the update path, metrics logging, W&B sync, and Hub push succeeded.

### Qwen3.5 9B option-B 2-iteration production-shape smoke

- Job: [`6a0fbffbb33ece92698bfe69`](https://huggingface.co/jobs/ZeterMordio/6a0fbffbb33ece92698bfe69)
- Final model/metrics: [`ZeterMordio/anchor-negotiation-sdpo-qwen35-2iter-gen96`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-2iter-gen96)
- Script source: [`ZeterMordio/anchor-negotiation-sdpo-qwen35-smoke/train_negotiation_sdpo_qwen35.py`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-smoke/blob/main/train_negotiation_sdpo_qwen35.py)
- Local commits: `22ec2ad` parser/fast-path/update optimizations and `2648a5a` seller privacy-guard false-positive fix.
- Config: `Qwen/Qwen3.5-9B` buyer + seller, `NUM_ITERS=2`, `BATCH_SIZE=16`, `GROUP_SIZE=8`, `MAX_TURNS=6`, option-B native thinking, `NATIVE_THINK_TOKENS=300`, `NATIVE_FINAL_TOKENS=96`, `ROLLOUT_MAX_LENGTH=UPDATE_MAX_LENGTH=3072`, `LR=3e-6`, `WARMUP_STEPS=10`, `SDPO_LAMBDA=0.9 -> 0.88`, strict feedback, CPU-state AdamW, `UPDATE_LENGTH_BUCKETING=1`, and self-teacher forward in `torch.inference_mode()`.
- Dependency decision: the requested Qwen3.5 fast-path packages were tested but disabled for this run. `causal-conv1d`/`flash-linear-attention` failed through combinations of missing NVCC for source builds, torch/cu130 requiring a newer driver than the HF A100 image exposed, and ABI-mismatched prebuilt wheels under torch 2.6. The known-good torch 2.6/cu124 path was used; startup diagnostics confirmed fallback to Transformers' slower torch gated-delta path.
- Result: completed in 49.2 min, pushed `iter-1`, `iter-2`, and final `main`. Iter0: reward `-0.0744`, deal `46.9%`, format errors `12/128`, role confusions `3`, first-offer ratio `0.820`, peak reserved VRAM `79.8GB`. Iter1: reward `-0.1196`, deal `38.3%`, format errors `19/128`, role confusions `0`, first-offer ratio `0.706`, peak reserved VRAM `80.3GB`. Early stop triggered after 2 consecutive buyer-format-warning iterations.
- Runtime analysis: average iteration time was `~20.9 min`: rollout `~661s` and update `~595s`. Update timers show pretokenize/collate are already negligible (`<1%`), while backward is dominant (`~56%` of update), policy+teacher forwards are `~29%`, and CPU AdamW optimizer is `~13%`. Length bucketing plus inference-mode teacher did not make the update cheap because full-parameter 9B backward and CPU optimizer remain the bottlenecks.
- Next safe objective-preserving run: keep the objective unchanged but improve stability and throughput by (1) lowering LR to `1e-6` or increasing warmup, (2) slowing SDPO handoff / using more GRPO-heavy early iterations because format errors rose despite low effective warmup LR, (3) reducing `NATIVE_THINK_TOKENS` after measuring whether long hidden thinking improves reward, and (4) using larger hardware (`a100x4`/`l40sx4`) only with a deliberate single-GPU-vs-sharded plan because the current A100x1 run nearly saturates 80GB.

---

## 2026-05-18: SDPO Serious-Run Optimizer Defaults

Updated the SDPO defaults for the next serious Qwen3-8B ref-free run:

- `LR=5e-6`: bolder coming-real-run default requested after the ref-free and optimizer-stability changes; still below the negotiation paper's `3e-5`, which previously collapsed dense Qwen in this repo.
- `SDPO_LAMBDA=0.5`: balanced GRPO/SDPO mixture for the coming real runs.
- `NUM_ITERS=60`: longer run to match prior extended-run planning and give the stronger SDPO signal time to matter.
- `WARMUP_STEPS=10`: with one CPU AdamW step per production-shaped iteration, this warms over roughly the first sixth of a 60-iteration run.
- `WEIGHT_DECAY=0.01`: standard mild AdamW decay for full-parameter tuning, now applied in both CPU-state AdamW and CUDA AdamW paths.
- `GRAD_CLIP_NORM=1.0`: made the existing hardcoded clip configurable and logged.
- `ROLLOUT_MAX_LENGTH=3072`, `UPDATE_MAX_LENGTH=3072`: modestly reduces prompt truncation for 6-turn, 300-token negotiations while avoiding the memory hit of jumping straight to 4096.
- W&B / metrics now include `train/optimizer_global_step`, active optimizer LR as `train/lr`, and `lr_last`, useful during warmup.

---

## 2026-05-23: Repo Health / SDPO-First Cleanup

Committed the Qwen3.5 SDPO smoke logs intentionally under `runs/qwen35_sdpo_smokes/` with a run-card README rather than leaving anonymous `job_*.log` files in the parent `QwenGT/` directory. The archived logs document the VRAM/format-error tradeoffs behind the current SDPO/Qwen3.5 recommendations:

- `job_6a0d34952dc5b1243da50b26.log`: `GEN_BATCH_LIMIT=128`, `LR=5e-6`, Qwen3.5-9B buyer+seller, completed 2 iterations and pushed [`ZeterMordio/anchor-negotiation-sdpo-qwen35-vram-2iter-20260520-041204`](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-vram-2iter-20260520-041204), but reward/deal/format metrics were poor.
- `job_6a0f8d7fb33ece92698bfba9.log`: real-shape clean native-finalizer test at `GEN_BATCH_LIMIT=8`, useful for memory/finalizer context.
- `job_6a0f9688b33ece92698bfc1b.log`: real-shape clean native-finalizer test at `GEN_BATCH_LIMIT=96`, reached iter0 reward `-0.0483`, deal `50.8%`, buyer format errors `21/128`, and peak reserved VRAM around `83GB`.

Repo organization decisions:

- `train_negotiation_sdpo.py` and `train_negotiation_sdpo_qwen35.py` are the active training path.
- `train_negotiation_pure.py` stays active as the pure-GRPO baseline.
- `deprecated/train_negotiation_dual_role.py` now holds the older SPIRAL/RLVR dual-role script. It is still useful for comparison/provenance, but not the preferred path for the current negotiation-improvement goal.
- `deprecated/engineering_notebook.md` holds the original notebook because `JOURNAL.md` is the maintained source of truth.
- `tools/backfill_pure_to_wandb.py` remains as a support utility rather than a top-level training script.
- A local `/Users/tosha/code/QwenGT/AGENTS.md` copy was created from `LOCAL_NOTES.md` for future agent discoverability. It is outside this Git repo and should stay local unless sanitized intentionally.

Health-check updates:

- `README.md` now states SDPO+GRPO buyer-only training as the main direction and points dual-role/spiral users to `deprecated/`.
- `eval_negotiation.py` now defaults to the SDPO model/seller setup, uses the safer Qwen3.5 numeric action parser, and supports both CausalLM checkpoints and Qwen3.5 ImageTextToText wrappers via `AutoProcessor` + `AutoModelForImageTextToText` fallback logic.
- Local syntax import smoke could not run fully on this machine because the local environment lacks `torch`; run `python3 -m py_compile ...` inside the HF Jobs/uv environment before a GPU evaluation launch.
