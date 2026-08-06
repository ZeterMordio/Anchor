# Anchor

Anchor trains Qwen buyers for bilateral price negotiation with reinforcement learning from verifiable rewards (RLVR). It reproduces the buyer-only protocol from [*Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards*](https://huggingface.co/papers/2604.09855) and tests whether hindsight feedback through self-distillation policy optimization (SDPO) improves on pure group relative policy optimization (GRPO).

The buyer has a private budget. A frozen seller has a private cost. They bargain over products drawn from [AmazonPriceHistory](https://github.com/TianXiaSJTU/AmazonPriceHistory), and only the buyer receives gradient updates.

This is active research code. The training pipeline works end to end, but the current Qwen3.5 SDPO runs have not shown a learning improvement yet.

## Method

Each episode works as follows:

1. Sample a product, buyer budget, and seller cost.
2. Let the buyer make the first offer to a frozen seller.
3. Hide each model's private reasoning from its counterparty.
4. Enforce the seller's cost floor and the buyer's budget and protocol rules.
5. Update only the buyer tokens.

The buyer reward is

```text
R = (budget - final_price) / abs(budget - cost)
```

The reward is clipped to `[-1, 1]`. A failed negotiation scores `0`; buyer format, protocol, or budget violations score `-1`.

Pure GRPO uses the episode reward for group-relative advantages. The SDPO path also evaluates the sampled buyer response under hindsight verifier feedback. The log-probability gap from that same-policy teacher supplies token-level credit. The main launcher starts with a GRPO-heavy mixture and moves to an even GRPO/SDPO split by iteration 20. The update is on-policy and does not load a separate reference policy.

## Current status

| Run | Outcome |
| --- | --- |
| [Pure GRPO, Qwen3 4B](https://huggingface.co/ZeterMordio/anchor-negotiation-pure) | Completed the planned 42 iterations. This is the reproduction baseline. |
| [Qwen3.5 9B SDPO fastpath smoke](https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-qwen35-fastpath-2iter-20260523-045622) | Completed two production-shaped iterations and early-stopped on format warnings. Mean reward moved from `-0.1019` to `-0.1905`; deal rate moved from `46.1%` to `35.2%`. |

The Qwen3.5 run validates model loading, rollouts, dense updates, checkpointing, and telemetry on the pinned CUDA fastpath image. It does not establish that SDPO improves negotiation quality. See [`JOURNAL.md`](JOURNAL.md) for run configurations, failed approaches, timing data, and the reasoning behind current defaults.

## Run the main experiment

The supported production path uses Hugging Face Jobs and the prebuilt Qwen3.5 image. It requires:

- an authenticated [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) CLI;
- `HF_TOKEN` and `WANDB_API_KEY` configured as Hugging Face Jobs secrets;
- access to the `a100-large` Jobs flavor.

Preview the exact upload and job commands first. The launcher is dry-run by default.

```bash
git clone https://github.com/ZeterMordio/Anchor.git
cd Anchor
uv run --with huggingface_hub python tools/launch_qwen35_fastpath_sdpo.py --dry-run
```

Review the model, iteration count, output repositories, timeout, and hardware in the printed command. To submit the paid job:

```bash
uv run --with huggingface_hub python tools/launch_qwen35_fastpath_sdpo.py --execute
```

Useful overrides are exposed as launcher flags:

```bash
uv run --with huggingface_hub python tools/launch_qwen35_fastpath_sdpo.py --help
```

The launcher deliberately fixes dense training to `a100-large`, saves every 10 iterations, uses a finite timeout, and uploads the exact training script used by the job. The Docker stack and rebuild instructions live in [`docker/qwen35-fastpath/README.md`](docker/qwen35-fastpath/README.md).

## Other entry points

| File | Use |
| --- | --- |
| [`train_negotiation_sdpo_qwen35.py`](train_negotiation_sdpo_qwen35.py) | Main Qwen3.5 9B SDPO plus GRPO experiment. |
| [`train_negotiation_sdpo.py`](train_negotiation_sdpo.py) | Qwen3 8B fallback and control. |
| [`train_negotiation_pure.py`](train_negotiation_pure.py) | Pure buyer-only GRPO baseline. |
| [`eval_negotiation.py`](eval_negotiation.py) | Checkpoint evaluation against the frozen seller on the held-out split. |
| [`tools/`](tools/) | HF Jobs launcher, fastpath canary, and maintenance utilities. |
| [`deprecated/`](deprecated/) | Archived dual-role and SPIRAL experiments. |

The training scripts are standalone and configured through environment variables. Their defaults are part of the experiment definition and are logged with each run. `train_negotiation_sdpo_qwen35.py` currently defaults to Qwen3.5 9B, 60 iterations, batches of 16 products with 8 rollouts each, six turns, and strict feedback that does not reveal the seller's private cost.

## Local checks

Syntax-check the active scripts without downloading model weights:

```bash
python3 -m py_compile \
  train_negotiation_pure.py \
  train_negotiation_sdpo.py \
  train_negotiation_sdpo_qwen35.py \
  eval_negotiation.py
```

Run the lightweight launcher tests:

```bash
uv run --with pytest pytest -q tests/test_launch_qwen35_fastpath_sdpo.py
```

The LoRA and training-module tests also require PyTorch and Transformers. GPU execution uses the pinned image rather than a local macOS environment.

## Scope and limitations

- Dense Qwen3.5 training keeps a live buyer and seller on one GPU. The current path needs an A100 with 80 GB of memory.
- The scripts fetch the dataset from GitHub at runtime, so runs depend on network access and the upstream file layout.
- Evaluation and training share the seeded `802/128` split across all 18 dataset categories.
- There is no installable package or stable API. The repository contains research scripts and their experiment log.
- No software license has been added.
