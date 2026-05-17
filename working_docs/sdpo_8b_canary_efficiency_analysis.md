# SDPO 8B Production-Shape Canary Efficiency Analysis

Date: 2026-05-17

This is a temporary/working document for planning efficiency improvements before the main 8B SDPO run.

## Run references

- HF Job: <https://huggingface.co/jobs/ZeterMordio/6a0a3504e7940de6ee6cdd40>
- W&B run: <https://wandb.ai/chalk/anchor-negotiation-sdpo/runs/67dqjlos>
- Output repo: <https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-8b-canary-prodshape>
- Metrics file: <https://huggingface.co/ZeterMordio/anchor-negotiation-sdpo-8b-canary-prodshape/blob/main/metrics.json>

## Canary configuration

```text
MODEL_NAME=Qwen/Qwen3-8B
SELLER_MODEL_NAME=Qwen/Qwen3-8B
NUM_ITERS=1
BATCH_SIZE=16
GROUP_SIZE=8
MAX_TURNS=6
MAX_NEW_TOKENS=300
LR=1e-6
KL_COEF=0.01
SDPO_LAMBDA=0.9
SDPO_FEEDBACK_MODE=strict
SDPO_ADV_CLIP=5.0
BUYER_TEMP=1.0
SELLER_TEMP=0.7
CHECKPOINT_EVERY=0
GEN_BATCH_LIMIT=128
GRADIENT_CHECKPOINTING=1
USE_LIGER=1
OPTIMIZER=adamw_cpu
hardware=a100-large
```

## W&B / metrics summary

```text
Status: COMPLETED
Total wall time: 1928.2s = 32.1 min

Training iteration time: 1765.8s = 29.4 min
  rollout: 423.3s = 7.1 min
  update: 1342.4s = 22.4 min

Reward: 0.0690
Deal rate: 57.8%
Mean price: $203.67
Mean turns: 6.04
First offer ratio: 0.8509
Price overshoot rate: 8.6%
Role confusions: 0

SDPO tokens: 45,720
SDPO mean |A|: 0.5100
SDPO demos: 112

VRAM current: 32.8GB
VRAM peak: 70.1GB
```

Outcome distribution:

```text
DEAL_SELLER_ACCEPTS: 48
DEAL_BUYER_ACCEPTS: 26
BUYER_QUIT: 14
SELLER_QUIT: 11
BUYER_BUDGET_VIOLATION: 11
BUYER_DEAL_INVALID_SELLER_OFFER: 10
NO_PRIOR_BUYER_OFFER: 4
SELLER_CANNOT_ACCEPT_BELOW_COST: 2
BUYER_DEAL_PRICE_MISMATCH: 2
```

Derived metrics:

```text
episodes: 128
rollout_s_per_episode: 3.31
iter_s_per_episode: 13.80
update_s_per_episode: 10.49
approx_buyer_turns: ~386.5
update_s_per_approx_buyer_turn: ~3.47
sdpo_tokens_per_episode: 357.2
sdpo_tokens_per_sec_update: 34.1
vram_headroom_gb: ~15.0
42_iter_train_hours_at_current_speed: ~20.6h excluding checkpoint/upload overhead
60_iter_train_hours_at_current_speed: ~29.4h excluding checkpoint/upload overhead
```

## What `GEN_BATCH_LIMIT` means

`GEN_BATCH_LIMIT` is **not** the total number of episodes, although in the production canary it happened to equal `BATCH_SIZE × GROUP_SIZE = 16 × 8 = 128`.

In code, it is the **maximum number of active prompts passed to `model.generate()` in one generation sub-batch**:

```python
for batch_start in range(0, len(prompts_text_list), GEN_BATCH_LIMIT):
    batch_prompts = prompts_text_list[batch_start : batch_start + GEN_BATCH_LIMIT]
    inputs = tokenizer(batch_prompts, padding=True, ...)
    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, ...)
```

During rollout, every negotiation round does:

1. collect prompts for all currently active buyers
2. call `generate_batched(...)` for buyer model
3. parse buyer actions
4. collect prompts for sellers still active
5. call `generate_batched(...)` for seller model
6. parse seller actions and regulate seller

So if 128 episodes are active and `GEN_BATCH_LIMIT=128`, round-0 buyer generation is one big `generate()` call of up to 128 prompts. If `GEN_BATCH_LIMIT=32`, it becomes four calls of 32 prompts. If many episodes end early, later rounds may have fewer active prompts.

`GEN_BATCH_LIMIT` controls **rollout generation microbatching**, not the training/update batch.

Increasing it can improve rollout throughput until VRAM or generation inefficiency becomes limiting. In the canary:

```text
Rollout: 128 episodes in 423s = 3.3s/episode
Peak VRAM total: 70.1GB
```

The 8B tiny full-format smoke was:

```text
2 episodes in 48s = 24.1s/episode
```

So `GEN_BATCH_LIMIT=128` is already doing its job. Rollout batching is not the main problem.

## How rollout and update currently work sequentially

Current iteration structure:

```text
for iteration:
    1. sample products
    2. expand products into BATCH_SIZE × GROUP_SIZE episodes
    3. rollout all episodes with current buyer + frozen seller
    4. update buyer using saved buyer turns
    5. compute/log metrics
    6. checkpoint/final save
```

### Rollout phase

For each turn round up to `MAX_TURNS=6`:

```text
buyer generation for all active episodes
parse buyer actions
seller generation for all active non-terminal episodes
parse/regulate seller actions
```

Rollout is **turn-parallel** but still chronologically sequential. Turn 2 cannot be generated until turn 1 has been parsed, because seller/buyer context depends on prior actions. Within each turn, active episodes are batched.

### Update phase

The update is the inefficient part.

The current code does:

```python
for group in num_groups:               # 16 groups
    group_eps = episodes[g*G:(g+1)*G]  # G = 8 rollouts per product
    compute group advantages
    build feedback for each episode

    for each episode in group:
        for each buyer turn in episode:
            policy forward/backward
            ref forward
            teacher forward
            loss.backward()

    optimizer_step()
```

The key issue is that **the update processes buyer turns one at a time**.

For every buyer turn, it runs roughly:

1. policy model forward with gradients
2. ref model forward no-grad
3. teacher-conditioned buyer model forward no-grad
4. backward through policy
5. later, one CPU AdamW step per group

That means hundreds of small 8B forwards/backwards per iteration.

From W&B/canary:

```text
episodes: 128
mean_turns: 6.04 total role turns
approx buyer turns: ~386
update_time: 1342s
update per buyer turn: ~3.47s
SDPO tokens: 45,720
update throughput: ~34 SDPO tokens/sec
```

This is the bottleneck.

## Are we optimizing the A100?

Partially.

### Rollout: mostly yes

Generation is batched at `GEN_BATCH_LIMIT=128`, and rollout speed improved massively vs tiny smoke. Peak VRAM also suggests the A100 is being used aggressively during rollout.

### Update: no

The update is under-optimized for A100 because:

1. it is per-turn sequential
2. it uses batch size 1 tokenized sequences for logprob/backward
3. it recomputes prompts/tokenization repeatedly
4. it does three model passes per buyer turn
5. it uses CPU AdamW once per group, causing CPU/GPU transfer overhead
6. it likely has Python overhead and poor GPU occupancy because each work item is too small

The A100 likes large matrix multiplies over batches. The current update feeds it many single-sample forwards.

## Should we increase `BATCH_SIZE`?

Not as the first optimization.

Increasing `BATCH_SIZE` increases number of products/groups/episodes and therefore almost linearly increases the number of buyer-turn updates. It does **not** make the update more batched in the current code.

Example:

```text
BATCH_SIZE=16 => 16 groups => 128 episodes => update 1342s
BATCH_SIZE=32 => 32 groups => 256 episodes => likely update ~2680s
```

Unless the update is rewritten to batch buyer turns, increasing `BATCH_SIZE` mostly increases cost.

## Best speedups, in order

### 1. Batch the update over buyer turns within a group

This is the big one.

Instead of:

```text
for ep:
  for buyer_turn:
    tokenize one sequence
    forward one sequence
```

Collect all buyer turns in a group into a list and batch them.

Current group size is 8 episodes. Each episode has ~3 buyer turns, so a group has ~24 buyer-turn training examples.

Add a microbatch env var, for example:

```text
UPDATE_MICROBATCH_SIZE=4 or 8
```

For each microbatch:

- tokenize all prompt+completion texts together with padding
- compute policy logprobs for batch
- compute ref logprobs for batch
- compute teacher logprobs for batch
- compute losses per row
- backward once per microbatch

Expected speedup: **2×–5× on update**, depending on padding and VRAM.

Risk: medium, but it preserves the objective if implemented carefully.

### 2. Pre-tokenize all prompt/completion/teacher examples before model forward

Currently the innermost loop repeatedly calls:

- `apply_chat_template`
- `tokenizer(prompt_text)`
- `tokenizer(full_text)`
- `tokenizer(teacher_prompt_text)`
- `tokenizer(teacher_full_text)`

This is CPU/Python overhead, not GPU work. It also makes profiling harder.

Build a flat list of training examples first:

```python
examples = [
    {
      "full_text": prompt + completion,
      "prompt_len": ...,
      "teacher_full_text": teacher_prompt + completion,
      "teacher_prompt_len": ...,
      "grpo_adv": ...,
    },
    ...
]
```

Then batch-tokenize.

Expected speedup: modest alone, but important for batched update.

### 3. Reduce CPU AdamW steps by stepping once per iteration, or every N groups

Currently:

```text
16 groups => 16 CPU AdamW steps per iteration
```

With 8B parameters, CPU AdamW is expensive because it touches every trainable parameter and copies gradients/update tensors.

If we accumulate gradients over all groups and do:

```text
1 optimizer step per iteration
```

then CPU AdamW overhead drops by about 16×.

But this changes optimizer update frequency. It is equivalent to a larger effective batch for optimizer stepping. The policy objective is still the same data, but gradient accumulation dynamics differ. Preserve update scale by dividing losses appropriately.

Safer compromise:

```text
OPTIM_STEP_EVERY_GROUPS=4
```

So 16 groups gives 4 optimizer steps instead of 16.

Expected speedup: potentially large if CPU AdamW is a major part of the 1342s update.

Risk: medium. Needs smoke/canary. It may stabilize training because larger-batch updates are less noisy, but it changes dynamics.

### 4. Instrument timing before making more assumptions

W&B currently reports update as one 1342s block, but not how much is:

- tokenization
- policy forward/backward
- ref forward
- teacher forward
- CPU AdamW
- grad clipping

Add timers around:

```text
update/tokenize
update/policy_forward_backward
update/ref_forward
update/teacher_forward
update/optimizer_step
```

And log per-group progress to W&B. This does not speed training directly, but prevents blind optimization.

### 5. Test CUDA AdamW only if we can fit it

Current peak with CPU AdamW:

```text
70.1GB / 85.1GB
~15GB headroom
```

CUDA AdamW would need roughly:

- exp_avg fp32: ~32.8GB
- exp_avg_sq fp32: ~32.8GB
- plus optimizer overhead

So it almost certainly does **not** fit on a single A100 for full 8B. CPU AdamW is slow but likely necessary unless we use:

- multi-GPU sharding/FSDP/DeepSpeed
- 8-bit optimizer
- LoRA
- lower-precision optimizer states

Those change implementation/training assumptions. For strict full fine-tuning on one A100, CPU AdamW is the safe path.

### 6. Multi-GPU is the true full-finetune efficiency path

If the goal is ideally efficient rented hardware, single A100 full 8B + ref + CPU AdamW is not ideal. The efficient full-finetune setup is:

- FSDP/DeepSpeed ZeRO across 2–4 GPUs
- optimizer states sharded across GPUs/CPU/NVMe as needed
- batched update examples

But this is a bigger rewrite than optimizing the current script.

## Concrete recommendation before main run

Do **not** launch the 42-iter main yet.

First implement a **batched-update canary**.

### Change A: flatten buyer turns into examples

Inside each group:

```text
group_eps -> buyer_turn_examples
```

### Change B: update microbatching

Add:

```text
UPDATE_MICROBATCH_SIZE=4
```

Run policy/ref/teacher over microbatches.

### Change C: optimizer step frequency

Add:

```text
OPTIM_STEP_EVERY_GROUPS=4
```

or initially keep per-group step to isolate the batching speedup.

### Change D: timing logs

Log:

```text
perf/update_tokenize_s
perf/update_policy_s
perf/update_ref_s
perf/update_teacher_s
perf/update_backward_s
perf/update_optimizer_s
perf/update_examples
perf/update_tokens_per_s
```

### Test sequence

1. 8B full-format tiny smoke:

   ```text
   BATCH_SIZE=1 GROUP_SIZE=2 NUM_ITERS=1
   ```

2. production-shape canary:

   ```text
   BATCH_SIZE=16 GROUP_SIZE=8 NUM_ITERS=1 UPDATE_MICROBATCH_SIZE=4
   ```

3. If VRAM < 80GB, try:

   ```text
   UPDATE_MICROBATCH_SIZE=8
   ```

4. Only then run main.

## Expected target

Current:

```text
rollout: 7.1 min
update: 22.4 min
total iter: 29.4 min
```

Realistic optimized target without changing training method:

```text
rollout: 7 min
update: 6–12 min
total iter: 13–19 min
```

That would cut cost by roughly **1.5×–2.3×**.

If CPU AdamW step reduction is safe, further speedup may be possible.

Main message: `GEN_BATCH_LIMIT=128` already optimizes rollout. The real waste is the **unbatched sequential SDPO update plus repeated CPU AdamW stepping**.
