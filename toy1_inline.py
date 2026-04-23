"""Toy Run 1: Self-contained GRPO pipeline validation.
Qwen3-1.7B + LoRA. 3 iterations. Rule-based seller. Saves to /tmp/toy1_model.
"""
import os, sys, re, json, random, time, urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.randn(2, 2).cuda() @ torch.randn(2, 2).cuda()
    print("CUDA test OK")
else:
    print("FATAL: No CUDA")
    sys.exit(1)

# ─── Dataset ──────────────────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory/main/data/AmazonHistoryPrice/"
CATS = ["electronics", "beauty", "books", "home-kitchen", "sports-outdoors"]


def load_products():
    all_items = []
    for cat in CATS:
        url = BASE + f"{cat}.json"
        try:
            req = urllib.request.urlopen(url, timeout=15)
            items = json.loads(req.read().decode("utf-8"))
        except Exception as e:
            print(f"  Skip {cat}: {e}")
            continue
        for idx, it in enumerate(items):
            try:
                lp = float(it["list_price"].replace("$", "").replace(",", ""))
                cost = float(it["lowest_price"].replace("$", "").replace(",", ""))
                if lp <= 0 or cost <= 0:
                    continue
                all_items.append({
                    "codename": f"{cat}_{idx}", "title": it.get("title", "")[:80],
                    "list_price": lp, "cost": cost, "budget": round(lp * 0.8, 2),
                })
            except Exception:
                continue
    random.seed(42)
    random.shuffle(all_items)
    return all_items[: int(len(all_items) * 0.9)], all_items[int(len(all_items) * 0.9):]


# ─── Prompts / Action / Reward ───────────────────────────────────────────────
BUYER_SYS = """You are a buyer negotiating for the lowest price.
Format EXACTLY as:
Thought: <reasoning>
Talk: <what you say>
Action: [BUY] $<price> (1x <codename>)
Actions: [BUY], [DEAL], [REJECT], [QUIT]"""

ACTION_RE = re.compile(r'\[(BUY|SELL|DEAL|REJECT|QUIT)\](?:\s*\$([\d,.]+))?', re.I)


def build_prompt(p):
    user = (
        f"Product: {p['title']}\nList price: ${p['list_price']:.2f}\n"
        f"Your budget: ${p['budget']:.2f}\nMake your first offer."
    )
    return [{"role": "system", "content": BUYER_SYS}, {"role": "user", "content": user}]


def extract_action(text):
    m = ACTION_RE.search(text)
    if m:
        ps = m.group(2)
        return {"type": m.group(1).upper(), "price": float(ps.replace(",", "")) if ps else None}
    return {"type": "UNKNOWN", "price": None}


def seller_responds(act, p):
    cost, budget, lp, code = p["cost"], p["budget"], p["list_price"], p["codename"]
    at, price = act["type"], act["price"] or 0
    if at in ("QUIT", "UNKNOWN"):
        return None, True
    if at == "BUY":
        if price >= cost * 1.2:
            return price, True
        return None, False
    return None, False


def compute_reward(fp, budget, cost):
    if fp is None:
        return 0.0
    d = abs(budget - cost)
    return max(-1.0, min(1.0, (budget - fp) / d)) if d > 1e-6 else 0.0


# ─── Generation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def gen_completions(model, tok, prompts, G=4, max_new=80, temp=1.0):
    device = model.device
    all_res = []
    for prompt in prompts:
        pt = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(pt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        res = []
        for _ in range(G):
            out = model.generate(
                **pids, max_new_tokens=max_new, do_sample=True, temperature=temp,
                top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
            gt = out.sequences[0][pids["input_ids"].shape[1]:]
            text = tok.decode(gt, skip_special_tokens=True)
            sc = torch.stack(out.scores, dim=1)
            lp = F.log_softmax(sc, dim=-1)
            tok_lp = torch.gather(lp[0], 1, gt.unsqueeze(-1)).squeeze(-1)
            res.append({
                "text": text, "gen_tokens": gt, "token_lp": tok_lp,
                "prompt_text": pt, "prompt_ids": pids["input_ids"],
                "logprob_sum": tok_lp.sum().item(),
            })
        all_res.append(res)
    return all_res


# ─── GRPO Update ──────────────────────────────────────────────────────────────
def grpo_update(model, ref, tok, batch_data, opt, eps=0.2):
    model.train()
    losses = []
    for item in batch_data:
        pt, gt, adv = item["prompt_text"], item["gen_tokens"], item["advantage"]
        full = pt + item["text"]
        fids = tok(full, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        plen = tok(pt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)["input_ids"].shape[1]

        out = model(**fids)
        lp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        tok_lp = torch.gather(lp, 2, fids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            rout = ref(**fids)
            rlp = F.log_softmax(rout.logits[:, :-1, :], dim=-1)
            rtok_lp = torch.gather(rlp, 2, fids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

        mask = fids["attention_mask"][:, 1:].clone()
        mask[:, :plen - 1] = 0

        ratio = torch.exp(tok_lp - rtok_lp)
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        loss = -(torch.min(ratio * adv, clipped * adv) * mask).sum() / (mask.sum() + 1e-8)
        losses.append(loss)

    avg = sum(losses) / len(losses)
    opt.zero_grad()
    avg.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return avg.item()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n[1/5] Loading dataset...")
    train, _ = load_products()
    print(f"  Products: {len(train)}")

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\n[2/5] Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    print(f"  Device: {next(model.parameters()).device}")

    print("\n[3/5] Applying LoRA...")
    lcfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    print("\n[4/5] Loading reference model...")
    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

    N_ITER, BS, G = 3, 4, 4
    print(f"\n[5/5] Training: {N_ITER} iters, BS={BS}, G={G}")
    metrics = []
    t0 = time.time()

    for it in range(N_ITER):
        t1 = time.time()
        print(f"\n--- Iter {it} ---")
        prods = random.sample(train, min(BS, len(train)))
        prompts = [build_prompt(p) for p in prods]

        print("  Generating...")
        comps = gen_completions(model, tok, prompts, G=G, max_new=80)

        rewards_all = []
        for i, p in enumerate(prods):
            grp_r, grp_d = [], []
            for comp in comps[i]:
                act = extract_action(comp["text"])
                fp, done = seller_responds(act, p)
                r = compute_reward(fp, p["budget"], p["cost"])
                grp_r.append(r)
                grp_d.append({**comp, "action": act, "reward": r, "final_price": fp})
            m_r = sum(grp_r) / len(grp_r)
            for gd in grp_d:
                gd["advantage"] = gd["reward"] - m_r
            rewards_all.extend(grp_d)

        print("  Updating...")
        loss = grpo_update(model, ref, tok, rewards_all, opt)

        mr = sum(d["reward"] for d in rewards_all) / len(rewards_all)
        ma = sum(d["advantage"] for d in rewards_all) / len(rewards_all)
        dr = sum(1 for d in rewards_all if d["final_price"] is not None) / len(rewards_all)
        fp_vals = [d["final_price"] for d in rewards_all if d["final_price"] is not None]
        mp = sum(fp_vals) / max(1, len(fp_vals))
        elapsed = time.time() - t1

        print(f"  Loss={loss:.4f}  Reward={mr:.4f}  Adv={ma:.4f}  Deal={dr:.1%}  "
              f"MeanPrice=${mp:.2f}  Time={elapsed:.1f}s")
        metrics.append({"it": it, "loss": loss, "reward": mr, "adv": ma,
                        "deal": dr, "price": mp, "time": elapsed})

        s = rewards_all[0]
        print(f"  Sample: {s['action']['type']} ${s['action']['price']} -> R={s['reward']:.4f}")
        print(f"  Text: {s['text'][:100]}")

    # Save
    save_dir = "/tmp/toy1_model"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Push
    hub_id = os.environ.get("HUB_MODEL_ID", "ZeterMordio/anchor-toy1")
    try:
        from huggingface_hub import HfApi, create_repo
        print(f"\nPushing to {hub_id}...")
        create_repo(hub_id, exist_ok=True)
        HfApi().upload_folder(folder_path=save_dir, repo_id=hub_id, repo_type="model")
        print("Push OK!")
    except Exception as e:
        print(f"Push failed: {e}")

    print(f"\nDone! Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
