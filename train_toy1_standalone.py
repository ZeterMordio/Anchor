"""
Toy Run 1: Self-contained GRPO pipeline validation.
Single-turn buyer + rule seller. Qwen3-1.7B with LoRA. 5 iterations.
Saves model to HF Hub at the end.
"""
import os, sys, re, json, random, time, traceback, urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi, create_repo

# ─── Diagnose ─────────────────────────────────────────────────────────────────
print(f"torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    x = torch.randn(2, 2).cuda()
    print(f"CUDA test OK: {x.device}")
else:
    print("FATAL: No CUDA!")
    sys.exit(1)

# ─── Dataset ──────────────────────────────────────────────────────────────────
BASE_URL = (
    "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory/main/data/AmazonHistoryPrice/"
)
CATEGORIES = [
    "automotive", "baby-products", "beauty", "books", "electronics",
    "health-personal-care", "home-kitchen", "industrial-scientific",
    "movies-tv", "music", "other", "patio-lawn-garden", "pet-supplies",
    "software", "sports-outdoors", "tools-home-improvement",
]


def parse_price(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").strip())


def load_products(cache_dir: str = "./data_cache", split_seed: int = 42):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    all_items = []
    for cat in CATEGORIES:
        local = Path(cache_dir) / f"{cat}.json"
        if not local.exists():
            try:
                urllib.request.urlretrieve(BASE_URL + f"{cat}.json", local)
            except Exception as e:
                print(f"  Skip {cat}: {e}")
                continue
        with open(local, encoding="utf-8") as f:
            items = json.load(f)
        for idx, it in enumerate(items):
            try:
                lp = parse_price(it.get("list_price", "0"))
                cost = parse_price(it.get("lowest_price", "0"))
                if lp <= 0 or cost <= 0:
                    continue
                all_items.append({
                    "codename": f"{cat}_{idx}",
                    "title": it.get("title", ""),
                    "description": it.get("description", "")[:200],
                    "category": cat,
                    "list_price": lp,
                    "cost": cost,
                    "budget": round(lp * 0.8, 2),
                    "mi": round(lp * 0.8, 2) > cost,
                })
            except Exception:
                continue
    random.seed(split_seed)
    random.shuffle(all_items)
    train = all_items[: int(len(all_items) * 0.86)]
    test = all_items[len(train):]
    print(f"Products: {len(all_items)}  train={len(train)} test={len(test)}  MI={sum(p['mi'] for p in all_items)}")
    return train, test


# ─── Prompts ──────────────────────────────────────────────────────────────────
BUYER_SYS = """You are a buyer negotiating for the best possible price.
Format your reply EXACTLY as:
Thought: <your strategic reasoning>
Talk: <what you say to the seller>
Action: [BUY] $<price> (1x <codename>)

Actions: [BUY], [DEAL], [REJECT], [QUIT]"""


def build_prompt(p):
    inv = f"Inventory List\n- codename: {p['codename']}\n  title: {p['title']}\n  list_price: ${p['list_price']:.2f}"
    need = f"Shopping List\n- codename: {p['codename']}\n  budget_limit: ${p['budget']:.2f}"
    user = f"{inv}\n\n{need}\n\nNow I am the seller, you are the buyer. Negotiate in 6 turns. Make your first offer now."
    return [{"role": "system", "content": BUYER_SYS}, {"role": "user", "content": user}]


# ─── Action / Reward ──────────────────────────────────────────────────────────
ACTION_RE = re.compile(r'\[(BUY|SELL|DEAL|REJECT|QUIT)\](?:\s*\$([\d,\.]+))?(?:\s*\(([^)]*)\))?', re.I)


def extract_action(text):
    m = ACTION_RE.search(text)
    if m:
        ps = m.group(2)
        return {"type": m.group(1).upper(), "price": float(ps.replace(",", "")) if ps else None, "raw": m.group(0)}
    return {"type": "UNKNOWN", "price": None, "raw": text[-120:]}


def seller_responds(action, p):
    cost, budget, lp, code = p["cost"], p["budget"], p["list_price"], p["codename"]
    at, price = action["type"], action["price"] or 0
    if at == "QUIT":
        return "Buyer quit.", None, True
    if at == "UNKNOWN":
        return "Invalid format.", None, True
    if at == "DEAL":
        return ("Deal!", price, True) if price >= cost else ("Below cost.", None, True)
    if at == "BUY":
        if price >= cost * 1.3:
            return f"[DEAL] ${price:.0f} (1x {code})", price, True
        elif price >= cost:
            c = (price + lp) / 2
            return f"[SELL] ${c:.0f} (1x {code})", None, False
        c = (price + lp * 0.95) / 2
        return f"[SELL] ${c:.0f} (1x {code})", None, False
    return "[REJECT]", None, False


def compute_reward(fp, budget, cost):
    if fp is None:
        return 0.0
    d = abs(budget - cost)
    if d < 1e-6:
        return 0.0
    return max(-1.0, min(1.0, (budget - fp) / d))


# ─── Generation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def gen_completions(model, tok, prompts, G=4, max_new=100, temp=1.0):
    all_res = []
    for prompt in prompts:
        pt = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(pt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        res = []
        for _ in range(G):
            out = model.generate(
                **pids, max_new_tokens=max_new, do_sample=True, temperature=temp,
                top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
            gt = out.sequences[0][pids["input_ids"].shape[1]:]
            gt_text = tok.decode(gt, skip_special_tokens=True)
            sc = torch.stack(out.scores, dim=1)
            lp = F.log_softmax(sc, dim=-1)
            tok_lp = torch.gather(lp[0], 1, gt.unsqueeze(-1)).squeeze(-1)
            res.append({
                "text": gt_text, "logprob_sum": tok_lp.sum().item(),
                "token_log_probs": tok_lp, "gen_tokens": gt,
                "prompt_text": pt, "prompt_ids": pids["input_ids"],
            })
        all_res.append(res)
    return all_res


# ─── GRPO update ────────────────────────────────────────────────────────────────
def grpo_update(model, ref_model, tok, batch_data, opt, eps=0.2):
    model.train()
    tot = 0.0
    for item in batch_data:
        pt, gt, adv = item["prompt_text"], item["gen_tokens"], item["advantage"]
        full_text = pt + item["text"]
        full_ids = tok(full_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        pids = tok(pt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)["input_ids"]
        plen = pids.shape[1]

        out = model(**full_ids)
        lp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        tok_lp = torch.gather(lp, 2, full_ids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            rout = ref_model(**full_ids)
            rlp = F.log_softmax(rout.logits[:, :-1, :], dim=-1)
            rtok_lp = torch.gather(rlp, 2, full_ids["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)

        mask = full_ids["attention_mask"][:, 1:].clone()
        mask[:, :plen - 1] = 0

        ratio = torch.exp(tok_lp - rtok_lp)
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        surr1, surr2 = ratio * adv, clipped * adv
        loss = -(torch.min(surr1, surr2) * mask).sum() / (mask.sum() + 1e-8)
        tot += loss

    avg = tot / len(batch_data)
    opt.zero_grad()
    avg.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return avg.item()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda"
    print("\n[1/6] Loading dataset...")
    train_products, _ = load_products()
    print(f"Train products: {len(train_products)}")

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\n[2/6] Tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"\n[3/6] Loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    print(f"  Model device: {next(model.parameters()).device}")

    print("\n[4/6] Applying LoRA...")
    lcfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    print("\n[5/6] Loading reference model...")
    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

    N_ITER, BS, G, MAX_NEW = 5, 8, 4, 100
    print(f"\n[6/6] Training: {N_ITER} iters, BS={BS}, G={G}")

    metrics_log = []
    t0 = time.time()

    for it in range(N_ITER):
        t1 = time.time()
        print(f"\n--- Iter {it} ---")
        prods = random.sample(train_products, min(BS, len(train_products)))
        prompts = [build_prompt(p) for p in prods]

        print("  Generating...")
        comps = gen_completions(model, tok, prompts, G=G, max_new=MAX_NEW)

        rewards_all = []
        for i, p in enumerate(prods):
            grp_r, grp_d = [], []
            for comp in comps[i]:
                act = extract_action(comp["text"])
                _, fp, done = seller_responds(act, p)
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

        print(f"  Loss={loss:.4f}  Reward={mr:.4f}  Adv={ma:.4f}  Deal={dr:.1%}  MeanPrice=${mp:.2f}  Time={time.time()-t1:.1f}s")
        metrics_log.append({"it": it, "loss": loss, "reward": mr, "adv": ma, "deal": dr, "price": mp, "time": time.time() - t1})

        s = rewards_all[0]
        print(f"  Sample: {s['action']['type']} ${s['action']['price']} -> R={s['reward']:.4f}  |  {s['text'][:80]}")

    # Save
    save_dir = "/app/toy1_model"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Push to hub
    hub_id = os.environ.get("HUB_MODEL_ID", "ZeterMordio/anchor-toy1")
    try:
        print(f"\nPushing to {hub_id} ...")
        create_repo(hub_id, exist_ok=True)
        api = HfApi()
        api.upload_folder(folder_path=save_dir, repo_id=hub_id, repo_type="model")
        print("Push OK!")
    except Exception as e:
        print(f"Push failed: {e}")

    print(f"\nDone! Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
