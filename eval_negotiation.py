"""
eval_negotiation.py — Evaluation for trained negotiation models.
RLVR paper §4.1 metrics + §5 benchmarking protocol.

Usage:
    MODEL_PATH="ZeterMordio/anchor-negotiation-dual-role" python eval_negotiation.py

Key variables:
  N_TEST=128        # test products (paper size)
  N_ROLL=4          # rollouts per product (paper: 4)
  BUYER_MODEL       # trained buyer model (from hub or local)
  SELLER_MODEL      # frozen seller baseline (default: base Qwen3-4B)
  BUYER_FIRST=1     # 1=paper protocol (buyer starts), 0=reverse
"""
import os, sys, re, json, random, time
from pathlib import Path
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config ────────────────────────────────────────────────────────────────
BUYER_PATH = os.environ.get("BUYER_PATH", "ZeterMordio/anchor-negotiation-dual-role")
SELLER_PATH = os.environ.get("SELLER_PATH", "Qwen/Qwen3-4B")
N_TEST       = int(os.environ.get("N_TEST", "128"))   # paper test split
N_ROLL       = int(os.environ.get("N_ROLL", "4"))       # paper eval §5: 4
MAX_TURNS    = int(os.environ.get("MAX_TURNS", "6"))
BUYER_FIRST  = os.environ.get("BUYER_FIRST", "1") == "1"
BUYER_TEMP   = float(os.environ.get("BUYER_TEMP", "1.0"))
SELLER_TEMP  = float(os.environ.get("SELLER_TEMP", "0.7"))
MAX_NEW      = int(os.environ.get("MAX_NEW", "4000"))    # paper: 4k eval
SEED         = int(os.environ.get("SEED", "42"))
OUT          = os.environ.get("OUT", "eval_results.json")

random.seed(SEED)
