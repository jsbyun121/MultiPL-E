#!/usr/bin/env python3
"""
Count tokens for a Hugging Face dataset split.
- Works with: hf://jusjinuk/julia-manuals (and any similar dataset)
- Reports: total tokens, min, Q1, median, Q3, max
- Defaults: split=train, tokenizer=cl100k_base, text column auto-detected

Usage:
  python hf_count_tokens.py jusjinuk/julia-manuals
  python hf_count_tokens.py jusjinuk/julia-manuals --split train
  python hf_count_tokens.py jusjinuk/julia-manuals --text-col text
"""

import argparse
from pathlib import Path
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

def autodetect_text_col(example, column_names):
    """
    Heuristic: prefer a column literally named 'text' if present; otherwise,
    pick the string column with the longest content in this example.
    """
    if "text" in column_names and isinstance(example.get("text", None), str):
        return "text"
    str_cols = [c for c in column_names if isinstance(example.get(c, None), str)]
    if not str_cols:
        raise ValueError("No string/text-like columns found. Please pass --text-col explicitly.")
    # choose the longest string in the first example
    return max(str_cols, key=lambda c: len(example[c]))

def main():
    ap = argparse.ArgumentParser(description="Count tokens for a HF dataset split.")
    ap.add_argument("dataset", help="HF dataset id (e.g., jusjinuk/julia-manuals)")
    ap.add_argument("--split", default="train", help="Split name (default: train)")
    ap.add_argument("--text-col", default=None, help="Name of the text column (auto-detect if omitted)")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples to scan")
    ap.add_argument("--tokenizer", default="cl100k_base", help="tiktoken encoding (default: cl100k_base)")
    args = ap.parse_args()

    # Load dataset (materialized, not streaming, so we can compute exact quantiles)
    ds = load_dataset(args.dataset, split=args.split)

    if len(ds) == 0:
        raise ValueError(f"Dataset {args.dataset} (split={args.split}) is empty.")

    # Auto-detect text column if needed
    if args.text_col is None:
        args.text_col = autodetect_text_col(ds[0], ds.column_names)

    enc = tiktoken.get_encoding(args.tokenizer)
    token_counts = []
    n_samples = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)

    for i in tqdm(range(n_samples), desc="Tokenizing"):
        ex = ds[i]
        text = ex.get(args.text_col, "")
        if not isinstance(text, str):
            # if it's a list of strings or something similar, join safely
            if isinstance(text, (list, tuple)):
                text = "\n".join(map(str, text))
            else:
                text = str(text)
        n = len(enc.encode(text))
        token_counts.append(n)

    arr = np.array(token_counts, dtype=np.int64)
    total = int(arr.sum())

    # Compute robust quantiles
    q1 = float(np.percentile(arr, 25))
    q2 = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))

    print("\n=== Dataset ===")
    print(f"id: {args.dataset}")
    print(f"split: {args.split}")
    print(f"text column: {args.text_col}")
    print(f"samples counted: {n_samples:,}")

    print("\n=== Token Stats (tiktoken: {args.tokenizer}) ===")
    print(f"total_tokens: {total:,}")
    print(f"min: {int(arr.min()):,}")
    print(f"Q1: {q1:,.1f}")
    print(f"median(Q2): {q2:,.1f}")
    print(f"Q3: {q3:,.1f}")
    print(f"max: {int(arr.max()):,}")

if __name__ == "__main__":
    main()
