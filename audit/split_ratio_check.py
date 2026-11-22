"""
Go/no-go check for the tokenization-split hypothesis.

1) Find where the splice signal (GT / AG) sits in each class, from the data itself.
2) For each tokenizer, count how often that dinucleotide is split across a token boundary.

Usage (Colab):
    pip install transformers pandas
    python split_ratio_check.py --csv train.csv
Expects a CSV with columns: sequence, label
"""
import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

MODELS = {
    "DNABERT-2": "zhihan1996/DNABERT-2-117M",
    "NT-v2": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
}


def dinuc_profile(seqs, dinuc):
    """Fraction of sequences having `dinuc` starting at each position."""
    L = min(len(s) for s in seqs)
    counts = np.zeros(L - 1)
    for s in seqs:
        for i in range(L - 1):
            if s[i:i + 2] == dinuc:
                counts[i] += 1
    return counts / len(seqs)


def token_spans(tok, seq):
    """(start, end) of each token in base coordinates; None if tokens don't rebuild seq."""
    tokens = tok.tokenize(seq)
    if "".join(tokens) != seq:
        return None
    spans, pos = [], 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return spans


def is_split(spans, p):
    """Dinucleotide at (p, p+1) is split if p is the last base of its token."""
    for a, b in spans:
        if a <= p < b:
            return p == b - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min_freq", type=float, default=0.9)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["sequence"] = df["sequence"].str.upper()
    print(df["label"].value_counts(), "\n")

    # Step 1: locate the splice signal per class
    sites = []  # (label, dinuc, position)
    for label, g in df.groupby("label"):
        for dinuc in ["GT", "AG"]:
            prof = dinuc_profile(g["sequence"].tolist(), dinuc)
            top = np.argsort(prof)[::-1][:3]
            print(f"label={label} {dinuc} top positions:",
                  [(int(p), round(float(prof[p]), 3)) for p in top])
            if prof[top[0]] >= args.min_freq:
                sites.append((label, dinuc, int(top[0])))
    print("\nDetected sites:", sites, "\n")

    # Step 2: split ratio per tokenizer
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        for label, dinuc, p in sites:
            g = df[df["label"] == label]
            split = intact = skipped = 0
            for s in g["sequence"]:
                if s[p:p + 2] != dinuc:
                    continue
                spans = token_spans(tok, s)
                r = is_split(spans, p) if spans else None
                if r is None:
                    skipped += 1
                elif r:
                    split += 1
                else:
                    intact += 1
            total = split + intact
            ratio = split / total if total else float("nan")
            print(f"[{name}] label={label} {dinuc}@{p}: "
                  f"split={split} intact={intact} skipped={skipped} ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
