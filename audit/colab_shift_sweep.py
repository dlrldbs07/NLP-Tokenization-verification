"""
Step 2: shift sweep.

Crop a fixed-length window at 7 different offsets from the same 400bp sequence:

    seq_s = seq[s : WIN + s]        s = 0..6,  WIN = 394

Every condition has the SAME length and contains ONLY real sequence
(no synthetic padding). Position p in the original maps to p - s.
So the sequence content is held (almost) constant and only the token
boundaries move. s = 6 is the frame-preserving control: the window moved
6 bases, but a blocked 6-mer tokenizer sees the same frame as s = 0.

For each (model, class, shift) it reports how often the splice dinucleotide
falls across a token boundary, and for BPE it reports how many sequences
flip between split and intact across shifts (= how much paired data we get).

Tokenizers only — no GPU, no model weights.
"""

# ============================================================
# CELL 1 — config
# ============================================================
CSV_PATH = "train.csv"
WIN      = 394          # window length: 400 - max_shift
SHIFTS   = range(0, 7)  # 0..6
N_PER_CLASS = 2000      # subsample for speed; None = use all

# (label, dinucleotide, position in the ORIGINAL 400bp sequence)
SITES = [(0, "AG", 198),   # acceptor
         (1, "GT", 200)]   # donor

MODELS = {
    "DNABERT-2": "zhihan1996/DNABERT-2-117M",
    "NT-v2":     "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
}


# ============================================================
# CELL 2 — load data
# ============================================================
import pandas as pd

df = pd.read_csv(CSV_PATH)
df["sequence"] = df["sequence"].str.upper()
print(df.shape, df["label"].value_counts().to_dict())


# ============================================================
# CELL 3 — helpers
# ============================================================
from transformers import AutoTokenizer

def token_spans(tok, seq):
    """(start, end) of each token in base coordinates; None if tokens don't rebuild seq."""
    toks = [t.lstrip("#").lstrip("_") for t in tok.tokenize(seq)]
    if "".join(toks) != seq:
        return None
    spans, pos = [], 0
    for t in toks:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return spans

def split_info(spans, p):
    """-> (is_split, offset of p inside its token, length of that token)"""
    for a, b in spans:
        if a <= p < b:
            return (p == b - 1), p - a, b - a
    return None, None, None


# ============================================================
# CELL 4 — sweep
# ============================================================
import numpy as np
from collections import defaultdict

rows = []
flip = {}   # model -> label -> list of (#split shifts out of 7)

for name, path in MODELS.items():
    print(f"\n=== {name} ===")
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print("  tokenizer load failed:", type(e).__name__, str(e)[:300])
        continue

    flip[name] = {}
    for label, dinuc, p in SITES:
        g = df[df["label"] == label]
        g = g[g["sequence"].str[p:p + 2] == dinuc]
        if N_PER_CLASS:
            g = g.sample(min(N_PER_CLASS, len(g)), random_state=0)
        seqs = g["sequence"].tolist()

        per_seq_split = np.zeros(len(seqs), dtype=int)
        for s in SHIFTS:
            q = p - s
            n_split = n_intact = n_bad = 0
            offs, tlens = [], []
            for i, full in enumerate(seqs):
                sub = full[s:WIN + s]
                spans = token_spans(tok, sub)
                if spans is None:
                    n_bad += 1
                    continue
                sp, off, tlen = split_info(spans, q)
                if sp is None:
                    n_bad += 1
                elif sp:
                    n_split += 1
                    per_seq_split[i] += 1
                    offs.append(off); tlens.append(tlen)
                else:
                    n_intact += 1
                    offs.append(off); tlens.append(tlen)
            tot = n_split + n_intact
            rows.append(dict(model=name, label=label, dinuc=dinuc, shift=s,
                             split=n_split, intact=n_intact, bad=n_bad,
                             ratio=n_split / tot if tot else float("nan"),
                             mean_tok_len=float(np.mean(tlens)) if tlens else float("nan")))
            print(f"  label={label} {dinuc} shift={s}: split={n_split:5d} "
                  f"intact={n_intact:5d} ratio={rows[-1]['ratio']:.3f} "
                  f"mean_token_len={rows[-1]['mean_tok_len']:.2f}")

        flip[name][label] = per_seq_split
        n_shifts = len(list(SHIFTS))
        both = int(((per_seq_split > 0) & (per_seq_split < n_shifts)).sum())
        print(f"  -> sequences that are split in SOME shifts and intact in others: "
              f"{both} / {len(seqs)}  (paired comparison uses these)")

res = pd.DataFrame(rows)
res.to_csv("shift_sweep.csv", index=False)
print("\nsaved shift_sweep.csv")
print(res.pivot_table(index=["model", "label"], columns="shift", values="ratio").round(3))
