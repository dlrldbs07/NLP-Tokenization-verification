"""
Colab go/no-go check: how often does the splice signal (GT/AG) fall across
a token boundary?

Paste each CELL into a separate Colab cell and run top to bottom.
No GPU needed — only tokenizers are loaded, not models.
"""

# ============================================================
# CELL 1 — install
# ============================================================
# !pip -q install transformers pandas


# ============================================================
# CELL 2 — point at your data
# ============================================================
# Upload a CSV with a sequence column and a label column, then set the path.
# In Colab: click the folder icon on the left → upload → right-click → copy path.
#
# from google.colab import files
# files.upload()

CSV_PATH = "train.csv"   # <-- change this


# ============================================================
# CELL 3 — load and inspect
# ============================================================
import pandas as pd

df = pd.read_csv(CSV_PATH)
print(df.shape)
print(df.columns.tolist())
print(df.head(2))

# auto-detect the columns
seq_col = next((c for c in df.columns
                if df[c].astype(str).str.fullmatch(r"[ACGTNacgtn]+").mean() > 0.9), None)
lab_col = next((c for c in df.columns if c != seq_col and df[c].nunique() <= 10), None)
print("sequence column:", seq_col, "| label column:", lab_col)

assert seq_col and lab_col, "컬럼 자동 탐지 실패 — 아래 두 줄을 직접 지정하세요"
# seq_col, lab_col = "sequence", "label"

df = df[[seq_col, lab_col]].rename(columns={seq_col: "sequence", lab_col: "label"})
df["sequence"] = df["sequence"].str.upper()
print(df["label"].value_counts())
print("sequence lengths:", df["sequence"].str.len().value_counts().head())


# ============================================================
# CELL 4 — step 1: where is the splice signal?
# ============================================================
import numpy as np

def dinuc_profile(seqs, dinuc):
    """Fraction of sequences having `dinuc` starting at each position."""
    L = min(len(s) for s in seqs)
    counts = np.zeros(L - 1)
    for s in seqs:
        for i in range(L - 1):
            if s[i:i + 2] == dinuc:
                counts[i] += 1
    return counts / len(seqs)

MIN_FREQ = 0.9
sites = []   # (label, dinuc, position)

for label, g in df.groupby("label"):
    seqs = g["sequence"].tolist()
    for dinuc in ["GT", "AG"]:
        prof = dinuc_profile(seqs, dinuc)
        top = np.argsort(prof)[::-1][:3]
        print(f"label={label} n={len(seqs)} {dinuc} top:",
              [(int(p), round(float(prof[p]), 3)) for p in top])
        if prof[top[0]] >= MIN_FREQ:
            sites.append((label, dinuc, int(top[0])))

print("\nDetected sites:", sites)


# ============================================================
# CELL 5 — step 2: how often is that signal split?
# ============================================================
from transformers import AutoTokenizer

MODELS = {
    "DNABERT-2": "zhihan1996/DNABERT-2-117M",
    "NT-v2":     "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
}

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

def is_split(spans, p):
    """Dinucleotide at (p, p+1) is split if p is the last base of its token."""
    for a, b in spans:
        if a <= p < b:
            return p == b - 1
    return None

for name, path in MODELS.items():
    print(f"\n=== {name} ===")
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print("  토크나이저 로드 실패:", type(e).__name__, str(e)[:300])
        continue

    demo = df["sequence"].iloc[0][:60]
    print("  sample tokens:", tok.tokenize(demo)[:12])

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
        print(f"  label={label} {dinuc}@{p}: split={split} intact={intact} "
              f"skipped={skipped} ratio={ratio:.3f}")
