"""
HW2 analysis — everything the paper needs beyond the main results table.

Runs on what run_full.py already saved (results/*.csv and emb_cache/*.npz).
No GPU, no re-extraction; takes well under a minute, so it is safe to rerun
while drafting.

Produces, into results/:
    fig_mcc_by_offset.(png|pdf)      macro-MCC vs offset, per model
    fig_contrasts.(png|pdf)          paired contrasts with 95% CIs
    table_ablation_offset_probe.csv  shared probe vs one probe per offset
    table_flips.csv                  how often correctness flips at the split
    examples_failures.txt            sampled sequences, including failures
    tables.tex                       the above as LaTeX, ready to paste

    !python analyze.py
"""
import os, glob, json, random, textwrap
import numpy as np, pandas as pd

SEED = 20260101
random.seed(SEED); np.random.seed(SEED)

DRIVE = "/content/drive/MyDrive/nlp_hw2"
BASE = DRIVE if os.path.isdir(DRIVE) else "."
RES = os.path.join(BASE, "results")
CACHE = os.path.join(BASE, "emb_cache")
if not os.path.isdir(RES):
    RES, CACHE = "results", "emb_cache"
print("reading from:", RES, "and", CACHE)

SHIFTS = list(range(7))
SITES = {0: ("AG", 198, "acceptor"), 1: ("GT", 200, "donor")}
SPLIT_OFFSET = {0: 1, 1: 3}
ADJACENT = {0: (0, 2), 1: (2, 4)}
B_BOOT = 2000

report = json.load(open(os.path.join(RES, "run_report.json")))
MODELS = report["env"]["models_loaded"]
print("models:", MODELS)


def boot_ci(values, sids, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sids)
    idx = {s: np.where(sids == s)[0] for s in uniq}
    means = np.empty(B)
    for b in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        means[b] = values[np.concatenate([idx[s] for s in draw])].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return values.mean(), lo, hi


def load_correctness(model, pool):
    p = os.path.join(RES, f"correctness__{model}__{pool}.csv")
    df = pd.read_csv(p)
    df.columns = [c if not c.isdigit() else int(c) for c in df.columns]
    return df.set_index(["sid", "label"])


# ===================================================================
# 1. Figures
# ===================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mcc = pd.read_csv(os.path.join(RES, "table_mcc_by_offset.csv"))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
for ax, pool in zip(axes, ["local", "global"]):
    sub = mcc[mcc["pool"] == pool]
    for m in MODELS:
        s = sub[sub["model"] == m].sort_values("offset")
        ax.plot(s["offset"], s["macro_mcc"], marker="o", ms=4, label=m)
    for lab, off in SPLIT_OFFSET.items():
        ax.axvline(off, color="0.8", lw=6, zorder=0)
        ax.text(off, ax.get_ylim()[0], f" {SITES[lab][2]}\n split",
                fontsize=6, va="bottom", color="0.45")
    ax.set_title(f"{pool} pooling", fontsize=10)
    ax.set_xlabel("window offset $s$")
    ax.grid(alpha=.25, lw=.5)
axes[0].set_ylabel("macro-MCC (full test set)")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, f"fig_mcc_by_offset.{ext}"), dpi=200)
plt.close(fig)
print("wrote fig_mcc_by_offset")

prim = pd.read_csv(os.path.join(RES, "table_primary_contrasts.csv"))
pl = prim[prim["pool"] == "local"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.2, 0.52 * len(pl) + 1.1))
ypos = np.arange(len(pl))[::-1]
for y, r in zip(ypos, pl.itertuples()):
    sig = r.hi < 0 or r.lo > 0
    ax.plot([r.lo, r.hi], [y, y], color="C3" if sig else "0.55", lw=2)
    ax.plot(r.d, y, "o", ms=5, color="C3" if sig else "0.35")
ax.axvline(0, color="k", lw=.8)
ax.set_yticks(ypos)
ax.set_yticklabels([f"{r.model} · {r.cls}" for r in pl.itertuples()], fontsize=8)
ax.set_xlabel("paired contrast $d$  (split offset − adjacent non-split)")
ax.set_title("local pooling, 95% CI, sequence-level bootstrap", fontsize=9)
ax.grid(axis="x", alpha=.25, lw=.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, f"fig_contrasts.{ext}"), dpi=200)
plt.close(fig)
print("wrote fig_contrasts")

# ===================================================================
# 2. Flip analysis — the effect expressed as counts, not just a mean
# ===================================================================
rows = []
for m in MODELS:
    for pool in ("local", "global"):
        w = load_correctness(m, pool)
        for lab, (dinuc, pos, nm) in SITES.items():
            if lab not in w.index.get_level_values("label"):
                continue
            sub = w.xs(lab, level="label")
            sp = SPLIT_OFFSET[lab]; a, b = ADJACENT[lab]
            both_adj_right = (sub[a] == 1) & (sub[b] == 1)
            lost = int((both_adj_right & (sub[sp] == 0)).sum())
            kept = int((both_adj_right & (sub[sp] == 1)).sum())
            both_adj_wrong = (sub[a] == 0) & (sub[b] == 0)
            gained = int((both_adj_wrong & (sub[sp] == 1)).sum())
            rows.append(dict(model=m, pool=pool, cls=nm, n=len(sub),
                             right_at_both_adjacent=kept + lost,
                             lost_at_split=lost, kept_at_split=kept,
                             gained_at_split=gained,
                             loss_rate=round(lost / max(kept + lost, 1), 4)))
flips = pd.DataFrame(rows)
flips.to_csv(os.path.join(RES, "table_flips.csv"), index=False)
print("\n=== flips: of sequences correct at BOTH adjacent offsets, "
      "how many fail at the splitting offset ===")
print(flips[flips["pool"] == "local"].to_string(index=False))

# ===================================================================
# 3. Ablation — shared probe vs one probe per offset
#    Separates "the representation lost the signal" from
#    "the single fixed probe is misaligned to the shifted frame".
# ===================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef


def macro_mcc(y, p, n_cls=3):
    return float(np.mean([matthews_corrcoef((y == c).astype(int),
                                            (p == c).astype(int))
                          for c in range(n_cls)]))


def load_cache(model, split):
    hits = glob.glob(os.path.join(CACHE, f"{model}__{split}__*.npz"))
    if not hits:
        return None
    z = np.load(max(hits, key=os.path.getmtime))
    return z["local"], z["global"], z["sid"], z["off"], z["lab"]


abl = []
for m in MODELS:
    tr, te = load_cache(m, "train"), load_cache(m, "test")
    if tr is None or te is None:
        print(f"  no cache for {m}, skipping offset-probe ablation")
        continue
    for pool_i, pool in ((0, "local"), (1, "global")):
        Xtr, otr, ytr = tr[pool_i].astype(np.float32), tr[3], tr[4]
        Xte, ote, yte = te[pool_i].astype(np.float32), te[3], te[4]
        for s in SHIFTS:
            mtr, mte = otr == s, ote == s
            sc = StandardScaler().fit(Xtr[mtr])
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(
                sc.transform(Xtr[mtr]), ytr[mtr])
            pred = clf.predict(sc.transform(Xte[mte]))
            abl.append(dict(model=m, pool=pool, offset=s, probe="per-offset",
                            macro_mcc=round(macro_mcc(yte[mte], pred), 4)))
        print(f"  {m}/{pool}: per-offset probes done")

if abl:
    ab = pd.DataFrame(abl)
    shared = mcc.rename(columns={"macro_mcc": "shared"})
    comp = ab.merge(shared[["model", "pool", "offset", "shared"]],
                    on=["model", "pool", "offset"], how="left")
    comp["delta_vs_shared"] = (comp["macro_mcc"] - comp["shared"]).round(4)
    comp.to_csv(os.path.join(RES, "table_ablation_offset_probe.csv"),
                index=False)
    print("\n=== ablation: per-offset probe vs the single shared probe "
          "(local pooling) ===")
    print(comp[comp["pool"] == "local"].to_string(index=False))

# ===================================================================
# 4. Sampled examples, including failures
# ===================================================================
test_csv = None
for cand in (os.path.join(BASE, "test.csv"), "test.csv",
             os.path.join(DRIVE, "test.csv")):
    if os.path.exists(cand):
        test_csv = cand
        break

lines = []
if test_csv is None:
    lines.append("test.csv not found — rerun with it beside this script")
else:
    tdf = pd.read_csv(test_csv)
    seq_col = next(c for c in tdf.columns
                   if tdf[c].astype(str).str.fullmatch(r"[ACGTNacgtn]+").mean() > .9)
    tdf["seq"] = tdf[seq_col].str.upper()
    tdf = tdf[tdf["seq"].str.len() == 400].reset_index(drop=True)
    tdf["sid"] = np.arange(len(tdf))
    seq_by_sid = dict(zip(tdf["sid"], tdf["seq"]))

    PRIMARY = MODELS[0] if "NT-v2" not in MODELS else "NT-v2"
    w = load_correctness(PRIMARY, "local")
    for lab, (dinuc, pos, nm) in SITES.items():
        if lab not in w.index.get_level_values("label"):
            continue
        sub = w.xs(lab, level="label")
        sp = SPLIT_OFFSET[lab]; a, b = ADJACENT[lab]
        lost = sub[(sub[a] == 1) & (sub[b] == 1) & (sub[sp] == 0)]
        always = sub[(sub[SHIFTS] == 0).all(axis=1)]
        for title, pool_df, k in [
                (f"{nm}: correct at adjacent offsets, WRONG at the splitting "
                 f"offset {sp}", lost, 4),
                (f"{nm}: wrong at every offset (not a splitting effect)",
                 always, 3)]:
            lines.append("=" * 72)
            lines.append(f"{PRIMARY} / local — {title}")
            lines.append(f"  population: {len(pool_df)} sequences")
            ids = sorted(pool_df.index.tolist())
            pick = random.Random(SEED).sample(ids, min(k, len(ids)))
            for sid in pick:
                s = seq_by_sid.get(sid)
                if s is None:
                    continue
                # integrity check: this sid must be the sequence the probe
                # scored. If sid numbering drifted (e.g. test was subsampled
                # in run_full but not here) the signal will not be where the
                # label says, and the example would be wrong in the paper.
                if s[pos:pos + 2] != dinuc:
                    lines.append(
                        f"  !! sid={sid}: expected {dinuc} at {pos}, found "
                        f"'{s[pos:pos+2]}'. sid numbering does not match "
                        f"run_full.py — do not quote these examples.")
                    continue
                ctx = s[pos - 12:pos + 14]
                mark = " " * 12 + "^^"
                lines.append(f"  sid={sid}  {dinuc}@{pos}")
                lines.append(f"    ...{ctx}...")
                lines.append(f"       {mark}")
                lines.append("    correct by offset: " +
                             " ".join(f"{o}:{int(pool_df.loc[sid, o])}"
                                      for o in SHIFTS))
            lines.append("")

open(os.path.join(RES, "examples_failures.txt"), "w").write("\n".join(lines))
print("\nwrote examples_failures.txt")
print("\n".join(lines[:28]))

# ===================================================================
# 5. LaTeX for the tables
# ===================================================================
tex = []
tex.append("% --- main results: paired contrasts (local pooling) ---")
cols = ["model", "cls", "n", "d", "lo", "hi", "same_frame"]
sub = prim[prim["pool"] == "local"][cols]
tex.append(sub.to_latex(index=False, escape=True, float_format="%.4f"))
tex.append("\n% --- macro-MCC by offset ---")
piv = mcc[mcc["pool"] == "local"].pivot(index="model", columns="offset",
                                        values="macro_mcc")
tex.append(piv.to_latex(escape=True, float_format="%.4f"))
cross_p = os.path.join(RES, "table_cross_model.csv")
if os.path.exists(cross_p):
    tex.append("\n% --- cross-model differences ---")
    tex.append(pd.read_csv(cross_p).to_latex(index=False, escape=True,
                                             float_format="%.4f"))
open(os.path.join(RES, "tables.tex"), "w").write("\n".join(tex))
print("wrote tables.tex")

print("\nfiles in", RES)
for f in sorted(os.listdir(RES)):
    print("   ", f)
