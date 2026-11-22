r"""Redraw the two paper figures at their final printed size.

Authoring at 9in and letting LaTeX shrink to \textwidth makes every label
smaller than intended, so these are drawn at ACL dimensions directly:
\textwidth ~ 6.3in for the two-panel figure, \columnwidth ~ 3.1in for the
forest plot. Numbers are the full-run results (seed 20260101).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = list(range(7))
SPLIT = {1: "acceptor", 3: "donor"}

MCC = {
    "local": {
        "NT-v2":     [0.5739, 0.5534, 0.5662, 0.5508, 0.5607, 0.5548, 0.5760],
        "DNABERT-2": [0.5193, 0.5155, 0.5198, 0.5180, 0.5148, 0.5218, 0.5240],
        "DNABERT":   [0.5836, 0.5853, 0.5874, 0.5857, 0.5829, 0.5828, 0.5887],
    },
    "global": {
        "NT-v2":     [0.3471, 0.3217, 0.3246, 0.3410, 0.3253, 0.3288, 0.3438],
        "DNABERT-2": [0.2609, 0.2585, 0.2633, 0.2614, 0.2600, 0.2634, 0.2615],
        "DNABERT":   [0.3696, 0.3721, 0.3751, 0.3719, 0.3708, 0.3661, 0.3609],
    },
}

CONTRASTS = [  # label, d, lo, hi      (local pooling)
    ("NT-v2 · acceptor",      -0.0379, -0.0584, -0.0184),
    ("NT-v2 · donor",         -0.0480, -0.0703, -0.0252),
    ("DNABERT-2 · acceptor",  -0.0074, -0.0174,  0.0021),
    ("DNABERT-2 · donor",      0.0005, -0.0084,  0.0099),
    ("DNABERT · acceptor",    -0.0026, -0.0079,  0.0026),
    ("DNABERT · donor",        0.0035, -0.0030,  0.0094),
    ("one-hot (b) · acceptor", -0.0026, -0.0158, 0.0105),
    ("one-hot (b) · donor",   -0.0495, -0.0693, -0.0317),
]

COL = {"NT-v2": "#1f77b4", "DNABERT-2": "#ff7f0e", "DNABERT": "#2ca02c"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 7.5})

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.45))
for ax, pool in zip(axes, ["local", "global"]):
    vals = MCC[pool]
    lo = min(min(v) for v in vals.values())
    hi = max(max(v) for v in vals.values())
    pad = (hi - lo) * 0.30
    ax.set_ylim(lo - pad * 0.45, hi + pad)          # headroom for the labels
    for off, name in SPLIT.items():
        ax.axvspan(off - 0.17, off + 0.17, color="0.88", zorder=0, lw=0)
        ax.text(off, hi + pad * 0.42, f"{name}\nsplit", fontsize=6.3,
                ha="center", va="center", color="0.35", linespacing=0.95)
    for m, v in vals.items():
        ax.plot(S, v, marker="o", ms=3.2, lw=1.3, color=COL[m], label=m,
                zorder=3)
    ax.set_title(f"{pool} pooling")
    ax.set_xlabel("window offset $s$")
    ax.set_xticks(S)
    ax.grid(alpha=.3, lw=.4)
    ax.tick_params(length=2.5)
axes[0].set_ylabel("macro-MCC")
# one legend for both panels, below the axes so it covers no data
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.075, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(f"fig_mcc_by_offset.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig_mcc_by_offset")

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(3.1, 2.75))
y = np.arange(len(CONTRASTS))[::-1]
for yi, (lab, d, lo, hi) in zip(y, CONTRASTS):
    sig = hi < 0 or lo > 0
    c = "#c0392b" if sig else "0.45"
    ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
    ax.plot([lo, lo], [yi - .14, yi + .14], color=c, lw=1.1)
    ax.plot([hi, hi], [yi - .14, yi + .14], color=c, lw=1.1)
    ax.plot(d, yi, "o", ms=3.6, color=c, zorder=3)
ax.axvline(0, color="k", lw=.7, zorder=1)
ax.axhline(1.5, color="0.8", lw=.7, ls=":")          # separate the baseline
ax.set_yticks(y)
ax.set_yticklabels([c[0] for c in CONTRASTS], fontsize=7)
ax.set_xlabel("paired contrast $d$")
ax.grid(axis="x", alpha=.3, lw=.4)
ax.tick_params(length=2.5)
ax.set_ylim(-0.6, len(CONTRASTS) - 0.4)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig_contrasts.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote fig_contrasts")

# ---------------------------------------------------------------- checks
for pool, vals in MCC.items():
    for m, v in vals.items():
        print(f"  {pool:<7}{m:<11} min {min(v):.4f} @ s={int(np.argmin(v))}  "
              f"max {max(v):.4f} @ s={int(np.argmax(v))}")
