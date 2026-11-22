"""Emit ACL-style LaTeX for the HW2 tables, from the run's actual numbers.

Numbers are transcribed from results/*.csv of the 2026-11-20 full run
(seed 20260101, train 15000 / dev 4562 / test 4562). Any derived value
(means, percentages) is computed here rather than typed, so the arithmetic
in the paper is checked rather than trusted.
"""
import numpy as np

# ---------------------------------------------------------------- source data
MCC_LOCAL = {   # macro-MCC per offset, local pooling, full test set
    "NT-v2":     [0.5739, 0.5534, 0.5662, 0.5508, 0.5607, 0.5548, 0.5760],
    "DNABERT-2": [0.5193, 0.5155, 0.5198, 0.5180, 0.5148, 0.5218, 0.5240],
    "DNABERT":   [0.5836, 0.5853, 0.5874, 0.5857, 0.5829, 0.5828, 0.5887],
}
MCC_BASE_B = [0.3822, 0.4231, 0.4224, 0.4235, 0.4277, 0.4308, 0.4117]
MCC_BASE_A = 0.6775

PRIMARY = [  # model, class, n, d, lo, hi, same-frame, sf_lo, sf_hi
    ("NT-v2",     "acceptor",  951, -0.0379, -0.0584, -0.0184, -0.0042, -0.0126, 0.0042),
    ("NT-v2",     "donor",    1010, -0.0480, -0.0703, -0.0252,  0.0079, -0.0030, 0.0198),
    ("DNABERT-2", "acceptor",  951, -0.0074, -0.0174,  0.0021,  0.0053, -0.0074, 0.0179),
    ("DNABERT-2", "donor",    1010,  0.0005, -0.0084,  0.0099,  0.0050, -0.0059, 0.0158),
    ("DNABERT",   "acceptor",  951, -0.0026, -0.0079,  0.0026,  0.0021, -0.0074, 0.0126),
    ("DNABERT",   "donor",    1010,  0.0035, -0.0030,  0.0094,  0.0000, -0.0109, 0.0109),
]

FLIPS = [  # model, class, right at both adjacent, lost at split
    ("NT-v2",     "acceptor", 747, 53),
    ("NT-v2",     "donor",    704, 65),
    ("DNABERT-2", "acceptor", 762, 11),
    ("DNABERT-2", "donor",    768,  8),
    ("DNABERT",   "acceptor", 787,  0),
    ("DNABERT",   "donor",    831,  2),
]

CROSS = [  # pooling, class, n, delta, lo, hi
    ("local",  "acceptor",  951, -0.0305, -0.0521, -0.0095),
    ("local",  "donor",    1010, -0.0485, -0.0728, -0.0248),
    ("global", "acceptor",  951,  0.0063, -0.0279,  0.0421),
    ("global", "donor",    1010,  0.0000, -0.0317,  0.0322),
]

BASE_CONTRAST = [  # class, n, d, lo, hi  (baseline b, window-indexed one-hot)
    ("acceptor",  951, -0.0026, -0.0158,  0.0105),
    ("donor",    1010, -0.0495, -0.0693, -0.0317),
]

ABLATION = {  # gain of a per-offset probe over the single shared probe, local
    "NT-v2":     [0.0295, 0.0524, 0.0444, 0.0451, 0.0368, 0.0506, 0.0253],
    "DNABERT-2": [-0.0059, -0.0047, -0.0059, -0.0082, -0.0067, -0.0078, -0.0163],
    "DNABERT":   [0.0010, 0.0046, -0.0005, 0.0019, 0.0040, 0.0041, -0.0046],
}

ERRMIX = [  # source, class, where, p(acceptor), p(donor), p(non-splice)
    ("baseline (b)", "donor", "adjacent", 0.0683, 0.7475, 0.1842),
    ("baseline (b)", "donor", "split",    0.0614, 0.6980, 0.2406),
    ("NT-v2",        "donor", "adjacent", 0.0347, 0.7965, 0.1688),
    ("NT-v2",        "donor", "split",    0.0366, 0.7485, 0.2149),
    ("baseline (b)", "acceptor", "adjacent", 0.7829, 0.0499, 0.1672),
    ("baseline (b)", "acceptor", "split",    0.7802, 0.0473, 0.1725),
    ("NT-v2",        "acceptor", "adjacent", 0.8496, 0.0189, 0.1314),
    ("NT-v2",        "acceptor", "split",    0.8118, 0.0284, 0.1598),
]

SPLIT_OFF = {"acceptor": 1, "donor": 3}


def ci(lo, hi):
    return f"[{lo:+.4f}, {hi:+.4f}]"


def bold_if(sig, text):
    return r"\textbf{" + text + "}" if sig else text


out = []
A = out.append

# ---------------------------------------------------------------- Table 1
A(r"""% ================= Table 1: primary contrasts =================
\begin{table*}[t]
\centering\small
\begin{tabular}{llrrlrl}
\toprule
Model & Class & $n$ & $d$ & 95\% CI & same-frame & 95\% CI \\
\midrule""")
for m, c, n, d, lo, hi, sf, sflo, sfhi in PRIMARY:
    sig = hi < 0 or lo > 0
    A(f"{m} & {c} & {n} & {bold_if(sig, f'{d:+.4f}')} & {bold_if(sig, ci(lo, hi))} "
      f"& {sf:+.4f} & {ci(sflo, sfhi)} \\\\")
A(r"""\bottomrule
\end{tabular}
\caption{Paired contrasts under local pooling. $d$ compares correctness at the
offset where NT splits the canonical signal (offset~1 for acceptors, offset~3
for donors) against the mean of the two adjacent non-splitting offsets, per
sequence. The same-frame column is the offset-6 minus offset-0 contrast, which
shifts the window by six bases while restoring NT's original reading frame.
Intervals are 95\% percentile bootstrap over $B=2{,}000$ resamples at the
original-sequence level. Bold marks intervals excluding zero.}
\label{tab:primary}
\end{table*}
""")

# ---------------------------------------------------------------- Table 2
A(r"""% ================= Table 2: flip counts =================
\begin{table}[t]
\centering\small
\begin{tabular}{llrrr}
\toprule
Model & Class & correct at & lost at & rate \\
      &       & both adj.  & split   &      \\
\midrule""")
for m, c, both, lost in FLIPS:
    rate = 100.0 * lost / both
    A(f"{m} & {c} & {both} & {lost} & {rate:.1f}\\% \\\\")
A(r"""\bottomrule
\end{tabular}
\caption{Of the sequences a model classifies correctly at \emph{both} adjacent
non-splitting offsets, how many it then fails at the splitting offset. Local
pooling, eligible test sequences.}
\label{tab:flips}
\end{table}
""")

# ---------------------------------------------------------------- Table 3
A(r"""% ================= Table 3: cross-model =================
\begin{table}[t]
\centering\small
\begin{tabular}{llrrl}
\toprule
Pooling & Class & $n$ & $\Delta$ & 95\% CI \\
\midrule""")
for pool, c, n, dl, lo, hi in CROSS:
    sig = hi < 0 or lo > 0
    A(f"{pool} & {c} & {n} & {bold_if(sig, f'{dl:+.4f}')} & "
      f"{bold_if(sig, ci(lo, hi))} \\\\")
A(r"""\bottomrule
\end{tabular}
\caption{Cross-model difference $\Delta$ between NT's paired contrast and
DNABERT-2's, on the same sequences and the same NT-defined offsets.}
\label{tab:cross}
\end{table}
""")

# ---------------------------------------------------------------- Table 4
means = {m: float(np.mean(v)) for m, v in MCC_LOCAL.items()}
mean_b = float(np.mean(MCC_BASE_B))
A(r"""% ================= Table 4: task difficulty =================
\begin{table}[t]
\centering\small
\begin{tabular}{lrr}
\toprule
Representation & macro-MCC & tokenizer \\
\midrule""")
A(f"one-hot, original coords & \\textbf{{{MCC_BASE_A:.4f}}} & none \\\\")
for m in ["DNABERT", "NT-v2", "DNABERT-2"]:
    A(f"{m} (frozen, local) & {means[m]:.4f} & yes \\\\")
A(f"one-hot, window coords & {mean_b:.4f} & none \\\\")
A(r"""\bottomrule
\end{tabular}
\caption{Task difficulty. Model rows are the mean over the seven offsets; the
one-hot rows use the same 40\,bp readout interval with no model at all. A
tokenizer-free linear classifier on raw bases outperforms every frozen genomic
language model representation tested.}
\label{tab:difficulty}
\end{table}
""")

# ---------------------------------------------------------------- Table 5
A(r"""% ================= Table 5: tokenizer-free control =================
\begin{table}[t]
\centering\small
\begin{tabular}{lrrl}
\toprule
Class & $n$ & $d$ & 95\% CI \\
\midrule""")
for c, n, d, lo, hi in BASE_CONTRAST:
    sig = hi < 0 or lo > 0
    A(f"{c} & {n} & {bold_if(sig, f'{d:+.4f}')} & {bold_if(sig, ci(lo, hi))} \\\\")
A(r"""\bottomrule
\end{tabular}
\caption{The same paired contrast computed for the window-indexed one-hot
baseline, which has no tokenizer. The acceptor contrast is flat; the donor
contrast matches NT's in both sign and magnitude, so the donor effect cannot be
attributed to token boundaries.}
\label{tab:basecontrast}
\end{table}
""")

# ---------------------------------------------------------------- Table 6
A(r"""% ================= Table 6: per-offset probe ablation =================
\begin{table}[t]
\centering\small
\begin{tabular}{lrrrrrrr}
\toprule
Model & \multicolumn{7}{c}{offset} \\
\cmidrule(lr){2-8}
 & 0 & 1 & 2 & 3 & 4 & 5 & 6 \\
\midrule""")
for m in ["NT-v2", "DNABERT-2", "DNABERT"]:
    cells = []
    for s, v in enumerate(ABLATION[m]):
        txt = f"{v:+.3f}"
        cells.append(r"\underline{" + txt + "}" if s in (1, 3) else txt)
    A(f"{m} & " + " & ".join(cells) + r" \\")
A(r"""\bottomrule
\end{tabular}
\caption{Change in macro-MCC from training a separate probe per offset instead
of one shared probe, local pooling. Underlined columns are the offsets at which
NT splits a canonical signal. Offsets 0 and 6 share a reading frame, and are
where NT gains least.}
\label{tab:ablation}
\end{table}
""")

# ---------------------------------------------------------------- Table 7
A(r"""% ================= Table 7: error composition =================
\begin{table}[t]
\centering\small
\begin{tabular}{llrrr}
\toprule
Source & Offsets & acc. & don. & non-spl. \\
\midrule""")
last = None
for src, c, where, pa, pd_, pn in ERRMIX:
    if last is not None and last != c:
        A(r"\midrule")
    last = c
    lbl = (f"split ({SPLIT_OFF[c]})" if where == "split" else "adjacent")
    A(f"{src} & {lbl} & {pa:.4f} & {pd_:.4f} & {pn:.4f} \\\\")
A(r"""\bottomrule
\end{tabular}
\caption{Composition of predictions on eligible sequences of one true class, at
the splitting offset versus its two neighbours. Upper block: true donors. Lower
block: true acceptors. Both NT and the tokenizer-free baseline lose donors to
the non-splice class, not to the acceptor class.}
\label{tab:errmix}
\end{table}
""")

# ---------------------------------------------------------------- figures
A(r"""% ================= Figures =================
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_mcc_by_offset.pdf}
\caption{macro-MCC on the full test set at each window offset. Shaded columns
mark the offsets at which NT splits a canonical signal.}
\label{fig:mcc}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_contrasts.pdf}
\caption{Paired contrasts with 95\% confidence intervals, local pooling.}
\label{fig:contrasts}
\end{figure}
""")

open("tables_acl.tex", "w").write("\n".join(out))

# ---------------------------------------------------------------- checks
print("derived values (computed, not transcribed):")
for m, v in means.items():
    print(f"  {m:<10} mean macro-MCC over offsets = {v:.4f}")
print(f"  baseline(b) mean                     = {mean_b:.4f}")
print(f"  baseline(a)                          = {MCC_BASE_A:.4f}")
print()
print("flip rates:")
for m, c, both, lost in FLIPS:
    print(f"  {m:<10} {c:<9} {lost:3d}/{both:3d} = {100*lost/both:5.2f}%")
print()
gap = MCC_BASE_A - max(means.values())
print(f"one-hot(a) exceeds the best frozen model by {gap:.4f} macro-MCC")
print("\nwrote tables_acl.tex")
