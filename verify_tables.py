"""Check every number hard-coded in make_tables.py / make_figures.py
against results/*.csv produced by run_full.py and analyze.py.

Those two scripts hold the run's numbers as Python literals, transcribed
from the result CSVs, and compute only derived quantities (per-offset
means, flip percentages) themselves. Transcription can go wrong silently,
so this script re-reads the CSVs and compares value by value. It writes
nothing and has no side effects.

Usage:
    python verify_tables.py [--results results] [--scripts .]

Exit status 0 if every literal matches, 1 otherwise.
"""
import argparse
import ast
import os
import sys

import pandas as pd

TOL = 5e-5          # the CSVs are rounded to 4 decimal places
PROBLEMS = []
CHECKED = 0


# ---------------------------------------------------------------- literals
def literals(path):
    """Module-level NAME = <literal> assignments, without importing."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        try:
            out[tgt.id] = ast.literal_eval(node.value)
        except ValueError:
            pass            # not a literal (a function, a comprehension)
    return out


# ---------------------------------------------------------------- comparing
def eq(label, literal, frame, column):
    """Compare one literal against exactly one matching CSV row."""
    global CHECKED
    CHECKED += 1
    if len(frame) == 0:
        PROBLEMS.append(f"{label}: no matching row in the CSV")
        return
    if len(frame) > 1:
        PROBLEMS.append(f"{label}: {len(frame)} matching CSV rows, expected 1")
        return
    actual = frame.iloc[0][column]
    if abs(float(literal) - float(actual)) > TOL:
        PROBLEMS.append(f"{label}: script says {literal}, CSV says {actual}")


def pick(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k].astype(str).str.strip() == str(v)
    return df[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--scripts", default=".")
    a = ap.parse_args()

    need = ["table_primary_contrasts.csv", "table_mcc_by_offset.csv",
            "table_cross_model.csv", "table_baselines.csv",
            "table_error_composition.csv", "table_flips.csv",
            "table_ablation_offset_probe.csv"]
    missing = [f for f in need if not os.path.exists(os.path.join(a.results, f))]
    if missing:
        sys.exit("missing result files in %r: %s\nRun run_full.py then "
                 "analyze.py first." % (a.results, ", ".join(missing)))

    csv = {f[:-4]: pd.read_csv(os.path.join(a.results, f)) for f in need}
    prim = csv["table_primary_contrasts"]
    mcc = csv["table_mcc_by_offset"]
    cross = csv["table_cross_model"]
    base = csv["table_baselines"]
    emix = csv["table_error_composition"]
    flips = csv["table_flips"]
    abl = csv["table_ablation_offset_probe"]

    T = literals(os.path.join(a.scripts, "make_tables.py"))
    F = literals(os.path.join(a.scripts, "make_figures.py"))

    # -- macro-MCC per offset ------------------------------------------
    for m, vals in T["MCC_LOCAL"].items():
        for s, v in enumerate(vals):
            eq(f"make_tables MCC_LOCAL[{m}][{s}]", v,
               pick(mcc, model=m, pool="local", offset=s), "macro_mcc")
    for pool, models in F["MCC"].items():
        for m, vals in models.items():
            for s, v in enumerate(vals):
                eq(f"make_figures MCC[{pool}][{m}][{s}]", v,
                   pick(mcc, model=m, pool=pool, offset=s), "macro_mcc")

    # -- primary and same-frame contrasts, local pooling ---------------
    for m, cls, n, d, lo, hi, sf, sflo, sfhi in T["PRIMARY"]:
        row = pick(prim, model=m, pool="local", cls=cls)
        for name, lit, col in [("n", n, "n"), ("d", d, "d"),
                               ("lo", lo, "lo"), ("hi", hi, "hi"),
                               ("same_frame", sf, "same_frame"),
                               ("sf_lo", sflo, "sf_lo"),
                               ("sf_hi", sfhi, "sf_hi")]:
            eq(f"make_tables PRIMARY[{m},{cls}].{name}", lit, row, col)

    # -- cross-model differences ---------------------------------------
    for pool, cls, n, delta, lo, hi in T["CROSS"]:
        row = pick(cross, pool=pool, cls=cls)
        for name, lit, col in [("n", n, "n"), ("delta", delta, "delta"),
                               ("lo", lo, "lo"), ("hi", hi, "hi")]:
            eq(f"make_tables CROSS[{pool},{cls}].{name}", lit, row, col)

    # -- flips ----------------------------------------------------------
    for m, cls, denom, lost in T["FLIPS"]:
        row = pick(flips, model=m, pool="local", cls=cls)
        eq(f"make_tables FLIPS[{m},{cls}].denominator", denom, row,
           "right_at_both_adjacent")
        eq(f"make_tables FLIPS[{m},{cls}].lost", lost, row, "lost_at_split")

    # -- baselines ------------------------------------------------------
    eq("make_tables MCC_BASE_A", T["MCC_BASE_A"],
       pick(base, baseline="onehot_original_coords"), "macro_mcc")
    for s, v in enumerate(T["MCC_BASE_B"]):
        eq(f"make_tables MCC_BASE_B[{s}]", v,
           pick(base, baseline="onehot_window_coords", offset=s), "macro_mcc")
    for cls, n, d, lo, hi in T["BASE_CONTRAST"]:
        row = pick(base, baseline="onehot_window_coords_contrast", cls=cls)
        for name, lit, col in [("n", n, "n"), ("d", d, "d"),
                               ("lo", lo, "lo"), ("hi", hi, "hi")]:
            eq(f"make_tables BASE_CONTRAST[{cls}].{name}", lit, row, col)

    # -- offset-specific probe ablation ---------------------------------
    for m, vals in T["ABLATION"].items():
        for s, v in enumerate(vals):
            eq(f"make_tables ABLATION[{m}][{s}]", v,
               pick(abl, model=m, pool="local", offset=s), "delta_vs_shared")

    # -- error composition ----------------------------------------------
    # The CSV spells these out ("baseline(b) one-hot", "split offset");
    # the scripts abbreviate. Match on a normalized prefix instead.
    nonsplice = [c for c in emix.columns
                 if c.startswith("pred_")
                 and c not in ("pred_acceptor", "pred_donor")]
    if len(nonsplice) != 1:
        PROBLEMS.append("table_error_composition.csv: cannot identify the "
                        "non-splice prediction column, found %r" % nonsplice)
    else:
        src = emix["source"].astype(str)
        where = emix["where"].astype(str)
        for tag, cls, place, p_acc, p_don, p_non in T["ERRMIX"]:
            key = "baseline" if "baseline" in tag else tag
            m = (src.str.startswith(key)
                 & (emix["cls"].astype(str) == cls)
                 & where.str.startswith(place))
            row = emix[m]
            for name, lit, col in [("acceptor", p_acc, "pred_acceptor"),
                                   ("donor", p_don, "pred_donor"),
                                   ("non-splice", p_non, nonsplice[0])]:
                eq(f"make_tables ERRMIX[{tag},{cls},{place}].{name}",
                   lit, row, col)

    # -- the forest plot in make_figures --------------------------------
    for label, d, lo, hi in F["CONTRASTS"]:
        name, cls = [x.strip() for x in label.split("·")]
        if name.startswith("one-hot"):
            row = pick(base, baseline="onehot_window_coords_contrast", cls=cls)
            cols = ("d", "lo", "hi")
        else:
            row = pick(prim, model=name, pool="local", cls=cls)
            cols = ("d", "lo", "hi")
        for lit, col in zip((d, lo, hi), cols):
            eq(f"make_figures CONTRASTS[{label}].{col}", lit, row, col)

    # -- report ----------------------------------------------------------
    print(f"compared {CHECKED} values against {a.results}/")
    if PROBLEMS:
        print(f"\n{len(PROBLEMS)} MISMATCH(ES):")
        for p in PROBLEMS:
            print("  -", p)
        print("\nThe paper's tables and figures do not match the result "
              "files. Fix the literals before using them.")
        return 1
    print("all literals in make_tables.py and make_figures.py match the "
          "result CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
