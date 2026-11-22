"""
HW2 full run — extraction + probe + main results.

Scope differs from the pilot in one way that matters: dev and test are used in
FULL. The width of the paired-contrast confidence interval is set by the number
of ELIGIBLE TEST sequences, not by training size, so spending compute there buys
more precision per GPU-minute than enlarging the training split would.

Embeddings are cached per (model, split, config) and the cache lives on Drive,
so a disconnect costs only the current split.

    !python run_full.py

Writes results/*.csv — those files are what the paper's tables are built from.
Everything is seeded; every number printed is a real measurement.
"""

# ============================================================
# CELL 1 — install + config
# ============================================================
# transformers 5.x breaks the remote code these checkpoints ship with
#   NT-v2     -> EsmConfig has no attribute 'is_decoder'
#   DNABERT-2 -> BertConfig has no attribute 'pad_token_id'
# Pin the MAJOR version only. An exact 4.44.x pin fails on Colab's Python 3.13
# because its `tokenizers` dependency has no cp313 wheel and the Rust build
# fails; the newest 4.x does have wheels.
# !pip -q install -U "transformers<5" einops scikit-learn pandas

SEED        = 20260101
WIN         = 394          # 400 - max_shift
SHIFTS      = list(range(7))
READOUT     = (180, 220)   # pooling interval in ORIGINAL coordinates
N_TRAIN     = 15000        # stratified subsample; None = all 36,496
N_DEV       = None         # full
N_TEST      = None         # full — this is what sets the CI width
BATCH       = 32
C_GRID      = [0.01, 0.1, 1.0, 10.0]
B_BOOT      = 2000

# (label, dinucleotide, position in the ORIGINAL 400 bp sequence)
SITES = [(0, "AG", 198),   # acceptor
         (1, "GT", 200)]   # donor
# NT-defined splitting offset per class, VERIFIED from token spans in CELL 5
SPLIT_OFFSET = {0: 1, 1: 3}
ADJACENT     = {0: (0, 2), 1: (2, 4)}

MODELS = {
    # NT's config auto_map has no "AutoModel" entry, so AutoModel silently
    # falls back to the built-in ESM, which lacks NT's gated FFN and fails on a
    # weight-shape mismatch. AutoModelForMaskedLM pulls the repo's own
    # modeling_esm.py; .esm is the encoder underneath the MLM head.
    "NT-v2":     dict(ckpt="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                      kind="blocked", loader="mlm", base_attr="esm"),
    # DNABERT-2 ships a Triton flash-attention kernel that does not compile on
    # a T4 under triton 3.x. Its forward picks the path at call time with
    #     if self.p_dropout or flash_attn_qkvpacked_func is None:  -> PyTorch
    # so nulling that module global selects the plain PyTorch branch, which
    # computes the same exact attention (flash attention is an exact kernel,
    # not an approximation). Slower, identical numerics.
    "DNABERT-2": dict(ckpt="zhihan1996/DNABERT-2-117M", kind="bpe",
                      loader="auto", base_attr=None, no_triton=True),
    "DNABERT":   dict(ckpt="zhihan1996/DNA_bert_6", kind="overlap6",
                      loader="auto", base_attr=None),
}

DATA_DIR  = "."   # folder holding train.csv / dev.csv / test.csv
# Cache and results go to Drive so a runtime restart does not cost GPU hours.
# Set DRIVE to None to keep everything in the ephemeral runtime instead.
DRIVE     = "/content/drive/MyDrive/nlp_hw2"
RESULT_DIR = "results"

import os, time, json, gc, random
import numpy as np, pandas as pd
import torch

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

import transformers
print("transformers:", transformers.__version__)
if int(transformers.__version__.split(".")[0]) >= 5:
    print("  !! transformers 5.x will fail to load NT-v2 and DNABERT-2.")
    print("  !! Run:  !pip -q install -U 'transformers<5'   then rerun this script.")

# Mount Drive if it is configured, and keep the cache there.
if DRIVE:
    if not os.path.exists("/content/drive"):
        try:
            from google.colab import drive as _gdrive
            _gdrive.mount("/content/drive")
        except Exception as e:
            print("  Drive mount failed, falling back to local cache:", e)
            DRIVE = None
if DRIVE and os.path.isdir(os.path.dirname(DRIVE)):
    os.makedirs(DRIVE, exist_ok=True)
    CACHE_DIR = os.path.join(DRIVE, "emb_cache")
    RESULT_DIR = os.path.join(DRIVE, "results")
    # fall back to Drive copies of the data if they are not in the runtime
    if not os.path.exists(os.path.join(DATA_DIR, "train.csv")) \
            and os.path.exists(os.path.join(DRIVE, "train.csv")):
        DATA_DIR = DRIVE
else:
    CACHE_DIR = "emb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
print("data:", DATA_DIR, "| cache:", CACHE_DIR, "| results:", RESULT_DIR)


# ============================================================
# CELL 2 — load data
# ============================================================
def load_split(name, n):
    """Find <name>.csv, auto-detect the sequence/label columns, subsample."""
    path = None
    for cand in (f"{DATA_DIR}/{name}.csv", f"{DATA_DIR}/{name}.tsv"):
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        raise FileNotFoundError(f"no {name}.csv under {DATA_DIR}")
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    seq_col = next(c for c in df.columns
                   if df[c].astype(str).str.fullmatch(r"[ACGTNacgtn]+").mean() > 0.9)
    lab_col = next(c for c in df.columns if c != seq_col and df[c].nunique() <= 10)
    df = df[[seq_col, lab_col]].rename(columns={seq_col: "seq", lab_col: "label"})
    df["seq"] = df["seq"].str.upper()
    df = df[df["seq"].str.len() == 400].reset_index(drop=True)
    if n is not None and len(df) > n:
        # stratified subsample, reproducible
        per = n // df["label"].nunique()
        parts = [g.sample(min(len(g), per), random_state=SEED)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).sort_index().reset_index(drop=True)
    df["sid"] = np.arange(len(df))          # original-sequence id, used by bootstrap
    return df

train = load_split("train", N_TRAIN)
dev   = load_split("dev",   N_DEV)
test  = load_split("test",  N_TEST)
for nm, d in [("train", train), ("dev", dev), ("test", test)]:
    print(nm, len(d), dict(d["label"].value_counts().sort_index()))


# ============================================================
# CELL 3 — eligibility + windows
# ============================================================
def eligible_mask(df):
    """True where the canonical signal sits at its standard coordinate."""
    m = np.zeros(len(df), dtype=bool)
    for lab, dinuc, pos in SITES:
        sel = df["label"].values == lab
        has = df["seq"].str[pos:pos + 2].values == dinuc
        m |= (sel & has)
    return m

for nm, d in [("train", train), ("dev", dev), ("test", test)]:
    d["eligible"] = eligible_mask(d)
    for lab, dinuc, pos in SITES:
        sel = d["label"].values == lab
        print(f"{nm} label={lab} {dinuc}@{pos}: "
              f"{(sel & d['eligible'].values).sum()} / {sel.sum()}")

def windows(seq):
    return [seq[s:s + WIN] for s in SHIFTS]


# ============================================================
# CELL 4 — token spans (base coordinates) per tokenizer family
# ============================================================
from transformers import AutoTokenizer, AutoModel

def to_model_input(text, kind):
    """DNABERT expects space-separated overlapping 6-mers; the others take raw."""
    if kind == "overlap6":
        return " ".join(text[i:i + 6] for i in range(len(text) - 5))
    return text

def token_spans(tok, text, kind):
    """
    (start, end) in BASE coordinates for every non-special token, in order.
    Returns None if the tokenization cannot be mapped (e.g. an [UNK] appears),
    so those sequences are skipped rather than silently mis-pooled.
    """
    if kind == "overlap6":
        n_k = len(text) - 5
        return [(i, i + 6) for i in range(n_k)]
    # strip subword markers before measuring lengths — DNABERT-2's BPE emits
    # them, and without this the reconstruction check below fails on every
    # sequence and silently drops the whole model. (Same handling as HW1.)
    pieces = [t.lstrip("#").lstrip("_").lstrip("▁")
              for t in tok.tokenize(text)]
    if "".join(pieces) != text:
        return None
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans

def n_prefix_special(tok, text, kind):
    """How many special tokens the encoder prepends, so span i lines up with
    input_ids[i + n_prefix]."""
    enc = tok(to_model_input(text, kind), return_tensors=None)["input_ids"]
    n_body = len(token_spans(tok, text, kind) or [])
    ids = tok.convert_ids_to_tokens(enc)
    specials = set(tok.all_special_tokens)
    k = 0
    while k < len(ids) and ids[k] in specials:
        k += 1
    assert len(enc) - k >= n_body, "token count mismatch — check the tokenizer"
    return k

# --- verify the split arithmetic instead of assuming it ---------------------
def split_status(tok, text, kind, dinuc_pos_in_window):
    """True if the dinucleotide starting at that window position is cut across
    two tokens."""
    spans = token_spans(tok, text, kind)
    if spans is None:
        return None
    p = dinuc_pos_in_window
    for a, b in spans:
        if a <= p < b:
            return p == b - 1      # last base of its token -> next base is elsewhere
    return None

# --- cheap guard: can every tokenizer be mapped back to base coordinates? ---
# Run this BEFORE the 40-minute extraction, not after.
print("--- tokenizer span check (all models) ---")
_demo = train["seq"].iloc[0][:WIN]
for _m, _cfg in MODELS.items():
    _t = AutoTokenizer.from_pretrained(_cfg["ckpt"], trust_remote_code=True)
    _sp = token_spans(_t, _demo, _cfg["kind"])
    if _sp is None:
        print(f"  {_m:<12} FAILED to map tokens to bases — fix before extracting")
        print(f"               sample tokens: {_t.tokenize(_demo)[:8]}")
    else:
        print(f"  {_m:<12} ok  {len(_sp)} tokens, covers {_sp[-1][1]} bases "
              f"(expected {WIN if _cfg['kind'] != 'overlap6' else WIN - 5})")
    del _t

print("\n--- verifying NT split offsets from actual token spans ---")
_tok_nt = AutoTokenizer.from_pretrained(MODELS["NT-v2"]["ckpt"], trust_remote_code=True)
_probe = train[train["eligible"]].groupby("label").head(50)
for lab, dinuc, pos in SITES:
    sub = _probe[_probe["label"] == lab]
    for s in SHIFTS:
        stats = [split_status(_tok_nt, r.seq[s:s + WIN], "blocked", pos - s)
                 for r in sub.itertuples()]
        stats = [x for x in stats if x is not None]
        rate = 100.0 * sum(stats) / max(len(stats), 1)
        flag = "  <-- SPLIT" if rate > 50 else ""
        print(f"  label={lab} {dinuc} offset={s}: split {rate:5.1f}% (n={len(stats)}){flag}")


# ============================================================
# CELL 5 — embedding extraction  (the heavy stage; timed)
# ============================================================
from transformers import AutoConfig

# Current transformers reads the RoPE base from config.rope_theta. NT's config
# predates that field; the older ESM implementation had the base hardcoded to
# 10000, so writing 10000.0 reproduces the original behaviour exactly rather
# than changing the model. Recorded here because it belongs in the
# reproducibility appendix.
CONFIG_SHIMS = {"rope_theta": 10000.0}
SHIMS_APPLIED = {}

from transformers import AutoModelForMaskedLM
import sys

def disable_triton_attention():
    """Null the module global that selects DNABERT-2's Triton kernel, so its
    forward falls into the PyTorch branch. Returns the modules patched."""
    patched = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.endswith("bert_layers"):
            continue
        if getattr(mod, "flash_attn_qkvpacked_func", None) is not None:
            mod.flash_attn_qkvpacked_func = None
            patched.append(name.split(".")[-2] + "." + name.split(".")[-1])
    return patched


def load_model(model_name):
    cfg = MODELS[model_name]
    tok = AutoTokenizer.from_pretrained(cfg["ckpt"], trust_remote_code=True)
    conf = AutoConfig.from_pretrained(cfg["ckpt"], trust_remote_code=True)
    applied = []
    for key, val in CONFIG_SHIMS.items():
        if not hasattr(conf, key):
            setattr(conf, key, val)
            applied.append(f"{key}={val}")
    if applied:
        SHIMS_APPLIED[model_name] = applied
        print(f"  [shim] {model_name}: added {', '.join(applied)} to config")
    cls = AutoModelForMaskedLM if cfg.get("loader") == "mlm" else AutoModel
    model = cls.from_pretrained(cfg["ckpt"], config=conf, trust_remote_code=True)
    if cfg.get("no_triton"):
        hit = disable_triton_attention()
        if hit:
            SHIMS_APPLIED.setdefault(model_name, []).append(
                "flash_attn_qkvpacked_func=None (PyTorch attention path)")
            print(f"  [shim] {model_name}: Triton attention disabled in {hit}")
    # strip the LM head: we want encoder hidden states, not vocabulary logits
    if cfg.get("base_attr") and hasattr(model, cfg["base_attr"]):
        model = getattr(model, cfg["base_attr"])
    model = model.to(DEV).eval()
    return tok, model, cfg["kind"]


def hidden_of(out):
    """Final-layer hidden states, whatever the model's output container is."""
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states:
        return out.hidden_states[-1]
    h = out[0]
    if h.dim() != 3:
        raise RuntimeError(f"unexpected model output shape {tuple(h.shape)}")
    return h

@torch.no_grad()
def smoke_test(model_name):
    """One batch of two, before committing to the long loop."""
    tok, model, kind = load_model(model_name)
    w = train["seq"].iloc[0][:WIN]
    spans = token_spans(tok, w, kind)
    enc = tok([to_model_input(w, kind)], return_tensors="pt", padding=True).to(DEV)
    out = model(**{k: v for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")})
    h = hidden_of(out)
    n_pre = n_prefix_special(tok, w, kind)
    conf = AutoConfig.from_pretrained(MODELS[model_name]["ckpt"],
                                      trust_remote_code=True)
    len_ok = h.shape[1] >= n_pre + len(spans)
    # guard against grabbing MLM logits instead of hidden states: the last
    # dimension must be the hidden size, never the vocabulary size
    dim_ok = h.shape[-1] == getattr(conf, "hidden_size", h.shape[-1])
    note = ("" if dim_ok else
            f"  <-- got vocab-sized output ({h.shape[-1]}), not hidden states")
    print(f"  {model_name:<12} hidden {tuple(h.shape)}  spans {len(spans)}  "
          f"prefix {n_pre}  -> {'OK' if (len_ok and dim_ok) else 'MISMATCH'}{note}")
    del model; gc.collect(); torch.cuda.empty_cache()
    return len_ok and dim_ok

print("--- model smoke test ---")
OK_MODELS = []
for _m in MODELS:
    try:
        if smoke_test(_m):
            OK_MODELS.append(_m)
    except Exception as e:
        print(f"  {_m:<12} LOAD FAILED: {type(e).__name__}: {str(e)[:200]}")
print("usable models:", OK_MODELS)
if not OK_MODELS:
    raise RuntimeError("no model loaded — stop here and report the errors above")


@torch.no_grad()
def extract(model_name, df, log):
    tok, model, kind = load_model(model_name)
    n_pre = n_prefix_special(tok, df["seq"].iloc[0][:WIN], kind)

    local_out, global_out, keep_sid, keep_off, keep_lab = [], [], [], [], []
    t0 = time.time()
    rows = list(df.itertuples())
    for s in SHIFTS:
        lo, hi = READOUT[0] - s, READOUT[1] - s     # readout in window coordinates
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            texts, spans_b, sids, labs = [], [], [], []
            for r in chunk:
                w = r.seq[s:s + WIN]
                sp = token_spans(tok, w, kind)
                if sp is None:
                    continue                      # unmappable tokenization: skip
                texts.append(to_model_input(w, kind))
                spans_b.append(sp); sids.append(r.sid); labs.append(r.label)
            if not texts:
                continue
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            out = model(**{k: v for k, v in enc.items()
                           if k in ("input_ids", "attention_mask")})
            h = hidden_of(out)
            h = h.float()
            mask = enc["attention_mask"].bool()
            for j, sp in enumerate(spans_b):
                n_body = len(sp)
                body = h[j, n_pre:n_pre + n_body]
                sel = [k for k, (a, b) in enumerate(sp) if a < hi and b > lo]
                if not sel:
                    continue
                local_out.append(body[sel].mean(0).half().cpu().numpy())
                valid = mask[j].sum().item()
                global_out.append(h[j, :valid].mean(0).half().cpu().numpy())
                keep_sid.append(sids[j]); keep_off.append(s); keep_lab.append(labs[j])
        print(f"  {model_name} offset {s} done  ({time.time()-t0:.0f}s)")

    if not keep_sid:
        raise RuntimeError(
            f"{model_name}: every sequence was skipped. The tokenizer's output "
            f"could not be mapped back to base coordinates — check CELL 4.")
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
    log.append(dict(model=model_name, n_rows=len(keep_sid), seconds=round(dt, 1),
                    peak_gpu_gb=round(mem, 2),
                    sec_per_1k=round(1000 * dt / max(len(keep_sid), 1), 2)))
    del model; gc.collect(); torch.cuda.empty_cache()
    return (np.stack(local_out), np.stack(global_out),
            np.array(keep_sid), np.array(keep_off), np.array(keep_lab))

def extract_cached(model_name, split_name, df, log):
    """Skip work that already finished. The key covers everything that would
    change the embeddings, so a config change invalidates the cache."""
    key = f"{model_name.replace('/', '_')}__{split_name}__n{len(df)}" \
          f"__win{WIN}__ro{READOUT[0]}-{READOUT[1]}__seed{SEED}.npz"
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        z = np.load(path)
        print(f"  cached: {path}  ({len(z['sid']):,} rows)")
        log.append(dict(model=model_name, split=split_name,
                        n_rows=int(len(z["sid"])), seconds=0.0,
                        peak_gpu_gb=0.0, sec_per_1k=0.0, cached=True))
        return z["local"], z["global"], z["sid"], z["off"], z["lab"]
    out = extract(model_name, df, log)
    log[-1]["split"] = split_name
    log[-1]["cached"] = False
    np.savez_compressed(path, local=out[0], **{"global": out[1]},
                        sid=out[2], off=out[3], lab=out[4])
    print(f"  saved: {path}")
    return out

TIMING = []
CACHE = {}
for mname in OK_MODELS:
    print(f"\n=== {mname} ===")
    CACHE[mname] = {}
    for split_name, d in [("train", train), ("dev", dev), ("test", test)]:
        print(f" {split_name}:")
        CACHE[mname][split_name] = extract_cached(mname, split_name, d, TIMING)
print("\n--- timing ---")
print(pd.DataFrame(TIMING).to_string(index=False))


# ============================================================
# CELL 6 — probe
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

def macro_mcc(y_true, y_pred, n_cls=3):
    """One-vs-rest MCC averaged over classes, as stated in the proposal."""
    vals = []
    for c in range(n_cls):
        vals.append(matthews_corrcoef((y_true == c).astype(int),
                                      (y_pred == c).astype(int)))
    return float(np.mean(vals))

def fit_probe(Xtr, ytr, sid_tr, Xdv, ydv, off_dv):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xdv_s = sc.transform(Xtr), sc.transform(Xdv)
    # equal weight per ORIGINAL sequence, not per window
    cnt = pd.Series(sid_tr).value_counts()
    w = (1.0 / cnt.reindex(sid_tr).values)
    best, best_score = None, -np.inf
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xtr_s, ytr, sample_weight=w)
        pred = clf.predict(Xdv_s)
        score = np.mean([macro_mcc(ydv[off_dv == s], pred[off_dv == s])
                         for s in SHIFTS])
        print(f"    C={C:<6} mean val macro-MCC across offsets = {score:.4f}")
        if score > best_score:
            best, best_score, best_C = clf, score, C
    print(f"    -> chose C={best_C}  (val macro-MCC {best_score:.4f})")
    return sc, best, best_C

PROBES = {}
for mname in OK_MODELS:
    for pool_idx, pool in [(0, "local"), (1, "global")]:
        Xtr, sid_tr, _, ytr = (CACHE[mname]["train"][pool_idx],
                               CACHE[mname]["train"][2],
                               CACHE[mname]["train"][3],
                               CACHE[mname]["train"][4])
        Xdv, _, off_dv, ydv = (CACHE[mname]["dev"][pool_idx],
                               CACHE[mname]["dev"][2],
                               CACHE[mname]["dev"][3],
                               CACHE[mname]["dev"][4])
        print(f"\n  probe: {mname} / {pool}")
        PROBES[(mname, pool)] = fit_probe(Xtr.astype(np.float32), ytr, sid_tr,
                                          Xdv.astype(np.float32), ydv, off_dv)


# ============================================================
# CELL 7 — per-offset correctness and the paired contrasts
# ============================================================
elig_sids = set(test.loc[test["eligible"], "sid"].values)

def correctness_table(mname, pool):
    sc, clf, _ = PROBES[(mname, pool)]
    X, _, sid, off, y = (CACHE[mname]["test"][0 if pool == "local" else 1],
                         None, CACHE[mname]["test"][2],
                         CACHE[mname]["test"][3], CACHE[mname]["test"][4])
    pred = clf.predict(sc.transform(X.astype(np.float32)))
    df = pd.DataFrame(dict(sid=sid, offset=off, label=y, pred=pred,
                           correct=(pred == y).astype(float)))
    # macro-MCC per offset on the FULL test set
    mcc = {s: macro_mcc(y[off == s], pred[off == s]) for s in SHIFTS}
    wide = df[df["sid"].isin(elig_sids)].pivot_table(
        index=["sid", "label"], columns="offset", values="correct")
    # keep the predicted class too: a 0/1 correctness table cannot say WHAT a
    # wrong prediction was, and the confusion at the splitting offset is what
    # distinguishes a boundary effect from a readout aliasing artifact
    pw = df[df["sid"].isin(elig_sids)].pivot_table(
        index=["sid", "label"], columns="offset", values="pred")
    return wide.dropna(), mcc, pw.dropna()

def contrasts(wide, label):
    sub = wide.xs(label, level="label")
    sp = SPLIT_OFFSET[label]; a, b = ADJACENT[label]
    d = sub[sp].values - 0.5 * (sub[a].values + sub[b].values)
    same_frame = sub[6].values - sub[0].values
    return d, same_frame, sub.index.values

RESULTS = {}
MCC_ROWS = []
for mname in OK_MODELS:
    for pool in ("local", "global"):
        wide, mcc, pw = correctness_table(mname, pool)
        RESULTS[(mname, pool)] = dict(wide=wide, mcc=mcc, pred=pw)
        # per-sequence correctness is the raw material for every later table;
        # saving it means the analysis can be rerun without touching the GPU
        wide.to_csv(os.path.join(RESULT_DIR,
                                 f"correctness__{mname}__{pool}.csv"))
        pw.to_csv(os.path.join(RESULT_DIR,
                               f"predictions__{mname}__{pool}.csv"))
        for s, v in mcc.items():
            MCC_ROWS.append(dict(model=mname, pool=pool, offset=s,
                                 macro_mcc=v))
        print(f"\n{mname} / {pool}")
        print("  macro-MCC per offset:",
              {s: round(v, 4) for s, v in mcc.items()})
        for lab, nm in [(0, "acceptor"), (1, "donor")]:
            if lab not in wide.index.get_level_values("label"):
                continue
            d, sf, _ = contrasts(wide, lab)
            base = wide.xs(lab, level="label")[list(ADJACENT[lab])].values.mean()
            print(f"  {nm}: n={len(d)}  mean d={d.mean():+.4f}  "
                  f"same-frame={sf.mean():+.4f}  adjacent-offset acc={base:.4f}")


# ============================================================
# CELL 8 — sequence-level clustered bootstrap
# ============================================================
def boot_ci(values, sids, B=B_BOOT, seed=SEED):
    """Resample ORIGINAL SEQUENCES, not windows."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sids)
    idx_by_sid = {s: np.where(sids == s)[0] for s in uniq}
    means = np.empty(B)
    for b in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx_by_sid[s] for s in draw])
        means[b] = values[take].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return values.mean(), lo, hi

print("\n=== primary contrasts, 95% CI (sequence-level bootstrap) ===")
summary = []
for mname in OK_MODELS:
    for pool in ("local", "global"):
        wide = RESULTS[(mname, pool)]["wide"]
        for lab, nm in [(0, "acceptor"), (1, "donor")]:
            if lab not in wide.index.get_level_values("label"):
                continue
            d, sf, sids = contrasts(wide, lab)
            m, lo, hi = boot_ci(d, sids)
            ms, los, his = boot_ci(sf, sids)
            verdict = ("below zero" if hi < 0 else
                       "above zero" if lo > 0 else "contains zero")
            summary.append(dict(model=mname, pool=pool, cls=nm, n=len(d),
                                d=round(m, 4), lo=round(lo, 4), hi=round(hi, 4),
                                verdict=verdict,
                                same_frame=round(ms, 4),
                                sf_lo=round(los, 4), sf_hi=round(his, 4)))
print(pd.DataFrame(summary).to_string(index=False))
pd.DataFrame(summary).to_csv(
    os.path.join(RESULT_DIR, "table_primary_contrasts.csv"), index=False)
pd.DataFrame(MCC_ROWS).to_csv(
    os.path.join(RESULT_DIR, "table_mcc_by_offset.csv"), index=False)

print("\n=== cross-model difference (Analysis II) ===")
cross = []
if not {"NT-v2", "DNABERT-2"} <= set(OK_MODELS):
    print("  skipped — needs both NT-v2 and DNABERT-2, and one did not load")
for pool in ("local", "global") if {"NT-v2", "DNABERT-2"} <= set(OK_MODELS) else ():
    for lab, nm in [(0, "acceptor"), (1, "donor")]:
        wa = RESULTS[("NT-v2", pool)]["wide"]
        wb = RESULTS[("DNABERT-2", pool)]["wide"]
        if lab not in wa.index.get_level_values("label"):
            continue
        da, _, sa = contrasts(wa, lab)
        db, _, sb = contrasts(wb, lab)
        common = np.intersect1d(sa, sb)
        da = da[np.isin(sa, common)]; db = db[np.isin(sb, common)]
        delta = da - db
        m, lo, hi = boot_ci(delta, common)
        cross.append(dict(pool=pool, cls=nm, n=len(common),
                          delta=round(m, 4), lo=round(lo, 4), hi=round(hi, 4),
                          verdict=("below zero" if hi < 0 else
                                   "above zero" if lo > 0 else "contains zero")))
print(pd.DataFrame(cross).to_string(index=False))
pd.DataFrame(cross).to_csv(
    os.path.join(RESULT_DIR, "table_cross_model.csv"), index=False)


# ============================================================
# CELL 9 — non-gLM baseline (sanity: is the task trivially easy?)
# ============================================================
MP = {"A": 0, "C": 1, "G": 2, "T": 3}

def onehot(seqs, lo, hi):
    X = np.zeros((len(seqs), (hi - lo) * 4), dtype=np.float32)
    for i, sq in enumerate(seqs):
        for j, ch in enumerate(sq[lo:hi]):
            if ch in MP:
                X[i, j * 4 + MP[ch]] = 1.0
    return X

base_rows = []

# (a) fixed ORIGINAL coordinates. Offset-invariant by construction, so this is
#     not a shift control — it is the task-difficulty reference: how well the
#     label is predicted from the raw bases around the site, with no model.
Xtr = onehot(train["seq"].values, *READOUT)
Xte = onehot(test["seq"].values, *READOUT)
sc_b = StandardScaler().fit(Xtr)
clf_b = LogisticRegression(C=1.0, max_iter=3000).fit(
    sc_b.transform(Xtr), train["label"].values)
pred_b = clf_b.predict(sc_b.transform(Xte))
mcc_b = macro_mcc(test["label"].values, pred_b)
acc_b = float((pred_b == test["label"].values).mean())
print(f"\nbaseline (a) one-hot of original [{READOUT[0]},{READOUT[1]}) — "
      f"no model, no tokenizer")
print(f"  macro-MCC {mcc_b:.4f}   accuracy {acc_b:.4f}")
base_rows.append(dict(baseline="onehot_original_coords", offset="all",
                      macro_mcc=round(mcc_b, 4), accuracy=round(acc_b, 4)))

# (b) fixed WINDOW coordinates: the same slice of each cropped window, so the
#     bases inside it shift with the offset exactly as the models' input does.
#     There is no tokenizer here, so any offset effect this shows is pure
#     positional/terminal-context confound — the one named in Limitations.
print(f"\nbaseline (b) one-hot of window [{READOUT[0]},{READOUT[1]}) — "
      f"shifts with the offset, still no tokenizer")
Xtr_b = np.concatenate([onehot([s[k:k + WIN] for s in train["seq"].values],
                               *READOUT) for k in SHIFTS])
ytr_b = np.tile(train["label"].values, len(SHIFTS))
sid_b = np.tile(train["sid"].values, len(SHIFTS))
sc_b2 = StandardScaler().fit(Xtr_b)
w_b = 1.0 / pd.Series(sid_b).map(pd.Series(sid_b).value_counts()).values
clf_b2 = LogisticRegression(C=1.0, max_iter=3000).fit(
    sc_b2.transform(Xtr_b), ytr_b, sample_weight=w_b)

corr, preds_b = {}, {}
for k in SHIFTS:
    Xk = onehot([s[k:k + WIN] for s in test["seq"].values], *READOUT)
    pk = clf_b2.predict(sc_b2.transform(Xk))
    preds_b[k] = pk
    corr[k] = (pk == test["label"].values).astype(float)
    m = macro_mcc(test["label"].values, pk)
    print(f"  offset {k}: macro-MCC {m:.4f}")
    base_rows.append(dict(baseline="onehot_window_coords", offset=k,
                          macro_mcc=round(m, 4),
                          accuracy=round(float(corr[k].mean()), 4)))

elig_arr = test["eligible"].values
for lab, nm in [(0, "acceptor"), (1, "donor")]:
    sel = elig_arr & (test["label"].values == lab)
    if sel.sum() == 0:
        continue
    sp = SPLIT_OFFSET[lab]; a, b = ADJACENT[lab]
    d_b = corr[sp][sel] - 0.5 * (corr[a][sel] + corr[b][sel])
    sids_b = test["sid"].values[sel]
    m, lo, hi = boot_ci(d_b, sids_b)
    verdict = ("below zero" if hi < 0 else
               "above zero" if lo > 0 else "contains zero")
    print(f"  {nm}: n={sel.sum()}  d={m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"-> {verdict}")
    base_rows.append(dict(baseline="onehot_window_coords_contrast", offset=sp,
                          cls=nm, n=int(sel.sum()), d=round(m, 4),
                          lo=round(lo, 4), hi=round(hi, 4), verdict=verdict))

pd.DataFrame(base_rows).to_csv(
    os.path.join(RESULT_DIR, "table_baselines.csv"), index=False)


# ============================================================
# CELL 9b — what IS the wrong answer at the splitting offset?
#
# A drop in accuracy says nothing about its cause. If a donor at offset 3 is
# misread specifically as an ACCEPTOR, that points at the two signals landing
# on the same readout index rather than at a token boundary: the donor GT sits
# at readout index 20-s and the acceptor AG at 18-s, so donor at s=3 and
# acceptor at s=1 coincide. The baseline indexes by WINDOW position and is
# exposed to that; the models pool over a window fixed in ORIGINAL coordinates
# and are not. Comparing their error composition tests that explanation.
# ============================================================
CLSNAME = {0: "acceptor", 1: "donor", 2: "non-splice"}

def error_mix(pred_at_split, pred_at_adj, truth, tag, cls_lab):
    """Composition of predictions at the splitting offset vs its neighbours,
    over sequences of one true class."""
    out = []
    for where, p in (("split offset", pred_at_split),
                     ("adjacent offsets", pred_at_adj)):
        n = len(p)
        row = dict(source=tag, cls=CLSNAME[cls_lab], where=where, n=int(n))
        for c in (0, 1, 2):
            row[f"pred_{CLSNAME[c]}"] = round(float((p == c).mean()), 4)
        out.append(row)
    return out

conf_rows = []
for lab in (0, 1):
    sp = SPLIT_OFFSET[lab]; a, b = ADJACENT[lab]
    sel = test["eligible"].values & (test["label"].values == lab)
    if sel.sum() == 0:
        continue
    conf_rows += error_mix(preds_b[sp][sel],
                           np.concatenate([preds_b[a][sel], preds_b[b][sel]]),
                           lab, "baseline(b) one-hot", lab)
    for mname in OK_MODELS:
        pw = RESULTS[(mname, "local")]["pred"]
        if lab not in pw.index.get_level_values("label"):
            continue
        sub = pw.xs(lab, level="label")
        conf_rows += error_mix(sub[sp].values,
                               np.concatenate([sub[a].values, sub[b].values]),
                               lab, f"{mname} local", lab)

conf = pd.DataFrame(conf_rows)
conf.to_csv(os.path.join(RESULT_DIR, "table_error_composition.csv"),
            index=False)
print("\n=== what the wrong answers are, at the splitting offset vs its "
      "neighbours ===")
print(conf.to_string(index=False))


# ============================================================
# CELL 10 — run summary
# ============================================================
tdf = pd.DataFrame(TIMING)
pilot_rows = tdf["n_rows"].sum()
full_rows = 36496 * len(SHIFTS)
print("\n================ SCOPING REPORT ================")
print(tdf.to_string(index=False))
print(f"\nforward passes this run : {pilot_rows:,}")
print(f"full train set would be : {full_rows:,} per model "
      f"({full_rows/max(pilot_rows,1):.1f}x this pilot)")
timed = [r for r in TIMING if not r.get("cached") and r["seconds"] > 0]
if not timed:
    print("  (everything came from cache — no fresh timing to extrapolate from)")
for mname in {r["model"] for r in timed}:
    rs = [r for r in timed if r["model"] == mname]
    rate = sum(r["seconds"] for r in rs) / sum(r["n_rows"] for r in rs)
    print(f"  {mname:<12} {rate*1000:6.1f} s / 1k passes   "
          f"-> full train {rate*full_rows/60:6.1f} min, "
          f"all three splits ~{rate*full_rows*1.5/60:6.1f} min")
print("\nCEILING CHECK — if accuracy at the adjacent offsets is already ~1.0,")
print("the paired contrast cannot move and the training-subset fallback applies.")
for (mname, pool), r in RESULTS.items():
    if pool != "local":
        continue
    w = r["wide"]
    for lab, nm in [(0, "acceptor"), (1, "donor")]:
        if lab in w.index.get_level_values("label"):
            acc = w.xs(lab, level="label")[list(ADJACENT[lab])].values.mean()
            mark = "  <-- CEILING RISK" if acc > 0.98 else ""
            print(f"  {mname:<12} {nm:<9} adjacent-offset accuracy {acc:.4f}{mark}")

import transformers
env = dict(transformers=transformers.__version__, torch=torch.__version__,
           device=(torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"),
           config_shims=SHIMS_APPLIED, models_loaded=OK_MODELS,
           models_failed=[m for m in MODELS if m not in OK_MODELS])
print("\n--- environment (for the reproducibility appendix) ---")
for k, v in env.items():
    print(f"  {k}: {v}")

env["n_train"] = int(len(train)); env["n_dev"] = int(len(dev))
env["n_test"] = int(len(test)); env["B_boot"] = B_BOOT
env["win"] = WIN; env["readout"] = list(READOUT); env["shifts"] = SHIFTS
json.dump(dict(timing=TIMING, summary=summary, cross=cross,
               baselines=base_rows, seed=SEED, env=env),
          open(os.path.join(RESULT_DIR, "run_report.json"), "w"), indent=2)
print(f"\nwrote {RESULT_DIR}/ :")
for f in sorted(os.listdir(RESULT_DIR)):
    print("   ", f)
