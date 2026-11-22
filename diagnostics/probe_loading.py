"""
Model-loading diagnostics only. No data, no extraction, no probe.
Runs in ~2-3 minutes. Purpose: find a way to load NT and DNABERT-2.

    !python probe_loading.py
"""
import os, traceback, json
import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
import transformers

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("transformers", transformers.__version__, "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu")
try:
    import triton
    print("triton", triton.__version__)
except Exception as e:
    print("triton: not importable:", e)
print("=" * 70)

DEMO = "ACGT" * 98 + "AC"          # 394 bp


def show_config(ckpt):
    print(f"\n### config: {ckpt}")
    try:
        c = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    except Exception as e:
        print("   config load failed:", type(e).__name__, str(e)[:200])
        return None
    keys = ["model_type", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "vocab_size",
            "architectures", "auto_map", "add_bias_fnn", "emb_layer_norm_before",
            "token_dropout", "position_embedding_type", "use_flash_attn",
            "attn_implementation", "rope_theta", "is_decoder"]
    for k in keys:
        if hasattr(c, k):
            print(f"   {k:28} = {getattr(c, k)}")
    return c


def try_load(tag, fn):
    print(f"\n--- attempt: {tag}")
    try:
        m = fn()
        m = m.to(DEV).eval()
        n = sum(p.numel() for p in m.parameters())
        print(f"    LOADED  params={n/1e6:.1f}M")
        return m
    except Exception as e:
        msg = str(e).replace("\n", " ")[:260]
        print(f"    FAILED  {type(e).__name__}: {msg}")
        return None


def forward_check(tag, ckpt, model, kind):
    if model is None:
        return False
    try:
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        text = (" ".join(DEMO[i:i + 6] for i in range(len(DEMO) - 5))
                if kind == "overlap6" else DEMO)
        enc = tok([text], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**{k: v for k, v in enc.items()
                           if k in ("input_ids", "attention_mask")})
        h = out[0] if not hasattr(out, "last_hidden_state") else out.last_hidden_state
        print(f"    forward OK  hidden={tuple(h.shape)}")
        return True
    except Exception as e:
        print(f"    forward FAILED  {type(e).__name__}: {str(e)[:220]}")
        return False
    finally:
        del model
        torch.cuda.empty_cache()


WORKING = {}

# ===================================================================
# 1. Nucleotide Transformer — the primary model, must work
# ===================================================================
print("\n" + "=" * 70)
print("NUCLEOTIDE TRANSFORMER")
print("=" * 70)

NT_CANDIDATES = [
    # v2 family: gated FFN, currently mismatching the built-in ESM
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    # v1 family: plain ESM, SAME non-overlapping 6-mer tokenization,
    # so it preserves the experimental design if v2 cannot be loaded
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
]

for ckpt in NT_CANDIDATES:
    c = show_config(ckpt)
    if c is None:
        continue
    m = try_load("AutoModel + trust_remote_code", lambda:
                 AutoModel.from_pretrained(ckpt, trust_remote_code=True))
    if m is None:
        m = try_load("AutoModelForMaskedLM", lambda:
                     AutoModelForMaskedLM.from_pretrained(ckpt, trust_remote_code=True))
    if m is None:
        # if the checkpoint is gated and the class is not, say so explicitly
        inter = getattr(c, "intermediate_size", None)
        print(f"    note: config intermediate_size={inter}; a checkpoint weight of "
              f"2x that indicates a gated FFN the loaded class does not implement")
    if forward_check(ckpt, ckpt, m, "blocked"):
        WORKING["NT"] = ckpt
        # check the tokenizer is still blocked 6-mer, which the design requires
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        pieces = [t.lstrip("#").lstrip("_").lstrip("▁")
                  for t in tok.tokenize(DEMO)]
        lens = sorted({len(p) for p in pieces})
        print(f"    tokenizer: {len(pieces)} tokens, piece lengths {lens} "
              f"{'(blocked 6-mer OK)' if lens == [6] or lens[-1] == 6 else '(CHECK)'}")
        break

# ===================================================================
# 2. DNABERT-2 — Triton flash-attention fails on T4
# ===================================================================
print("\n" + "=" * 70)
print("DNABERT-2")
print("=" * 70)
D2 = "zhihan1996/DNABERT-2-117M"
show_config(D2)

def _plain():
    c = AutoConfig.from_pretrained(D2, trust_remote_code=True)
    return AutoModel.from_pretrained(D2, config=c, trust_remote_code=True)

def _no_flash():
    """Most forks of this repo gate the Triton path on a config flag."""
    c = AutoConfig.from_pretrained(D2, trust_remote_code=True)
    for flag in ("use_flash_attn", "use_triton", "flash_attn"):
        setattr(c, flag, False)
    return AutoModel.from_pretrained(D2, config=c, trust_remote_code=True)

def _no_remote():
    """Plain HF BertModel. Only valid if the weight names line up; the load
    report printed below says whether anything was left uninitialised."""
    return AutoModel.from_pretrained(D2, trust_remote_code=False)

for tag, fn in [("as-is", _plain),
                ("flash/triton flags off", _no_flash),
                ("built-in BertModel, no remote code", _no_remote)]:
    m = try_load(tag, fn)
    if forward_check(tag, D2, m, "bpe"):
        WORKING["DNABERT-2"] = tag
        break

# ===================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(json.dumps(WORKING, indent=2) if WORKING else "nothing loaded")
for need in ("NT", "DNABERT-2"):
    if need not in WORKING:
        print(f"  !! {need} still unresolved")
