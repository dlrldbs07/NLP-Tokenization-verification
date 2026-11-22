"""
Find DNABERT-2's cached remote code and show how it picks its attention path.
Fast, no GPU work. Also tries CPU, where the Triton kernel should be bypassed.

    !python dump_dnabert2.py
"""
import os, glob, re, torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

CKPT = "zhihan1996/DNABERT-2-117M"

# make sure the remote code is on disk
AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
try:
    AutoModel.from_pretrained(CKPT, trust_remote_code=True)
except Exception:
    pass

roots = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/**/DNABERT-2-117M*"),
    recursive=True) + glob.glob(os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/**/*DNABERT*"),
    recursive=True)
roots = sorted({r for r in roots if os.path.isdir(r)})
print("cached module dirs:")
for r in roots:
    print("  ", r, os.listdir(r))

path = None
for r in roots:
    cand = os.path.join(r, "bert_layers.py")
    if os.path.exists(cand):
        path = cand
        break
if path is None:
    for cand in glob.glob(os.path.expanduser(
            "~/.cache/huggingface/modules/**/bert_layers.py"), recursive=True):
        path = cand
        break

print("\nbert_layers.py:", path)
if not path:
    raise SystemExit("not found — paste the 'cached module dirs' listing above")

src = open(path).read()
print("lines:", len(src.splitlines()))

print("\n--- every line mentioning flash / triton / attn impl ---")
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r"flash|triton|unpad|FLASH|IMPL", line):
        print(f"{i:5d}: {line.rstrip()[:150]}")

print("\n--- the self-attention class, in full ---")
m = re.search(r"class BertUnpadSelfAttention.*?(?=\nclass )", src, re.S)
if not m:
    m = re.search(r"class \w*SelfAttention.*?(?=\nclass )", src, re.S)
if m:
    body = m.group(0).splitlines()
    for line in body[:110]:
        print("   ", line.rstrip()[:150])
else:
    print("   (no SelfAttention class matched — paste the grep above instead)")

print("\n--- does it run on CPU? (Triton needs CUDA, so this may bypass it) ---")
try:
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModel.from_pretrained(CKPT, trust_remote_code=True).eval()
    demo = "ACGT" * 98 + "AC"
    enc = tok([demo], return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items()
                       if k in ("input_ids", "attention_mask")})
    h = out[0] if not hasattr(out, "last_hidden_state") else out.last_hidden_state
    print(f"    CPU forward OK  hidden={tuple(h.shape)}")
    import time
    t0 = time.time()
    with torch.no_grad():
        for _ in range(4):
            model(**{k: v for k, v in enc.items()
                     if k in ("input_ids", "attention_mask")})
    print(f"    CPU speed: {(time.time()-t0)/4*1000:.0f} ms per sequence "
          f"(batch of 1)")
except Exception as e:
    print(f"    CPU forward FAILED  {type(e).__name__}: {str(e)[:200]}")
