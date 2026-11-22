# Alignment sensitivity to splice-signal splitting in genomic language models

Code for the paper *Auditing Alignment Sensitivity to Splice-Signal Splitting
in Genomic Language Models* (CSE4057, Hanyang University).

The experiment evaluates three frozen genomic encoders — Nucleotide
Transformer v2, DNABERT-2 and DNABERT — with linear probes on seven shifted
394-bp windows per GUE splice-site sequence, and compares them against two
nucleotide one-hot baselines.

---

## 1. Environment

The reported run used:

| | version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 4.57.6 |
| GPU | NVIDIA Tesla T4 (Google Colab) |

NumPy, pandas, scikit-learn and matplotlib are also required; their exact
versions were not recorded in the run report for the reported run.

```bash
pip install -U "transformers<5" einops scikit-learn pandas matplotlib
```

**Pin the major version only.** An exact pin such as `transformers==4.44.2`
fails on Python 3.13: its `tokenizers` dependency has no `cp313` wheel, the
Rust build fails, and pip can silently leave a 5.x release installed. The
checkpoints' remote code does not run under Transformers 5.x — NT-v2 raises
`'EsmConfig' object has no attribute 'is_decoder'` (or `rope_theta`) and
DNABERT-2 raises `'BertConfig' object has no attribute 'pad_token_id'`.

All checkpoints are loaded with `trust_remote_code=True`.

### Compatibility adjustments applied in the code

1. **`rope_theta`** is set to `10000.0` when absent from a model config.
2. **NT-v2** is loaded through `AutoModelForMaskedLM`, after which its `.esm`
   encoder is used. The checkpoint's `auto_map` has no `AutoModel` entry, so
   `AutoModel` falls back to the built-in ESM implementation, which lacks the
   gated FFN and fails with a `[4096, 512]` vs `[2048, 512]` shape mismatch.
   `diagnostics/probe_loading.py` reproduces the diagnosis.
3. **DNABERT-2** has its module-global `flash_attn_qkvpacked_func` set to
   `None`, which selects the PyTorch attention path instead of the Triton
   kernel. The Triton kernel raises a `CompilationError` on a T4.
   `diagnostics/dump_dnabert2.py` dumps the cached `bert_layers.py` so the
   branch condition can be inspected directly.

Bitwise equivalence between the two attention implementations was not
verified.

---

## 2. Data

The GUE reconstructed splice-site dataset is **not included here.** Obtain it
from the DNABERT-2 release (Zhou et al., 2024) and place the official splits
in the data directory as:

```
train.csv   36,496 rows
dev.csv      4,562 rows
test.csv     4,562 rows
```

Each file has one sequence column (400 bp) and one label column
(0 = acceptor, 1 = donor, 2 = non-splice). `run_full.py` detects the column
names; it does not assume a fixed header.

Partitions are used as published. They are **not** deduplicated — see the
paper's revision plan.

---

## 3. Configuration

Edit the config block at the top of `run_full.py` (and the matching paths in
`analyze.py`):

```python
DATA_DIR   = "."                              # holds train/dev/test.csv
DRIVE      = "/content/drive/MyDrive/nlp_hw2" # cache + results; None = ephemeral
RESULT_DIR = "results"
```

Pooled representations are cached under `DRIVE`, so a runtime restart does not
cost GPU hours. `analyze.py` reuses that cache.

> Cache filenames encode model, partition, sample count, window length,
> readout interval and seed, but not data contents or checkpoint revision, and
> `analyze.py` selects caches by filename and modification time. **Use a clean,
> run-specific cache directory.** A stale cache will be picked up silently.

---

## 4. Seeds

A single seed, **`20260101`**, is used throughout: Python, NumPy and PyTorch
RNGs, the stratified training subsample (15,000 sequences, 5,000 per class),
the bootstrap resampling, and the failure-case example sampling.

Bootstrap: `B = 2000` percentile resamples, drawn at the original-sequence
level within each eligible class.

A fixed seed is set, but deterministic GPU execution is **not** enforced, so
bitwise reproducibility across hardware or library versions is not claimed.
Probe fitting and all statistics run on CPU and are deterministic given the
cached representations.

---

## 5. Reproducing the paper

Run in this order. `analyze.py` requires the representation cache written by
`run_full.py`.

```bash
python run_full.py        # extraction, probes, contrasts, baselines   (~2 h on a T4)
python analyze.py         # flips, offset-probe ablation, failure cases, figures
python verify_tables.py   # checks the paper's tables against results/*.csv
python make_tables.py     # writes tables_acl.tex
python make_figures.py    # writes fig_mcc_by_offset.pdf, fig_contrasts.pdf
```

### Which command produces which table

Every number in the paper comes from a CSV in `results/`. `make_tables.py`
only formats them (see §6).

| Paper | Produced by | Result file | Rows to read |
|---|---|---|---|
| Table 1 — primary & same-frame contrasts | `run_full.py` | `table_primary_contrasts.csv` | `pool == "local"` |
| Table 2 — per-sequence flips | `analyze.py` | `table_flips.csv` | `pool == "local"` |
| Table 3 — cross-model differences | `run_full.py` | `table_cross_model.csv` | all |
| Table 4 — task performance | `run_full.py` | `table_baselines.csv`, `table_mcc_by_offset.csv` | `baseline == "onehot_original_coords"` / `"onehot_window_coords"`; `pool == "local"` |
| Table 5 — baseline (b) contrasts | `run_full.py` | `table_baselines.csv` | `baseline == "onehot_window_coords_contrast"` |
| Table 6 — donor error composition | `run_full.py` | `table_error_composition.csv` | `cls == "donor"` |
| Table 7 — offset-specific probes | `analyze.py` | `table_ablation_offset_probe.csv` | `pool == "local"`, column `delta_vs_shared` |
| Table 8 — failure cases | `analyze.py` | `examples_failures.txt` | seeded samples per failure category |
| Figure 1 — macro-MCC by offset | `run_full.py` → `make_figures.py` | `table_mcc_by_offset.csv` | all |
| Figure 2 — contrast forest plot | `run_full.py` → `make_figures.py` | `table_primary_contrasts.csv`, `table_baselines.csv` | `pool == "local"` |

Global-pooling results quoted in the text are the `pool == "global"` rows of
the same files.

`run_full.py` also writes `run_report.json` and, per model and pooling scheme,
`correctness__<model>__<pool>.csv` and `predictions__<model>__<pool>.csv` —
the per-sequence, per-offset correctness and predicted classes that all paired
contrasts are computed from.

### Other scripts

| | |
|---|---|
| `pilot_pipeline.py` | subsample pilot used to measure throughput before the main run |
| `diagnostics/probe_loading.py` | NT-v2 loading-path diagnosis |
| `diagnostics/dump_dnabert2.py` | dumps DNABERT-2's cached `bert_layers.py` |
| `audit/colab_split_check.py` | canonical-signal position detection |
| `audit/split_ratio_check.py` | per-tokenizer splitting-rate measurement |
| `audit/colab_shift_sweep.py` | splitting status across window offsets |

The full-training-split tokenizer audit reported in the paper's Method section
comes from the `audit/` scripts, run separately from the main pipeline.

---

## 6. A limitation of the table and figure scripts

`make_tables.py` and `make_figures.py` do **not** read `results/*.csv`. They
hold the run's numbers as Python literals, transcribed by hand from the result
CSVs of the 2026-11-20 run, and compute only derived quantities (per-offset
means, flip percentages) themselves.

`verify_tables.py` exists because of that. It parses the literals out of both
scripts without importing them, re-reads every result CSV, and compares value
by value — 218 comparisons — exiting non-zero on any mismatch:

```bash
$ python verify_tables.py --results results
compared 218 values against results/
all literals in make_tables.py and make_figures.py match the result CSVs.
```

Run it before trusting `tables_acl.tex` or either figure. If the literals and
the CSVs disagree, the CSVs are correct.

---

## 7. Running this from scratch

On a fresh machine (a new Colab notebook with a T4):

```bash
git clone <this repository>
cd <this repository>
pip install -U "transformers<5" einops scikit-learn pandas matplotlib
# place train.csv, dev.csv, test.csv in DATA_DIR (see §2)
# set DRIVE to an empty directory (see §3)
python run_full.py && python analyze.py && python verify_tables.py
```

Expected cost on a T4: 506,604 model–window instances, 168,868 per model,
about 1.92 hours of extraction. Per 1,000 windows: roughly 3.6 s (NT-v2),
8.2 s (DNABERT-2), 29.0 s (DNABERT). Peak PyTorch-allocated GPU memory:
about 0.50, 0.67 and 0.83 GB respectively.

The timings above come from the original non-cached execution. Once a cache
exists, `run_report.json` records cache hits and zero extraction time, so it
cannot be used to estimate the original cost.
