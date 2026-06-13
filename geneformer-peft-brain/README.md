# Geneformer PEFT — Brain CellxGene Cell-Type Classification

Fine-tuning the [Geneformer](https://huggingface.co/ctheodoris/Geneformer) single-cell foundation
model on brain single-cell RNA-seq data (CellxGene `.h5ad`) for **cell-type classification**, using
**parameter-efficient fine-tuning (PEFT)** — QLoRA, LoRA, DoRA, LoftQ, PiSSA, OLoRA, IA³, AdaLoRA,
and QDoRA. Designed for imbalanced, multi-study, multi-donor brain datasets.

---

## Table of contents
- [What this does](#what-this-does)
- [Quick start](#quick-start)
- [Environment setup](#environment-setup)
- [Data requirements](#data-requirements)
- [Configuration](#configuration)
- [PEFT methods](#peft-methods)
- [Pipeline details](#pipeline-details)
- [Outputs](#outputs)
- [Metrics](#metrics)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## What this does

Geneformer represents each cell as a **rank-ordered list of gene tokens** (genes sorted by
normalized expression), which removes library-size effects and transfers across studies. This
project freezes the pretrained model and trains small PEFT adapters (≈1–3 % of parameters) on top,
so a fine-tune fits comfortably on a 6 GB GPU.

The pipeline has three commands:

1. **`src.check_splits`** — verify the train/val/test split has no donor leakage.
2. **`src.train`** — tokenize, fine-tune with PEFT, evaluate, calibrate confidence, save the adapter.
3. **`src.inference`** — load the adapter and write predictions back into a test `.h5ad`.

---

## Quick start

Run from the project directory (the folder containing `src/` and `configs/`).

```bash
# 1. Verify splits (fast — reads only obs)
python -m src.check_splits

# 2. Train (tokenizes on first run, then fine-tunes; writes to results/)
python -m src.train

# 3. Inference on the test set → results/predictions_test.h5ad
python -m src.inference
```

Switch PEFT method or precision by editing **one line** in `configs/config.yaml`:

```yaml
peft:
  method: "qlora"   # qlora | qdora | lora | loftq | dora | pissa | olora | ia3 | adalora
```

> **GPU note (important).** `qlora`/`qdora` need a CUDA GPU + `bitsandbytes`. The HuggingFace repo
> root model is the large **316 M** Geneformer, which is too slow on a 6 GB GPU (~100 s/step).
> For ≤8 GB GPUs use the small **Geneformer-V1-10M** checkpoint and set `model_version: V1`
> (see [Configuration](#configuration)). On Turing GPUs (e.g. RTX 2060) set `bf16: false, fp16: true`.

---

## Environment setup

Tested on **Windows + Python 3.8 + NVIDIA RTX 2060 (6 GB)**.

```bash
# 1. PyTorch with CUDA — the default pip wheel is CPU-only and cannot use the GPU.
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# 2. Core dependencies
pip install -r requirements.txt
#    peft, datasets, accelerate, bitsandbytes, transformers, scanpy, anndata, loompy, wandb, ...

# 3. Geneformer (tokenizer + dictionaries). The plain `pip git+` install can fail on the
#    partial-clone filter, so clone fully then install locally:
git clone https://huggingface.co/ctheodoris/Geneformer
pip install ./Geneformer --no-build-isolation
```

If you are on Python < 3.10 (Geneformer declares `requires-python >=3.10`), see
[Troubleshooting](#troubleshooting) for the exact flags used here.

---

## Data requirements

Place AnnData files under `data/` and point `configs/config.yaml` at them:

```
data/
  cxg_train.h5ad
  cxg_validation.h5ad
  cxg_test.h5ad
```

| Requirement | Why |
|-------------|-----|
| `X` = **raw integer counts** (not log-normalized) | Geneformer ranks raw expression; log-counts give wrong rankings. The pipeline checks `uns['log1p']` and the value magnitude. |
| Ensembl gene IDs available | Geneformer's vocabulary is Ensembl-based. The pipeline writes `var['ensembl_id']` from `var.index` (or `data.ensembl_id_column`) automatically. |
| Total counts per cell | Written to `obs['n_counts']` automatically (from `X` row sums) if missing. |
| Cell-type labels in `obs` | Column set by `data.label_column` (default `cell_type`). |
| A donor column in `obs` | Column set by `data.donor_column` (default `donor_id`), used by `check_splits`. |

> **Donor-stratified splits matter.** Cells from one donor share genetic + batch signal. If a donor
> appears in both train and test, the model can memorize donor-specific patterns and test metrics
> become inflated — they no longer reflect generalization to *new* donors. Always run
> `python -m src.check_splits` first; it exits non-zero on leakage. If your dataset has very few
> donors, a clean split may be impossible — treat metrics as pipeline checks only.

---

## Configuration

All settings live in `configs/config.yaml`. Key knobs:

```yaml
# Model — for low-VRAM GPUs point this at a local Geneformer-V1-10M checkpoint and set V1.
# (The path below is machine-specific; change it on another machine, or use "ctheodoris/Geneformer".)
model_name: ".../Geneformer/Geneformer-V1-10M"
model_version: "V1"          # V1 -> gc30M dict, len 2048; V2 -> gc104M dict, len 4096

data:
  train_path / val_path / test_path: "data/cxg_*.h5ad"
  label_column: "cell_type"
  donor_column: "donor_id"
  ensembl_id_column: null     # null = use var.index
  min_cells_per_class: 50     # drop cell types with fewer than this many cells
  tokenizer_nproc: 1          # 1 on Windows (tokenizer multiprocessing can hang)

peft:
  method: "qlora"
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: "all-linear"

training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4      # effective batch = 32
  learning_rate: 2.0e-4               # NOTE: 2.0e-4, not 2e-4 — YAML parses 2e-4 as a string
  num_train_epochs: 4
  max_steps: -1                       # >0 caps steps (quick smoke test); -1 = full epochs
  bf16: false                         # Turing GPUs: false; Ampere+: true
  fp16: true                          # Turing GPUs: true
  gradient_checkpointing: true
  class_weighted_loss: true
  class_weight_power: 0.5             # soften balanced weights (0=uniform, 1=raw balanced)
  label_smoothing_factor: 0.1
  neftune_noise_alpha: 5
  calibrate_temperature: true         # fit confidence calibration on val after training
  calibration_method: "temperature"  # "temperature" or "vector" (per-class)
  hierarchical_loss: false            # add coarse-level loss (needs coarse_map_path)
  check_splits_before_train: true     # abort if donor leakage is detected
  eval_steps: 200
  save_steps: 200
```

### Quick smoke test
To validate the whole workflow in minutes (not a real result), set:
`num_train_epochs: 1`, `max_steps: 150`, `eval_steps: 50`, `save_steps: 50` and use the
**V1-10M** model. This runs train → eval → test → calibration → save in roughly 10 minutes on a
6 GB GPU.

---

## PEFT methods

| Method | Idea | Best for |
|--------|------|----------|
| `qlora` | 4-bit NF4 base + LoRA | lowest VRAM (≤16 GB) |
| `qdora` | DoRA on a 4-bit base | best accuracy/VRAM trade-off |
| `lora` | low-rank adapters | standard baseline |
| `loftq` | SVD-aware quantized init | few-epoch budgets |
| `dora` | weight-decomposed LoRA | closest to full fine-tune |
| `pissa` / `olora` | SVD / QR initialization | fast convergence, small data |
| `ia3` | learned scale vectors | fewest parameters |
| `adalora` | prunes rank during training | concentrate budget on key layers |

Flags (`use_dora`, `init_lora_weights`, quantization, AdaLoRA schedule) are derived automatically
from `method` — you only change that one line.

---

## Pipeline details

**Tokenization** uses Geneformer's `TranscriptomeTokenizer` and is cached in `data/tokenized/`
(set `retokenize: true` to force). Each split is first *prepared* (adds `ensembl_id`/`n_counts`,
drops rare classes) into `data/filtered/`. The string cell-type labels the tokenizer emits are
mapped to integer ids via the saved `label_map.json`.

**Training** uses the HuggingFace `Trainer` with early stopping on `f1_macro`, class-weighted loss
(softened), optional label smoothing / NEFTune / hierarchical loss, cosine LR schedule, and
gradient clipping. The best checkpoint is reloaded and evaluated on the test set.

**Calibration** fits a temperature (or per-class vector) on the validation logits after training and
saves it to `calibration.json`. Inference divides logits by it before softmax, so the `confidence`
column is calibrated. This does not change predictions.

---

## Outputs

After training, `results/geneformer_peft_brain/` contains:

| File | Description |
|------|-------------|
| `final_model/` | PEFT adapter weights (load with `PeftModel.from_pretrained`) |
| `label_map.json` | `{label2id, id2label}` mapping |
| `calibration.json` | fitted temperature/vector + ECE before/after |
| `config_used.yaml` | exact config used (reproducibility) |
| `test_per_class_metrics.csv` | precision/recall/F1 per cell type |
| `test_confusion_matrix.npy` (+ `_labels.npy`) | confusion matrix |

Inference writes `results/predictions_test.h5ad` with new `obs` columns
`predicted_label`, `predicted_label_id`, `confidence`.

---

## Metrics

| Metric | Meaning |
|--------|---------|
| `balanced_accuracy` | mean per-class recall — best headline for imbalanced data |
| `f1_macro` | F1 averaged equally over classes — model-selection metric |
| `f1_weighted` | F1 weighted by frequency — comparable to published numbers |
| `accuracy` | raw accuracy — misleading under imbalance |

`f1_macro` is used for early stopping because `f1_weighted`/`accuracy` are dominated by the majority
class and can look high while rare cell types are missed.

---

## Project structure

```
configs/
  config.yaml                # all hyperparameters and paths
  coarse_map.example.json    # template for hierarchical loss
src/
  check_splits.py            # donor-leakage verification
  data_processing.py         # prepare + tokenize + cache + label remap
  model_peft.py              # PEFT config factory + quantization
  train.py                   # training entry point
  inference.py               # batch inference → predictions h5ad
  calibration.py             # temperature / vector scaling
  utils.py                   # config, label mapping, metrics, coarse mapping
data/        # .h5ad inputs + tokenized cache (gitignored)
results/     # model, metrics, predictions (gitignored)
```

---

## Troubleshooting

**`torch.cuda.is_available()` is False but you have a GPU** — the default torch wheel is CPU-only.
Install the CUDA build: `pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121`.

**Training is extremely slow (~100 s/step)** — you are on the 316 M model. Use Geneformer-V1-10M and
`model_version: V1`; it is ~30× smaller and fits a 6 GB GPU.

**`bitsandbytes` / quantization error** — modern `bitsandbytes` (≥0.43) has Windows CUDA wheels.
If 4-bit is unavailable, use a non-quantized method (`lora`/`dora`/`loftq`).

**Installing Geneformer on Python 3.8** — Geneformer declares `requires-python >=3.10`. To install
on 3.8: full `git clone`, then
`pip install ./Geneformer --no-build-isolation --ignore-requires-python --no-deps`, plus
`pip install loompy`. Pin `setuptools==69.5.1` if metadata generation fails on the license field.
If `import geneformer` errors on `tdigest`/`ray`/`optuna` (only needed for embedding extraction),
trim `geneformer/__init__.py` to import only `tokenizer` and `collator_for_classification`.

**`AssertionError: 'ensembl_id' / 'n_counts' column missing`** — newer Geneformer needs these in
`var`/`obs`; the pipeline adds them automatically during preparation.

**`ValueError` with `learning_rate`** — write `2.0e-4`, not `2e-4` (YAML parses the latter as a string).

**Tokenization hangs on Windows** — set `tokenizer_nproc: 1`.
