# Geneformer PEFT — Brain CellxGene Cell Type Classification

Fine-tuning [Geneformer](https://huggingface.co/ctheodoris/Geneformer) on brain single-cell RNA-seq data for cell type classification using parameter-efficient fine-tuning (PEFT). Designed for CellxGene `.h5ad` datasets with imbalanced, multi-study brain data.

---

## Background

**Geneformer** [1] is a transformer foundation model (BERT architecture) pretrained on 29.9 million single-cell transcriptomes. Instead of raw counts, it represents each cell as a **ranked list of gene tokens** — genes are sorted by expression level and converted to Ensembl-based integer IDs. This rank-based encoding removes library-size bias and generalizes across studies and sequencing protocols.

Fine-tuning Geneformer with full weight updates is expensive (~110 M parameters). **PEFT** methods inject small trainable adapters while keeping pretrained weights frozen, reducing GPU memory by 60–90% with minimal accuracy loss.

---

## Features

| Feature | Details |
|---------|---------|
| **7 PEFT methods** | `qlora`, `loftq`, `lora`, `dora`, `pissa`, `olora`, `ia3` — switched by one config line |
| **4-bit QLoRA** [2] | NF4 quantization applied at model load time (correct pipeline) |
| **LoftQ** [3] | SVD-based adapter init that minimizes quantization error; better than QLoRA when training budget is limited |
| **Class-weighted loss** | Sklearn balanced weights for skewed brain data (excitatory neurons often ~60–70% of cells) |
| **Label smoothing** [9] | Reduces overconfidence on noisy cross-study CellxGene annotations |
| **NEFTune** [8] | Adds uniform noise to embeddings during training — improves generalization at zero accuracy cost |
| **Cosine LR + warmup** | Cosine decay schedule with configurable warmup ratio |
| **Tokenization caching** | Skip re-tokenization when `.dataset` files already exist (`retokenize: false`) |
| **Raw count validation** | Checks h5ad input for log-normalization via h5py without loading the full matrix |
| **Stable label mapping** | Label↔ID map derived from `obs` categorical order (matches tokenizer), saved as JSON for inference |
| **Per-class metrics** | Classification report (F1, precision, recall per cell type) as CSV + confusion matrix as `.npy` |
| **Balanced accuracy** | Reported as main metric alongside macro/weighted F1 |
| **WandB logging** | Optional experiment tracking |
| **Reproducibility** | `seed` set globally (Python/NumPy/Torch/Trainer/DataLoader) and config copied to output dir |

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (bf16 + gradient checkpointing); CPU-only is possible for inference only
- For QLoRA/LoftQ: GPU with `bitsandbytes` support (NVIDIA; Linux preferred — Windows requires a special `bitsandbytes-windows` build)

### Installation

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---------|---------|
| `geneformer` | Tokenizer + `BertForSequenceClassification` (from HuggingFace Hub) |
| `peft` | LoRA, QLoRA, LoftQ, DoRA, IA³ adapters |
| `bitsandbytes` | 4-bit NF4 quantization for QLoRA/LoftQ |
| `transformers` | HuggingFace Trainer, TrainingArguments, EarlyStoppingCallback |
| `accelerate` | Mixed-precision and multi-GPU backend |
| `scanpy` / `anndata` | Reading `.h5ad` files |
| `datasets` | HuggingFace Dataset format for tokenized data |

---

## Data Preparation

### File format

Place your AnnData files in `data/` and update paths in `configs/config.yaml`:

```
data/
  brain_cellxgene_train.h5ad
  brain_cellxgene_validation.h5ad
  brain_cellxgene_test.h5ad
```

### Requirements for the h5ad files

| Requirement | Why |
|-------------|-----|
| `X` must be **raw integer counts** (not log-normalized) | Geneformer tokenizes by ranking raw expression; log-counts produce wrong gene rankings |
| `var_names` must be **Ensembl gene IDs** | Geneformer's token vocabulary is built on Ensembl IDs |
| Cell type labels must be in `obs` as a column | Configure the column name with `data.label_column` in config |
| Labels should be consistent across train/val/test | The label→int mapping is fixed from the train file's categorical order |

The pipeline validates that `X` is not log-normalized:
- Checks `uns` for a `"log1p"` key
- Checks if the max expression value in a sample row is suspiciously small (< 20)

> **Important for valid metrics:** Split data by `donor_id` or `sample_id`, not randomly by cell. Random cell-level splits inflate test F1 by 10–25 points because cells from the same donor appear in both train and test (label leakage through donor effects) [14].

### Tokenization

Geneformer's `TranscriptomeTokenizer` converts each cell to a ranked gene sequence. This step:
1. Copies the h5ad to a temp directory
2. Runs tokenization with `nproc` parallel workers
3. Saves a HuggingFace `Dataset` to `data/tokenized/{train,val,test}.dataset`

Set `retokenize: false` to reuse cached datasets on subsequent runs (saves 5–30 min depending on dataset size).

---

## Configuration

All hyperparameters live in `configs/config.yaml`. Annotated reference:

```yaml
model_name: "ctheodoris/Geneformer"   # HuggingFace model ID
output_dir: "./results/geneformer_peft_brain"
seed: 42                               # global seed (Python/NumPy/Torch/Trainer/DataLoader)

data:
  train_path: "data/brain_cellxgene_train.h5ad"
  val_path:   "data/brain_cellxgene_validation.h5ad"
  test_path:  "data/brain_cellxgene_test.h5ad"
  label_column: "cell_type"            # obs column used as classification target
  tokenizer_nproc: 4                   # parallel workers for TranscriptomeTokenizer
  retokenize: false                    # true = force re-tokenization even if cache exists

peft:
  method: "qlora"   # lora | qlora | loftq | dora | pissa | olora | ia3
  r: 16             # LoRA rank — higher = more parameters, slower convergence
  lora_alpha: 32    # scaling; effective scale = alpha/r (or alpha/sqrt(r) with rsLoRA)
  lora_dropout: 0.05
  target_modules: "all-linear"  # adapt every linear layer; ["query","value"] for minimal footprint
  loftq_iter: 1     # LoftQ only: SVD+quantize alternating rounds (1=fast, 4-8=more accurate)
  use_rslora: false # rsLoRA [7]: alpha/sqrt(r) scaling — stabilizes high-rank adapters
  layers_to_transform: null  # null = all layers; [6,7,8,9,10,11] = top 6 of 12 only
  layers_pattern: "layer"

training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4    # effective batch size = 8 × 4 = 32
  learning_rate: 2e-4
  num_train_epochs: 4
  bf16: true                        # bfloat16 mixed precision
  gradient_checkpointing: true      # trade compute for memory (~30% slower, ~40% less VRAM)
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  max_grad_norm: 1.0                # gradient clipping
  warmup_ratio: 0.05                # 5% of steps for linear LR warm-up
  lr_scheduler_type: "cosine"       # cosine decay after warm-up
  class_weighted_loss: true         # balanced sklearn weights to counter class imbalance
  label_smoothing_factor: 0.1       # soften one-hot targets [9]; 0.0 = off
  neftune_noise_alpha: 5            # NEFTune embedding noise [8]; null = off
  group_by_length: true             # pack same-length sequences per batch (1.5-2x throughput)
  save_steps: 200
  eval_steps: 200
  logging_steps: 50

wandb:
  enabled: true
  project: "geneformer-peft-brain"
  run_name: "qlora-brain-run-1"
```

---

## PEFT Methods

### How LoRA works [1]

LoRA injects two small matrices **A** (rank × hidden) and **B** (hidden × rank) alongside each frozen weight W. The forward pass computes `W·x + (B·A)·x·(alpha/r)`. Only A and B are trained — typically 0.1–1% of total parameters.

### Method comparison

| Method | VRAM | Accuracy | Best use case | Paper |
|--------|------|----------|---------------|-------|
| `qlora` | Lowest | Good | GPU ≤ 16 GB; fast experiments | [2] |
| `loftq` | Low | Better than QLoRA | GPU ≤ 24 GB; limited training epochs | [3] |
| `lora` | Medium | Good baseline | GPU ≥ 24 GB; standard reference | [1] |
| `dora` | Medium | Best LoRA variant | Closest behavior to full fine-tuning | [4] |
| `pissa` | Medium | Good; fast convergence | Small datasets, few epochs | [5] |
| `olora` | Medium | Similar to PiSSA | Faster init than PiSSA | [6] |
| `ia3` | Lowest | Moderate | Very small datasets; fewest params | [13] |

### Method details

**QLoRA** [2] — Quantizes the base model to 4-bit NF4 (Normal Float 4) at load time. Adapter weights stay in bf16. Reduces VRAM by ~4× versus full precision. Requires `quantization_config` to be passed at `from_pretrained`.

**LoftQ** [3] — Like QLoRA but initializes A and B via SVD to minimize `||W - Q(W) - B·A||`. Does **not** quantize at load time; quantization happens inside `get_peft_model`. Better starting point than QLoRA's random init, which matters most when training with few epochs.

**DoRA** [4] — Decomposes W into a magnitude vector and direction matrix, applies LoRA only to the direction. Empirically matches full fine-tuning behavior more closely than standard LoRA.

**PiSSA** [5] — Initializes B and A from the principal singular vectors of W (top-r SVD). Starts from the most informative weight subspace → faster convergence than random init.

**OLoRA** [6] — QR decomposition of W; Q becomes the frozen part, R becomes the adapter. Similar motivation to PiSSA, cheaper initialization.

**rsLoRA** [7] — Not a separate method but a scaling fix for high-rank LoRA. Changes the alpha/r scaling to alpha/sqrt(r), which stabilizes gradient flow at r=32+. Enable with `use_rslora: true`.

**IA³** [13] — Injects learned scale vectors into keys, values, and feed-forward layers. No matrix multiplication — fewest parameters of any method. Best for very small datasets.

### Switching methods

Change only `peft.method` in config — all flags (`use_dora`, `init_lora_weights`, `loftq_config`) are auto-derived:

```yaml
peft:
  method: "loftq"   # ← change this line only
```

---

## Training

```bash
python -m src.train
```

### Training pipeline

1. Label map built from `train.h5ad` `obs` categorical order → saved as `label_map.json`
2. All three splits tokenized (or loaded from cache if `retokenize: false`)
3. Base model loaded from HuggingFace Hub (with 4-bit quantization if QLoRA)
4. PEFT adapter applied via `get_peft_model`
5. Class weights computed from train label distribution (if `class_weighted_loss: true`)
6. HuggingFace Trainer runs with early stopping on `f1_macro` (patience = 6 eval cycles)
7. Best checkpoint loaded at end → evaluated on test set
8. Per-class metrics and confusion matrix saved

### Class-weighted loss [10]

Brain CellxGene data typically has 60–70% excitatory neurons. Without reweighting, the model optimizes for majority classes and ignores rare interneuron subtypes. Sklearn's `compute_class_weight("balanced")` is used by default. The weights are stored in a full-length vector indexed by label code to handle classes that appear in val/test but not in train.

### Label smoothing [9]

CellxGene annotations are harmonized across studies — some labels may be inconsistent or coarse. Label smoothing (0.1) prevents the model from becoming overconfident on potentially noisy targets, which also improves calibration.

### NEFTune [8]

Adds uniform random noise to embedding vectors during the forward pass (training only). Improves robustness and generalization with no accuracy cost. Controlled by `neftune_noise_alpha` (0 or null to disable).

### Early stopping behavior

- Metric: `f1_macro` — weights all cell types equally regardless of frequency
- Patience: 6 evaluation cycles without ≥ 0.01% improvement
- `save_total_limit: 7` ensures the best checkpoint is never deleted before it can be loaded

### Why f1_macro over f1_weighted?

`f1_weighted` is dominated by excitatory neurons and can score 0.90+ while completely failing on rare interneuron subtypes. `f1_macro` weights every class equally, exposing poor rare-class performance.

---

## Output Files

After training, `results/geneformer_peft_brain/` contains:

| File | Description |
|------|-------------|
| `final_model/` | Saved PEFT adapter weights (load with `PeftModel.from_pretrained`) |
| `label_map.json` | `{"label2id": {...}, "id2label": {...}}` mapping for inference |
| `config_used.yaml` | Exact config used — for reproducibility |
| `test_per_class_metrics.csv` | F1, precision, recall, support per cell type |
| `test_confusion_matrix.npy` | Confusion matrix array (rows = true, cols = predicted) |
| `test_confusion_matrix_labels.npy` | Cell type name array matching matrix axis order |

### Load confusion matrix in Python

```python
import numpy as np
cm = np.load("results/geneformer_peft_brain/test_confusion_matrix.npy")
labels = np.load("results/geneformer_peft_brain/test_confusion_matrix_labels.npy", allow_pickle=True)
```

---

## Inference

```bash
python -m src.inference
```

Loads the trained adapter from `final_model/`, reads `label_map.json` (no re-tokenization of train set needed), runs batch inference on the test set, and writes three new columns into the test `.h5ad`:

| Column | Type | Description |
|--------|------|-------------|
| `predicted_label_id` | int | Predicted class index |
| `predicted_label` | str | Predicted cell type name (from `label_map.json`) |
| `confidence` | float | Softmax probability of the top prediction (uncalibrated — see [11]) |

Output file: `results/predictions_test.h5ad`

---

## Metrics Reference

| Metric | Definition | Notes |
|--------|-----------|-------|
| `accuracy` | Fraction of cells correctly classified | Misleading when imbalanced |
| `balanced_accuracy` | Mean per-class recall | Best single headline metric for brain data |
| `f1_macro` | Mean F1 across all classes equally | Main training/selection metric |
| `f1_weighted` | F1 weighted by class frequency | Comparable to published benchmarks |
| `precision` | Weighted average precision | — |
| `recall` | Weighted average recall | — |

---

## Project Structure

```
configs/
  config.yaml              # all hyperparameters and data paths
src/
  train.py                 # training entry point
  inference.py             # batch inference on test set
  model_peft.py            # PEFT method setup (LoRA/QLoRA/LoftQ/DoRA/IA³/...)
  data_processing.py       # tokenization pipeline, caching, raw count validation
  utils.py                 # config loading, label mapping, metrics saving
data/                      # place .h5ad files here (not committed)
results/                   # model outputs, metrics, predictions (not committed)
```

---

## Troubleshooting

**`ValueError: log-normalized data detected`**
→ Your h5ad `X` contains log-counts. Geneformer requires raw counts. Run `sc.pp.normalize_total` + `sc.pp.log1p` only after training/tokenization.

**`bitsandbytes` not found / quantization error on Windows**
→ QLoRA requires a special Windows build. Either install `bitsandbytes-windows` or switch to `method: lora` or `method: loftq` (LoftQ does not need bitsandbytes at load time).

**CUDA out of memory**
→ Reduce `per_device_train_batch_size` to 4 or 2. Increase `gradient_accumulation_steps` proportionally to maintain effective batch size. Ensure `gradient_checkpointing: true`.

**Tokenization hangs or is slow**
→ Set `tokenizer_nproc: 1` on Windows (multiprocessing limitations). After the first run, set `retokenize: false` to use the cached datasets.

**`KeyError: 'length'`**
→ Set `group_by_length: false`. Some Geneformer tokenizer versions do not write a `length` column.

**Test F1 much higher than expected**
→ Check that train/val/test are split by `donor_id`, not randomly by cell. Random cell splits inflate F1 by 10–25 points [14].

---

## References

> Links below are from training knowledge (cutoff Aug 2025). Verify at `arxiv.org/abs/<ID>` before citing.

**PEFT Methods**

[1] Hu, E.J. et al. **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022. https://arxiv.org/abs/2106.09685

[2] Dettmers, T. et al. **QLoRA: Efficient Finetuning of Quantized LLMs.** NeurIPS 2023. https://arxiv.org/abs/2305.14314

[3] Liu, Y. et al. **LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models.** ICLR 2024. https://arxiv.org/abs/2310.08659

[4] Liu, S.H. et al. **DoRA: Weight-Decomposed Low-Rank Adaptation.** ICML 2024. https://arxiv.org/abs/2402.09353

[5] Meng, F. et al. **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.** 2024. https://arxiv.org/abs/2404.02948

[6] Büyükacar, A., Doğan, G. **OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models.** 2024. https://arxiv.org/abs/2406.01775

[7] Kalajdzievski, D. **A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.** 2023. https://arxiv.org/abs/2312.03732

**Training Techniques**

[8] Jain, N. et al. **NEFTune: Noisy Embeddings Improve Instruction Finetuning.** 2023. https://arxiv.org/abs/2310.05914

[9] Müller, R., Kornblith, S., Hinton, G. **When Does Label Smoothing Help?** NeurIPS 2019. https://arxiv.org/abs/1906.02629

[10] Cui, Y. et al. **Class-Balanced Loss Based on Effective Number of Samples.** CVPR 2019. https://arxiv.org/abs/1901.05555

[11] Guo, C. et al. **On Calibration of Modern Neural Networks.** ICML 2017. https://arxiv.org/abs/1706.04599

[12] Zhang, Q. et al. **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (AdaLoRA).** ICLR 2023. https://arxiv.org/abs/2303.10512

[13] Liu, H. et al. **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning (IA³).** NeurIPS 2022. https://arxiv.org/abs/2205.05638

**Foundation Models**

[14] Theodoris, C.V. et al. **Transfer learning enables predictions in network biology (Geneformer).** *Nature* 608, 616–624 (2023). https://doi.org/10.1038/s41586-023-06139-9

[15] Cui, H. et al. **scGPT: toward building a foundation model for single-cell multi-omics using generative AI.** *Nature Methods* (2024). https://doi.org/10.1038/s41592-024-02201-0

[16] Hao, M. et al. **Large Scale Foundation Model on Single-cell Transcriptomics (scFoundation).** 2024. https://arxiv.org/abs/2310.07497

**Evaluation**

[17] Luecken, M.D. et al. **Benchmarking atlas-level data integration in single-cell genomics.** *Nature Methods* (2022). https://doi.org/10.1038/s41592-021-01336-8
