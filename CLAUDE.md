# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Geneformer is installed from HuggingFace Hub)
pip install -r requirements.txt

# Run training (reads from configs/config.yaml)
python -m src.train

# Run inference on test set using a trained model
python -m src.inference
```

No test suite exists — validation is done during training via HuggingFace Trainer metrics.

## Architecture

This project fine-tunes the **Geneformer** foundational model (a BERT-based transformer pretrained on single-cell RNA-seq data) for cell type classification using parameter-efficient fine-tuning (PEFT).

### Data flow

1. **Input**: AnnData `.h5ad` files with gene expression matrices and cell type labels in `obs`
2. **Tokenization** (`src/data_processing.py`): Geneformer's official `TranscriptomeTokenizer` converts gene expression profiles into ranked integer token sequences; files are copied to a temp dir before tokenization
3. **Model** (`src/model_peft.py`): Loads `ctheodoris/Geneformer` (`BertForSequenceClassification`) from HuggingFace, then wraps with a PEFT adapter; QLoRA applies 4-bit NF4 quantization before wrapping
4. **Training** (`src/train.py`): HuggingFace `Trainer` with early stopping on F1-weighted; auto-detects number of classes from training data labels; logs to WandB
5. **Inference** (`src/inference.py`): Loads saved PEFT weights onto base model, runs batch inference, writes predicted labels and confidence scores back into the `.h5ad` file as new `obs` columns

### PEFT methods

Configured via `peft.method` in `configs/config.yaml`. Supported values:

| Value | Description |
|-------|-------------|
| `qlora` | 4-bit quantization + LoRA (default; most memory-efficient) |
| `lora` | Standard LoRA |
| `dora` | DoRA (weight-decomposed LoRA) |
| `pissa` | PiSSA (principal singular values) |
| `olora` | OLoRA (orthogonal initialization) |

### Key configuration knobs (`configs/config.yaml`)

- `data.label_col`: the `.obs` column name used as the classification label
- `data.train/val/test_path`: paths to `.h5ad` files
- `peft.r`, `peft.lora_alpha`, `peft.lora_dropout`: LoRA hyperparameters
- `training.num_train_epochs`, `training.learning_rate`, `training.per_device_train_batch_size`
- `wandb.enabled`: toggle WandB logging

### Output

Results land in `results/` (configured by `output_dir`). `src/utils.py:save_training_config()` copies the used `config.yaml` there for reproducibility. Inference adds `predicted_label_id`, `predicted_label`, and `confidence` columns to the test `.h5ad`.
