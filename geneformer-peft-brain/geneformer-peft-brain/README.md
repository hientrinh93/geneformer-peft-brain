# Geneformer + PEFT (QLoRA / DoRA / PiSSA / OLoRA) - Brain CellxGene

Fine-tuning Geneformer on brain single-cell data using Hugging Face PEFT.

## Features
- Full support for QLoRA, LoRA, DoRA, PiSSA, OLoRA
- Official Geneformer TranscriptomeTokenizer
- Gradient checkpointing + bf16
- Wandb logging
- Early stopping + detailed metrics (Accuracy, Macro F1, Weighted F1, Precision, Recall)
- Automatic test set evaluation

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python -m src.train
```

## Inference
```bash
python -m src.inference
```

## Project Structure
- `configs/` → configuration
- `src/` → all Python scripts
- `data/` → put your .h5ad files here
- `results/` → trained models and logs