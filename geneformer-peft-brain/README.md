# Geneformer + PEFT (QLoRA / DoRA / PiSSA / OLoRA) - Brain CellxGene

Fine-tuning Geneformer on brain single-cell data using Hugging Face PEFT.

## Features
- Support QLoRA, LoRA, DoRA, PiSSA, OLoRA
- Official Geneformer tokenizer
- Gradient checkpointing + bf16
- Wandb logging
- Early stopping + compute_metrics (Accuracy, F1, Precision, Recall)
- Test set evaluation

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
