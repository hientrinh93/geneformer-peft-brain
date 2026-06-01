import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_trainable_parameters(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Total parameters:     {total:,}\n")


def save_training_config(config: dict) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("configs/config.yaml", output_dir / "config_used.yaml")
    print(f"Config saved to {output_dir}/config_used.yaml")


def get_label_mapping_from_h5ad(adata_path: str, label_column: str) -> tuple[dict, dict]:
    """
    Build an ordered label <-> int mapping directly from the h5ad obs column.

    For categorical columns, the order follows adata.obs[col].cat.categories,
    which matches exactly what Geneformer's TranscriptomeTokenizer produces when
    it converts the column to integer codes.  Using a different order would cause
    the model to predict wrong class names during inference.

    Uses backed='r' (memory-mapped) so only the obs table is loaded — safe for
    multi-GB CellxGene files where loading the full expression matrix just to
    read cell type labels would be very slow.

    Returns:
        label2id: {"Excitatory neuron": 0, "Inhibitory neuron": 1, ...}
        id2label: {0: "Excitatory neuron", 1: "Inhibitory neuron", ...}
    """
    adata = sc.read_h5ad(adata_path, backed="r")
    try:
        col = adata.obs[label_column]
        if hasattr(col, "cat"):
            labels = list(col.cat.categories)
        else:
            labels = sorted(col.unique().tolist())
    finally:
        adata.file.close()

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


def save_per_class_metrics(
    true_labels: list,
    pred_labels: list,
    id2label: dict,
    output_dir: str,
) -> None:
    """
    Save per-class precision / recall / F1 as CSV and confusion matrix as .npy.
    Prints a summary table to stdout so results are visible in the training log.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    label_names = [id2label[i] for i in sorted(id2label.keys())]

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    report_path = Path(output_dir) / "test_per_class_metrics.csv"
    df.to_csv(report_path)

    cm = confusion_matrix(true_labels, pred_labels)
    np.save(Path(output_dir) / "test_confusion_matrix.npy", cm)
    np.save(Path(output_dir) / "test_confusion_matrix_labels.npy", np.array(label_names))

    print(f"\nPer-class metrics saved to {report_path}")
    print(
        classification_report(
            true_labels, pred_labels, target_names=label_names, zero_division=0
        )
    )
