# Lazy annotations so PEP 585 generics like tuple[dict, dict] don't crash on Python 3.8
# (annotations become strings and are never evaluated at runtime).
from __future__ import annotations

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


def build_coarse_mapping(label2id: dict, coarse_map_path: str):
    """
    Build the fine->coarse structures needed for hierarchical loss.

    coarse_map_path is a JSON file mapping each fine cell-type NAME to a coarse-category
    NAME, e.g. {"L2/3 IT": "Excitatory", "Pvalb": "Inhibitory", "Astro": "Non-neuronal"}.

    Returns (fine_to_coarse, agg_matrix, coarse2id):
      fine_to_coarse: LongTensor [n_fine]   — coarse id for each fine id
      agg_matrix:     FloatTensor [n_fine, n_coarse] — one-hot group membership; multiplying
                      fine softmax probs by this matrix marginalises them into coarse probs
      coarse2id:      dict coarse_name -> coarse id

    Raises ValueError listing any fine labels missing from the map, so a partial map can't
    silently send classes into a wrong/implicit group.
    """
    import torch

    with open(coarse_map_path, "r", encoding="utf-8") as f:
        fine_to_coarse_name = json.load(f)

    n_fine = len(label2id)
    # Order fine ids 0..n_fine-1; id2label inverse for name lookup
    id2name = {i: name for name, i in label2id.items()}

    missing = [id2name[i] for i in range(n_fine) if id2name[i] not in fine_to_coarse_name]
    if missing:
        raise ValueError(
            f"coarse_map at {coarse_map_path} is missing {len(missing)} fine label(s): "
            f"{missing}. Every fine cell type must map to a coarse category."
        )

    # Assign coarse ids in first-appearance order over the fine id sequence (stable/deterministic)
    coarse2id = {}
    fine_to_coarse = []
    for i in range(n_fine):
        cname = fine_to_coarse_name[id2name[i]]
        if cname not in coarse2id:
            coarse2id[cname] = len(coarse2id)
        fine_to_coarse.append(coarse2id[cname])

    n_coarse = len(coarse2id)
    agg = torch.zeros(n_fine, n_coarse, dtype=torch.float32)
    for fi, ci in enumerate(fine_to_coarse):
        agg[fi, ci] = 1.0

    fine_to_coarse_t = torch.tensor(fine_to_coarse, dtype=torch.long)
    print(f"Hierarchical loss: {n_fine} fine classes -> {n_coarse} coarse groups")
    return fine_to_coarse_t, agg, coarse2id


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

    # Pass the full label set explicitly. After rare-class filtering, the test split may
    # contain only a subset of the trained classes; without `labels`, sklearn infers the
    # class count from the data and errors when it doesn't match the 11 target_names.
    label_ids = sorted(id2label.keys())
    label_names = [id2label[i] for i in label_ids]

    report = classification_report(
        true_labels,
        pred_labels,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    report_path = Path(output_dir) / "test_per_class_metrics.csv"
    df.to_csv(report_path)

    cm = confusion_matrix(true_labels, pred_labels, labels=label_ids)
    np.save(Path(output_dir) / "test_confusion_matrix.npy", cm)
    np.save(Path(output_dir) / "test_confusion_matrix_labels.npy", np.array(label_names))

    print(f"\nPer-class metrics saved to {report_path}")
    print(
        classification_report(
            true_labels, pred_labels, labels=label_ids,
            target_names=label_names, zero_division=0,
        )
    )
