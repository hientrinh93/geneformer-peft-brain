"""
Temperature scaling for confidence calibration (Guo et al. 2017).

A fine-tuned classifier's softmax probabilities are usually NOT calibrated: a model that
outputs 0.99 confidence is rarely right 99% of the time. QLoRA/DoRA models in particular
tend to be over-confident. Temperature scaling fixes this with a single scalar T > 0:

    calibrated_probs = softmax(logits / T)

T > 1 softens over-confident predictions; T < 1 sharpens under-confident ones. T is fit on
the VALIDATION set (never train or test) by minimising negative log-likelihood. It does not
change which class is predicted (argmax is unchanged) — only the confidence numbers — so
accuracy/F1 are untouched while the `confidence` column becomes trustworthy.
"""

import json
from pathlib import Path

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """
    Find the temperature T that minimises NLL of softmax(logits / T) against labels.

    Optimises log_T (so T = exp(log_T) stays strictly positive) with LBFGS, which converges
    in a handful of iterations for this 1-parameter problem.
    """
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.tensor(np.asarray(labels), dtype=torch.long)

    log_T = torch.zeros(1, requires_grad=True)  # start at T = 1 (no scaling)
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / log_T.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def _ece_from_probs(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE given already-computed probabilities (shared by temperature and vector scaling)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == np.asarray(labels)).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def expected_calibration_error(
    logits: np.ndarray, labels: np.ndarray, T: float = 1.0, n_bins: int = 15
) -> float:
    """Expected Calibration Error after dividing logits by scalar T (lower is better)."""
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32) / T
    probs = torch.softmax(logits_t, dim=-1).numpy()
    return _ece_from_probs(probs, labels, n_bins)


def fit_vector_scaling(
    logits: np.ndarray, labels: np.ndarray, max_iter: int = 200
) -> tuple:
    """
    Vector scaling (Guo et al. 2017): calibrated_logits = w * logits + b, with a PER-CLASS
    scale vector w and bias vector b. More expressive than a single temperature — it can fix
    classes that are individually over- or under-confident (common in imbalanced brain data
    where rare types are systematically under-confident). Still does not change argmax order
    enough to matter in practice for accuracy, but recalibrates per-class confidence.

    Optimises w (init 1) and b (init 0) by LBFGS on NLL. Returns (w, b) as lists.
    """
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.tensor(np.asarray(labels), dtype=torch.long)
    n_classes = logits_t.shape[1]

    w = torch.ones(n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t * w + b, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach().tolist(), b.detach().tolist()


def fit_and_save_calibration(
    logits: np.ndarray, labels: np.ndarray, output_dir: str, method: str = "temperature"
) -> dict:
    """
    Fit the chosen calibration on validation logits, report ECE before/after, save to
    calibration.json. method: "temperature" (single scalar) | "vector" (per-class w,b).
    Inference reads calibration.json to produce calibrated confidence scores.
    """
    probs_before = torch.softmax(torch.tensor(np.asarray(logits), dtype=torch.float32), dim=-1).numpy()
    ece_before = _ece_from_probs(probs_before, labels)

    if method == "vector":
        w, b = fit_vector_scaling(logits, labels)
        calib = {"method": "vector", "w": w, "b": b}
    else:
        T = fit_temperature(logits, labels)
        calib = {"method": "temperature", "temperature": T}

    probs_after = apply_calibration(
        torch.tensor(np.asarray(logits), dtype=torch.float32), calib
    )
    probs_after = torch.softmax(probs_after, dim=-1).numpy()
    ece_after = _ece_from_probs(probs_after, labels)

    calib["ece_before"] = ece_before
    calib["ece_after"] = ece_after
    out_path = Path(output_dir) / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)

    detail = f"T={calib['temperature']:.4f}" if method != "vector" else "per-class w,b"
    print(
        f"Calibration ({method}): {detail} | "
        f"ECE {ece_before:.4f} -> {ece_after:.4f} (saved to {out_path})"
    )
    return calib


def load_calibration(output_dir: str) -> dict:
    """Load calibration.json; returns an identity calibration if none was saved."""
    path = Path(output_dir) / "calibration.json"
    if not path.exists():
        return {"method": "temperature", "temperature": 1.0}
    with open(path) as f:
        return json.load(f)


def apply_calibration(logits, calib: dict):
    """
    Apply a loaded calibration to a logits tensor, returning CALIBRATED LOGITS
    (caller still applies softmax). Supports both temperature and vector scaling.
    """
    if calib.get("method") == "vector":
        w = torch.tensor(calib["w"], dtype=logits.dtype, device=logits.device)
        b = torch.tensor(calib["b"], dtype=logits.dtype, device=logits.device)
        return logits * w + b
    T = float(calib.get("temperature", 1.0))
    return logits / T
