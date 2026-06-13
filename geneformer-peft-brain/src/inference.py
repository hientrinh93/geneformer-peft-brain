import json
import pickle
from pathlib import Path

import scanpy as sc
import torch
from datasets import load_from_disk
from transformers import BertForSequenceClassification
from geneformer import (
    DataCollatorForCellClassification,
    TOKEN_DICTIONARY_FILE,
    TOKEN_DICTIONARY_FILE_30M,
)
from peft import PeftModel

from src.data_processing import (
    GeneformerDataset,
    _remap_labels_to_ids,
    check_raw_counts,
    tokenize_with_official_tokenizer,
)
from src.calibration import apply_calibration, load_calibration
from src.model_peft import get_bnb_config, get_compute_dtype, needs_quantization
from src.utils import load_config


def load_trained_model(model_path: str, config: dict, num_labels: int):
    """
    Load base Geneformer and apply the saved PEFT adapter.

    For QLoRA models, the base must be loaded with the same 4-bit quantization
    config used during training — otherwise the adapter weights (trained on top
    of a quantized base) are applied to a full-precision base, silently changing
    the numerical baseline and producing different predictions.
    """
    load_kwargs = {"num_labels": num_labels}
    if needs_quantization(config):
        load_kwargs["quantization_config"] = get_bnb_config(get_compute_dtype(config))
        print("Loading base model with 4-bit NF4 quantization (matching training conditions)...")

    base_model = BertForSequenceClassification.from_pretrained(
        config["model_name"], **load_kwargs
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model


def main():
    config = load_config()
    model_path = Path(config["output_dir"]) / "final_model"

    # Load label map saved during training — guarantees the same int<->string mapping
    # without re-tokenizing the training set
    label_map_path = Path(config["output_dir"]) / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Label map not found at {label_map_path}. "
            "Run training first (train.py saves this file automatically)."
        )
    with open(label_map_path) as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = label_map["label2id"]
    num_labels = len(id2label)

    model = load_trained_model(str(model_path), config, num_labels)
    print(f"Loaded model from {model_path}")

    # Load tokenized test data from cache; tokenize only if the cache is missing
    token_dir = Path("./data/tokenized")
    test_dataset_path = token_dir / "test.dataset"

    if not test_dataset_path.exists():
        print("Tokenized test data not found, tokenizing now...")
        token_dir.mkdir(parents=True, exist_ok=True)
        # Validate raw counts before tokenizing to catch normalized data early
        check_raw_counts(config["data"]["test_path"])
        tokenize_with_official_tokenizer(
            config["data"]["test_path"],
            str(token_dir),
            "test",
            config["data"]["label_column"],
            nproc=config["data"].get("tokenizer_nproc", 4),
            # Must match the version used at train time or token IDs won't align with the model
            model_version=config.get("model_version", "V1"),
        )

    test_hf = load_from_disk(str(test_dataset_path))
    # Map the tokenizer's string 'label' to int ids (the collator builds int tensors and
    # chokes on strings). This also drops any cell whose class isn't in label2id — keeping
    # the dataset aligned with the prepared h5ad loaded below.
    test_hf = _remap_labels_to_ids(test_hf, label2id)
    test_dataset = GeneformerDataset(test_hf)

    # Carry obs/var metadata alongside predictions. Use the PREPARED test file (rare classes
    # dropped, same cells as the tokenized dataset) so prediction count matches obs rows.
    prepared_test = Path("./data/filtered/test_prepared.h5ad")
    adata_path = str(prepared_test) if prepared_test.exists() else config["data"]["test_path"]
    test_adata = sc.read_h5ad(adata_path)
    if test_adata.n_obs != len(test_dataset):
        print(
            f"WARNING: test h5ad has {test_adata.n_obs} cells but tokenized set has "
            f"{len(test_dataset)} — prediction columns may not align."
        )

    tok_dict_file = (
        TOKEN_DICTIONARY_FILE_30M if config.get("model_version", "V1") == "V1"
        else TOKEN_DICTIONARY_FILE
    )
    with open(tok_dict_file, "rb") as f:
        token_dictionary = pickle.load(f)
    data_collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = config["training"].get("per_device_train_batch_size", 8)
    predictions, confidences = [], []

    # Calibration fitted on the validation set during training (calibration.json):
    # either a single temperature or per-class vector scaling. Applied to logits before
    # softmax to yield calibrated confidence scores. Falls back to identity if no file.
    calib = load_calibration(config["output_dir"])
    print(f"Applying calibration ({calib.get('method', 'temperature')}) to confidence scores")

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_items = [
                test_dataset[j]
                for j in range(i, min(i + batch_size, len(test_dataset)))
            ]
            batch = data_collator(batch_items)
            # Only feed tensors the model's forward accepts. The collator also returns
            # 'labels' and 'length'; passing 'length' raises TypeError (the Trainer drops
            # such columns automatically, but this manual loop must filter them itself).
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }

            logits = model(**inputs).logits
            probs = torch.softmax(apply_calibration(logits, calib), dim=-1)

            predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            confidences.extend(probs.max(dim=-1).values.cpu().numpy())

    test_adata.obs["predicted_label_id"] = predictions
    test_adata.obs["predicted_label"] = [id2label.get(p, "Unknown") for p in predictions]
    # Note: confidence scores are uncalibrated softmax probabilities — QLoRA models tend
    # to be overconfident. Apply temperature scaling on val set for calibrated probabilities.
    test_adata.obs["confidence"] = confidences

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    output_file = out_dir / "predictions_test.h5ad"
    test_adata.write_h5ad(output_file)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
