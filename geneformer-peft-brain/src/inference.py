import json
from pathlib import Path

import scanpy as sc
import torch
from datasets import load_from_disk
from geneformer import BertForSequenceClassification, DataCollatorForCellClassification
from peft import PeftModel

from src.data_processing import (
    GeneformerDataset,
    check_raw_counts,
    tokenize_with_official_tokenizer,
)
from src.model_peft import get_bnb_config, needs_quantization
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
        load_kwargs["quantization_config"] = get_bnb_config()
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
        )

    test_hf = load_from_disk(str(test_dataset_path))
    test_dataset = GeneformerDataset(test_hf)

    # Keep the original h5ad to carry all obs/var metadata alongside predictions
    test_adata = sc.read_h5ad(config["data"]["test_path"])

    data_collator = DataCollatorForCellClassification()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = config["training"].get("per_device_train_batch_size", 8)
    predictions, confidences = [], []

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_items = [
                test_dataset[j]
                for j in range(i, min(i + batch_size, len(test_dataset)))
            ]
            batch = data_collator(batch_items)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

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
