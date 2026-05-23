import torch
from pathlib import Path
import json
from src.utils import load_config
from src.data_processing import load_and_tokenize_data
from peft import PeftModel
from geneformer import BertForSequenceClassification, DataCollatorForCellClassification
import scanpy as sc

def load_trained_model(model_path: str, config, num_labels: int):
    base_model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=num_labels)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model

def main():
    config = load_config()
    model_path = Path(config["output_dir"]) / "final_model"

    # Get label mapping from train data
    train_dataset, _, _ = load_and_tokenize_data(config)
    label_col = "label"
    unique_labels = train_dataset.dataset.unique(label_col)
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)

    model = load_trained_model(str(model_path), config, num_labels)
    print("Model loaded successfully!")

    # Load and tokenize test data
    _, _, test_dataset = load_and_tokenize_data(config)

    data_collator = DataCollatorForCellClassification()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    confidences = []

    print("Running inference on test set...")
    with torch.no_grad():
        for i in range(0, len(test_dataset), 8):
            batch = data_collator([test_dataset[j] for j in range(i, min(i + 8, len(test_dataset)))])
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            confs = probs.max(dim=-1).values.cpu().numpy()
            
            predictions.extend(pred_ids.tolist())
            confidences.extend(confs.tolist())

    # Load original test adata to save predictions
    test_adata = sc.read_h5ad(config["data"]["test_path"])
    test_adata.obs["predicted_label_id"] = predictions
    test_adata.obs["predicted_label"] = [id_to_label.get(p, "Unknown") for p in predictions]
    test_adata.obs["confidence"] = confidences

    output_file = "results/predictions_test.h5ad"
    test_adata.write_h5ad(output_file)
    print(f"✅ Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
