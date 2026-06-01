import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from geneformer import BertForSequenceClassification, DataCollatorForCellClassification
from src.data_processing import load_and_tokenize_data
from src.model_peft import get_bnb_config, needs_quantization, prepare_model
from src.utils import (
    get_label_mapping_from_h5ad,
    load_config,
    save_per_class_metrics,
    save_training_config,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        # balanced_accuracy = mean per-class recall; standard headline metric in scRNA-seq papers
        # unlike f1_weighted, it weights every class equally regardless of cell count
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


def build_weighted_trainer_class(class_weights: torch.Tensor):
    """
    Factory returning a Trainer subclass with class-weighted cross-entropy loss.

    label_smoothing_factor is explicitly threaded through from self.args so it
    applies even when compute_loss is overridden.  The default Trainer applies
    smoothing inside its own compute_loss; overriding that method without
    forwarding the arg silently drops label smoothing.
    """
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = class_weights.to(outputs.logits.device)
            loss = F.cross_entropy(
                outputs.logits,
                labels,
                weight=weights,
                label_smoothing=self.args.label_smoothing_factor,
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def main():
    config = load_config()

    # set_seed covers Python/NumPy/Torch; seed and data_seed in TrainingArguments
    # cover the Trainer's internal dataloader shuffle and evaluation sampling
    set_seed(config.get("seed", 42))

    if config.get("wandb", {}).get("enabled", False):
        os.environ["WANDB_PROJECT"] = config["wandb"]["project"]

    save_training_config(config)

    label2id, id2label = get_label_mapping_from_h5ad(
        config["data"]["train_path"], config["data"]["label_column"]
    )
    num_labels = len(label2id)
    print(f"Detected {num_labels} classes")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(
            {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            f, indent=2,
        )
    print(f"Label map saved to {label_map_path}")

    print("Loading and tokenizing datasets...")
    train_dataset, val_dataset, test_dataset = load_and_tokenize_data(config)

    load_kwargs = dict(
        num_labels=num_labels,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    if needs_quantization(config):
        load_kwargs["quantization_config"] = get_bnb_config()
        print("Loading model with 4-bit NF4 quantization (QLoRA)...")

    model = BertForSequenceClassification.from_pretrained(config["model_name"], **load_kwargs)

    print(f"Applying PEFT method: {config['peft']['method']}")
    model = prepare_model(model, config)
    # PEFT's built-in method reports accurate trainable counts for quantized layers
    model.print_trainable_parameters()

    TrainerClass = Trainer
    if config["training"].get("class_weighted_loss", False):
        raw_labels = np.array(train_dataset.dataset["label"])
        present_classes = np.unique(raw_labels)
        weights_present = compute_class_weight("balanced", classes=present_classes, y=raw_labels)

        # Build a full num_labels-length weight vector indexed by label code.
        # If a class exists in val/test but not in train, compute_class_weight returns
        # a shorter array and F.cross_entropy would silently index out of bounds.
        full_weights = np.ones(num_labels, dtype=np.float32)
        for cls, wt in zip(present_classes, weights_present):
            full_weights[int(cls)] = wt
        class_weights = torch.tensor(full_weights, dtype=torch.float32)

        TrainerClass = build_weighted_trainer_class(class_weights)
        print(
            f"Class-weighted loss enabled — {len(present_classes)}/{num_labels} classes "
            f"present in train split"
        )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        warmup_ratio=config["training"].get("warmup_ratio", 0.05),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 0),
        dataloader_pin_memory=config["training"].get("dataloader_pin_memory", False),
        # label_smoothing threads through to WeightedTrainer.compute_loss via self.args
        label_smoothing_factor=config["training"].get("label_smoothing_factor", 0.0),
        # NEFTune: uniform noise on input embeddings during training (Jain et al. 2023)
        neftune_noise_alpha=config["training"].get("neftune_noise_alpha", None),
        # group_by_length: pack similarly-lengthed sequences per batch to cut padding waste
        group_by_length=config["training"].get("group_by_length", False),
        length_column_name="length",
        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        load_best_model_at_end=True,
        # f1_macro weights every class equally — correct for imbalanced brain data
        # f1_weighted is dominated by excitatory neurons and hides poor rare-class performance
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        label_names=["labels"],
        # Pass seed explicitly so Trainer's dataloader shuffle matches config seed
        seed=config.get("seed", 42),
        data_seed=config.get("seed", 42),
        report_to="wandb" if config.get("wandb", {}).get("enabled", False) else "none",
        run_name=config.get("wandb", {}).get("run_name"),
        # save_total_limit must be >= early_stopping_patience + 1 so the best
        # checkpoint is never deleted before load_best_model_at_end can load it
        save_total_limit=7,
    )

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForCellClassification(),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=6,
            # require at least 0.01% improvement to reset patience counter;
            # prevents premature stopping on noisy eval fluctuations
            early_stopping_threshold=1e-4,
        )],
    )

    print("Starting training...")
    trainer.train()

    if test_dataset is not None:
        print("Evaluating on test set...")
        # trainer.predict() returns raw logits for per-class breakdown;
        # trainer.evaluate() only returns aggregated metrics
        test_output = trainer.predict(test_dataset)
        pred_labels = test_output.predictions.argmax(axis=-1)
        true_labels = test_output.label_ids
        print("Test metrics:", test_output.metrics)
        save_per_class_metrics(true_labels, pred_labels, id2label, config["output_dir"])

    trainer.save_model(f"{config['output_dir']}/final_model")
    print("Training complete.")


if __name__ == "__main__":
    main()
