from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from geneformer import DataCollatorForCellClassification, BertForSequenceClassification
from src.model_peft import prepare_model
from src.data_processing import load_and_tokenize_data
from src.utils import load_config, print_trainable_parameters, save_training_config

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

def main():
    config = load_config()
    save_training_config(config)

    print("Loading and tokenizing datasets...")
    train_dataset, val_dataset, test_dataset = load_and_tokenize_data(config)

    # Auto detect num_labels
    label_col = "label"
    num_labels = len(train_dataset.dataset.unique(label_col))
    print(f"Detected {num_labels} classes")

    model = BertForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    print("Applying PEFT...")
    model = prepare_model(model, config)
    print_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 0),
        dataloader_pin_memory=config["training"].get("dataloader_pin_memory", False),

        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],

        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        report_to="wandb" if config.get("wandb", {}).get("enabled", False) else "none",
        run_name=config.get("wandb", {}).get("run_name"),
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForCellClassification(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    if test_dataset is not None:
        print("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        print("Test results:", test_results)

    trainer.save_model(f"{config['output_dir']}/final_model")
    print("Training completed!")


if __name__ == "__main__":
    main()
