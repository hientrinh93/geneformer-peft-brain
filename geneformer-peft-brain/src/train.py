import json
import math
import os
import pickle
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
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from transformers import BertForSequenceClassification
from geneformer import (
    DataCollatorForCellClassification,
    TOKEN_DICTIONARY_FILE,
    TOKEN_DICTIONARY_FILE_30M,
)
from src.calibration import fit_and_save_calibration
from src.check_splits import check_donor_leakage
from src.data_processing import load_and_tokenize_data
from src.model_peft import get_bnb_config, get_compute_dtype, needs_quantization, prepare_model
from src.utils import (
    build_coarse_mapping,
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


def build_custom_trainer_class(
    class_weights=None,
    fine_to_coarse=None,
    agg_matrix=None,
    coarse_head=None,
    hier_lambda: float = 0.0,
):
    """
    Factory returning a Trainer subclass with a custom loss supporting two optional terms:

    1. Class-weighted cross-entropy (class_weights not None) — upweights rare cell types.
    2. Hierarchical loss (fine_to_coarse not None) — adds lambda * coarse-level CE so that
       confusing cell types across DIFFERENT broad categories is penalised more than confusing
       siblings within one category. Two ways to produce coarse logits:
         - marginalise (coarse_head is None): p_coarse = p_fine @ agg_matrix. Zero extra
           params; coarse prediction is fully determined by the fine head.
         - learned head (coarse_head given): a separate Linear(hidden_size -> n_coarse) reads
           the CLS representation directly. More capacity — the coarse task can shape the shared
           encoder features independently of the fine head, which often regularises better.

    label_smoothing_factor is threaded through from self.args so it still applies even though
    compute_loss is overridden (the default Trainer applies smoothing in its own compute_loss).
    """
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            # The learned coarse head needs the encoder's hidden states; request them only then
            # to avoid the extra memory when marginalising.
            need_hidden = coarse_head is not None
            outputs = (
                model(**inputs, output_hidden_states=True) if need_hidden else model(**inputs)
            )
            logits = outputs.logits

            weights = class_weights.to(logits.device) if class_weights is not None else None
            loss = F.cross_entropy(
                logits,
                labels,
                weight=weights,
                label_smoothing=self.args.label_smoothing_factor,
            )

            if fine_to_coarse is not None:
                coarse_target = fine_to_coarse.to(logits.device)[labels]
                if coarse_head is not None:
                    # CLS token of the last hidden layer is Bert's sequence representation
                    pooled = outputs.hidden_states[-1][:, 0]
                    coarse_logits = coarse_head(pooled)
                    coarse_loss = F.cross_entropy(coarse_logits, coarse_target)
                else:
                    p_coarse = F.softmax(logits, dim=-1) @ agg_matrix.to(logits.device)
                    # epsilon before log avoids -inf when a coarse prob underflows to 0
                    coarse_loss = F.nll_loss(torch.log(p_coarse + 1e-12), coarse_target)
                loss = loss + hier_lambda * coarse_loss

            return (loss, outputs) if return_outputs else loss

    return CustomTrainer


class AdaLoRACallback(TrainerCallback):
    """
    Drives AdaLoRA's dynamic rank reallocation during training.

    AdaLoRA must call model.update_and_allocate(global_step) once per optimizer step,
    AFTER gradients are computed but BEFORE they are zeroed — it reads .grad to estimate
    each singular value's importance, then prunes the budget accordingly.

    on_pre_optimizer_step fires at exactly that moment and once per optimizer step (so it
    respects gradient_accumulation_steps).  It requires transformers >= 4.42; on older
    versions this hook is never called and AdaLoRA silently degrades to a fixed-rank
    adapter — the assertion below makes that failure loud instead of silent.
    """

    def __init__(self, model):
        self.model = model
        self._fired = False

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._fired = True
        self.model.update_and_allocate(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        assert self._fired, (
            "AdaLoRACallback.on_pre_optimizer_step never fired — your transformers "
            "version likely predates 4.42. AdaLoRA rank reallocation did NOT run. "
            "Upgrade transformers or switch peft.method to 'lora'."
        )


def main():
    config = load_config()

    # set_seed covers Python/NumPy/Torch; seed and data_seed in TrainingArguments
    # cover the Trainer's internal dataloader shuffle and evaluation sampling
    set_seed(config.get("seed", 42))

    if config.get("wandb", {}).get("enabled", False):
        os.environ["WANDB_PROJECT"] = config["wandb"]["project"]

    save_training_config(config)

    # Pre-train guard: refuse to train on donor-leaking splits (metrics would be invalid).
    # check_donor_leakage returns n_leaks (>0 = leakage) or -1 (donor column missing).
    if config["training"].get("check_splits_before_train", True):
        n_leaks = check_donor_leakage(config)
        if n_leaks > 0:
            raise SystemExit(
                f"Aborting: {n_leaks} donor(s) leak across splits — fix the split or set "
                "training.check_splits_before_train: false to override (not recommended)."
            )
        if n_leaks < 0:
            print(
                "WARNING: could not verify donor splits (donor column missing). "
                "Proceeding, but metrics may be invalid if splits share donors."
            )

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
    train_dataset, val_dataset, test_dataset = load_and_tokenize_data(config, label2id)

    # AdaLoRA needs the total number of OPTIMIZER steps up front so its budget scheduler
    # knows the horizon for tinit/tfinal/deltaT. Compute it from the train set size here
    # (before prepare_model builds the AdaLoraConfig) and inject it into the peft config.
    is_adalora = config["peft"]["method"] == "adalora"
    if is_adalora:
        eff_batch = (
            config["training"]["per_device_train_batch_size"]
            * config["training"]["gradient_accumulation_steps"]
        )
        steps_per_epoch = math.ceil(len(train_dataset) / eff_batch)
        total_step = steps_per_epoch * config["training"]["num_train_epochs"]
        config["peft"]["adalora_total_step"] = total_step
        print(f"AdaLoRA: computed total_step = {total_step} ({steps_per_epoch} steps/epoch)")

    load_kwargs = dict(
        num_labels=num_labels,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    if needs_quantization(config):
        load_kwargs["quantization_config"] = get_bnb_config(get_compute_dtype(config))
        print("Loading model with 4-bit NF4 quantization (QLoRA)...")

    model = BertForSequenceClassification.from_pretrained(config["model_name"], **load_kwargs)
    # Capture hidden size before PEFT wrapping for the optional learned coarse head
    hidden_size = model.config.hidden_size

    print(f"Applying PEFT method: {config['peft']['method']}")
    model = prepare_model(model, config)
    # PEFT's built-in method reports accurate trainable counts for quantized layers
    model.print_trainable_parameters()

    # Custom-loss terms (class weighting, hierarchical loss) both override compute_loss and
    # pop labels, which would drop AdaLoRA's built-in orthogonal-regularisation term (it is
    # added inside AdaLoraModel.forward only when the model computes the loss itself).
    # So both terms are mutually exclusive with AdaLoRA — AdaLoRA uses its own regularised loss.
    want_weighted = config["training"].get("class_weighted_loss", False)
    want_hier = config["training"].get("hierarchical_loss", False)
    if is_adalora and (want_weighted or want_hier):
        print(
            "NOTE: class_weighted_loss / hierarchical_loss are ignored for AdaLoRA — its "
            "orthogonal regularisation requires the model's built-in loss."
        )
        want_weighted = want_hier = False

    TrainerClass = Trainer
    class_weights = None
    agg_matrix = fine_to_coarse = coarse_head = None
    hier_lambda = 0.0

    if want_weighted:
        raw_labels = np.array(train_dataset.dataset["label"])
        present_classes = np.unique(raw_labels)
        weights_present = compute_class_weight("balanced", classes=present_classes, y=raw_labels)

        # Soften raw "balanced" weights with a power < 1 to avoid extreme penalisation
        # of majority classes.  Raw balanced weights = n_total / (n_classes * n_i), which
        # can give excitatory neurons a weight of ~0.1 and rare types ~20x.  Raising to
        # weight_power (default 0.5 = sqrt) compresses the range while keeping the
        # minority-class preference.  Set weight_power=1.0 in config to restore raw behavior.
        weight_power = config["training"].get("class_weight_power", 0.5)
        weights_present = weights_present ** weight_power

        # Build a full num_labels-length weight vector indexed by label code.
        # If a class exists in val/test but not in train, compute_class_weight returns
        # a shorter array and F.cross_entropy would silently index out of bounds.
        full_weights = np.ones(num_labels, dtype=np.float32)
        for cls, wt in zip(present_classes, weights_present):
            full_weights[int(cls)] = wt
        class_weights = torch.tensor(full_weights, dtype=torch.float32)
        print(
            f"Class-weighted loss enabled — {len(present_classes)}/{num_labels} classes "
            f"present in train split (weight_power={weight_power})"
        )

    if want_hier:
        coarse_map_path = config["training"]["coarse_map_path"]
        hier_lambda = config["training"].get("hierarchical_lambda", 0.3)
        hier_mode = config["training"].get("hierarchical_mode", "marginalize")
        fine_to_coarse, agg_matrix, coarse2id = build_coarse_mapping(label2id, coarse_map_path)
        if hier_mode == "learned_head":
            # Attach the head to the model so its params land in model.parameters() and are
            # picked up by the Trainer's optimizer (new modules default to requires_grad=True).
            # It's an auxiliary training-only head — not needed at inference, so it is not saved.
            coarse_head = torch.nn.Linear(hidden_size, len(coarse2id))
            model.coarse_head = coarse_head
            print(
                f"Hierarchical loss enabled — mode=learned_head, lambda={hier_lambda}, "
                f"head={hidden_size}->{len(coarse2id)}"
            )
        else:
            print(f"Hierarchical loss enabled — mode=marginalize, lambda={hier_lambda}")

    if want_weighted or want_hier:
        TrainerClass = build_custom_trainer_class(
            class_weights=class_weights,
            fine_to_coarse=fine_to_coarse,
            agg_matrix=agg_matrix,
            coarse_head=coarse_head,
            hier_lambda=hier_lambda,
        )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        # float() guards the YAML scientific-notation gotcha (2e-4 parses as a str, not float)
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["num_train_epochs"],
        # max_steps > 0 caps training (overrides epochs); -1 disables. Used for quick smoke tests.
        max_steps=config["training"].get("max_steps", -1),
        bf16=config["training"].get("bf16", False),
        # fp16 for Turing/older GPUs (e.g. RTX 2060) that lack bf16 support
        fp16=config["training"].get("fp16", False),
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
        # Pass seed explicitly so Trainer's dataloader shuffle matches config seed.
        # (data_seed is intentionally omitted — it requires accelerate >= 1.1.0, which drops
        # Python 3.8 support; `seed` alone already seeds the dataloader sampler.)
        seed=config.get("seed", 42),
        report_to="wandb" if config.get("wandb", {}).get("enabled", False) else "none",
        run_name=config.get("wandb", {}).get("run_name"),
        # save_total_limit must be >= early_stopping_patience + 1 so the best
        # checkpoint is never deleted before load_best_model_at_end can load it
        save_total_limit=7,
    )

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=6,
        # require at least 0.01% improvement to reset patience counter;
        # prevents premature stopping on noisy eval fluctuations
        early_stopping_threshold=1e-4,
    )]
    if is_adalora:
        # Must reference the actual PEFT model so update_and_allocate hits the AdaLoRA layers
        callbacks.append(AdaLoRACallback(model))

    # Newer Geneformer's collator requires the token dictionary (for pad-token handling).
    # Must match the model_version used for tokenization (V1 -> gc30M vocab, V2 -> gc104M).
    tok_dict_file = (
        TOKEN_DICTIONARY_FILE_30M if config.get("model_version", "V1") == "V1"
        else TOKEN_DICTIONARY_FILE
    )
    with open(tok_dict_file, "rb") as f:
        token_dictionary = pickle.load(f)

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForCellClassification(token_dictionary=token_dictionary),
        callbacks=callbacks,
    )

    print("Starting training...")
    resume_from = config["training"].get("resume_from_checkpoint", None)
    if resume_from:
        # resume_from_checkpoint can be:
        #   True  → Trainer auto-detects the latest checkpoint in output_dir
        #   str   → explicit path to a specific checkpoint folder
        print(f"Resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from or False)

    # Fit temperature on the VALIDATION set (not test) so inference confidence is calibrated.
    # Done after load_best_model_at_end so we calibrate the model that will actually be saved.
    if config["training"].get("calibrate_temperature", True):
        # calibration_method: "temperature" (single scalar) | "vector" (per-class scale+bias)
        calib_method = config["training"].get("calibration_method", "temperature")
        print(f"Fitting calibration ({calib_method}) on validation set...")
        val_output = trainer.predict(val_dataset)
        fit_and_save_calibration(
            val_output.predictions, val_output.label_ids, config["output_dir"], calib_method
        )

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
