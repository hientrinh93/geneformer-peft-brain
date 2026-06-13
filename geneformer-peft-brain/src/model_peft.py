import torch
from peft import LoraConfig, IA3Config, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def get_compute_dtype(config: dict):
    """
    Pick the mixed-precision compute dtype from config.
    bf16 is only safe on Ampere+ (A100, RTX 30xx/40xx). Turing GPUs (e.g. RTX 2060) and
    older have no bf16 support, so fall back to fp16 there. Driven by training.bf16.
    """
    return torch.bfloat16 if config.get("training", {}).get("bf16", False) else torch.float16


def get_bnb_config(compute_dtype=None) -> BitsAndBytesConfig:
    """
    4-bit NF4 quantization config used by QLoRA/QDoRA at model-load time.
    compute_dtype should match the training precision (fp16 on Turing, bf16 on Ampere+).
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype if compute_dtype is not None else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def needs_quantization(config: dict) -> bool:
    """
    Return True when the base model must be loaded with BnB quantization_config.
    QLoRA and QDoRA both quantize the base to 4-bit at load time; LoftQ handles
    quantization internally inside get_peft_model, so it does NOT need
    quantization_config at the from_pretrained call.
    """
    return config["peft"]["method"] in ("qlora", "qdora")


def get_peft_config(config: dict):
    """
    Build the PEFT config purely from config["peft"]["method"].
    Flags (use_dora, init_lora_weights) are derived automatically so the user
    only needs to change the method name in config.yaml.
    """
    method = config["peft"]["method"]
    p = config["peft"]

    if method == "ia3":
        # IA3 learns per-channel scale vectors — far fewer params than LoRA.
        # Good when the dataset is very small or VRAM is extremely limited.
        return IA3Config(
            task_type="SEQ_CLS",
            target_modules=["key", "value", "intermediate.dense"],
            feedforward_modules=["intermediate.dense"],
        )

    if method == "adalora":
        # AdaLoRA starts every adapter at rank `init_r`, then prunes the least-important
        # singular values during training so the rank budget concentrates on the layers
        # that matter most.  Requires total_step (number of optimizer steps) so the budget
        # scheduler knows the training horizon — train.py computes and injects it.
        from peft import AdaLoraConfig

        total_step = p.get("adalora_total_step")
        if not total_step:
            raise ValueError(
                "AdaLoRA requires peft.adalora_total_step to be set. "
                "train.py computes this automatically from the dataset; if you call "
                "get_peft_config directly, set it manually."
            )
        target_r = p["r"]                                # final AVERAGE rank per adapter
        # `or` (not dict.get default) so an explicit null in config.yaml also falls back
        # to the computed default rather than passing None into AdaLoraConfig.
        init_r = p.get("adalora_init_r") or int(target_r * 1.5)   # higher starting rank
        # tinit: warm-up steps with no pruning; tfinal: steps before end to freeze the mask.
        # Defaults give ~10% warm-up and ~15% final-freeze, which is the paper's regime.
        tinit = p.get("adalora_tinit") or int(total_step * 0.1)
        tfinal = p.get("adalora_tfinal") or int(total_step * 0.15)
        return AdaLoraConfig(
            task_type="SEQ_CLS",
            init_r=init_r,
            target_r=target_r,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=p.get("adalora_deltaT", 10),          # steps between rank reallocations
            lora_alpha=p["lora_alpha"],
            lora_dropout=p["lora_dropout"],
            target_modules=p["target_modules"],
            # orth_reg_weight penalises non-orthogonal singular vectors so the SVD-style
            # decomposition stays meaningful; added to the loss inside AdaLoraModel.forward.
            orth_reg_weight=p.get("adalora_orth_reg_weight", 0.5),
            total_step=total_step,
        )

    # --- LoRA family: derive flags from method name ---
    # qdora = DoRA on a 4-bit-quantized base (weight decomposition + QLoRA memory savings)
    use_dora = method in ("dora", "qdora")

    if method == "pissa":
        init_lora_weights = "pissa"       # SVD init from W; fast convergence
    elif method == "olora":
        init_lora_weights = "olora"       # QR init; similar to pissa, cheaper
    elif method == "loftq":
        init_lora_weights = "loftq"       # alternating SVD+quantize init
    else:                                  # lora, qlora, dora, qdora
        init_lora_weights = True           # standard: B=0, A=Kaiming-uniform

    kwargs = dict(
        task_type="SEQ_CLS",
        r=p["r"],
        lora_alpha=p["lora_alpha"],
        lora_dropout=p["lora_dropout"],
        target_modules=p["target_modules"],
        bias="none",
        use_dora=use_dora,
        init_lora_weights=init_lora_weights,
        # rsLoRA: scales by alpha/sqrt(r) for stable high-rank training (Kalajdzievski 2023)
        use_rslora=p.get("use_rslora", False),
    )

    # Layer targeting is only valid when restricting to specific layers. PEFT rejects
    # layers_pattern when target_modules is a string (e.g. "all-linear"), so only pass the
    # layer-targeting args when the user actually set layers_to_transform.
    layers_to_transform = p.get("layers_to_transform", None)
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = layers_to_transform
        kwargs["layers_pattern"] = p.get("layers_pattern", None)

    if method == "loftq":
        from peft import LoftQConfig
        # loftq_iter: number of alternating SVD+quantize rounds.
        # 1 = fast setup; 4-8 = more accurate init (better accuracy, slower).
        # Configurable via config.yaml peft.loftq_iter.
        loftq_iter = p.get("loftq_iter", 1)
        kwargs["loftq_config"] = LoftQConfig(loftq_bits=4, loftq_iter=loftq_iter)

    return LoraConfig(**kwargs)


def prepare_model(model, config: dict):
    """
    Wrap the base model with the configured PEFT adapter.

    QLoRA path: model was already loaded with BnB quantization_config (4-bit).
        prepare_model_for_kbit_training enables gradients on non-quantized params
        and casts LayerNorm to fp32 for numerical stability.

    LoftQ path: model was loaded in full precision. get_peft_model internally
        quantizes each weight and computes SVD of the quantization error to
        initialize A and B — no prepare_model_for_kbit_training needed.

    All other methods: standard get_peft_model call, no quantization.
    """
    method = config["peft"]["method"]
    use_gc = config["training"].get("gradient_checkpointing", False)

    if method in ("qlora", "qdora"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gc
        )

    peft_config = get_peft_config(config)
    model = get_peft_model(model, peft_config)
    return model
