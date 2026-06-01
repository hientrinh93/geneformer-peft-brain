import torch
from peft import LoraConfig, IA3Config, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config used by QLoRA at model-load time."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def needs_quantization(config: dict) -> bool:
    """
    Return True when the base model must be loaded with BnB quantization_config.
    Only QLoRA quantizes at load time; LoftQ handles quantization internally inside
    get_peft_model, so it does NOT need quantization_config at the from_pretrained call.
    """
    return config["peft"]["method"] == "qlora"


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

    # --- LoRA family: derive flags from method name ---
    use_dora = method == "dora"

    if method == "pissa":
        init_lora_weights = "pissa"       # SVD init from W; fast convergence
    elif method == "olora":
        init_lora_weights = "olora"       # QR init; similar to pissa, cheaper
    elif method == "loftq":
        init_lora_weights = "loftq"       # alternating SVD+quantize init
    else:                                  # lora, qlora, dora
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
        # Restrict adapter to specific transformer layers (None = all layers)
        layers_to_transform=p.get("layers_to_transform", None),
        layers_pattern=p.get("layers_pattern", None),
    )

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

    if method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gc
        )

    peft_config = get_peft_config(config)
    model = get_peft_model(model, peft_config)
    return model
