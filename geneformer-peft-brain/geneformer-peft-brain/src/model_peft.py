from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

def get_bnb_config():
    """4-bit quantization config for QLoRA"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def get_peft_config(config):
    """PEFT config supporting all variants"""
    peft_params = config["peft"]
    return LoraConfig(
        task_type="SEQ_CLS",
        r=peft_params["r"],
        lora_alpha=peft_params["lora_alpha"],
        lora_dropout=peft_params["lora_dropout"],
        target_modules=peft_params["target_modules"],
        bias="none",
        use_dora=peft_params["use_dora"],
        init_lora_weights=peft_params["init_lora_weights"],
    )

def prepare_model(model, config):
    """Prepare model with selected PEFT method"""
    if config["peft"]["method"] == "qlora":
        print("Applying 4-bit quantization for QLoRA...")
        model = prepare_model_for_kbit_training(model)

    peft_config = get_peft_config(config)
    model = get_peft_model(model, peft_config)

    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")

    return model
