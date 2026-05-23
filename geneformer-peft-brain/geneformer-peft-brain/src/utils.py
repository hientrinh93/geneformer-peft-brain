from pathlib import Path
import yaml
import shutil

def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def print_trainable_parameters(model):
    """Print number of trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Total parameters:     {total:,}\n")

def save_training_config(config):
    """Save used config to output directory"""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("configs/config.yaml", output_dir / "config_used.yaml")
    print(f"Config saved to {output_dir}/config_used.yaml")
