import scanpy as sc
import tempfile
import shutil
from pathlib import Path
from geneformer import TranscriptomeTokenizer
from datasets import load_from_disk
from torch.utils.data import Dataset

class GeneformerDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def tokenize_with_official_tokenizer(adata_path: str, output_dir: str, prefix: str, label_column: str):
    """Use official Geneformer tokenizer"""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        shutil.copy(adata_path, temp_dir / Path(adata_path).name)
        
        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={label_column: "label"},
            nproc=4
        )
        tokenizer.tokenize_data(
            data_directory=str(temp_dir),
            output_directory=output_dir,
            output_prefix=prefix,
            file_format="h5ad"
        )
        print(f"✅ Tokenized {prefix} dataset successfully")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_and_tokenize_data(config):
    """Load and tokenize train/val/test using official tokenizer"""
    token_dir = Path("./data/tokenized")
    token_dir.mkdir(parents=True, exist_ok=True)
    label_col = config["data"]["label_column"]

    tokenize_with_official_tokenizer(config["data"]["train_path"], str(token_dir), "train", label_col)
    tokenize_with_official_tokenizer(config["data"]["val_path"], str(token_dir), "val", label_col)
    tokenize_with_official_tokenizer(config["data"]["test_path"], str(token_dir), "test", label_col)

    train_hf = load_from_disk(str(token_dir / "train.dataset"))
    val_hf = load_from_disk(str(token_dir / "val.dataset"))
    test_hf = load_from_disk(str(token_dir / "test.dataset"))

    return (GeneformerDataset(train_hf), 
            GeneformerDataset(val_hf), 
            GeneformerDataset(test_hf))
