import shutil
import tempfile
from pathlib import Path

import h5py
import scanpy as sc
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
from torch.utils.data import Dataset


class GeneformerDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def check_raw_counts(adata_path: str) -> None:
    """
    Fast sanity check that the h5ad file contains raw integer counts.

    Geneformer ranks genes by expression magnitude, so normalized or log-transformed
    values will produce wrong token sequences.  This check reads only the HDF5
    metadata (no matrix loading) so it is safe for multi-GB files.

    Raises ValueError if log1p normalization is detected.
    Prints a warning if expression values look suspiciously low.
    """
    with h5py.File(adata_path, "r") as f:
        uns_keys = list(f.get("uns", {}).keys())
        if "log1p" in uns_keys:
            raise ValueError(
                f"{adata_path}: 'log1p' key found in adata.uns — data is log-normalized.\n"
                "Geneformer requires raw integer counts.  Fix:\n"
                "  import numpy as np\n"
                "  adata.X = np.expm1(adata.X)   # reverse log1p\n"
                "  del adata.uns['log1p']\n"
                "  adata.write_h5ad(path)"
            )

        # Read just the first few stored values to check order of magnitude.
        # Works for both dense (Dataset) and sparse (Group with 'data' key) X.
        x = f.get("X")
        if x is None:
            return
        if isinstance(x, h5py.Dataset):
            # h5py.Dataset has no .flat — read the first row and flatten instead
            sample_vals = np.asarray(x[0]).ravel()[:500]
        elif "data" in x:
            sample_vals = x["data"][:500]
        else:
            return

        max_val = float(sample_vals.max()) if len(sample_vals) > 0 else 0
        if max_val < 20:
            print(
                f"WARNING {Path(adata_path).name}: max expression in sample = {max_val:.2f}. "
                "Raw counts are usually much higher.  Verify X contains raw read counts."
            )


def tokenize_with_official_tokenizer(
    adata_path: str,
    output_dir: str,
    prefix: str,
    label_column: str,
    nproc: int = 4,
    force: bool = False,
) -> None:
    """
    Tokenize a single h5ad file with Geneformer's TranscriptomeTokenizer.

    Skips tokenization if the output dataset directory already exists and force=False,
    so repeated training runs do not re-tokenize unnecessarily.
    """
    out_path = Path(output_dir) / f"{prefix}.dataset"

    if out_path.exists() and not force:
        print(f"Found cached tokenized data at {out_path}, skipping.")
        return

    temp_dir = Path(tempfile.mkdtemp())
    try:
        shutil.copy(adata_path, temp_dir / Path(adata_path).name)
        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={label_column: "label"},
            nproc=nproc,
        )
        tokenizer.tokenize_data(
            data_directory=str(temp_dir),
            output_directory=output_dir,
            output_prefix=prefix,
            file_format="h5ad",
        )
        print(f"Tokenized {prefix} → {out_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_and_tokenize_data(config: dict) -> tuple:
    """
    Tokenize (or load from cache) train / val / test splits.

    Raw-count validation is run before tokenization so problems are caught early
    rather than after a long tokenization step.
    """
    token_dir = Path("./data/tokenized")
    token_dir.mkdir(parents=True, exist_ok=True)

    label_col = config["data"]["label_column"]
    nproc = config["data"].get("tokenizer_nproc", 4)
    force = config["data"].get("retokenize", False)

    for split, path_key in [
        ("train", "train_path"),
        ("val", "val_path"),
        ("test", "test_path"),
    ]:
        check_raw_counts(config["data"][path_key])
        tokenize_with_official_tokenizer(
            config["data"][path_key],
            str(token_dir),
            split,
            label_col,
            nproc=nproc,
            force=force,
        )

    train_hf = load_from_disk(str(token_dir / "train.dataset"))
    val_hf = load_from_disk(str(token_dir / "val.dataset"))
    test_hf = load_from_disk(str(token_dir / "test.dataset"))

    return (
        GeneformerDataset(train_hf),
        GeneformerDataset(val_hf),
        GeneformerDataset(test_hf),
    )
