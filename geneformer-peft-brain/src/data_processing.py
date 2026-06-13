import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
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


def prepare_adata_for_tokenization(
    adata_path: str,
    output_path: str,
    label_column: str,
    min_cells: int = 50,
    ensembl_col: str = None,
) -> bool:
    """
    Prepare an h5ad for Geneformer tokenization. Two steps:

    1. Ensure `var` has an 'ensembl_id' column. Newer Geneformer (gc104M) requires the
       Ensembl IDs in an explicit `var['ensembl_id']` column rather than reading var.index.
       Source: `ensembl_col` if provided and present, otherwise var.index.

    2. Optionally drop rare classes (< `min_cells`). Geneformer fine-tuning on <50-cell
       classes is unreliable (too few examples; blows up balanced loss weights), and these
       classes only add noise to the confusion matrix.

    Writes a new file to `output_path` only if something changed (ensembl column added and/or
    classes dropped). Returns True if a new file was written (caller tokenizes it), False if
    the original file is already tokenization-ready (caller uses the original).
    """
    adata = sc.read_h5ad(adata_path)
    changed = False

    if "ensembl_id" not in adata.var.columns:
        if ensembl_col and ensembl_col in adata.var.columns:
            adata.var["ensembl_id"] = adata.var[ensembl_col].astype(str).values
            print(f"Added var['ensembl_id'] from var['{ensembl_col}']")
        else:
            adata.var["ensembl_id"] = adata.var.index.astype(str)
            print("Added var['ensembl_id'] from var.index")
        changed = True

    # Geneformer needs total read counts per cell in obs['n_counts'] for rank normalization.
    if "n_counts" not in adata.obs.columns:
        counts = adata.X.sum(axis=1)
        # X may be sparse (np.matrix from .sum) or dense ndarray — flatten to 1-D either way.
        adata.obs["n_counts"] = np.asarray(counts).ravel()
        print("Added obs['n_counts'] from X row sums")
        changed = True

    if min_cells > 0:
        counts = adata.obs[label_column].value_counts()
        rare = counts[counts < min_cells].index.tolist()
        if rare:
            n_before = adata.n_obs
            adata = adata[~adata.obs[label_column].isin(rare)].copy()
            # Remove dropped categories from the Categorical dtype so they don't appear
            # in the tokenizer's category list and shift integer codes for remaining classes.
            if hasattr(adata.obs[label_column], "cat"):
                adata.obs[label_column] = adata.obs[label_column].cat.remove_unused_categories()
            print(
                f"Dropped {len(rare)} rare class(es) with < {min_cells} cells: {rare}\n"
                f"  Cells: {n_before} -> {adata.n_obs} ({n_before - adata.n_obs} removed)"
            )
            changed = True

    if changed:
        adata.write_h5ad(output_path)
    return changed


def tokenize_with_official_tokenizer(
    adata_path: str,
    output_dir: str,
    prefix: str,
    label_column: str,
    nproc: int = 4,
    force: bool = False,
    model_version: str = "V1",
) -> None:
    """
    Tokenize a single h5ad file with Geneformer's TranscriptomeTokenizer.

    Skips tokenization if the output dataset directory already exists and force=False,
    so repeated training runs do not re-tokenize unnecessarily.

    model_version controls the V1 vs V2 tokenizer behaviour:
      V1 → context length 2048, no special tokens (TranscriptomeTokenizer defaults)
      V2 → context length 4096, <cls>/<eos> special tokens, V2 gene dictionaries
    The V2 path requires a geneformer package that ships V2 dictionaries and accepts the
    model_version argument; otherwise it raises a clear error rather than silently
    tokenizing with V1 rules (which would mismatch a V2 model and corrupt predictions).
    """
    out_path = Path(output_dir) / f"{prefix}.dataset"

    if out_path.exists() and not force:
        print(f"Found cached tokenized data at {out_path}, skipping.")
        return

    # Pass model_version explicitly. The tokenizer auto-selects the matching gene/token
    # dictionaries and sequence length per version (V1 -> gc30M dicts, len 2048, no special
    # tokens; V2 -> gc104M dicts, len 4096, <cls>/<eos>). If omitted, it DEFAULTS TO V2, which
    # would tokenize against the wrong vocabulary for a V1 model.
    tokenizer_kwargs = dict(
        custom_attr_name_dict={label_column: "label"},
        nproc=nproc,
        model_version=model_version.upper(),
    )

    temp_dir = Path(tempfile.mkdtemp())
    try:
        shutil.copy(adata_path, temp_dir / Path(adata_path).name)
        try:
            tokenizer = TranscriptomeTokenizer(**tokenizer_kwargs)
        except TypeError as e:
            raise RuntimeError(
                "Failed to build a V2 TranscriptomeTokenizer — your installed geneformer "
                "package does not support the V2 arguments (model_version / special_token). "
                "Upgrade geneformer to a V2-capable release, or set model_version: V1 in "
                f"config.yaml.\nOriginal error: {e}"
            ) from e
        tokenizer.tokenize_data(
            data_directory=str(temp_dir),
            output_directory=output_dir,
            output_prefix=prefix,
            file_format="h5ad",
        )
        print(f"Tokenized {prefix} -> {out_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _remap_labels_to_ids(hf_dataset, label2id: dict):
    """
    Map the tokenizer's string 'label' column to integer class ids using label2id.

    Newer Geneformer copies the raw obs value (e.g. "leukocyte") into the dataset's 'label'
    column instead of an integer code. The model needs integer labels, so we convert here.
    Cells whose label is not in label2id (e.g. a class dropped from train but present in
    val/test) are removed — we cannot score a class the model has no output unit for.
    """
    if label2id is None:
        return hf_dataset
    if len(hf_dataset) > 0 and isinstance(hf_dataset[0]["label"], str):
        hf_dataset = hf_dataset.filter(lambda ex: ex["label"] in label2id)
        hf_dataset = hf_dataset.map(lambda ex: {"label": label2id[ex["label"]]})
    return hf_dataset


def load_and_tokenize_data(config: dict, label2id: dict = None) -> tuple:
    """
    Tokenize (or load from cache) train / val / test splits.

    Raw-count validation and rare-class filtering are run before tokenization so
    problems are caught early rather than after a long tokenization step.

    Rare-class filtering: controlled by data.min_cells_per_class (default 50).
    Cells whose cell type has fewer than that many examples in the split are removed
    before tokenization.  Filtered files are written to data/filtered/ so the
    originals are untouched.  Set min_cells_per_class: 0 to disable.
    """
    token_dir = Path("./data/tokenized")
    token_dir.mkdir(parents=True, exist_ok=True)

    label_col = config["data"]["label_column"]
    nproc = config["data"].get("tokenizer_nproc", 4)
    force = config["data"].get("retokenize", False)
    min_cells = config["data"].get("min_cells_per_class", 50)
    model_version = config.get("model_version", "V1")
    ensembl_col = config["data"].get("ensembl_id_column")

    # Always created: even with no rare-class filtering, we may need to add the ensembl_id
    # column required by the tokenizer, which writes a prepared copy here.
    filter_dir = Path("./data/filtered")
    filter_dir.mkdir(parents=True, exist_ok=True)

    for split, path_key in [
        ("train", "train_path"),
        ("val", "val_path"),
        ("test", "test_path"),
    ]:
        raw_path = config["data"][path_key]
        check_raw_counts(raw_path)

        # Prepare each split: add ensembl_id column if missing, drop rare classes (if enabled).
        # If nothing needed changing, the original file is already tokenization-ready.
        tokenize_path = raw_path
        prepared_path = str(filter_dir / f"{split}_prepared.h5ad")
        was_prepared = prepare_adata_for_tokenization(
            raw_path, prepared_path, label_col, min_cells, ensembl_col
        )
        if was_prepared:
            tokenize_path = prepared_path

        tokenize_with_official_tokenizer(
            tokenize_path,
            str(token_dir),
            split,
            label_col,
            nproc=nproc,
            force=force,
            model_version=model_version,
        )

    train_hf = _remap_labels_to_ids(load_from_disk(str(token_dir / "train.dataset")), label2id)
    val_hf = _remap_labels_to_ids(load_from_disk(str(token_dir / "val.dataset")), label2id)
    test_hf = _remap_labels_to_ids(load_from_disk(str(token_dir / "test.dataset")), label2id)

    return (
        GeneformerDataset(train_hf),
        GeneformerDataset(val_hf),
        GeneformerDataset(test_hf),
    )
