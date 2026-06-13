"""
Donor-stratified split verification.

Run BEFORE training to confirm the train/val/test h5ad files do not leak the same
donor across splits, and to inspect per-split class balance.

    python -m src.check_splits

Why this matters for brain CellxGene data
------------------------------------------
Cells from the same donor share genetic background, batch/technical artefacts, and
disease state.  If cells from donor D appear in BOTH train and test, the model can
memorise donor-specific signal and score artificially high on test — the metric no
longer reflects generalisation to NEW donors, which is the real deployment scenario.

A valid benchmark splits by donor: every donor's cells live in exactly one split.
This script detects violations of that rule and exits non-zero if any are found, so it
can be wired into CI or a pre-train sanity check.
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import scanpy as sc

from src.utils import load_config


def _read_obs(path: str, columns: list) -> pd.DataFrame:
    """
    Read only the requested obs columns using backed mode (memory-mapped).

    backed='r' loads the obs table without the expression matrix, so this is fast
    even for multi-GB CellxGene files.
    """
    adata = sc.read_h5ad(path, backed="r")
    try:
        present = [c for c in columns if c in adata.obs.columns]
        missing = [c for c in columns if c not in adata.obs.columns]
        df = adata.obs[present].copy()
    finally:
        adata.file.close()
    return df, missing


def check_donor_leakage(config: dict) -> int:
    """
    Verify no donor_id appears in more than one split.

    Returns the number of leaking donors (0 = clean).  Also prints per-split class
    distribution so imbalance is visible before training.
    """
    donor_col = config["data"].get("donor_column", "donor_id")
    label_col = config["data"]["label_column"]

    splits = {
        "train": config["data"]["train_path"],
        "val": config["data"]["val_path"],
        "test": config["data"]["test_path"],
    }

    donor_to_splits = defaultdict(set)
    any_donor_col_missing = False

    print("=" * 70)
    print(f"Split verification - donor column: '{donor_col}', label: '{label_col}'")
    print("=" * 70)

    for split, path in splits.items():
        if not Path(path).exists():
            print(f"\n[{split}] MISSING FILE: {path} - skipping")
            continue

        df, missing = _read_obs(path, [donor_col, label_col])
        print(f"\n[{split}] {path}")
        print(f"  cells: {len(df)}")

        if donor_col in missing:
            any_donor_col_missing = True
            print(
                f"  WARNING: donor column '{donor_col}' not found in obs. "
                f"Cannot check leakage for this split.\n"
                f"  Available obs columns: {list(df.columns)}"
            )
        else:
            donors = df[donor_col].astype(str).unique()
            print(f"  unique donors: {len(donors)}")
            for d in donors:
                donor_to_splits[d].add(split)

        if label_col in df.columns:
            counts = df[label_col].value_counts()
            print(f"  classes: {len(counts)}")
            # Show the 3 largest and 3 smallest classes to reveal imbalance at a glance
            print("    largest :", dict(counts.head(3)))
            print("    smallest:", dict(counts.tail(3)))

    if any_donor_col_missing:
        print(
            f"\nCannot complete leakage check - donor column '{donor_col}' missing in "
            f"at least one split. Set data.donor_column in config.yaml to the correct name."
        )
        return -1

    # A donor that maps to more than one split is a leak
    leaks = {d: sorted(s) for d, s in donor_to_splits.items() if len(s) > 1}

    print("\n" + "=" * 70)
    if leaks:
        print(f"DATA LEAKAGE DETECTED - {len(leaks)} donor(s) span multiple splits:")
        for d, where in list(leaks.items())[:20]:
            print(f"  donor {d}: {where}")
        if len(leaks) > 20:
            print(f"  ... and {len(leaks) - 20} more")
        print(
            "\nFix: re-split the data by donor so each donor's cells are in exactly one "
            "split. Metrics computed on a leaking split overestimate generalisation."
        )
    else:
        print("OK - no donor appears in more than one split. Splits are donor-stratified.")
    print("=" * 70)

    return len(leaks)


def main():
    config = load_config()
    n_leaks = check_donor_leakage(config)
    # Non-zero exit on leakage (or missing donor column) so this can gate training in CI
    sys.exit(1 if n_leaks != 0 else 0)


if __name__ == "__main__":
    main()
