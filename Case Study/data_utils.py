# data_utils.py
"""
Utility helpers for loading the fixed train / test split of the board-game dataset.

Usage
-----
from data_utils import load_split

X_train, X_test, y_train, y_test = load_split(
    csv_path="data_fullbool.csv",          # or "data_numeric.csv"
    target_col="avg_rating",
)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


# ----------------------------------------------------------------------
# Core loader
# ----------------------------------------------------------------------
def load_split(
    csv_path: str | Path = "C:/users/morit/Desktop/Coding/Diplomarbeit/code/Machine Learning/data.csv", 
    id_col: str = "bgg_id",
    target_col: str = "avg_rating",
    train_ids_path: str | Path = "C:/users/morit/Desktop/Coding/Diplomarbeit/code/Machine Learning/train_ids.txt",
    test_ids_path: str | Path = "C:/users/morit/Desktop/Coding/Diplomarbeit/code/Machine Learning/test_ids.txt",
    as_numpy: bool = False,
) -> Tuple:
    """
    Load the frozen train / test split for *any* version of the dataset.

    Parameters
    ----------
    csv_path : str | Path
        Path to the dataset CSV (full-boolean or numeric-only).
    id_col : str
        Column in the CSV that contains the unique game ID (default "bgg_id").
    target_col : str
        Name of the target column (default "avg_rating").
    train_ids_path, test_ids_path : str | Path
        Paths to the one-ID-per-line text files created earlier.
    as_numpy : bool
        If True, return NumPy arrays instead of pandas objects.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
        Either pandas DataFrames / Series or NumPy arrays, depending on
        `as_numpy`.
    """
    csv_path = Path(csv_path)
    train_ids_path = Path(train_ids_path)
    test_ids_path = Path(test_ids_path)

    # 1. Read the fixed ID lists
    train_ids = np.loadtxt(train_ids_path, dtype=int)
    test_ids = np.loadtxt(test_ids_path, dtype=int)

    # Basic sanity check to avoid leakage
    intersect = np.intersect1d(train_ids, test_ids)
    if intersect.size > 0:
        raise ValueError(
            f"IDs present in both train and test split: {intersect[:10]}"
        )

    # 2. Load the dataset
    df = pd.read_csv(csv_path)

    # 3. Build boolean masks
    train_mask = df[id_col].isin(train_ids)
    test_mask = df[id_col].isin(test_ids)

    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("One of the splits ended up empty—check ID files.")

    # 4. Slice into features / target
    drop_cols = [id_col, target_col, "name"]
    X_train = df.loc[train_mask].drop(columns=drop_cols, errors="ignore")
    y_train = df.loc[train_mask, target_col]

    X_test = df.loc[test_mask].drop(columns=drop_cols, errors="ignore")
    y_test = df.loc[test_mask, target_col]

    if as_numpy:
        return (
            X_train.to_numpy(),
            X_test.to_numpy(),
            y_train.to_numpy(),
            y_test.to_numpy(),
        )
    return X_train, X_test, y_train, y_test



# ----------------------------------------------------------------------
# Convenience: quick shapes printer
# ----------------------------------------------------------------------
def print_split_shapes(csv_path: str | Path, **kwargs) -> None:
    """Small helper to eyeball the row / column counts."""
    Xt, Xv, yt, yv = load_split(csv_path, **kwargs)
    print(
        f"Loaded '{csv_path}':\n"
        f"  Train: {Xt.shape[0]} rows, {Xt.shape[1]} features\n"
        f"  Test : {Xv.shape[0]} rows, {Xv.shape[1]} features"
    )


# ----------------------------------------------------------------------
# CLI usage  -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect fixed train/test split.")
    parser.add_argument("csv", help="Dataset CSV file to inspect.")
    args = parser.parse_args()
    print_split_shapes(args.csv)
