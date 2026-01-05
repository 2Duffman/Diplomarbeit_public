# pfi_all_models.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


from model_training import save_estimator, train_all_default_models, fit_nn_1layer, fit_nn_3layers, load_estimator, load_all_models
from data_utils import load_split
from PFI import permutation_importance_df, grouped_permutation_importance


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "data.csv"     
TARGET = "avg_rating"
SCORING = "neg_mean_squared_error"               
N_REPEATS = 10               
OUTPUT_TXT = "pfi_results.txt"
SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to build groups for Grouped PFI
# ──────────────────────────────────────────────────────────────────────────────
def build_groups_for_grouped_pfi(X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group booleans by prefix before the first underscore (e.g., mechanic_*, category_*),
    and keep numerical features as one-feature groups so ALL features are represented.
    """
    groups: Dict[str, List[str]] = {}

    # Identify boolean-like columns (actual bool dtype or 0/1 dummies)
    bool_cols = []
    for c in X.columns:
        s = X[c]
        if s.dtype == bool:
            bool_cols.append(c)
        else:
            # treat as boolean dummy if values are subset of {0,1}
            vals = pd.Series(s.unique())
            try:
                if set(vals.dropna().astype(int).unique()).issubset({0, 1}):
                    bool_cols.append(c)
            except Exception:
                pass

    # Group booleans by prefix
    for c in bool_cols:
        if "_" in c:
            prefix = c.split("_", 1)[0]
        else:
            prefix = c  # no underscore → its own group name
        groups.setdefault(prefix, []).append(c)

    # Add numeric features as their own single-feature groups (so they’re included)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    for c in num_cols:
        if c not in bool_cols:  # avoid duplicates if a numeric got flagged as bool
            groups[c] = [c]

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    
    
    # Load once and get a consistent test split for all models
    X_train, X_test, y_train, y_test = load_split(csv_path=DATASET_PATH, target_col=TARGET)

    # Fit all default models (same as your comparison set)
    models = load_all_models("artifacts")

    # Build groups for Grouped PFI once, from the (named) test-frame columns
    groups = build_groups_for_grouped_pfi(X_test)

    # Open a single text file and append all results
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Permutation Feature Importance (PFI) and Grouped PFI\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Target: {TARGET}\n")
        f.write(f"Scoring: {SCORING}\n")
        f.write(f"Repeats: {N_REPEATS}\n")
        f.write(f"Subsample: None (full test set, n={len(X_test)})\n")
        f.write(f"Random state: {SEED}\n")
        f.write(f"Total features: {X_test.shape[1]}\n\n")

        for model_name, est in models.items():
            print("Processing model:", model_name)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("=" * 80 + "\n\n")

            # -------------------------------
            # Standard PFI (no subsampling)
            # -------------------------------
            f.write(">> Permutation Feature Importance (PFI)\n")
            pfi_df = permutation_importance_df(
                est,
                X_test,
                y_test,
                scoring=SCORING,
                n_repeats=N_REPEATS,
                subsample=None,         # <— no subsampling
                random_state=SEED,
                n_jobs=-1,
            )

            # Save top and full tables for clarity
            f.write("\nTop 25 features by PFI:\n")
            f.write(pfi_df.head(25).to_string(index=False))
            f.write("\n\nFull PFI table:\n")
            f.write(pfi_df.to_string(index=False))
            f.write("\n")

            # ---------------------------------
            # Grouped PFI (no subsampling)
            # ---------------------------------
            f.write("\n>> Grouped Permutation Feature Importance (Grouped PFI)\n")
            gfi_df = grouped_permutation_importance(
                est,
                X_test,
                y_test,
                groups=groups,
                scoring=SCORING,
                n_repeats=N_REPEATS,
                subsample=None,         # <— no subsampling
                random_state=SEED,
            )

            f.write("\nTop 25 groups by Grouped PFI:\n")
            f.write(gfi_df.head(25).to_string(index=False))
            f.write("\n\nFull Grouped PFI table:\n")
            f.write(gfi_df.to_string(index=False))
            f.write("\n")

    print(f"Saved all PFI and Grouped PFI results to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
