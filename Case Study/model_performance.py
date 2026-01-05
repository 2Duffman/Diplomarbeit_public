# pfi_all_models.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


from model_training import save_estimator, train_all_default_models, fit_nn_1layer, fit_nn_3layers, load_estimator, load_all_models
from data_utils import load_split
from PFI import permutation_importance_df, grouped_permutation_importance


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "data.csv"     # adjust if your CSV lives elsewhere
TARGET = "avg_rating"
N_REPEATS = 10                # increase for more stable estimates
OUTPUT_TXT = "pfi_results.txt"
SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    
    
    # Load once and get a consistent test split for all models
    X_train, X_test, y_train, y_test = load_split(csv_path=DATASET_PATH, target_col=TARGET)

    # Fit all default models (same as your comparison set)
    models = load_all_models("artifacts")

    results = []

    for name, model in models.items():
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        rmse  = root_mean_squared_error(y_test,  y_pred_test)
        mae  = mean_absolute_error(y_test,  y_pred_test)
        r2  = r2_score(y_test,  y_pred_test)

        results.append({
            "model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    # Build DataFrame
    df = pd.DataFrame(results).set_index("model")
    # Optional: order by test RMSE
    df = df.sort_values("R2")

    # Create LaTeX table string
    latex_table = df.to_latex(
        float_format="%.3f",       # 3 decimal places
        column_format="lrrr",   # 1 left + 3 right-aligned columns
        caption="Performance of the trained regression models",
        label="tab:model_performance"
    )

    # Save to file so you can \input it in LaTeX
    with open("model_performance.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
