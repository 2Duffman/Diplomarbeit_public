from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt

from data_utils import load_split
from model_training import load_all_models
from ALE_clean import ale_1d, ale_1d_binary


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "data.csv"
TARGET = "avg_rating"
TRAIN_IDS_PATH = "train_ids.txt"
TEST_IDS_PATH = "test_ids.txt"
ARTIFACTS_DIR = "artifacts"

OUT_DIR = Path("ale_plots_testing2")

MODEL_WHITELIST = None  # set to None for all models
K_CONT = 50

# display cropping for continuous plots
X_Q_LO = 0.01
X_Q_HI = 0.99
X_Q_HI_BY_FEATURE = {"max_time": 0.95, "min_age": 1}
NO_LEFT_CROP = {"min_age", "min_players", "complexity"}  # keep left edge at min for these

# binary plots: fix y-scale per model
BINARY_Y_PAD = 1.10  # 10% headroom

FEATURES_TO_PLOT: List[str] = [
    "year",
    "category_Expansion for Base-game",
    "complexity",
    "category_Wargame",
    "mechanic_Roll_/_Spin_and_Move",
    "category_Miniatures",
    "mechanic_Dice_Rolling",
    "mechanic_Cooperative_Game",
    "min_age",
    "max_time",
    "min_players",
]

def safe_name(s: str) -> str:
    s = s.strip()
    return re.sub(r"[^\w\-\.]+", "_", s)


def is_binary_series(s: pd.Series) -> bool:
    """True if s is bool or a 0/1 dummy (ignoring NaNs)."""
    if s.dtype == bool:
        return True
    vals = pd.unique(s.dropna())
    if len(vals) == 0:
        return False
    try:
        vals_i = set(pd.Series(vals).astype(int).unique())
        return vals_i.issubset({0, 1})
    except Exception:
        return False


def finite_quantile_limits(feature: str, x: pd.Series, q_lo: float, q_hi_default: float) -> Optional[Tuple[float, float]]:
    """Quantile limits from finite numeric values only."""
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None

    q_hi = X_Q_HI_BY_FEATURE.get(feature, q_hi_default)
    lo, hi = np.quantile(arr, [q_lo, q_hi])

    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        return None
    return float(lo), float(hi)



def finite_min(x: pd.Series) -> Optional[float]:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.min(arr))


def plot_ale(
    edges: np.ndarray,
    ale_bin: np.ndarray,
    feature: str,
    title: str,
    out_png: Path,
    x_series_for_xlim: Optional[pd.Series] = None,
    binary_ymax: Optional[float] = None,
):
    """
    Plot ALE.
    - continuous: step plot with optional x cropping
    - binary: single bar (1 vs 0), colored by sign, optional fixed y-axis
    """
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    is_binary_like = (len(ale_bin) == 2 and len(edges) == 3)

    if is_binary_like:
        ale0 = float(ale_bin[0])
        ale1 = float(ale_bin[1])
        delta = ale1 - ale0  # effect when switching 0 -> 1

        color = "green" if delta >= 0 else "red"
        ax.bar([0], [delta], width=0.6, color=color)
        ax.set_xticks([0])
        ax.set_xticklabels(["1 vs 0"])
        ax.set_xlim(-0.6, 0.6)

        if binary_ymax is not None and np.isfinite(binary_ymax) and binary_ymax > 0:
            ymax = BINARY_Y_PAD * float(binary_ymax)
            ax.set_ylim(-ymax, ymax)

        ax.text(
            0, delta,
            f"{delta:.3f}",
            ha="center",
            va="bottom" if delta >= 0 else "top"
        )
        ax.set_ylabel("ALE difference (centered)")

    else:
        x_step = edges
        y_step = np.r_[ale_bin, ale_bin[-1]]
        ax.step(x_step, y_step, where="post")

        if x_series_for_xlim is not None:
            lims = finite_quantile_limits(feature, x_series_for_xlim, X_Q_LO, X_Q_HI)
            if lims is not None:
                lo, hi = lims
                if feature in NO_LEFT_CROP:
                    mn = finite_min(x_series_for_xlim)
                    if mn is not None:
                        lo = mn
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    ax.set_xlim(lo, hi)

        ax.set_ylabel("ALE (centered)")
        if feature == "min_players":
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # Ensure the plot extends a bit beyond the last integer so the last step is visible
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, x1 + 0.5)
    ax.set_xlabel(feature)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(
        csv_path=DATASET_PATH,
        target_col=TARGET,
        train_ids_path=TRAIN_IDS_PATH,
        test_ids_path=TEST_IDS_PATH,
    )

    missing = [f for f in FEATURES_TO_PLOT if f not in X_test.columns]
    if missing:
        raise KeyError(f"FEATURES_TO_PLOT contains missing columns: {missing}")

    models = load_all_models(ARTIFACTS_DIR)
    print("Loaded models:", list(models.keys()), flush=True)

    if MODEL_WHITELIST is not None:
        models = {k: v for k, v in models.items() if k in MODEL_WHITELIST}
        print("Filtered models:", list(models.keys()), flush=True)

    for model_name, est in models.items():
        print(f"\n=== ALE for model: {model_name} ===", flush=True)
        model_dir = OUT_DIR / safe_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: fixed y-axis scale for binary plots (per model)
        max_abs_delta = 0.0
        for feat in FEATURES_TO_PLOT:
            if not is_binary_series(X_test[feat]):
                continue
            try:
                edges_b, ale_bin_b = ale_1d_binary(est, X_test, feat)
                if len(ale_bin_b) == 2:
                    d = float(ale_bin_b[1] - ale_bin_b[0])
                    if np.isfinite(d):
                        max_abs_delta = max(max_abs_delta, abs(d))
            except Exception:
                pass

        binary_ymax = max_abs_delta if max_abs_delta > 0 else None

        # Pass 2: generate plots
        for feat in FEATURES_TO_PLOT:
            try:
                if is_binary_series(X_test[feat]):
                    edges, ale_bin = ale_1d_binary(est, X_test, feat)
                else:
                    edges, ale_bin = ale_1d(est, X_test, feat, K=K_CONT)

                out_png = model_dir / f"ALE__{safe_name(model_name)}__{safe_name(feat)}.png"
                title = f"ALE — {model_name} — {feat}"

                plot_ale(
                    edges=edges,
                    ale_bin=ale_bin,
                    feature=feat,
                    title=title,
                    out_png=out_png,
                    x_series_for_xlim=X_test[feat],
                    binary_ymax=binary_ymax,
                )
                print("  saved", out_png, flush=True)

            except Exception as e:
                print(f"  FAILED {feat}: {e}", flush=True)

    print("\nDone. ALE plots saved in:", OUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
