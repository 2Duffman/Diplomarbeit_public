from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from data_utils import load_split
from model_training import load_all_models, KerasPipeline


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "data.csv"
TARGET = "avg_rating"
TRAIN_IDS_PATH = "train_ids.txt"
TEST_IDS_PATH = "test_ids.txt"
ARTIFACTS_DIR = "artifacts"

MODEL_XGB_KEY = "XGBRegressor"  
MODEL_NN3_KEY = "NN_3_layers"           

ROLL_SPIN_MOVE_COL = "mechanic_Roll_/_Spin_and_Move"
COOP_COL = "mechanic_Cooperative_Game"

OUT_DIR = Path("local_candidates")
OUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _to_dense_float32(Z):
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float32)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict robustly for sklearn Pipelines and your KerasPipeline."""
    if isinstance(model, KerasPipeline):
        Xt = model.preprocessor.transform(X)
        Xt = _to_dense_float32(Xt)
        y = model.model.predict(Xt, verbose=0)
        return np.asarray(y).reshape(-1)

    y = model.predict(X)
    return np.asarray(y).reshape(-1)


def _flag_topk(df: pd.DataFrame, col: str, k: int, largest: bool = True) -> pd.Series:
    idx = df[col].nlargest(k).index if largest else df[col].nsmallest(k).index
    return df.index.isin(idx)


def _save(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def _as_bool01(series: pd.Series) -> np.ndarray:
    """Convert a series that is 0/1, bool, or possibly NaN into a clean boolean array."""
    return series.fillna(0).astype(int).astype(bool).to_numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────────────────────
def build_local_master_table() -> pd.DataFrame:
    # 1) Load split (this defines the authoritative test row order)
    X_train, X_test, y_train, y_test = load_split(
        csv_path=DATASET_PATH,
        target_col=TARGET,
        train_ids_path=TRAIN_IDS_PATH,
        test_ids_path=TEST_IDS_PATH,
    )

    # 2) Load models
    models = load_all_models(ARTIFACTS_DIR)
    print("Available model keys:", list(models.keys()))

    if MODEL_XGB_KEY not in models or MODEL_NN3_KEY not in models:
        raise KeyError(
            f"Need MODEL_XGB_KEY={MODEL_XGB_KEY} and MODEL_NN3_KEY={MODEL_NN3_KEY}. "
            f"Available: {list(models.keys())}"
        )

    xgb = models[MODEL_XGB_KEY]
    nn3 = models[MODEL_NN3_KEY]

    # 3) Predict (aligned with X_test row order)
    y_pred_xgb = _predict(xgb, X_test).astype(float)
    y_pred_nn3 = _predict(nn3, X_test).astype(float)

    # 4) Load raw data ONCE and align meta by the SAME indices as X_test/y_test
    raw = pd.read_csv(DATASET_PATH)
    meta = raw.loc[X_test.index].copy()  # <-- critical fix (no joining via test_ids)

    # 5) Build master table (keep row order as-is)
    df = pd.DataFrame({
        "y_true": np.asarray(y_test, dtype=float),  # avg_rating for test rows
        "y_pred_xgb": y_pred_xgb,
        "y_pred_nn3": y_pred_nn3,
    })

    # attach identifiers + a few context columns (only if present)
    for col in ["name", "bgg_id", "year", "complexity", "max_time", "min_time",
                "min_players", "max_players", "min_age"]:
        if col in meta.columns:
            df[col] = meta[col].values

    # errors + disagreement
    df["err_xgb"] = df["y_pred_xgb"] - df["y_true"]
    df["err_nn3"] = df["y_pred_nn3"] - df["y_true"]
    df["abs_err_xgb"] = df["err_xgb"].abs()
    df["abs_err_nn3"] = df["err_nn3"].abs()
    df["y_pred_mean"] = 0.5 * (df["y_pred_xgb"] + df["y_pred_nn3"])
    df["abs_err_mean"] = 0.5 * (df["abs_err_xgb"] + df["abs_err_nn3"])
    df["model_disagreement"] = (df["y_pred_xgb"] - df["y_pred_nn3"]).abs()

    # 6) Exact mechanic flags
    if ROLL_SPIN_MOVE_COL not in meta.columns:
        raise KeyError(f"Expected column '{ROLL_SPIN_MOVE_COL}' not found in data.")
    if COOP_COL not in meta.columns:
        raise KeyError(f"Expected column '{COOP_COL}' not found in data.")

    df["in_roll_spin_and_move"] = _as_bool01(meta[ROLL_SPIN_MOVE_COL])
    df["in_cooperative"] = _as_bool01(meta[COOP_COL])

    # 7) Category flags 
    df["in_best100_true"] = _flag_topk(df, "y_true", 100, largest=True)
    df["in_worst100_true"] = _flag_topk(df, "y_true", 100, largest=False)

    df["in_best100_pred_xgb"] = _flag_topk(df, "y_pred_xgb", 100, largest=True)
    df["in_worst100_pred_xgb"] = _flag_topk(df, "y_pred_xgb", 100, largest=False)

    df["in_best100_pred_nn3"] = _flag_topk(df, "y_pred_nn3", 100, largest=True)
    df["in_worst100_pred_nn3"] = _flag_topk(df, "y_pred_nn3", 100, largest=False)

    df["in_best100_pred_mean"] = _flag_topk(df, "y_pred_mean", 100, largest=True)
    df["in_worst100_pred_mean"] = _flag_topk(df, "y_pred_mean", 100, largest=False)

    # Median-ish & typical buckets (small sets; flags indicate membership)
    med_true = float(np.median(df["y_true"]))
    median_band = df.loc[(df["y_true"] - med_true).abs() <= 0.10].copy()
    median20 = median_band.nsmallest(20, "abs_err_mean").index
    df["in_median20_true_lowerror"] = df.index.isin(median20)

    med_pred = float(np.median(df["y_pred_mean"]))
    tmp = df.copy()
    tmp["typical_score"] = (tmp["y_true"] - med_true).abs() + (tmp["y_pred_mean"] - med_pred).abs()
    typical30 = tmp.nsmallest(30, ["typical_score", "abs_err_mean"]).index
    df["in_typical30"] = df.index.isin(typical30)

    # Failure modes
    df["in_xgb_overpred50"] = _flag_topk(df, "err_xgb", 50, largest=True)
    df["in_xgb_underpred50"] = _flag_topk(df, "err_xgb", 50, largest=False)
    df["in_nn3_overpred50"] = _flag_topk(df, "err_nn3", 50, largest=True)
    df["in_nn3_underpred50"] = _flag_topk(df, "err_nn3", 50, largest=False)

    # Contrast cases
    df["in_disagreement_top100"] = _flag_topk(df, "model_disagreement", 100, largest=True)

    # Optional: quantile labels (helpful for “why interesting” narratives)
    df["true_decile"] = pd.qcut(df["y_true"], 10, labels=False, duplicates="drop")
    df["pred_mean_decile"] = pd.qcut(df["y_pred_mean"], 10, labels=False, duplicates="drop")
    df["abs_err_mean_decile"] = pd.qcut(df["abs_err_mean"], 10, labels=False, duplicates="drop")

    # 8) Save master + convenience subsets
    _save("local_master_table", df)
    _save("candidates_best100_true", df[df["in_best100_true"]])
    _save("candidates_worst100_true", df[df["in_worst100_true"]])
    _save("candidates_typical30", df[df["in_typical30"]])
    _save("candidates_disagreement_top100", df[df["in_disagreement_top100"]])

    print(f"Wrote: {OUT_DIR / 'local_master_table.csv'}")
    return df


def explain_games(df: pd.DataFrame, games: list[str], name_col: str = "name") -> None:
    """
    Given a list of game names, print their category memberships and key numbers.
    If names are ambiguous, use bgg_id instead (easy to add).
    """
    flag_cols = [c for c in df.columns if c.startswith("in_")]

    for g in games:
        sub = df[df[name_col].astype(str) == str(g)] if name_col in df.columns else pd.DataFrame()
        print("\n" + "=" * 80)
        print(f"Game: {g}")

        if sub.empty:
            print("Not found (name mismatch or not in test set). Try bgg_id.")
            continue

        for i, row in sub.iterrows():
            print("-" * 80)
            print(
                f"y_true={row['y_true']:.5f} | "
                f"y_pred_xgb={row['y_pred_xgb']:.5f} | y_pred_nn3={row['y_pred_nn3']:.5f}"
            )
            print(
                f"err_xgb={row['err_xgb']:+.5f} | err_nn3={row['err_nn3']:+.5f} | "
                f"disagreement={row['model_disagreement']:.5f}"
            )

            cats = [c for c in flag_cols if bool(row.get(c, False))]
            print("Categories:", ", ".join(cats) if cats else "(none of the top-k buckets)")

            for c in ["year", "complexity", "max_time", "min_age", "min_players", "max_players"]:
                if c in df.columns and pd.notna(row.get(c, np.nan)):
                    pct = float(df[c].rank(pct=True).loc[i])
                    print(f"{c}={row[c]} (≈{pct:.0%} percentile in test set)")


if __name__ == "__main__":
    master = build_local_master_table()

    # Example:
    # explain_games(master, ["Dream Runners", "Catan"])
