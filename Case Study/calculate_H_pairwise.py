from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_utils import load_split
from model_training import load_all_models


DATASET_PATH = "data.csv"
TARGET = "avg_rating"
TRAIN_IDS_PATH = "train_ids.txt"
TEST_IDS_PATH = "test_ids.txt"
ARTIFACTS_DIR = "artifacts"

OUT_DIR = Path("hjk_testing2_fast_df")

MODEL_WHITELIST = ["RandomForestRegressor", "NN_3_layers"]

# Monte-Carlo background size + evaluation set size
MC_SAMPLES = 64
EVAL_N = 1500
EVAL_BATCH_SIZE = 256
SEED = 0

# Feature set S for pairwise screening
"""
FEATURE_SET_S: List[str] = [
    "year",
    "complexity",
    "category_Expansion for Base-game",
    "max_time",
    "min_age",
    "min_time",
    "max_players",
    "min_players",
]
"""
##feature set with interesting mechanics/categories
FEATURE_SET_S = [
    "year",
    "complexity",
    "category_Expansion for Base-game",
    "category_Wargame",
    "category_Miniatures",
    "mechanic_Cooperative_Game",
    "mechanic_Dice_Rolling",
    "mechanic_Roll_/_Spin_and_Move",
    "max_time",
    "min_players",
]

def safe_name(s: str) -> str:
    s = s.strip()
    return re.sub(r"[^\w\-\.]+", "_", s)


def _to_dense_float32(Z):
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float32)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """
    Pipeline-safe prediction (expects DataFrame so ColumnTransformer can select by name).
    """
    from model_training import KerasPipeline

    if isinstance(model, KerasPipeline):
        Xt = model.preprocessor.transform(X)
        Xt = _to_dense_float32(Xt)
        pred = model.model.predict(Xt, verbose=0)
        return np.asarray(pred).reshape(-1)

    if hasattr(model, "named_steps") or hasattr(model, "feature_names_in_"):
        pred = model.predict(X)
        return np.asarray(pred).reshape(-1)

    pred = model.predict(X.to_numpy(dtype=np.float32))
    return np.asarray(pred).reshape(-1)


def is_binary_series(s: pd.Series) -> bool:
    """True if s is bool or 0/1 dummy (ignoring NaNs)."""
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


def _binary_levels(series: pd.Series):
    """Return (v0, v1) preserving dtype semantics."""
    if series.dtype == bool:
        return False, True
    return 0, 1




def _pd_j_for_eval_batch_df(model, X_bgM: pd.DataFrame, feat: str, xj_eval: np.ndarray) -> np.ndarray:
    """
    PD_j(xj_eval[b]) ≈ (1/M) Σ_m f(xj_eval[b], x_-j^(m)) using background rows X_bgM.
    Returns shape (B,).
    """
    M = len(X_bgM)
    B = len(xj_eval)

    rep = X_bgM.iloc[np.tile(np.arange(M), B)].copy()
    rep[feat] = np.repeat(xj_eval, M)

    pred = _predict(model, rep).reshape(B, M)
    return pred.mean(axis=1)


def _pd_jk_for_eval_batch_df(
    model,
    X_bgM: pd.DataFrame,
    feat_j: str,
    feat_k: str,
    xj_eval: np.ndarray,
    xk_eval: np.ndarray,
) -> np.ndarray:
    """
    PD_{jk}(xj_eval[b], xk_eval[b]) ≈ (1/M) Σ_m f(xj_eval[b], xk_eval[b], x_-jk^(m)).
    Returns shape (B,).
    """
    M = len(X_bgM)
    B = len(xj_eval)

    rep = X_bgM.iloc[np.tile(np.arange(M), B)].copy()
    rep[feat_j] = np.repeat(xj_eval, M)
    rep[feat_k] = np.repeat(xk_eval, M)

    pred = _predict(model, rep).reshape(B, M)
    return pred.mean(axis=1)


def compute_centered_pd_1d_on_eval(
    model,
    X_bg: pd.DataFrame,
    X_bgM: pd.DataFrame,
    X_eval: pd.DataFrame,
    feat: str,
    eval_batch_size: int,
) -> np.ndarray:
    """
    Compute centered PD_j evaluated at the empirical eval points:
      \tilde{PD}_j(x_j^{(i)}) for i=1..N_eval
    """
    n_eval = len(X_eval)

    # Binary fast path: only two PD values and lookup
    if is_binary_series(X_bg[feat]):
        v0, v1 = _binary_levels(X_bg[feat])

        tmp = X_bgM.copy()
        tmp[feat] = v0
        pd0 = float(np.mean(_predict(model, tmp)))

        tmp[feat] = v1
        pd1 = float(np.mean(_predict(model, tmp)))

        x01 = X_eval[feat].astype(int).to_numpy() if X_eval[feat].dtype != bool else X_eval[feat].astype(int).to_numpy()
        pd_vals = np.where(x01 == 1, pd1, pd0).astype(float)

    else:
        pd_vals = np.empty(n_eval, dtype=float)
        for start in range(0, n_eval, eval_batch_size):
            end = min(start + eval_batch_size, n_eval)
            Xb = X_eval.iloc[start:end]
            xj = Xb[feat].to_numpy()
            pd_vals[start:end] = _pd_j_for_eval_batch_df(model, X_bgM, feat, xj)

    # Center on eval points
    pd_vals = pd_vals - float(np.mean(pd_vals))
    return pd_vals


def compute_centered_pd_2d_on_eval(
    model,
    X_bg: pd.DataFrame,
    X_bgM: pd.DataFrame,
    X_eval: pd.DataFrame,
    feat_j: str,
    feat_k: str,
    eval_batch_size: int,
) -> np.ndarray:
    """
    Compute centered PD_{jk} evaluated at empirical eval points:
      \tilde{PD}_{jk}(x_j^{(i)}, x_k^{(i)}) for i=1..N_eval
    """
    n_eval = len(X_eval)

    # Binary-binary fast path: compute 4 PD values and lookup
    if is_binary_series(X_bg[feat_j]) and is_binary_series(X_bg[feat_k]):
        j0, j1 = _binary_levels(X_bg[feat_j])
        k0, k1 = _binary_levels(X_bg[feat_k])

        tmp = X_bgM.copy()

        tmp[feat_j], tmp[feat_k] = j0, k0
        pd00 = float(np.mean(_predict(model, tmp)))

        tmp[feat_j], tmp[feat_k] = j0, k1
        pd01 = float(np.mean(_predict(model, tmp)))

        tmp[feat_j], tmp[feat_k] = j1, k0
        pd10 = float(np.mean(_predict(model, tmp)))

        tmp[feat_j], tmp[feat_k] = j1, k1
        pd11 = float(np.mean(_predict(model, tmp)))

        xj01 = X_eval[feat_j].astype(int).to_numpy() if X_eval[feat_j].dtype != bool else X_eval[feat_j].astype(int).to_numpy()
        xk01 = X_eval[feat_k].astype(int).to_numpy() if X_eval[feat_k].dtype != bool else X_eval[feat_k].astype(int).to_numpy()

        pd_vals = np.empty(n_eval, dtype=float)
        # lookup
        mask00 = (xj01 == 0) & (xk01 == 0)
        mask01 = (xj01 == 0) & (xk01 == 1)
        mask10 = (xj01 == 1) & (xk01 == 0)
        mask11 = (xj01 == 1) & (xk01 == 1)
        pd_vals[mask00] = pd00
        pd_vals[mask01] = pd01
        pd_vals[mask10] = pd10
        pd_vals[mask11] = pd11

    else:
        pd_vals = np.empty(n_eval, dtype=float)
        for start in range(0, n_eval, eval_batch_size):
            end = min(start + eval_batch_size, n_eval)
            Xb = X_eval.iloc[start:end]
            xj = Xb[feat_j].to_numpy()
            xk = Xb[feat_k].to_numpy()
            pd_vals[start:end] = _pd_jk_for_eval_batch_df(model, X_bgM, feat_j, feat_k, xj, xk)

    # Center on eval points
    pd_vals = pd_vals - float(np.mean(pd_vals))
    return pd_vals


def compute_Hjk_for_feature_set(
    model,
    X_bg: pd.DataFrame,
    X_eval: pd.DataFrame,
    features: List[str],
    mc_samples: int = 64,
    eval_batch_size: int = 256,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute pairwise H_{jk}^2 for all j<k in `features`:

      H_{jk}^2 = E[(tildePD_{jk} - tildePD_j - tildePD_k)^2] / E[(tildePD_{jk})^2]

    where expectations are empirical means over eval points.
    PDs are estimated via Monte-Carlo background with size mc_samples, with binary shortcuts.
    """
    # fixed background subsample for this model
    if len(X_bg) > mc_samples:
        X_bgM = X_bg.sample(n=mc_samples, random_state=seed).copy()
    else:
        X_bgM = X_bg.copy()
        mc_samples = len(X_bgM)

    # Precompute centered univariate PDs on eval once (reuse across pairs)
    pd1_cache: Dict[str, np.ndarray] = {}
    for feat in features:
        pd1_cache[feat] = compute_centered_pd_1d_on_eval(
            model=model,
            X_bg=X_bg,
            X_bgM=X_bgM,
            X_eval=X_eval,
            feat=feat,
            eval_batch_size=eval_batch_size,
        )

    # Pairwise
    rows = []
    pairs = list(itertools.combinations(features, 2))

    for idx, (fj, fk) in enumerate(pairs, start=1):
        pdj = pd1_cache[fj]
        pdk = pd1_cache[fk]

        pdjk = compute_centered_pd_2d_on_eval(
            model=model,
            X_bg=X_bg,
            X_bgM=X_bgM,
            X_eval=X_eval,
            feat_j=fj,
            feat_k=fk,
            eval_batch_size=eval_batch_size,
        )

        denom = float(np.mean(pdjk ** 2))
        if not np.isfinite(denom) or denom <= 0:
            H2 = 0.0
        else:
            diff = pdjk - pdj - pdk
            num = float(np.mean(diff ** 2))
            H2 = num / denom
            H2 = float(np.clip(H2, 0.0, 1.0))  # MC / numeric guard

        rows.append({"feature_j": fj, "feature_k": fk, "Hjk2": H2, "Hjk": float(np.sqrt(H2))})
        print(f"    [{idx:02d}/{len(pairs)}] ({fj}, {fk}): Hjk2={H2:.4f}", flush=True)

    return pd.DataFrame(rows).sort_values("Hjk2", ascending=False).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(
        csv_path=DATASET_PATH,
        target_col=TARGET,
        train_ids_path=TRAIN_IDS_PATH,
        test_ids_path=TEST_IDS_PATH,
    )

    missing = [f for f in FEATURE_SET_S if f not in X_test.columns]
    if missing:
        raise KeyError(f"FEATURE_SET_S contains missing columns: {missing}")

    models = load_all_models(ARTIFACTS_DIR)
    print("Loaded models:", list(models.keys()), flush=True)

    if MODEL_WHITELIST is not None:
        models = {k: v for k, v in models.items() if k in MODEL_WHITELIST}
        print("Filtered models:", list(models.keys()), flush=True)

    # Fixed eval sample for comparability
    if len(X_test) > EVAL_N:
        X_eval = X_test.sample(n=EVAL_N, random_state=SEED).copy()
    else:
        X_eval = X_test.copy()

    X_bg = X_train

    for model_name, est in models.items():
        print(f"\n=== Pairwise Hjk for model: {model_name} ===", flush=True)
        model_dir = OUT_DIR / safe_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            hjk_df = compute_Hjk_for_feature_set(
                model=est,
                X_bg=X_bg,
                X_eval=X_eval,
                features=FEATURE_SET_S,
                mc_samples=MC_SAMPLES,
                eval_batch_size=EVAL_BATCH_SIZE,
                seed=SEED,
            )

            out_csv = model_dir / f"Hjk_pairs__S{len(FEATURE_SET_S)}__M{MC_SAMPLES}__Neval{len(X_eval)}.csv"
            hjk_df.to_csv(out_csv, index=False)

            print("\nTop 10 Hjk:", flush=True)
            print(hjk_df.head(10).to_string(index=False), flush=True)
            print("\n  saved", out_csv, flush=True)

        except Exception as e:
            print(f"  FAILED Hjk for {model_name}: {e}", flush=True)

    print("\nDone. Hjk tables saved in:", OUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
