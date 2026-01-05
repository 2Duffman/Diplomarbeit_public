from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from data_utils import load_split
from model_training import load_all_models


DATASET_PATH = "data.csv"
TARGET = "avg_rating"
TRAIN_IDS_PATH = "train_ids.txt"
TEST_IDS_PATH = "test_ids.txt"
ARTIFACTS_DIR = "artifacts"

OUT_DIR = Path("h_stat_testing_fast_df")

MODEL_WHITELIST = None

# Monte-Carlo + eval sizing (fast defaults)
MC_SAMPLES = 64          # used only for non-binary features
EVAL_N = 1500
EVAL_BATCH_SIZE = 256

SEED = 0
CENTER_F = True

TOP20_XGB: List[str] = [
    "year",
    "category_Expansion for Base-game",
    "complexity",
    "max_time",
    "min_age",
    "category_Wargame",
    "min_players",
    "min_time",
    "max_players",
    "implementation_avg_rating",
    "category_Miniatures",
    "designer_high_xp",
    "mechanic_Roll_/_Spin_and_Move",
    "publisher_high_xp",
    "designer_average_xp",
    "family_Crowdfunding",
    "compilation_count",
    "designer_low_xp",
    "publisher_(Self-Published)",
    "publisher_very_high_xp",
]


def safe_name(s: str) -> str:
    s = s.strip()
    return re.sub(r"[^\w\-\.]+", "_", s)


def _to_dense_float32(Z):
    if hasattr(Z, "toarray"):  # scipy sparse
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float32)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """
    Unified prediction helper (same spirit as your ALE code).
    Always accepts a DataFrame (important for pipelines selecting columns by name).
    """
    from model_training import KerasPipeline

    if isinstance(model, KerasPipeline):
        Xt = model.preprocessor.transform(X)
        Xt = _to_dense_float32(Xt)
        pred = model.model.predict(Xt, verbose=0)
        return np.asarray(pred).reshape(-1)

    # sklearn pipeline / estimator that expects DataFrame
    if hasattr(model, "named_steps") or hasattr(model, "feature_names_in_"):
        pred = model.predict(X)
        return np.asarray(pred).reshape(-1)

    # fallback
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


def _pd_minus_j_for_eval_batch_df(model, X_eval_batch: pd.DataFrame, feat: str, xj_samples: np.ndarray) -> np.ndarray:
    """
    PD_-j(x_-j) ≈ (1/M) Σ_m f(xj_samples[m], x_-j) for each eval row.
    Returns shape (B,).
    """
    B = len(X_eval_batch)
    M = len(xj_samples)

    rep = X_eval_batch.loc[X_eval_batch.index.repeat(M)].copy()
    rep[feat] = np.tile(xj_samples, B)

    pred = _predict(model, rep).reshape(B, M)
    return pred.mean(axis=1)

def compute_Hj_hybrid_df(
    model,
    X_bg: pd.DataFrame,
    X_eval: pd.DataFrame,
    features: List[str],
    mc_samples: int = 64,
    eval_batch_size: int = 256,
    seed: int = 0,
    center_f: bool = True,
) -> pd.DataFrame:
    """
    Compute Hj (one feature vs rest) using:
    - Binary features: fast 2-pred mixture formula
    - Non-binary: Monte-Carlo PD with M=mc_samples

    Uses DataFrames everywhere (pipeline-safe).
    """
    rng = np.random.default_rng(seed)

    # background subsample for MC (and for binary p1 estimate)
    if len(X_bg) > mc_samples:
        X_bgM = X_bg.sample(n=mc_samples, random_state=seed).copy()
    else:
        X_bgM = X_bg.copy()
        mc_samples = len(X_bgM)

    # f(x) on eval
    f_eval = _predict(model, X_eval).astype(float)
    if center_f:
        f_eval = f_eval - float(np.mean(f_eval))

    denom = float(np.mean(f_eval ** 2))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Non-finite/zero denominator. Try center_f=True or check predictions.")

    n_eval = len(X_eval)
    results = []

    for t, feat in enumerate(features, start=1):
        if feat not in X_eval.columns:
            raise KeyError(f"Feature {feat} not found in X columns.")

        # allocate
        pdj_vals = np.empty(n_eval, dtype=float)
        pdmj_vals = np.empty(n_eval, dtype=float)

        if is_binary_series(X_bg[feat]):
            # ── Binary fast path ────────────────────────────────────────────────
            v0, v1 = _binary_levels(X_bg[feat])

            # p1 from background
            x01 = X_bgM[feat].astype(int).to_numpy() if X_bgM[feat].dtype != bool else X_bgM[feat].astype(int).to_numpy()
            p1 = float(np.mean(x01))

            # PD_j(0), PD_j(1): two predictions on background complements
            tmp_bg = X_bgM.copy()
            tmp_bg[feat] = v0
            pdj0 = float(np.mean(_predict(model, tmp_bg)))
            tmp_bg[feat] = v1
            pdj1 = float(np.mean(_predict(model, tmp_bg)))

            # PD_j(xj) lookup on eval
            xj_eval_01 = X_eval[feat].astype(int).to_numpy() if X_eval[feat].dtype != bool else X_eval[feat].astype(int).to_numpy()
            pdj_vals[:] = np.where(xj_eval_01 == 1, pdj1, pdj0)

            # PD_-j(x_-j): exact mixture (1-p1)*f(v0,x_-j) + p1*f(v1,x_-j)
            for start in range(0, n_eval, eval_batch_size):
                end = min(start + eval_batch_size, n_eval)
                Xb = X_eval.iloc[start:end].copy()

                Xb[feat] = v0
                pred0 = _predict(model, Xb).astype(float)

                Xb[feat] = v1
                pred1 = _predict(model, Xb).astype(float)

                pdmj_vals[start:end] = (1.0 - p1) * pred0 + p1 * pred1

        else:
            # ── Non-binary MC path ─────────────────────────────────────────────
            xj_samples = X_bgM[feat].to_numpy()

            for start in range(0, n_eval, eval_batch_size):
                end = min(start + eval_batch_size, n_eval)
                Xb = X_eval.iloc[start:end]
                xj_eval = Xb[feat].to_numpy()

                pdj_vals[start:end] = _pd_j_for_eval_batch_df(model, X_bgM, feat, xj_eval)
                pdmj_vals[start:end] = _pd_minus_j_for_eval_batch_df(model, Xb, feat, xj_samples)

        # Center PD parts on eval points (empirical centering)
        pdj_vals = pdj_vals - float(np.mean(pdj_vals))
        pdmj_vals = pdmj_vals - float(np.mean(pdmj_vals))

        resid = f_eval - pdj_vals - pdmj_vals
        num = float(np.mean(resid ** 2))

        Hj2 = num / denom
        Hj2 = float(np.clip(Hj2, 0.0, 1.0))  # MC noise / numeric guard
        results.append({"feature": feat, "Hj2": Hj2, "Hj": float(np.sqrt(Hj2))})

        print(f"    [{t:02d}/{len(features)}] {feat}: Hj2={Hj2:.4f}", flush=True)

    return pd.DataFrame(results).sort_values("Hj2", ascending=False).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(
        csv_path=DATASET_PATH,
        target_col=TARGET,
        train_ids_path=TRAIN_IDS_PATH,
        test_ids_path=TEST_IDS_PATH,
    )

    missing = [f for f in TOP20_XGB if f not in X_test.columns]
    if missing:
        raise KeyError(f"TOP20_XGB contains missing columns: {missing}")

    models = load_all_models(ARTIFACTS_DIR)
    print("Loaded models:", list(models.keys()), flush=True)

    if MODEL_WHITELIST is not None:
        models = {k: v for k, v in models.items() if k in MODEL_WHITELIST}
        print("Filtered models:", list(models.keys()), flush=True)

    # fixed eval sample
    if len(X_test) > EVAL_N:
        X_eval = X_test.sample(n=EVAL_N, random_state=SEED).copy()
    else:
        X_eval = X_test.copy()

    X_bg = X_train

    for model_name, est in models.items():
        print(f"\n=== Hj (hybrid, DF-safe) for model: {model_name} ===", flush=True)
        model_dir = OUT_DIR / safe_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            hj_df = compute_Hj_hybrid_df(
                model=est,
                X_bg=X_bg,
                X_eval=X_eval,
                features=TOP20_XGB,
                mc_samples=MC_SAMPLES,
                eval_batch_size=EVAL_BATCH_SIZE,
                seed=SEED,
                center_f=CENTER_F,
            )

            out_csv = model_dir / f"Hj_top20__M{MC_SAMPLES}__Neval{len(X_eval)}.csv"
            hj_df.to_csv(out_csv, index=False)
            print("\nTop 10 Hj:", flush=True)
            print(hj_df.head(10).to_string(index=False), flush=True)
            print("\n  saved", out_csv, flush=True)

        except Exception as e:
            print(f"  FAILED Hj for {model_name}: {e}", flush=True)

    print("\nDone. Hj tables saved in:", OUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
