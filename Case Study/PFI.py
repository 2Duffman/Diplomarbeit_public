# interpretability.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.utils import check_random_state

try:
    # import your own types if placed in a different module
    from model_training import KerasPipeline
except Exception:
    class KerasPipeline:  # minimal fallback for type hints
        pass


def _maybe_subsample(
    X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
    subsample: Optional[int], random_state: int
) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
    if subsample is None or subsample >= len(X):
        return X, y
    rng = check_random_state(random_state)
    idx = rng.choice(len(X), size=subsample, replace=False)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return X.iloc[idx], y.iloc[idx]
    return X.iloc[idx], y[idx]


def permutation_importance_df(
    estimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    *,
    scoring: str = "r2",            # or "neg_mean_squared_error"
    n_repeats: int = 10,
    subsample: Optional[int] = 8000,  # speed-up for ~60k rows
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Computes feature-wise permutation importance and returns a tidy DataFrame:
    columns = [feature, importance_mean, importance_std, importance_norm, rank]
    Works with sklearn estimators/pipelines and your KerasPipeline wrapper.
    """
    X_use, y_use = _maybe_subsample(X, y, subsample, random_state)

    if not isinstance(X_use, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with named columns for interpretability.")

    feature_names = X_use.columns.tolist()

    if isinstance(estimator, KerasPipeline):
        # Custom permutation loop (since sklearn's utility won't know how to handle our wrapper)
        rng = check_random_state(random_state)

        # baseline
        baseline = _score_estimator(estimator, X_use, y_use, scoring)

        deltas = np.zeros((n_repeats, X_use.shape[1]), dtype=float)

        for r in range(n_repeats):
            for j, col in enumerate(feature_names):
                X_perm = X_use.copy()
                # permute one column
                shuffled = X_perm[col].to_numpy().copy()
                rng.shuffle(shuffled)
                X_perm[col] = shuffled
                score_p = _score_estimator(estimator, X_perm, y_use, scoring)
                deltas[r, j] = baseline - score_p  # drop in score -> importance
    else:
        # Use sklearn's fast implementation (works with Pipelines)
        pi = permutation_importance(
            estimator, X_use, y_use,
            scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
        )
        deltas = pi.importances.T  # shape (n_repeats, n_features)

    means = deltas.mean(axis=0)
    stds  = deltas.std(axis=0)
    # normalize by sum of positives for readability (avoid negative totals)
    total_pos = np.abs(means).sum() or 1.0
    norm = means / total_pos

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": means,
        "importance_std": stds,
        "importance_norm": norm,
    }).sort_values("importance_mean", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def grouped_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    groups: Dict[str, Sequence[str]],
    *,
    scoring: str = "r2",
    n_repeats: int = 10,
    subsample: Optional[int] = 8000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Group-level PFI: permutes *all columns in a group together*.
    Useful when many correlated booleans represent one concept (e.g., genre_*).

    groups: mapping { "group_name": [col1, col2, ...] }
    Returns DataFrame with [group, importance_mean, importance_std, importance_norm, size, rank]
    """
    X_use, y_use = _maybe_subsample(X, y, subsample, random_state)
    rng = check_random_state(random_state)

    # sanity check
    for g, cols in groups.items():
        missing = set(cols) - set(X_use.columns)
        if missing:
            raise ValueError(f"Group '{g}' contains missing columns: {sorted(missing)[:5]} ...")

    baseline = _score_estimator(estimator, X_use, y_use, scoring)
    group_names = list(groups.keys())
    deltas = np.zeros((n_repeats, len(group_names)), dtype=float)

    for r in range(n_repeats):
        for j, g in enumerate(group_names):
            X_perm = X_use.copy()
            # jointly permute columns in the group using a single permutation index
            idx = np.arange(len(X_perm))
            rng.shuffle(idx)
            for col in groups[g]:
                X_perm[col] = X_perm[col].to_numpy()[idx]
            score_p = _score_estimator(estimator, X_perm, y_use, scoring)
            deltas[r, j] = baseline - score_p

    means = deltas.mean(axis=0)
    stds  = deltas.std(axis=0)
    total_pos = np.abs(means).sum() or 1.0
    norm = means / total_pos
    sizes = [len(groups[g]) for g in group_names]

    df = pd.DataFrame({
        "group": group_names,
        "importance_mean": means,
        "importance_std": stds,
        "importance_norm": norm,
        "size": sizes,
    }).sort_values("importance_mean", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _score_estimator(estimator, X_df: pd.DataFrame, y, scoring: str) -> float:
    """Minimal scoring utility (R2 or neg-MSE). Extend as needed."""
    y_pred = estimator.predict(X_df)
    if scoring == "r2":
        from sklearn.metrics import r2_score
        return float(r2_score(y, y_pred))
    elif scoring == "neg_mean_squared_error":
        from sklearn.metrics import mean_squared_error
        return -float(mean_squared_error(y, y_pred))
    else:
        raise ValueError(f"Unsupported scoring: {scoring!r}")
