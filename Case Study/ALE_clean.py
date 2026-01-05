import numpy as np
import pandas as pd


def _to_dense_float32(Z):
    """Convert sparse/dense matrix to dense float32 numpy array."""
    if hasattr(Z, "toarray"):  # scipy sparse
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float32)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """
    Unified prediction helper.

    - KerasPipeline: always uses its preprocessor, then feeds float32 to keras model.
    - sklearn-ish models (pipelines, models with feature_names_in_): predict on DataFrame.
    - others: fall back to numpy float32.
    """
    from model_training import KerasPipeline

    # KerasPipeline MUST preserve column names for preprocessing
    if isinstance(model, KerasPipeline):
        Xt = model.preprocessor.transform(X)
        Xt = _to_dense_float32(Xt)
        pred = model.model.predict(Xt, verbose=0)
        return np.asarray(pred).reshape(-1)

    # sklearn pipeline / estimator that expects DataFrame
    if hasattr(model, "named_steps") or hasattr(model, "feature_names_in_"):
        pred = model.predict(X)
        return np.asarray(pred).reshape(-1)

    # array-based fallback
    Xn = X.to_numpy(dtype=np.float32)
    pred = model.predict(Xn)
    return np.asarray(pred).reshape(-1)


def ale_1d(model, X: pd.DataFrame, feat: str, K: int = 20):
    """
    1D ALE for a continuous feature using quantile-based bins.
    Returns (edges, ale_bin) where len(edges)=K+1 and len(ale_bin)=K.
    """
    x = pd.to_numeric(X[feat], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 2:
        raise ValueError(f"Not enough finite values in {feat} for ALE (valid={valid.sum()}).")

    xv = x[valid]
    edges = np.quantile(xv, np.linspace(0, 1, K + 1))

    # ensure strictly increasing edges (handle ties)
    for k in range(1, len(edges)):
        if edges[k] <= edges[k - 1]:
            edges[k] = edges[k - 1] + 1e-9

    # bin index only for valid rows
    bin_idx_v = np.digitize(xv, edges[1:-1], right=True) + 1
    bin_idx_v = np.clip(bin_idx_v, 1, K)
    rows_v = np.flatnonzero(valid)

    delta = np.zeros(K, dtype=float)
    for k in range(1, K + 1):
        rows = rows_v[bin_idx_v == k]
        if rows.size == 0:
            continue

        z_left, z_right = edges[k - 1], edges[k]
        X_left = X.iloc[rows].copy()
        X_right = X.iloc[rows].copy()
        X_left[feat] = z_left
        X_right[feat] = z_right

        diff = _predict(model, X_right) - _predict(model, X_left)
        m = float(np.nanmean(diff))
        if not np.isfinite(m):
            raise ValueError(f"{feat}: non-finite mean diff in ALE bin {k} (all NaN/inf?)")
        delta[k - 1] = m

    g_right = np.cumsum(delta)
    g_samples = g_right[bin_idx_v - 1]
    center = float(np.nanmean(g_samples))
    if not np.isfinite(center):
        raise ValueError(f"{feat}: non-finite centering term in ALE (all NaN/inf?)")

    ale_bin = g_right - center
    return edges, ale_bin


def ale_1d_binary(model, X: pd.DataFrame, feat: str):
    """
    Binary ALE for a 0/1 (or bool) feature.
    Computes the average effect of toggling 0 -> 1 *within the subgroup where x=1*,
    then centers the two levels (0 and 1) by their marginal mean.
    Returns (edges, ale_bin) with edges=[-0.5,0.5,1.5] and ale_bin length 2.
    """
    xi = X[feat]

    # numeric view for validation/masking
    if xi.dtype == bool:
        x01 = xi.astype(np.int8).to_numpy()
    else:
        x01 = pd.to_numeric(xi, errors="coerce").to_numpy(dtype=float)

    vals = np.unique(x01[np.isfinite(x01)])
    if not set(vals.astype(int)).issubset({0, 1}):
        raise ValueError(f"{feat} is not binary 0/1 (unique values: {vals[:10]})")

    mask1 = (x01 == 1)
    edges = np.array([-0.5, 0.5, 1.5], dtype=float)

    if mask1.sum() == 0:
        return edges, np.array([0.0, 0.0], dtype=float)

    X1 = X.loc[mask1].copy()
    X0 = X.loc[mask1].copy()

    # preserve dtype consistent with training
    if xi.dtype == bool:
        X1[feat] = True
        X0[feat] = False
    else:
        X1[feat] = 1
        X0[feat] = 0

    d = _predict(model, X1) - _predict(model, X0)
    delta = float(np.nanmean(d))
    if not np.isfinite(delta):
        raise ValueError(f"{feat}: non-finite delta in binary ALE (all NaN/inf?)")

    g0, g1 = 0.0, delta
    p1 = float(mask1.mean())
    mean_g = (1.0 - p1) * g0 + p1 * g1
    return edges, np.array([g0 - mean_g, g1 - mean_g], dtype=float)
