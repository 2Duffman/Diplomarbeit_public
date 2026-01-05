from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_split
from model_training import load_all_models


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "data.csv"
TARGET = "avg_rating"
TRAIN_IDS_PATH = "train_ids.txt"
TEST_IDS_PATH = "test_ids.txt"
ARTIFACTS_DIR = "artifacts"

MODEL_NAME = "XGBRegressor"     
OUT_DIR = Path("pdp_2d_plots_cropped")

SEED = 0

BG_FROM = "train"  
BG_N = 1000

GRID_CONT = 15
GRID_DISC_MAX = 12
PERCENTILES = (0.01, 0.99)
CENTER_PDP = True


FEATURE_MAX: dict[str, float] = {
    "min_players": 5,
    "min_age": 25,
    "max_time": 240,
    "max_players": 10,
}
FEATURE_MIN: dict[str, float] = {
    "max_players": 1,   # avoid 0
    "min_players": 1,   # avoid 0
}

# PDP pairs to generate
PDP_PAIRS: List[Tuple[str, str]] = [
    # S2 (interpretable context)
    ("complexity", "max_time"),
    ("category_Wargame", "min_players"),
    ("max_time", "min_players"),
    ("category_Expansion for Base-game", "complexity"),
    ("mechanic_Cooperative_Game", "min_players"),
    ("category_Miniatures", "mechanic_Dice_Rolling"),

    # strong constraint interactions from S=8
    ("min_age", "min_time"),
    ("min_time", "max_players"),
    ("max_players", "min_players"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    s = s.strip()
    return re.sub(r"[^\w\-\.]+", "_", s)


def _to_dense_float32(Z):
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float32)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """Pipeline-safe prediction: always pass a DataFrame (ColumnTransformer-by-name works)."""
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


def _finite_numeric(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]


def feature_type(name: str, s: pd.Series) -> str:
    """
    Decide how to treat a feature for 2D PDP grids:
    - binary: bool or 0/1
    - discrete: small integer support (player counts etc.)
    - continuous: everything else
    """
    if is_binary_series(s):
        return "binary"

    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "continuous"

    vals = np.unique(x.to_numpy(dtype=float))
    vals = vals[np.isfinite(vals)]

    # treat obvious integer-ish small support as discrete
    if vals.size > 0 and vals.size <= GRID_DISC_MAX and np.all(np.isclose(vals, np.round(vals))):
        return "discrete"

    # override: player-count features should be discrete even if support is bigger
    if name in {"min_players", "max_players"}:
        return "discrete"

    return "continuous"


def _grid_binary_like(s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    if s.dtype == bool:
        g_model = np.array([False, True], dtype=bool)
        g_plot = np.array([0.0, 1.0], dtype=float)
        return g_model, g_plot
    g_model = np.array([0, 1], dtype=int)
    g_plot = g_model.astype(float)
    return g_model, g_plot


def _apply_min_max_caps(name: str, grid: np.ndarray) -> np.ndarray:
    """Apply FEATURE_MIN/FEATURE_MAX to a grid (works for discrete and continuous)."""
    g = np.asarray(grid)
    if name in FEATURE_MIN:
        g = g[g >= FEATURE_MIN[name]]
    if name in FEATURE_MAX:
        g = g[g <= FEATURE_MAX[name]]
    return g


def _grid_discrete(name: str, s: pd.Series, max_unique: int) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = _finite_numeric(x)
    if x.size == 0:
        g = np.array([0], dtype=int)
        return g, g.astype(float)

    vals = np.unique(np.round(x).astype(int))
    vals = np.sort(vals)
    vals = _apply_min_max_caps(name, vals)

    if vals.size == 0:
        # fallback: include at least min if available
        mn = int(FEATURE_MIN.get(name, 0))
        mx = int(FEATURE_MAX.get(name, mn + 1))
        vals = np.array([mn, mx], dtype=int) if mx > mn else np.array([mn], dtype=int)

    # cap if too many
    if vals.size > max_unique:
        qs = np.linspace(0, 1, max_unique)
        g = np.quantile(vals, qs)
        g = np.unique(np.round(g).astype(int))
    else:
        g = vals

    g_plot = g.astype(float)
    return g, g_plot


def _grid_continuous(name: str, s: pd.Series, n: int, percentiles=(0.01, 0.99)) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = _finite_numeric(x)
    if x.size < 2:
        g = np.array([0.0], dtype=float)
        return g, g

    lo, hi = np.quantile(x, percentiles)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        lo, hi = float(np.min(x)), float(np.max(x))

    # apply caps
    if name in FEATURE_MIN:
        lo = max(lo, float(FEATURE_MIN[name]))
    if name in FEATURE_MAX:
        hi = min(hi, float(FEATURE_MAX[name]))

    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        # final fallback
        lo = float(np.min(x))
        hi = float(np.max(x))
        if name in FEATURE_MIN:
            lo = max(lo, float(FEATURE_MIN[name]))
        if name in FEATURE_MAX:
            hi = min(hi, float(FEATURE_MAX[name]))
        if not (lo < hi):
            hi = lo + 1e-6

    grid = np.linspace(lo, hi, n)
    grid = np.unique(grid)
    return grid, grid


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=float)
    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (mids[0] - c[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.r_[first, mids, last]


# ──────────────────────────────────────────────────────────────────────────────
# PDP computation
# ──────────────────────────────────────────────────────────────────────────────
def pdp_2d(
    model,
    X_bg: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    grid_x_model: np.ndarray,
    grid_y_model: np.ndarray,
) -> np.ndarray:
    """
    PD(x,y) ≈ (1/M) Σ_i f(x, y, X_{-xy}^{(i)})
    Returns Z with shape (len(grid_y), len(grid_x)).
    """
    M = len(X_bg)
    Ny = len(grid_y_model)
    base_idx = np.tile(np.arange(M), Ny)

    Z = np.zeros((Ny, len(grid_x_model)), dtype=float)

    for ix, vx in enumerate(grid_x_model):
        rep = X_bg.iloc[base_idx].copy()
        rep[feat_x] = vx
        rep[feat_y] = np.repeat(grid_y_model, M)
        pred = _predict(model, rep).astype(float).reshape(Ny, M)
        Z[:, ix] = pred.mean(axis=1)

    if CENTER_PDP:
        Z = Z - float(np.mean(Z))
    return Z


def apply_feasibility_mask(
    Z: np.ndarray,
    feat_x: str,
    feat_y: str,
    grid_x_plot: np.ndarray,
    grid_y_plot: np.ndarray,
) -> np.ndarray:
    """
    Mask impossible combinations for (max_players, min_players).
    Sets those cells to NaN so they are blank in the plot.
    """
    fx, fy = feat_x, feat_y
    if {fx, fy} == {"max_players", "min_players"}:
        Zm = Z.copy()
        # build mesh in plot coordinates
        Xg, Yg = np.meshgrid(grid_x_plot, grid_y_plot)
        # Determine which axis is which variable
        if fx == "max_players":
            maxp = Xg
            minp = Yg
        else:
            maxp = Yg
            minp = Xg
        mask = maxp < minp
        Zm[mask] = np.nan
        return Zm
    return Z


def plot_pdp_2d(
    Z: np.ndarray,
    feat_x: str,
    feat_y: str,
    grid_x_plot: np.ndarray,
    grid_y_plot: np.ndarray,
    ftype_x: str,
    ftype_y: str,
    out_png: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    continuous_pair = (
        ftype_x == "continuous"
        and ftype_y == "continuous"
        and grid_x_plot.size >= 3
        and grid_y_plot.size >= 3
    )

    if continuous_pair:
        Xg, Yg = np.meshgrid(grid_x_plot, grid_y_plot)
        cf = ax.contourf(Xg, Yg, Z, levels=14)
        cbar = fig.colorbar(cf, ax=ax)
    else:
        x_edges = _edges_from_centers(grid_x_plot)
        y_edges = _edges_from_centers(grid_y_plot)
        pcm = ax.pcolormesh(x_edges, y_edges, Z, shading="auto")
        cbar = fig.colorbar(pcm, ax=ax)

        if grid_x_plot.size <= 12:
            ax.set_xticks(grid_x_plot)
        if grid_y_plot.size <= 12:
            ax.set_yticks(grid_y_plot)

        # annotate if 2×2
        if grid_x_plot.size <= 2 and grid_y_plot.size <= 2:
            for iy in range(Z.shape[0]):
                for ix in range(Z.shape[1]):
                    if np.isfinite(Z[iy, ix]):
                        ax.text(grid_x_plot[ix], grid_y_plot[iy], f"{Z[iy, ix]:.2f}",
                                ha="center", va="center", fontsize=9)

    cbar.set_label("Centered PD" if CENTER_PDP else "Partial dependence")
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(
        csv_path=DATASET_PATH,
        target_col=TARGET,
        train_ids_path=TRAIN_IDS_PATH,
        test_ids_path=TEST_IDS_PATH,
    )

    models = load_all_models(ARTIFACTS_DIR)
    print("Loaded models:", list(models.keys()), flush=True)
    if MODEL_NAME not in models:
        raise KeyError(f"MODEL_NAME={MODEL_NAME} not found. Available: {list(models.keys())}")

    est = models[MODEL_NAME]
    model_dir = OUT_DIR / safe_name(MODEL_NAME)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_bg_full = X_train if BG_FROM.lower() == "train" else X_test
    X_bg = X_bg_full.sample(n=min(BG_N, len(X_bg_full)), random_state=SEED).copy()

    for a, b in PDP_PAIRS:
        if a not in X_bg.columns or b not in X_bg.columns:
            raise KeyError(f"Missing columns for pair ({a}, {b}).")

    print(f"\nComputing CROPPED 2D PDPs for model: {MODEL_NAME}", flush=True)
    print(f"Background: {BG_FROM} (n={len(X_bg)}), GRID_CONT={GRID_CONT}", flush=True)
    print("Caps:", FEATURE_MIN, FEATURE_MAX, flush=True)

    for (fx, fy) in PDP_PAIRS:
        sx, sy = X_bg[fx], X_bg[fy]
        tx, ty = feature_type(fx, sx), feature_type(fy, sy)

        # grids
        if tx == "binary":
            gx_model, gx_plot = _grid_binary_like(sx)
        elif tx == "discrete":
            gx_model, gx_plot = _grid_discrete(fx, sx, GRID_DISC_MAX)
        else:
            gx_model, gx_plot = _grid_continuous(fx, sx, GRID_CONT, PERCENTILES)

        if ty == "binary":
            gy_model, gy_plot = _grid_binary_like(sy)
        elif ty == "discrete":
            gy_model, gy_plot = _grid_discrete(fy, sy, GRID_DISC_MAX)
        else:
            gy_model, gy_plot = _grid_continuous(fy, sy, GRID_CONT, PERCENTILES)

        # sort plot grids
        x_order = np.argsort(gx_plot)
        y_order = np.argsort(gy_plot)
        gx_model, gx_plot = gx_model[x_order], gx_plot[x_order]
        gy_model, gy_plot = gy_model[y_order], gy_plot[y_order]

        print(f"  -> ({fx} × {fy}) types=({tx},{ty}) grid=({len(gx_model)}×{len(gy_model)})", flush=True)

        # compute + mask infeasible
        Z = pdp_2d(est, X_bg, fx, fy, gx_model, gy_model)
        Z = apply_feasibility_mask(Z, fx, fy, gx_plot, gy_plot)

        out_png = model_dir / f"PDP2D__{safe_name(MODEL_NAME)}__{safe_name(fx)}__X__{safe_name(fy)}.png"
        title = f"2D PDP — {MODEL_NAME}: {fx} × {fy}"
        plot_pdp_2d(Z, fx, fy, gx_plot, gy_plot, tx, ty, out_png, title)

        print("     saved", out_png, flush=True)

    print("\nDone. Saved plots in:", model_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
