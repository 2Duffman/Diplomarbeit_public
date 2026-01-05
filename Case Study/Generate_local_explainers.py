from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

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

OUT_DIR = Path("local_explanations")

SHAP_TAG = "shap_modelimpute"


CASES: List[int] = [
    432586,  # Thunder Road: Vendetta – Turbo Tina's Swag Pack
    11155,   # How to Be a Complete Bastard
    295770,  # Frosthaven
    11028,   # Lotto
    24278,   # Karambolage
    187740,  # Dominion: Fan-Edition I
    19377,   # Rock Paper Scissors Game
    36218,   # Dominion (canonical row)
]

# SHAP settings
BACKGROUND_SIZE = 200
MAX_EVALS = 2000
TOPK_DISPLAY = 15
RANDOM_SEED = 0

# ICE settings
ICE_FEATURES = [
    "year",
    "complexity",
    "min_time",
    "max_time",
    "min_players",
    "max_players",
    "min_age",
]
ICE_POINTS = 25

# ICE constraints
ICE_CONSTRAINTS = {
    "max_players": {"upper": 10},
    "max_time": {"upper": 240},
    "min_time": {"upper": 240},
    "min_age": {"upper": 25},
    "min_players": {"upper": 5},
    "year": {"lower": 1960},
}

# Long feature display-name overrides (ONLY for SHAP plots / SHAP CSV display)
LONG_GRID_FEATURE = (
    "family_Components_5_x_5_Grids_6_x_6_Grids_7_x_7_Grids_4_x_4_Grids_8_x_8_Grids_"
    "9_x_9_Grids_10_x_10_Grids_3_x_3_Grids_11_x_11_Grids_12_x_12_Grids_HexHex5_Grids_"
    "15_x_15_Grids_HexHex4_Grids_HexHex7_Grids_13_x_13_Grids_5_x_6_Grids_14_x_14_Grids_"
    "HexHex3_Grids_HexHex8_Grids_16_x_16_Grids_HexHex6_Grids_19_x_19_Grids_6_x_7_Grids_"
    "7_x_8_Grids_6_x_8_Grids_17_x_17_Grids_7_x_9_Grids_8_x_10_Grids_4_x_8_Grids_"
    "9_x_10_Grids_20_x_20_Grids_4x4x4_Grids_8_x_9_Grids_HexHex10_Grids_8_x_11_Grids_"
    "4_x_5_Grids_5_x_7_Grids_2_x_2_Grids_9_x_12_Grids_3x3x3_Grids_4_x_6_Grids_"
    "8_x_12_Grids_5x5x5_Grids_10_x_20_Grids_5_x_8_Grids_18_x_18_Grids_11_x_12_Grids_"
    "8_x_16_Grids_7_x_10_Grids_14_x_15_Grids_12_x_18_Grids_9_x_11_Grids_10_x_11_Grids_"
    "7_x_13_Grids_6_x_9_Grids_3_x_6_Grids_2_x_3_Grids_7_x_14_Grids_15_x_17_Grids_"
    "14_x_16_Grids_5_x_10_Grids_21_x_21_Grids_10_x_12_Grids_24_x_24_Grids_10_x_15_Grids_"
    "3_x_5_Grids_HexHex11_Grids_11_x_13_Grids_4_x_7_Grids_12_x_14_Grids_6_x_12_Grids_"
    "12_x_16_Grids_4x4x4x4_Grids_12_x_13_Grids"
)

LONG_ANIMALS_FEATURE = (
    "family_Animals_Wolves_Bears_Foxes_Elephants_Pandas_Deer_/_Antelope_Tigers_Lions_"
    "Hippopotamuses_Raccoons_Llamas_Giraffes_Rhinoceroses_Kangaroos_Mammoths_Zebras_"
    "Koalas_Moose_/_Elk_Sloths_Otters_Wombats_Coyotes"
)

LONG_MECHANIC_DIE_FEATURE = "mechanic_Die_Icon_Resolution_Re-rolling_and_Locking_Different_Dice_Movement"

LONG_MECHANIC_PURCHASE_FEATURE = "mechanic_Delayed_Purchase_Loans_Automatic_Resource_Growth_Ownership_Investment"

FEATURE_NAME_ALIASES: Dict[str, str] = {
    LONG_GRID_FEATURE: "family_Components_Grids",
    LONG_ANIMALS_FEATURE: "family_Animals_Mammals",
    LONG_MECHANIC_DIE_FEATURE: "mechanic_Die_Icon",
    LONG_MECHANIC_PURCHASE_FEATURE: "mechanic_Delayed_Purchase",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in " ._-–—()" else "_" for ch in str(s)).strip()


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def save_png(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _to_dense_float32(Z):
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=np.float32)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict for sklearn Pipelines and your KerasPipeline."""
    if isinstance(model, KerasPipeline):
        Xt = model.preprocessor.transform(X)
        Xt = _to_dense_float32(Xt)
        y = model.model.predict(Xt, verbose=0)
        return np.asarray(y).reshape(-1)

    y = model.predict(X)
    return np.asarray(y).reshape(-1)


def build_case_table(raw: pd.DataFrame, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    """
    Compact overview table of chosen cases using alignment via X_test.index.
    """
    meta = raw.loc[X_test.index, ["name", "bgg_id"]].copy()
    meta["row_index"] = X_test.index.to_numpy()

    y_true = y_test.to_numpy(dtype=float) if isinstance(y_test, pd.Series) else np.asarray(y_test, dtype=float)
    meta["y_true"] = y_true

    chosen = meta[meta["bgg_id"].isin(CASES)].copy()
    order = {bid: i for i, bid in enumerate(CASES)}
    chosen["_order"] = chosen["bgg_id"].map(order)
    chosen = chosen.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    missing = [bid for bid in CASES if bid not in set(chosen["bgg_id"].tolist())]
    if missing:
        print("WARNING: these bgg_ids were not found in the test set:", missing)

    return chosen


# ──────────────────────────────────────────────────────────────────────────────
# Model-imputer extraction for SHAP
# ──────────────────────────────────────────────────────────────────────────────
def _get_column_transformer(model) -> ColumnTransformer:
    """
    Return the fitted ColumnTransformer (preprocessor) for either sklearn Pipeline
    models (expects a step called 'prep' OR any ColumnTransformer step) or KerasPipeline.
    """
    if isinstance(model, KerasPipeline):
        prep = model.preprocessor
        if not isinstance(prep, ColumnTransformer):
            raise TypeError(f"KerasPipeline.preprocessor is not a ColumnTransformer: {type(prep)}")
        return prep

    if isinstance(model, Pipeline):
        # preferred: 'prep'
        if "prep" in model.named_steps and isinstance(model.named_steps["prep"], ColumnTransformer):
            return model.named_steps["prep"]

        # fallback: find first ColumnTransformer step
        for _, step in model.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return step

    raise TypeError(f"Unsupported model type for preprocessor extraction: {type(model)}")


def _get_num_imputer_and_cols(prep: ColumnTransformer) -> Tuple[List[str], SimpleImputer]:
    """
    Find the 'num' transformer inside the ColumnTransformer and return:
      (num_cols, fitted SimpleImputer)
    Assumes your training code names the numeric transformer 'num' and the imputer step 'imp'.
    """
    num_cols: Optional[List[str]] = None
    num_transformer = None

    for name, transformer, cols in prep.transformers_:
        if name == "num":
            num_cols = list(cols)
            num_transformer = transformer
            break

    if num_cols is None or num_transformer is None:
        raise KeyError("Could not find a 'num' transformer in the ColumnTransformer.")

    # num_transformer is typically a Pipeline(imp=SimpleImputer, sc=StandardScaler)
    imp: Optional[SimpleImputer] = None
    if isinstance(num_transformer, Pipeline):
        if "imp" in num_transformer.named_steps and isinstance(num_transformer.named_steps["imp"], SimpleImputer):
            imp = num_transformer.named_steps["imp"]
    elif isinstance(num_transformer, SimpleImputer):
        imp = num_transformer

    if imp is None:
        raise KeyError("Could not find fitted SimpleImputer (step 'imp') inside the 'num' transformer.")

    return num_cols, imp


def impute_like_model_to_numeric(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Make a numeric, finite DataFrame for SHAP that matches the model's numeric imputation:
      - Fill NaNs in numeric columns using the model's fitted SimpleImputer medians.
      - Convert bool -> 0/1.
      - Coerce any remaining non-numeric cols to numeric (coerce->NaN), fill remaining NaNs with 0.
      - float32 output.
    """
    prep = _get_column_transformer(model)
    num_cols, imp = _get_num_imputer_and_cols(prep)

    medians = pd.Series(imp.statistics_, index=num_cols)

    Xn = X.copy()

    # Ensure numeric cols are numeric so NaNs are detected
    for c in num_cols:
        if c in Xn.columns:
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")

    # Apply model-fitted imputation
    Xn[num_cols] = Xn[num_cols].fillna(medians)

    # bool -> int
    bool_cols = Xn.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        Xn[bool_cols] = Xn[bool_cols].astype(int)

    # Coerce any remaining non-numeric columns
    non_num_cols = Xn.columns.difference(Xn.select_dtypes(include=["number"]).columns)
    for c in non_num_cols:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")

    # Remaining NaNs (if any) -> 0 (this should be rare)
    Xn = Xn.fillna(0.0)

    return Xn.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# SHAP
# ──────────────────────────────────────────────────────────────────────────────
def make_predict_fn(model, feature_names_internal: List[str]):
    """
    SHAP permutation explainer passes numpy arrays; wrap into a DataFrame so your
    sklearn pipeline / KerasPipeline sees the right columns (original names).
    """
    def predict_fn(X):
        if isinstance(X, pd.DataFrame):
            Xdf = X
        else:
            Xdf = pd.DataFrame(np.asarray(X), columns=feature_names_internal)
        return _predict(model, Xdf)
    return predict_fn


def compute_shap_for_instance(
    model_name: str,
    model,
    X_background_num: pd.DataFrame,   # numeric, model-imputed
    x_instance_num: pd.DataFrame,     # numeric, model-imputed (1 row)
    x_instance_raw: pd.DataFrame,     # raw (for x_value export)
    out_prefix: Path,
    feature_names_display: List[str],
) -> None:
    """
    Saves (incrementally) with a new tag so it doesn't conflict with old SHAP files:
      - *.{SHAP_TAG}_values.csv
      - *.{SHAP_TAG}_waterfall.png
      - *.{SHAP_TAG}_bar.png
    """
    values_path = out_prefix.with_name(out_prefix.name + f".{SHAP_TAG}_values.csv")
    wf_path = out_prefix.with_name(out_prefix.name + f".{SHAP_TAG}_waterfall.png")
    bar_path = out_prefix.with_name(out_prefix.name + f".{SHAP_TAG}_bar.png")

    if values_path.exists() and wf_path.exists() and bar_path.exists():
        print(f"[SKIP] {SHAP_TAG} exists: {out_prefix.name}")
        return

    feature_names_internal = list(X_background_num.columns)
    predict_fn = make_predict_fn(model, feature_names_internal)

    # Numeric masker/background avoids np.isfinite(object) crashes
    masker = shap.maskers.Independent(X_background_num.to_numpy())

    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
        feature_names=feature_names_display,  # display names for plots
    )

    try:
        explanation = explainer(x_instance_num.to_numpy(), max_evals=MAX_EVALS)
    except TypeError:
        explanation = explainer(x_instance_num.to_numpy())

    exp0 = explanation[0]
    base_val = float(np.array(exp0.base_values).reshape(-1)[0])
    pred_val = float(predict_fn(x_instance_num)[0])

    # Export SHAP table (keep both original+display names)
    raw_vals = x_instance_raw.iloc[0].to_numpy()
    raw_vals = np.array([v if (v is None or isinstance(v, (int, float, np.number))) else str(v) for v in raw_vals])

    df_out = pd.DataFrame({
        "feature_original": feature_names_internal,
        "feature_display": feature_names_display,
        "x_value_raw": raw_vals,
        # this is what SHAP actually saw (already model-imputed, numeric)
        "x_value_used_by_shap": x_instance_num.iloc[0].to_numpy(),
        "shap_value": np.asarray(exp0.values),
    })
    df_out.insert(0, "model", model_name)
    df_out.insert(1, "base_value", base_val)
    df_out.insert(2, "prediction", pred_val)

    save_csv(values_path, df_out)

    # Waterfall plot (uses display names)
    plt.figure()
    shap.plots.waterfall(exp0, max_display=TOPK_DISPLAY, show=False)
    save_png(wf_path)

    # Bar plot (uses display names)
    plt.figure()
    shap.plots.bar(exp0, max_display=TOPK_DISPLAY, show=False)
    save_png(bar_path)

    print(f"[DONE] {SHAP_TAG}: {out_prefix.name}")


# ──────────────────────────────────────────────────────────────────────────────
# ICE
# ──────────────────────────────────────────────────────────────────────────────
def _get_ice_bounds(feature: str, values: np.ndarray) -> tuple[float, float]:
    """Compute bounds from constraints if provided, otherwise from data min/max."""
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    c = ICE_CONSTRAINTS.get(feature, {})
    if "lower" in c:
        lo = max(lo, float(c["lower"]))
    if "upper" in c:
        hi = min(hi, float(c["upper"]))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _make_ice_grid(train_col: pd.Series, feature: str, n_points: int = 25) -> np.ndarray:
    """
    Robust ICE grid:
      - apply constraints (clip to bounds)
      - use unique observed values if not too many
      - else use quantiles
      - if quantiles collapse (ties), fall back to linspace
      - if column is integer-like, snap to integers
    """
    vals = pd.to_numeric(train_col, errors="coerce").to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    lo, hi = _get_ice_bounds(feature, vals)
    vals = np.clip(vals, lo, hi)

    uniq = np.unique(vals)
    if uniq.size <= n_points:
        grid = uniq
    else:
        qs = np.linspace(0.0, 1.0, n_points)
        grid = np.unique(np.quantile(vals, qs))
        if grid.size < max(5, n_points // 3):  # too many ties -> collapse
            grid = np.linspace(lo, hi, n_points)

    # integer-like?
    if np.nanmax(np.abs(vals - np.round(vals))) < 1e-6:
        grid = np.unique(np.round(grid)).astype(float)

    grid = np.sort(grid.astype(float))
    return grid


def compute_ice_for_instance(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    x_instance: pd.DataFrame,  # 1-row DataFrame (raw)
    out_prefix: Path,
    features: List[str],
    n_points: int = 25,
) -> None:
    """
    Single-instance ICE (NICE curve): keep all other features fixed at x_instance,
    vary one feature over a grid taken from the training distribution (with constraints).
    Writes per feature:
      - *.ice_<feature>.csv
      - *.ice_<feature>.png
    """
    for feat in features:
        if feat not in X_train.columns or feat not in x_instance.columns:
            continue

        csv_path = out_prefix.with_name(out_prefix.name + f".ice_{feat}.csv")
        png_path = out_prefix.with_name(out_prefix.name + f".ice_{feat}.png")

        if csv_path.exists() and png_path.exists():
            print(f"[SKIP] ICE exists: {out_prefix.name} / {feat}")
            continue

        grid = _make_ice_grid(X_train[feat], feat, n_points=n_points)

        # Batch of counterfactuals
        X_batch = pd.concat([x_instance] * len(grid), ignore_index=True)
        X_batch[feat] = grid

        preds = _predict(model, X_batch).astype(float)

        # Original point (clipped only for plotting)
        x0 = float(pd.to_numeric(x_instance.iloc[0][feat], errors="coerce"))
        train_vals = pd.to_numeric(X_train[feat], errors="coerce").to_numpy(dtype=float)
        lo, hi = _get_ice_bounds(feat, train_vals)
        x0_plot = float(np.clip(x0, lo, hi))
        y0 = float(_predict(model, x_instance)[0])

        df_ice = pd.DataFrame({
            "model": model_name,
            "feature": feat,
            "grid_value": grid,
            "prediction": preds,
            "x_original": x0,
            "x_original_clipped_for_plot": x0_plot,
            "pred_original": y0,
            "grid_lower": lo,
            "grid_upper": hi,
        })
        save_csv(csv_path, df_ice)

        # Plot
        plt.figure()
        plt.plot(grid, preds)
        plt.scatter([x0_plot], [y0])
        plt.xlim(lo, hi)
        plt.xlabel(feat)
        plt.ylabel("prediction")
        save_png(png_path)

        print(f"[DONE] ICE: {out_prefix.name} / {feat}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    np.random.seed(RANDOM_SEED)
    ensure_dir(OUT_DIR)

    # Load split (authoritative alignment/order)
    X_train, X_test, y_train, y_test = load_split(
        csv_path=DATASET_PATH,
        target_col=TARGET,
        train_ids_path=TRAIN_IDS_PATH,
        test_ids_path=TEST_IDS_PATH,
    )

    # Load models
    models = load_all_models(ARTIFACTS_DIR)
    print("Available model keys:", list(models.keys()))
    if MODEL_XGB_KEY not in models or MODEL_NN3_KEY not in models:
        raise KeyError(
            f"Need MODEL_XGB_KEY={MODEL_XGB_KEY} and MODEL_NN3_KEY={MODEL_NN3_KEY}. "
            f"Available: {list(models.keys())}"
        )

    model_map: Dict[str, object] = {
        "xgb": models[MODEL_XGB_KEY],
        "nn3": models[MODEL_NN3_KEY],
    }

    # Sample ONE raw background (same rows), then impute per-model for SHAP
    X_bg_raw = X_train.sample(n=min(BACKGROUND_SIZE, len(X_train)), random_state=RANDOM_SEED)

    # Raw data for chosen-case overview (aligned via X_test.index)
    raw = pd.read_csv(DATASET_PATH)
    case_table = build_case_table(raw, X_test, y_test)



    # Precompute per-model SHAP background + feature display names
    shap_background_by_model: Dict[str, pd.DataFrame] = {}
    shap_display_names_by_model: Dict[str, List[str]] = {}

    for mname, model in model_map.items():
        X_bg_num = impute_like_model_to_numeric(model, X_bg_raw)
        shap_background_by_model[mname] = X_bg_num
        feature_names_internal = list(X_bg_num.columns)
        shap_display_names_by_model[mname] = [FEATURE_NAME_ALIASES.get(c, c) for c in feature_names_internal]

    # Iterate cases
    for _, row in case_table.iterrows():
        bgg_id = int(row["bgg_id"])
        name = str(row.get("name", f"bgg_{bgg_id}"))
        name_fs = safe_filename(name)

        idx = int(row["row_index"])
        x_raw = X_test.loc[[idx]]

        for mname, model in model_map.items():
            out_prefix = OUT_DIR / f"{mname}__{bgg_id}__{name_fs}"

           
            try:
                X_bg_num = shap_background_by_model[mname]
                x_num = impute_like_model_to_numeric(model, x_raw)

                compute_shap_for_instance(
                    model_name=mname,
                    model=model,
                    X_background_num=X_bg_num,
                    x_instance_num=x_num,
                    x_instance_raw=x_raw,
                    out_prefix=out_prefix,
                    feature_names_display=shap_display_names_by_model[mname],
                )
            except Exception as e:
                err_path = out_prefix.with_name(out_prefix.name + f".{SHAP_TAG}_ERROR.txt")
                save_text(err_path, f"{SHAP_TAG} failed for {mname} / {bgg_id} / {name}\n\n{repr(e)}\n")
                print(f"[ERROR] {SHAP_TAG} failed: {mname} / {bgg_id} ({name}). Wrote {err_path.name}")

            # --- ICE (raw; uses model pipeline directly; unchanged) ---
            try:
                compute_ice_for_instance(
                    model_name=mname,
                    model=model,
                    X_train=X_train,          # use train distribution for grids
                    x_instance=x_raw,         # keep other features fixed to this instance
                    out_prefix=out_prefix,
                    features=ICE_FEATURES,
                    n_points=ICE_POINTS,
                )
            except Exception as e:
                err_path = out_prefix.with_suffix(".ice_ERROR.txt")
                save_text(err_path, f"ICE failed for {mname} / {bgg_id} / {name}\n\n{repr(e)}\n")
                print(f"[ERROR] ICE failed: {mname} / {bgg_id} ({name}). Wrote {err_path.name}")

    print("[DONE] Finished (or attempted) SHAP + ICE. Results in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
