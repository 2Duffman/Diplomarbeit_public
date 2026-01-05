# model_training.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from data_utils import load_split


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def make_preprocessor(
    X_frame: pd.DataFrame,
    scale_numeric: bool,
    scale_boolean: bool = False,
) -> Tuple[ColumnTransformer, pd.Index, pd.Index]:
    """
    Builds a ColumnTransformer that:
      - imputes numeric with median and optionally scales
      - passes boolean (and other non-numeric) columns through, or optionally scales
    """
    num_cols = X_frame.select_dtypes(include="number").columns
    bool_cols = X_frame.columns.difference(num_cols)  # non-numerics

    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("sc", StandardScaler()))

    if scale_boolean:
        bool_transformer = Pipeline([("sc_bool", StandardScaler(with_mean=False))])
    else:
        bool_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("bool", bool_transformer, bool_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, bool_cols


# ──────────────────────────────────────────────────────────────────────────────
# Keras wrapper so NNs behave like a fitted sklearn Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class KerasPipeline:
    """
    Minimal wrapper that mimics a sklearn Pipeline:
    - stores a fitted preprocessor and a trained Keras model
    - exposes predict(X) that handles preprocessing internally
    """
    def __init__(self, preprocessor: ColumnTransformer, model: tf.keras.Model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            Xt = self.preprocessor.transform(X)
        else:
            # assume already preprocessed
            Xt = X
        return self.model.predict(Xt, verbose=0).ravel()

    # optional convenience if you ever want to fine-tune on new data
    def fit(self, X: pd.DataFrame, y: np.ndarray, **fit_kwargs):
        Xt = self.preprocessor.fit_transform(X)
        self.model.fit(Xt, y, **fit_kwargs)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    mse: float
    mae: float
    r2: float
    adjusted_r2: float
    n_test: int
    p_features: int
    time_seconds: Optional[float] = None

def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, p_features: int
) -> Metrics:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    # adjusted R^2 uses the *original* feature count (pre-transform)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p_features - 1)
    return Metrics(mse=mse, mae=mae, r2=r2, adjusted_r2=adj_r2, n_test=n, p_features=p_features)


# ──────────────────────────────────────────────────────────────────────────────
# Model builders 
# ──────────────────────────────────────────────────────────────────────────────

def get_default_models() -> Dict[str, object]:
    xgb_params = {
        "n_estimators": 592,
        "max_depth": 36,
        "learning_rate": 0.03424916510690319,
        "subsample": 0.8732399913040625,
        "colsample_bytree": 0.3595628004141089,
        "gamma": 0.03045351400058005,
        "reg_alpha": 2.7000220599828153,
        "reg_lambda": 6.977276466072381,
        "min_child_weight": 7,
    }
    ada_base_tree = DecisionTreeRegressor(max_depth=30, random_state=1503)

    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=0.0012),
        "XGBRegressor": xgb.XGBRegressor(**xgb_params, n_jobs=-1, random_state=1503),
        "AdaBoostRegressor": AdaBoostRegressor(estimator=ada_base_tree, n_estimators=100, random_state=1503),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=433, max_depth=34, bootstrap=False,
            min_samples_split=8, min_samples_leaf=1, max_features=0.16, random_state=1503
        ),
    }

def build_nn_3layer(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(166, activation="elu"),
        tf.keras.layers.Dropout(0.46845696300996187),
        tf.keras.layers.Dense(99, activation="elu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0022962224551903894),
                  loss="mean_squared_error")
    return model

def build_nn_1layer(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(6963, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.303e-05),
                  loss="mean_squared_error")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Fit functions (return fitted, ready-to-predict models)
# ──────────────────────────────────────────────────────────────────────────────

def _fit_sklearn_pipeline(
    base_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_numeric: bool,
    scale_boolean: bool = False,
) -> Pipeline:
    preprocessor, _, _ = make_preprocessor(X_train, scale_numeric, scale_boolean)
    pipe = Pipeline([("prep", preprocessor), ("model", base_model)])
    pipe.fit(X_train, y_train)
    return pipe

def fit_dummyregressor(data, target: str = "avg_rating") -> Pipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    return _fit_sklearn_pipeline(DummyRegressor(strategy="mean"), X_train, y_train, scale_numeric=False)

def fit_linearregression(data, target: str = "avg_rating") -> Pipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    return _fit_sklearn_pipeline(LinearRegression(), X_train, y_train, scale_numeric=True)

def fit_lasso(data, target: str = "avg_rating", alpha: float = 0.0012) -> Pipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    return _fit_sklearn_pipeline(Lasso(alpha=alpha), X_train, y_train, scale_numeric=True)

def fit_xgbregressor(data, target: str = "avg_rating", params: Optional[Dict] = None) -> Pipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    if params is None:
        params = {
            "n_estimators": 592, "max_depth": 36, "learning_rate": 0.03424916510690319,
            "subsample": 0.8732399913040625, "colsample_bytree": 0.3595628004141089,
            "gamma": 0.03045351400058005, "reg_alpha": 2.7000220599828153,
            "reg_lambda": 6.977276466072381, "min_child_weight": 7,
            "n_jobs": -1, "random_state": 1503,
        }
    model = xgb.XGBRegressor(**params)
    return _fit_sklearn_pipeline(model, X_train, y_train, scale_numeric=False)

def fit_adaboostregressor(data, target: str = "avg_rating") -> Pipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    ada_base = DecisionTreeRegressor(max_depth=30, random_state=1503)
    model = AdaBoostRegressor(estimator=ada_base, n_estimators=100, random_state=1503)
    return _fit_sklearn_pipeline(model, X_train, y_train, scale_numeric=False)

def fit_randomforestregressor(data, target: str = "avg_rating") -> Pipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    model = RandomForestRegressor(
        n_estimators=433, max_depth=34, bootstrap=False,
        min_samples_split=8, min_samples_leaf=1, max_features=0.16, random_state=1503
    )
    return _fit_sklearn_pipeline(model, X_train, y_train, scale_numeric=False)


# ──────────────────────────────────────────────────────────────────────────────
# NNs – return a KerasPipeline wrapper so usage matches sklearn pipelines
# ──────────────────────────────────────────────────────────────────────────────

def fit_nn_3layers(
    data, target: str = "avg_rating", epochs: int = 84, batch_size: int = 64,
    scale_boolean: bool = False
) -> KerasPipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    preprocessor, _, _ = make_preprocessor(X_train, scale_numeric=True, scale_boolean=scale_boolean)
    Xtr = preprocessor.fit_transform(X_train)
    model = build_nn_3layer(input_dim=Xtr.shape[1])
    model.fit(Xtr, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return KerasPipeline(preprocessor, model)

def fit_nn_1layer(
    data, target: str = "avg_rating", epochs: int = 17, batch_size: int = 64,
    scale_boolean: bool = False
) -> KerasPipeline:
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    preprocessor, _, _ = make_preprocessor(X_train, scale_numeric=True, scale_boolean=scale_boolean)
    Xtr = preprocessor.fit_transform(X_train)
    model = build_nn_1layer(input_dim=Xtr.shape[1])
    model.fit(Xtr, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return KerasPipeline(preprocessor, model)


# ──────────────────────────────────────────────────────────────────────────────
# One-call “train & evaluate” helpers with optional MLflow logging
# ──────────────────────────────────────────────────────────────────────────────

def _maybe_log_mlflow(params: Dict, metrics: Metrics, model_name: str, dataset_name: str | None):
    try:
        import mlflow, time
        with mlflow.start_run(nested=True):
            mlflow.log_param("model", model_name)
            if dataset_name:
                mlflow.log_param("dataset", dataset_name)
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("MSE", metrics.mse)
            mlflow.log_metric("MAE", metrics.mae)
            mlflow.log_metric("R2", metrics.r2)
            mlflow.log_metric("Adjusted_R2", metrics.adjusted_r2)
            if metrics.time_seconds is not None:
                mlflow.log_metric("Time_seconds", metrics.time_seconds)
    except Exception:
        # Keep silent if MLflow isn't configured; you can surface/log if preferred
        pass

def train_and_evaluate(
    data,
    model: Pipeline | KerasPipeline,
    target: str = "avg_rating",
    enable_mlflow: bool = False,
    model_name: str = "model",
    dataset_name: Optional[str] = None,
) -> Tuple[Pipeline | KerasPipeline, Metrics]:
    """
    Split data, evaluate a pre-fitted or unfitted model:
    - If the model is unfitted sklearn Pipeline, it will be fitted on X_train.
    - If it's a KerasPipeline returned by fit_nn_* it's already fitted.
    Returns the fitted model and metrics on X_test.
    """
    import time
    X_train, X_test, y_train, y_test = load_split(csv_path= data, target_col=target)
    p_features = X_test.shape[1]
    print("fitting and evaluating", model_name)
    start = time.time()

    # If sklearn Pipeline and not fitted yet, fit here
    needs_fit = isinstance(model, Pipeline) and not hasattr(model, "named_steps")  # conservative: fit anyway
    try:
        # try a quick predict to see if it's fitted
        if isinstance(model, Pipeline):
            _ = model.predict(X_test.head(1))
            needs_fit = False
    except Exception:
        needs_fit = True

    if needs_fit:
        if isinstance(model, Pipeline):
            model.fit(X_train, y_train)
        elif isinstance(model, KerasPipeline):
            # already fitted by our fit_nn_* builders
            pass

    y_pred = model.predict(X_test)
    end = time.time()

    metrics = evaluate_predictions(y_test, y_pred, p_features)
    metrics.time_seconds = (end - start)

    if enable_mlflow:
        _maybe_log_mlflow(params={}, metrics=metrics, model_name=model_name, dataset_name=dataset_name)

    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: train all defaults & report
# ──────────────────────────────────────────────────────────────────────────────

def train_all_default_models(
    data,
    target: str = "avg_rating",
    enable_mlflow: bool = False,
    dataset_name: Optional[str] = None,
) -> Dict[str, Tuple[Pipeline | KerasPipeline, Metrics]]:
    """
    Trains your configured set (Dummy, Linear, Lasso, XGB, AdaBoost, RF, NN-3, NN-1)
    and returns a dict: name -> (fitted_model, metrics)
    """
    results = {}
    print("Training all default models...")

    # tree-like / no scaling
    for name in ["DummyRegressor", "XGBRegressor", "AdaBoostRegressor", "RandomForestRegressor"]:
        print(f"Training {name}...")
        model = globals()[f"fit_{name.lower()}"](data, target=target)
        model, metrics = train_and_evaluate(
            data, model, target=target, enable_mlflow=enable_mlflow,
            model_name=name, dataset_name=dataset_name
        )
        results[name] = (model, metrics)

    # linear / scale numeric
    for name in ["LinearRegression", "Lasso"]:
        model = globals()[f"fit_{name.lower()}"](data, target=target)
        print(f"Training {name}...")
        model, metrics = train_and_evaluate(
            data, model, target=target, enable_mlflow=enable_mlflow,
            model_name=name, dataset_name=dataset_name
        )
        results[name] = (model, metrics)

    # neural nets
    nn3 = fit_nn_3layers(data, target=target)
    print("Training NN_3_layers...")
    nn3, m3 = train_and_evaluate(
        data, nn3, target=target, enable_mlflow=enable_mlflow,
        model_name="NN_3_layers", dataset_name=dataset_name
    )
    results["NN_3_layers"] = (nn3, m3)

    nn1 = fit_nn_1layer(data, target=target)
    print("Training NN_1_layer...")
    nn1, m1 = train_and_evaluate(
        data, nn1, target=target, enable_mlflow=enable_mlflow,
        model_name="NN_1_layer", dataset_name=dataset_name
    )
    results["NN_1_layer"] = (nn1, m1)

    return results


def save_estimator(estimator, path: str | Path):
    """
    Save any estimator:
      - sklearn Pipeline/estimator -> single .pkl via joblib
      - KerasPipeline -> folder with preprocessor.pkl + keras_model.keras (Keras v3 native)
      - raw xgboost.XGBRegressor -> prefer xgboost native .json
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # KerasPipeline: save pieces
    if isinstance(estimator, KerasPipeline):
        # ensure a directory path (e.g., artifacts/NN_3_layers)
        path.mkdir(parents=True, exist_ok=True)

        # 1) preprocessor
        joblib.dump(estimator.preprocessor, path / "preprocessor.pkl")

        # 2) keras model (native Keras format requires .keras or .h5 in Keras 3)
        keras_file = path / "keras_model.keras"
        estimator.model.save(keras_file, include_optimizer=True)

        # 3) simple metadata
        meta = {"type": "KerasPipeline", "format": "keras_v3", "file": keras_file.name}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return

    # Raw XGBRegressor (not wrapped in Pipeline)
    if isinstance(estimator, xgb.XGBRegressor):
        # Prefer native format for portability (".json" is typical)
        estimator.save_model(str(path))
        return

    # Fallback: sklearn Pipeline/estimator
    joblib.dump(estimator, path)
def load_estimator(path: str | Path):
    """
    Load estimator saved by save_estimator.
    Supports:
      - sklearn joblib (.pkl/.joblib)
      - xgboost native (.json/.ubj/.bin/.model)
      - KerasPipeline folder with 'preprocessor.pkl' + 'keras_model.keras'
      - Backward-compat: Keras SavedModel folder 'keras_model' if present
    """
    path = Path(path)

    # KerasPipeline folder?
    if path.is_dir():
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("type") == "KerasPipeline":
                preprocessor = joblib.load(path / "preprocessor.pkl")

                # preferred: native Keras file
                keras_file = path / meta.get("file", "keras_model.keras")
                if keras_file.exists():
                    model = tf.keras.models.load_model(keras_file)
                else:
                    # backward compat: SavedModel directory named 'keras_model'
                    legacy_dir = path / "keras_model"
                    if legacy_dir.exists():
                        model = tf.keras.models.load_model(legacy_dir)
                    else:
                        raise FileNotFoundError(
                            f"Keras model file not found in {path}. "
                            f"Looked for {keras_file.name} and 'keras_model' directory."
                        )

                # Recreate wrapper (import the real class)
                from model_training import KerasPipeline
                return KerasPipeline(preprocessor, model)

        # Otherwise, try loading a sklearn joblib inside the folder (optional convention)
        pkl = path / "model.pkl"
        if pkl.exists():
            return joblib.load(pkl)

        raise FileNotFoundError(f"Unrecognized saved model format in {path}")

    # File: could be sklearn .pkl or xgboost native
    if path.suffix in {".pkl", ".joblib"}:
        return joblib.load(path)

    if path.suffix in {".json", ".ubj", ".bin", ".model"}:
        booster = xgb.XGBRegressor()
        booster.load_model(str(path))
        return booster

    # Last resort: attempt joblib
    return joblib.load(path)

def load_all_models(artifacts_dir: str | Path = "artifacts") -> Dict[str, object]:
    """
    Loads all saved models from the given artifacts directory.
    Supports:
      - sklearn Pipelines (*.pkl, *.joblib)
      - xgboost models (*.json, *.model, etc.)
      - KerasPipeline folders (with preprocessor.pkl + keras_model.keras)
    Returns:
      dict: { model_name: loaded_model }
    """
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory '{artifacts_dir}' not found.")

    models = {}
    for item in artifacts_dir.iterdir():
        # Skip non-model files
        if item.name.startswith("."):
            continue

        name = item.stem if item.is_file() else item.name
        try:
            model = load_estimator(item)
            models[name] = model
        except Exception as e:
            print(f"Could not load {name}: {e}")

    if not models:
        print("No models loaded — check the artifacts directory path or file formats.")
    return models
