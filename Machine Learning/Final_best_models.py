import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import time
import mlflow.sklearn

# ──────────────────────────────────────────────────────────────────────────────
# Experiment setup
# ──────────────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "FINAL_COMPARISON_FOR_REAL"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ──────────────────────────────────────────────────────────────────────────────
# Load data (now with booleans) and split
# ──────────────────────────────────────────────────────────────────────────────
dataset = "data.csv"  # full dataset with boolean columns
# adjust the path to where your CSV actually lives
df = pd.read_csv(dataset)

# Drop identifier columns (keep as before)
df = df.drop(columns=["name", "bgg_id"])

# Target / features
y = df["avg_rating"]
X = df.drop(columns=["avg_rating"])

# Train/test split (same as before so results remain comparable)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1809
)

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing builders
# ──────────────────────────────────────────────────────────────────────────────
def make_preprocessor(X_frame: pd.DataFrame, scale_numeric: bool, scale_boolean: bool=False):
    """
    Builds a ColumnTransformer that:
      - imputes numeric with median and optionally scales
      - passes boolean (and other non-numeric) columns through, or optionally scales
    """
    num_cols  = X_frame.select_dtypes(include="number").columns
    bool_cols = X_frame.columns.difference(num_cols)  # anything non-numeric (e.g., bool, category)

    # numeric: impute median, optional StandardScaler
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("sc", StandardScaler()))

    # boolean/non-numeric: passthrough by default, optional StandardScaler (w/o centering)
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
# Models
# ──────────────────────────────────────────────────────────────────────────────
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

dummy_model        = DummyRegressor(strategy="mean")
lin_reg_model      = LinearRegression()
lasso_reg_model    = Lasso(alpha=0.0012)
xgb_model          = xgb.XGBRegressor(**xgb_params, n_jobs=-1, random_state=1503)
ada_base_tree      = DecisionTreeRegressor(max_depth=30, random_state=1503)
adaboost_model     = AdaBoostRegressor(estimator=ada_base_tree, n_estimators=100, random_state=1503)
rf_model           = RandomForestRegressor(
    n_estimators=433, max_depth=34, bootstrap=False,
    min_samples_split=8, min_samples_leaf=1, max_features=0.16, random_state=42
)

# Neural nets (trained on preprocessed arrays, not via scikit Pipeline)
def build_nn_3layer(input_dim: int):
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

def build_nn_1layer(input_dim: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(6963, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.303e-05),
                  loss="mean_squared_error")
    return model

# Group models and flags
MODELS = [
    ("DummyRegressor", dummy_model,        "tree_like"),  # no scaling needed
    ("LinearRegression", lin_reg_model,    "linear"),     # scale numeric
    ("Lasso", lasso_reg_model,             "linear"),     # scale numeric
    ("XGBRegressor", xgb_model,            "tree_like"),
    ("AdaBoostRegressor", adaboost_model,  "tree_like"),
    ("RandomForestRegressor", rf_model,    "tree_like"),
    ("NN_3_layers", "NN3",                 "nn"),         # handled specially
    ("NN_1_layer",  "NN1",                 "nn"),         # handled specially
]

# ──────────────────────────────────────────────────────────────────────────────
# Training loop with MLflow, using the new preprocessing per model type
# ──────────────────────────────────────────────────────────────────────────────
with mlflow.start_run():
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("test_size", 0.20)

    for model_name, model_obj, family in MODELS:
        with mlflow.start_run(nested=True):
            start = time.time()
            mlflow.log_param("model", model_name)

            # Decide scaling policy
            if family in ("linear", "nn"):
                SCALE_NUMERIC = True
            else:
                SCALE_NUMERIC = False
            SCALE_BOOLEAN = False  # as per your new pipeline

            mlflow.log_param("scale_numeric", SCALE_NUMERIC)
            mlflow.log_param("scale_boolean", SCALE_BOOLEAN)

            # Build the appropriate preprocessor
            preprocessor, num_cols, bool_cols = make_preprocessor(
                X_train, scale_numeric=SCALE_NUMERIC, scale_boolean=SCALE_BOOLEAN
            )
            mlflow.log_param("num_cols_count", len(num_cols))
            mlflow.log_param("bool_cols_count", len(bool_cols))

            if family == "nn":
                # Fit the preprocessor, transform to arrays, then train Keras
                Xtr = preprocessor.fit_transform(X_train)
                Xte = preprocessor.transform(X_test)

                # Build and train the right NN
                if model_obj == "NN3":
                    nn = build_nn_3layer(input_dim=Xtr.shape[1])
                    nn.fit(Xtr, y_train, epochs=84, batch_size=64, verbose=0)
                else:
                    nn = build_nn_1layer(input_dim=Xtr.shape[1])
                    nn.fit(Xtr, y_train, epochs=17, batch_size=64, verbose=0)

                # Evaluate
                y_pred = nn.predict(Xte).ravel()

                # Log shapes/info
                mlflow.log_param("preprocessed_n_features", Xtr.shape[1])

            else:
                # Wrap model in a Pipeline so preprocessing happens inside
                pipe = Pipeline([
                    ("prep", preprocessor),
                    ("model", model_obj),
                ])

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            n   = len(y_test)
            p   = X_test.shape[1]  # number of predictors pre-transform (consistent with your earlier calc)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            end = time.time()
            print(f"{model_name}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Adj_R2={adj_r2:.4f}, Time={end - start:.1f}s")y
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("Adjusted_R2", adj_r2)
            mlflow.log_metric("Time_seconds", end - start)
