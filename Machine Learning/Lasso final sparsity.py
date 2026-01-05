
"""
lasso_finalize.py
─────────────────
Turn an existing Optuna study (lasso_hpo.db) into two finished models:

    • lasso_dense.pkl   – α that minimises CV-MSE
    • lasso_sparse.pkl  – largest α within 1-SE of the best CV-MSE

"""

# ───────────────── imports ────────────────────────────────────────
import optuna, joblib, numpy as np, pandas as pd, pathlib, mlflow
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from data_utils import load_split                               # our helper

# ─────────────── config  (edit if paths differ) ──────────────────
DATA_CSV        = "data.csv"                         # full dataset (incl. booleans)
STUDY_DB        = "sqlite:///lasso_hpo.db"
STUDY_NAME      = "lasso_only"

TRACKING_URI    = r"file:///C:/Users/morit/Desktop/Coding/MLFlow/mlruns"
MLFLOW_EXPERIMENT = "Lasso_Final"
USE_MLFLOW      = False              # flip to False to skip MLflow logging

RANDOM_STATE    = 42
SCALE_NUMERIC   = True              # keep same preprocessing as in HPO
SCALE_BOOLEAN   = False

# ───────────────── data & preprocessor ────────────────────────────
X_train, X_test, y_train, y_test = load_split(DATA_CSV)

num_cols  = X_train.select_dtypes(include="number").columns
bool_cols = X_train.columns.difference(num_cols)

num_steps  = [("imp", SimpleImputer(strategy="median"))]
if SCALE_NUMERIC:
    num_steps.append(("sc", StandardScaler()))

bool_transformer = "passthrough"
if SCALE_BOOLEAN:
    bool_transformer = Pipeline([("sc_bool", StandardScaler(with_mean=False))])

preprocessor = ColumnTransformer(
    [("num",  Pipeline(num_steps),  num_cols),
     ("bool", bool_transformer,     bool_cols)]
)

# ───────────────── load study ─────────────────────────────────────
study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)

# ───── collect α, CV-MSE, std, and non-zero coeff count ──────────
records = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    alpha  = t.params["alpha"]
    mse    = -t.value                         # convert back to positive MSE
    std    = t.user_attrs["cv_mse_std"]
    # quick refit to count coefficients
    model  = Lasso(alpha=alpha, max_iter=10_000, random_state=RANDOM_STATE)
    Pipeline([("prep", preprocessor), ("model", model)]).fit(X_train, y_train)
    nz     = (np.abs(model.coef_) > 1e-6).sum()
    records.append((alpha, mse, std, nz))

arr = np.array(records,
               dtype=[("alpha", float), ("mse", float),
                      ("std", float), ("nz", int)])

# ───── identify dense & sparse alphas ─────────────────────────────
best_idx   = arr["mse"].argmin()
alpha_dense = arr["alpha"][best_idx]
best_mse    = arr["mse"][best_idx]
best_std    = arr["std"][best_idx]
one_se_thr  = best_mse + best_std

alpha_sparse = arr["alpha"][arr["mse"] <= one_se_thr].max()
nz_sparse    = arr["nz"][arr["alpha"] == alpha_sparse][0]

print("\n=== Lasso α selection ===")
print(f"Min-MSE α        : {alpha_dense:.5g}")
print(f"1-SE α (sparse)  : {alpha_sparse:.5g}  "
      f"({nz_sparse} non-zero coeffs)\n")

# ───── helper to fit, evaluate & save a pipeline ──────────────────
def build_and_save(tag, alpha):
    pipe = Pipeline([("prep", preprocessor),
                     ("model", Lasso(alpha=alpha, max_iter=10_000,
                                     random_state=RANDOM_STATE))])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "holdout_MSE": mean_squared_error(y_test, y_pred),
        "holdout_MAE": mean_absolute_error(y_test, y_pred),
        "holdout_R2" : r2_score(y_test, y_pred),
    }
    joblib.dump(pipe, f"lasso_{tag}.pkl")
    print(f"lasso_{tag}.pkl  saved  |  "
          f"MSE={metrics['holdout_MSE']:.4f}  "
          f"R²={metrics['holdout_R2']:.4f}")
    return metrics

metrics_dense  = build_and_save("dense",  alpha_dense)
metrics_sparse = build_and_save("sparse", alpha_sparse)

# ───── MLflow logging (optional) ──────────────────────────────────
if USE_MLFLOW:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="lasso_finalize"):
        mlflow.log_param("alpha_dense",  alpha_dense)
        mlflow.log_param("alpha_sparse", alpha_sparse)
        mlflow.log_param("nz_sparse",    nz_sparse)

        for k, v in metrics_dense.items():
            mlflow.log_metric(f"dense_{k}",  v)
        for k, v in metrics_sparse.items():
            mlflow.log_metric(f"sparse_{k}", v)

        mlflow.log_artifact("lasso_dense.pkl")
        mlflow.log_artifact("lasso_sparse.pkl")

        # link to the Optuna DB for reference
        if pathlib.Path("lasso_hpo.db").exists():
            mlflow.log_artifact("lasso_hpo.db")

print("\nDone.  Both final Lasso pipelines are ready for use.")
