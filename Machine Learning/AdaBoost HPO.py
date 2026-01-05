"""
Hyper-parameter optimisation for AdaBoost Regressor
────────────────────────────────────────────────────
• Uses fixed train/test IDs stored in train_ids.txt / test_ids.txt
• Saves (and resumes) Optuna study in sqlite:///adaboost_hpo.db
• Logs trials to MLflow if USE_MLFLOW = True
"""

from pathlib import Path
import warnings, optuna, numpy as np, pandas as pd, mlflow, joblib, pathlib
from optuna.integration.mlflow import MLflowCallback
from sklearn.pipeline   import Pipeline
from sklearn.compose    import ColumnTransformer
from sklearn.ensemble   import AdaBoostRegressor
from sklearn.tree       import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute      import SimpleImputer
from sklearn.metrics     import mean_squared_error, mean_absolute_error, r2_score
from data_utils          import load_split   # <- our helper module

warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIG ────────────────────────────────────────────────────────
DATA_CSV          = "data.csv"        # full dataset with booleans
USE_MLFLOW        = True
MLFLOW_EXPERIMENT = "AdaBoost_HPO_big_depth"
STUDY_NAME        = "adaboost_big_depth"
STORAGE           = "sqlite:///adaboost_hpo_big.db"
N_TRIALS          = 200
CV_FOLDS          = 5
RANDOM_STATE      = 42
# Hyper-parameters for AdaBoost
MAX_DEPTH       = (20, 40)                # max depth of base estimator
N_ESTIMATORS    = (50, 500)          # range for n_estimators
LEARNING_RATE  = (0.001, 0.3)        # range for learning rate
SCALE_NUMERIC     = False              # standardise numeric cols
SCALE_BOOLEAN     = False             # leave 0/1 booleans as-is
# ──────────────────────────────────────────────────────────────────
TRACKING_URI = r"file:///C:/Users/morit/Desktop/Coding/MLFlow/mlruns"
pathlib.Path(r"C:\Users\morit\Desktop\Coding\MLFlow\mlruns").mkdir(
    parents=True, exist_ok=True
)
mlflow.set_tracking_uri(TRACKING_URI)
# 1) Load split
X_train, X_test, y_train, y_test = load_split(DATA_CSV)

# 2) Preprocessing pipeline
num_cols  = X_train.select_dtypes(include="number").columns
bool_cols = X_train.columns.difference(num_cols)

num_steps  = [("imp", SimpleImputer(strategy="median"))]
if SCALE_NUMERIC:
    num_steps.append(("sc", StandardScaler()))

bool_steps = []
if SCALE_BOOLEAN:
    bool_transformer = Pipeline([("sc_bool", StandardScaler(with_mean=False))])
else:
    bool_transformer = "passthrough"

preprocessor = ColumnTransformer(
    [("num",  Pipeline(num_steps),  num_cols),
     ("bool", bool_transformer, bool_cols)]
)

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# 3) Optuna objective
def objective(trial: optuna.trial.Trial) -> float:
    max_depth = trial.suggest_int('max_depth', MAX_DEPTH[0], MAX_DEPTH[1])
    n_estimators = trial.suggest_int('n_estimators', N_ESTIMATORS[0], N_ESTIMATORS[1])
    learning_rate = trial.suggest_float('learning_rate', LEARNING_RATE[0], LEARNING_RATE[1], log=True)
    loss = trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
    base_estimator = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_STATE)
    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        loss=loss
    )
    pipe  = Pipeline([("prep", preprocessor), ("model", model)])

    scores = cross_val_score(
        pipe, X_train, y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        error_score="raise"
    )
    trial.set_user_attr("cv_mse_std", scores.std())
    return scores.mean()                      # maximise neg-MSE

# 4) Study (resume-able)
mlf_cb = None
if USE_MLFLOW:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    mlf_cb = MLflowCallback(metric_name="neg_MSE")

study = optuna.create_study(
    study_name=STUDY_NAME,
    direction="maximize",
    storage=STORAGE,
    load_if_exists=True
)

# 5) Optimisation loop
try:
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True,
                   callbacks=[mlf_cb] if mlf_cb else None)
except KeyboardInterrupt:
    print("Interrupted — progress saved to adaboost_hpo.db")

# 6) Fit best model on full train & evaluate hold-out
best_params = study.best_params
best_base = DecisionTreeRegressor(max_depth=best_params["max_depth"], random_state=RANDOM_STATE)
best_model = AdaBoostRegressor(
    estimator=best_base,
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    random_state=RANDOM_STATE
)
best_pipe  = Pipeline([
    ("prep", preprocessor),
    ("model", best_model)
])
best_pipe.fit(X_train, y_train)

y_pred = best_pipe.predict(X_test)
holdout_mse = mean_squared_error(y_test, y_pred)
holdout_mae = mean_absolute_error(y_test, y_pred)
holdout_r2  = r2_score(y_test, y_pred)
n = len(y_test)
p = X_test.shape[1]
holdout_adj_r2 = 1 - (1 - holdout_r2) * (n - 1) / (n - p - 1)
print(f"Hold-out adj. R² = {holdout_adj_r2:.4f}")

print(f"\nBest params = {best_params}")
print(f"Hold-out MSE = {holdout_mse:.4f}")
print(f"Hold-out MAE = {holdout_mae:.4f}")
print(f"Hold-out R²  = {holdout_r2 :.4f}")

joblib.dump(best_pipe, "best_adaboost_pipeline.pkl")
print("Saved -> best_adaboost_pipeline.pkl")

if USE_MLFLOW:
    with mlflow.start_run(run_id=study.best_trial.user_attrs.get("mlflow_run_id")):
        mlflow.log_param("max_depth", best_params["max_depth"])
        mlflow.log_param("n_estimators", best_params["n_estimators"])
        mlflow.log_param("learning_rate", best_params["learning_rate"])
        mlflow.log_metric("holdout_MSE", holdout_mse)
        mlflow.log_metric("holdout_MAE", holdout_mae)
        mlflow.log_metric("holdout_R2" , holdout_r2 )
        mlflow.log_metric("holdout_adj_R2", holdout_adj_r2)
        mlflow.log_artifact("best_adaboost_pipeline.pkl")
