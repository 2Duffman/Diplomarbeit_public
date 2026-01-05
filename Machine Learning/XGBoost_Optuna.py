import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

# Load your dataset (Replace with your actual dataset)
dataset = "data.csv"
# Load your dataset
data = pd.read_csv(f'./code/Machine Learning/{dataset}')
# Drop identifier columns (not used for prediction)
data = data.drop(columns=['name', 'bgg_id'])

# Separate target variable from predictors
y = data['avg_rating']
X = data.drop(columns=['avg_rating'])

# Impute missing values using median imputation
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale the imputed data using StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 600),  # Number of boosting rounds
        "max_depth": trial.suggest_int("max_depth", 3, 40),  # Depth of trees (based on RF & AdaBoost)
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),  # Step size shrinkage
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # Fraction of training data per tree
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),  # Feature sampling per tree
        "gamma": trial.suggest_float("gamma", 0, 5),  # Minimum loss reduction for a split
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),  # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),  # L2 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  # Minimum sum of weights in a node
        "tree_method": "hist",  # Faster training, especially on large datasets
        "objective": "reg:squarederror",  # Standard regression loss
    }

    # Create the XGBoost model
    model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=42)

    # Perform 5-fold cross-validation and return the mean R² score
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()
    return score  # Higher R² is better

# Create or load the study
study = optuna.create_study(
    direction="maximize"
)

# Run the optimization
study.optimize(objective, n_trials=150)  # Adjust n_trials as needed for overnight tuning

# Print the best parameters and score
print("Best Hyperparameters:", study.best_params_)
print("Best R² Score:", study.best_value)

# Train the best model on the entire training set
best_params = study.best_params_
best_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print("Test R² Score:", test_score)
