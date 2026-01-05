import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import mlflow
mlflow.set_experiment("XGB")

with mlflow.start_run():
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
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale the imputed data using StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    # Log parameters about preprocessing"
    
    mlflow.log_param("imputation_strategy", "no imputation")
    mlflow.log_param("scaler", "no scaling")
    
    mlflow.log_param("dataset_shape", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

        # Hyperparameter search space
    params = {'n_estimators': 592, 'max_depth': 36, 'learning_rate': 0.03424916510690319, 
    'subsample': 0.8732399913040625, 'colsample_bytree': 0.3595628004141089, 'gamma': 
    0.03045351400058005, 'reg_alpha': 2.7000220599828153, 'reg_lambda': 6.977276466072381, 
    'min_child_weight': 7}

        # Create the XGBoost model
    model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=42)
    mlflow.log_params(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate adjusted R²
    n = len(y_test)           # number of observations in the test set
    p = X_test.shape[1]       # number of predictors
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    
    # Log evaluation metrics with MLflow
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Adjusted_R2", adjusted_r2)

    print("MSE", mse)
    print("R2", r2)
    print("Adjusted_R2", adjusted_r2)
