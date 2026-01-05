import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
            # Start an MLflow run to track the experiment
mlflow.set_experiment("AdaBoost")
with mlflow.start_run():
    dataset = "data.csv"
    # Load your dataset
    data = pd.read_csv(f'./code/Machine Learning/{dataset}')
    mlflow.log_param("dataset", dataset)
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
    
    # Log parameters about preprocessing
    mlflow.log_param("imputation_strategy", "median")
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("dataset_shape", X_scaled.shape)
    
    # Split the dataset into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    mlflow.log_param("test_size", 0.2)
    
    # Define a base estimator for AdaBoost (using a simple decision tree)
    base_estimator = DecisionTreeRegressor(max_depth=30)
    mlflow.log_param("base_estimator", base_estimator)
    # Create the AdaBoost Regressor
    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=100,
        random_state=42
    )
    mlflow.log_param("n_estimators", 100)
    
    # Train the AdaBoost model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
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
    
    # Log the trained model using MLflow's sklearn integration
    mlflow.sklearn.log_model(model, "adaboost_regressor_model")
    
    # Print evaluation results to console
    
    print("AdaBoost Regressor MSE:", mse)
    print("AdaBoost Regressor R^2:", r2)
    print("AdaBoost Regressor Adjusted R^2:", adjusted_r2)
