import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Lasso_Regression")
alphas = [0.00001, 0.0001, 0.001, 0.01]
for alpha in alphas:
    with mlflow.start_run():
        dataset = "data.csv"
        # Load your dataset
        data = pd.read_csv(f'./code/Machine Learning/{dataset}')
        mlflow.log_param("dataset", dataset)
        # Drop identifier columns
        data = data.drop(columns=['name', 'bgg_id'])
        # Separate target variable from predictors
        y = data['avg_rating']
        X = data.drop(columns=['avg_rating'])
        # impute missing values in X
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        # scale the data
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        # Log imputation strategy and dataset info as parameters
        mlflow.log_param("imputation_strategy", "median")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("dataset_shape", X_scaled.shape)
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        mlflow.log_param("test_size", 0.2)
        # Create the ridge regression model
        model = Lasso(alpha=alpha)
        mlflow.log_param("alpha", alpha)
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Calculate the adjusted R² score
        n = len(y_test)          # number of observations
        p = X_test.shape[1]      # number of predictors
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        # Log evaluation metrics
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("Adjusted_R2", adjusted_r2)

        # Log the model using MLflow's sklearn integration
        mlflow.sklearn.log_model(model, "linear_regression_model")
        print("Mean Squared Error:", mse)
        print("R² Score:", r2)
        print("Adjusted R² Score:", adjusted_r2)