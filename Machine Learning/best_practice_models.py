import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Comparison 2")

with mlflow.start_run():
    dataset = "data_no_bool.csv"
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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1809)
    mlflow.log_param("test_size", 0.2)
    # Create the different models
    xgb_params = {'n_estimators': 592, 'max_depth': 36, 'learning_rate': 0.03424916510690319, 'subsample': 0.8732399913040625, 'colsample_bytree': 0.3595628004141089, 'gamma': 0.03045351400058005, 'reg_alpha': 2.7000220599828153, 'reg_lambda': 6.977276466072381, 'min_child_weight': 7}
    dummy_model = DummyRegressor(strategy="mean")
    LinRegModel = LinearRegression()
    LassoRegModel = Lasso(alpha=0.0012)
    XGBModel = xgb.XGBRegressor(**xgb_params, n_jobs=-1, random_state=1503)
    AdaBoostModel = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=30), n_estimators=100, random_state=1503)
    RandomForestModel = RandomForestRegressor( n_estimators=433, max_depth=34, bootstrap=False, min_samples_split=8, min_samples_leaf=1, max_features=0.16, random_state=42 )
    NN3LayerModel = tf.keras.Sequential([ tf.keras.layers.Input(shape=(X_train.shape[1],)), tf.keras.layers.Dense(166, activation='elu'), tf.keras.layers.Dropout(0.46845696300996187), tf.keras.layers.Dense(99, activation='elu'), tf.keras.layers.Dense(1, activation = 'linear') ])
    NN3LayerModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0022962224551903894), loss='mean_squared_error')
    NN1LayerModel = tf.keras.Sequential([tf.keras.layers.Input(shape=(X_train.shape[1],)), tf.keras.layers.Dense(6963, activation='relu'), tf.keras.layers.Dense(1, activation = 'linear') ])
    NN1LayerModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.303e-05), loss='mean_squared_error')
    # Add all models to a list
    models = [ dummy_model, LinRegModel, LassoRegModel, XGBModel, AdaBoostModel, RandomForestModel, NN3LayerModel, NN1LayerModel ]
    linear_models = [dummy_model, LinRegModel, LassoRegModel]

    for model in models:
        with mlflow.start_run(nested=True):
            if isinstance(model, tf.keras.Model):
                if model == NN3LayerModel:
                    model.fit(X_train, y_train, epochs=84, batch_size=64, verbose=0)
                    mlflow.log_param("model", "NN_"+str(len(model.layers)-2)+"_layers")
                if model == NN1LayerModel:
                    model.fit(X_train, y_train, epochs=17, batch_size=64, verbose=0)
                    mlflow.log_param("model", "NN_1_layer")
            else:
                model.fit(X_train, y_train)
                mlflow.log_param("model", model.__class__.__name__)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            n = len(y_test)          # number of observations
            p = X_test.shape[1]      # number of predictors
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            # Log evaluation metrics
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("Adjusted_R2", adjusted_r2)
            