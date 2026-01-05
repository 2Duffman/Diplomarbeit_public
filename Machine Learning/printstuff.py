import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

LinRegModel = LinearRegression()
RidgeRegModel = Ridge(alpha=3642)
LassoRegModel = Lasso(alpha=0.0032)
models = [LinRegModel, RidgeRegModel, LassoRegModel]

dataset = "data.csv"
# Load your dataset
data = pd.read_csv(f'./code/Machine Learning/{dataset}')
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

print(X_scaled.shape[1])