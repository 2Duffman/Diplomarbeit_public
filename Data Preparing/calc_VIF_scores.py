import mlflow
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assume X_imputed contains your numeric features
dataset = "data.csv"
    # Load your dataset
data = pd.read_csv(f'./code/Machine Learning/{dataset}')
# Drop identifier columns
data = data.drop(columns=['name', 'bgg_id'])
# Separate target variable from predictors
y = data['avg_rating']
X = data.drop(columns=['avg_rating'])
# impute missing values in X (mean imputation for numerical data)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# scale the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X_imputed.columns
vif_data["VIF"] = [variance_inflation_factor(X_imputed.values, i) for i in range(X_imputed.shape[1])]
with open("vif_output.txt", "w", encoding="utf-8") as f:
    f.write(vif_data.to_string())
