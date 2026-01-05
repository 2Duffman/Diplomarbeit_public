import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace this with your actual dataset)
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

def objective(trial):
    # Hyperparameter search space
    alpha = trial.suggest_float('alpha', 0.0001, 0.01, log=True)  # Fraction of features to use

    # Create the model with the current hyperparameters
    model = Lasso(alpha=alpha)

    # Perform 5-fold cross-validation and return the mean R^2 score
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()
    
    return score

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')  # We want to maximize R^2 score
study.optimize(objective, n_trials=100)  # You can increase the number of trials for better tuning

# Print the best parameters and score
print("Best Hyperparameters:", study.best_params)
print("Best R^2 Score:", study.best_value)

# Train the best model on the entire training set
best_params = study.best_params
best_model = Ridge(alpha=best_params['alpha'])
best_model.fit(X_train, y_train)

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print("Test R^2 Score:", test_score)
#Trial 69 finished with value: 0.48813810880284914 and parameters: {'max_features': 3630.3009627480174}. Best is trial 69 with value: 0.48813810880284914.