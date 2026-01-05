# Required Libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import mlflow
import optuna

# Function to compute adjusted R2
def adjusted_r2(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

# Load Data
data = pd.read_csv('./code/Machine Learning/data.csv')

# Drop identifier columns
data = data.drop(columns=['name', 'bgg_id'])

# Split features and target
X = data.drop(columns=['avg_rating'])
y = data['avg_rating']

# Imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standard Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Objective function for Optuna integrated with MLFlow
def objective(trial):
    #layer1_units = trial.suggest_int('layer1_units', 150, 512)
    layer1_units = 166
    #layer2_units = trial.suggest_int('layer2_units', 64, 256)
    layer2_units = 192
    #use_dropout = trial.suggest_categorical('use_dropout', [True, False])
    use_dropout = True
    #dropout_rate = trial.suggest_float('dropout_rate', 0.46, 0.5) if use_dropout else None
    dropout_rate = 0.46845696300996187 if use_dropout else None
    #layer3_units = trial.suggest_int('layer3_units', 60, 128)
    layer3_units = 99
    #activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
    activation = 'elu'
    #learning_rate = trial.suggest_float('learning_rate', 0.0015, 0.0045)
    learning_rate = 0.0022962224551903894
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    #batch_size = 64
    epochs = trial.suggest_int('epochs', 60, 100)
    # Define Neural Network Model

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(layer1_units, activation=activation))

    if use_dropout:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    else:
        model.add(tf.keras.layers.Dense(layer2_units, activation=activation))

    model.add(tf.keras.layers.Dense(layer3_units, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation = 'linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            'layer1_units': layer1_units,
            'layer2_units': layer2_units,
            'layer3_units': layer3_units,
            'use_dropout': use_dropout,
            'dropout_rate': dropout_rate,
            'activation': activation,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        })

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = adjusted_r2(y_test, y_pred, X_test.shape[1])

        mlflow.log_metrics({
            'MSE': mse,
            'R2': r2,
            'Adjusted_R2': adj_r2
        })

    return mse

# Execute Optuna study with MLFlow
mlflow.set_experiment("NN_optuna_study_epochs")
with mlflow.start_run(run_name="Optuna_Hyperparameter_Optimization"):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("Best_MSE", study.best_value)

print('Best Parameters:', study.best_params)
