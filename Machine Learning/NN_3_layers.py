# Required Libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import time

start = time.time()
# Function to compute adjusted R2
def adjusted_r2(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

# Load dataset
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

# Define Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(153, activation='elu',),
    tf.keras.layers.Dropout(0.4813525776897128),
    tf.keras.layers.Dense(87, activation='elu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0017898428936797518), loss='mse')

# Start MLFlow Run
    # Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)


# Predict on test set
y_pred = model.predict(X_test)

# Evaluate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(y_test, y_pred, X_test.shape[1])
end = time.time()
print(f'Time taken: {end - start} seconds')
    

print(f'Test MSE: {mse}')
print(f'Test R^2: {r2}')
print(f'Test Adjusted R^2: {adj_r2}')