import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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
    n_estimators = trial.suggest_int('n_estimators', 50, 500)  # Number of trees
    max_depth = trial.suggest_int('max_depth', 2, 50)  # Depth of each tree
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  # Minimum samples to split a node
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  # Minimum samples at a leaf node
    max_features = trial.suggest_float('max_features', 0.1, 1.0)  # Fraction of features to use
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])  # Use bootstrap sampling?

    # Create the model with the current hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )

    # Perform 5-fold cross-validation and return the mean R^2 score
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()
    
    return score

# Run the optimization with Optuna
study = optuna.create_study(direction='maximize')  # We want to maximize R^2 score
study.optimize(objective, n_trials=100)  # You can increase the number of trials for better tuning

# Print the best parameters and score
print("Best Hyperparameters:", study.best_params_)
print("Best R^2 Score:", study.best_value)

# Train the best model on the entire training set
best_params = study.best_params_
best_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    n_jobs=-1,
    random_state=42
)
best_model.fit(X_train, y_train)

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print("Test R^2 Score:", test_score)
"""[I 2025-03-10 23:20:00,033] A new study created in memory with name: no-name-4df48294-bd35-46f8-b47e-fffa3306bec7
[I 2025-03-10 23:23:25,419] Trial 0 finished with value: 0.6138118366448794 and parameters: {'n_estimators': 111, 'max_depth': 26, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.9641469788950932, 'bootstrap': True}. Best is trial 0 with value: 0.6138118366448794.
[I 2025-03-10 23:27:02,846] Trial 1 finished with value: 0.6129326476657628 and parameters: {'n_estimators': 255, 'max_depth': 41, 'min_samples_split': 16, 'min_samples_leaf': 7, 'max_features': 0.4919957267431294, 'bootstrap': True}. Best is trial 0 with value: 0.6138118366448794.
[I 2025-03-10 23:34:03,122] Trial 2 finished with value: 0.6109319915675167 and parameters: {'n_estimators': 223, 'max_depth': 28, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_features': 0.6824666618163883, 'bootstrap': False}. Best is trial 0 with value: 0.6138118366448794.
[I 2025-03-10 23:49:54,666] Trial 3 finished with value: 0.56740969711252 and parameters: {'n_estimators': 375, 'max_depth': 29, 'min_samples_split': 11, 'min_samples_leaf': 4, 'max_features': 0.9162749049387966, 'bootstrap': False}. Best is trial 0 with value: 0.6138118366448794.
[I 2025-03-10 23:57:40,545] Trial 4 finished with value: 0.6171609565114601 and parameters: {'n_estimators': 276, 'max_depth': 48, 'min_samples_split': 14, 'min_samples_leaf': 4, 'max_features': 0.5945195148420884, 'bootstrap': False}. Best is trial 4 with value: 0.6171609565114601.
[I 2025-03-11 00:08:13,964] Trial 5 finished with value: 0.5649180052435859 and parameters: {'n_estimators': 295, 'max_depth': 34, 'min_samples_split': 9, 'min_samples_leaf': 6, 'max_features': 0.925702678762652, 'bootstrap': False}. Best is trial 4 with value: 0.6171609565114601.
[I 2025-03-11 00:10:41,260] Trial 6 finished with value: 0.5438237366115203 and parameters: {'n_estimators': 490, 'max_depth': 7, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 0.47593405722801874, 'bootstrap': True}. Best is trial 4 with value: 0.6171609565114601.
[I 2025-03-11 00:16:37,442] Trial 7 finished with value: 0.6209776776816092 and parameters: {'n_estimators': 238, 'max_depth': 43, 'min_samples_split': 8, 'min_samples_leaf': 2, 'max_features': 0.8820619446580557, 'bootstrap': True}. Best is trial 7 with value: 0.6209776776816092.
[I 2025-03-11 00:25:43,552] Trial 8 finished with value: 0.547901474125864 and parameters: {'n_estimators': 258, 'max_depth': 25, 'min_samples_split': 20, 'min_samples_leaf': 5, 'max_features': 0.9554936596336294, 'bootstrap': False}. Best is trial 7 with value: 0.6209776776816092.
[I 2025-03-11 00:31:50,999] Trial 9 finished with value: 0.6208884053820499 and parameters: {'n_estimators': 281, 'max_depth': 39, 'min_samples_split': 18, 'min_samples_leaf': 1, 'max_features': 0.4649456530377516, 'bootstrap': False}. Best is trial 7 with value: 0.6209776776816092.
[I 2025-03-11 00:32:01,059] Trial 10 finished with value: 0.5775164017785606 and parameters: {'n_estimators': 50, 'max_depth': 14, 'min_samples_split': 3, 'min_samples_leaf': 9, 'max_features': 0.13801138153313502, 'bootstrap': True}. Best is trial 7 with value: 0.6209776776816092.
[I 2025-03-11 00:33:43,187] Trial 11 finished with value: 0.6293663440492759 and parameters: {'n_estimators': 161, 'max_depth': 48, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 0.3008594250046975, 'bootstrap': True}. Best is trial 11 with value: 0.6293663440492759.
[I 2025-03-11 00:34:47,534] Trial 12 finished with value: 0.629419599028242 and parameters: {'n_estimators': 158, 'max_depth': 49, 'min_samples_split': 7, 'min_samples_leaf': 2, 'max_features': 0.19362497550036584, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:35:46,215] Trial 13 finished with value: 0.6255970995298569 and parameters: {'n_estimators': 151, 'max_depth': 50, 'min_samples_split': 6, 'min_samples_leaf': 3, 'max_features': 0.18870345174691022, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:37:17,291] Trial 14 finished with value: 0.6291441779668108 and parameters: {'n_estimators': 154, 'max_depth': 46, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 0.2924261625519857, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:39:00,866] Trial 15 finished with value: 0.6291205925997552 and parameters: {'n_estimators': 165, 'max_depth': 35, 'min_samples_split': 6, 'min_samples_leaf': 1, 'max_features': 0.31119996851386206, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:39:38,886] Trial 16 finished with value: 0.6104681304296429 and parameters: {'n_estimators': 80, 'max_depth': 17, 'min_samples_split': 14, 'min_samples_leaf': 3, 'max_features': 0.3220203882109074, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:40:44,733] Trial 17 finished with value: 0.6047732288305449 and parameters: {'n_estimators': 196, 'max_depth': 35, 'min_samples_split': 8, 'min_samples_leaf': 10, 'max_features': 0.2108143187378712, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:43:51,444] Trial 18 finished with value: 0.6142596242131148 and parameters: {'n_estimators': 336, 'max_depth': 44, 'min_samples_split': 4, 'min_samples_leaf': 7, 'max_features': 0.3658300280825878, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:44:19,007] Trial 19 finished with value: 0.6218694104324756 and parameters: {'n_estimators': 110, 'max_depth': 50, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 0.10446001196379312, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:51:52,594] Trial 20 finished with value: 0.6220534259659323 and parameters: {'n_estimators': 390, 'max_depth': 40, 'min_samples_split': 7, 'min_samples_leaf': 3, 'max_features': 0.7083813955211771, 'bootstrap': True}. Best is trial 12 with value: 0.629419599028242.
[I 2025-03-11 00:53:21,919] Trial 21 finished with value: 0.6301134636735635 and parameters: {'n_estimators': 166, 'max_depth': 46, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 0.2607918566594495, 'bootstrap': True}. Best is trial 21 with value: 0.6301134636735635.
[I 2025-03-11 00:55:04,491] Trial 22 finished with value: 0.6306281709131739 and parameters: {'n_estimators': 192, 'max_depth': 46, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 0.24879040091965368, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 00:56:30,849] Trial 23 finished with value: 0.626791097783528 and parameters: {'n_estimators': 200, 'max_depth': 37, 'min_samples_split': 2, 'min_samples_leaf': 3, 'max_features': 0.2182186746357934, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 00:57:55,510] Trial 24 finished with value: 0.6276605862957528 and parameters: {'n_estimators': 115, 'max_depth': 45, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 0.38494679015185557, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 00:59:17,506] Trial 25 finished with value: 0.6220784901077513 and parameters: {'n_estimators': 199, 'max_depth': 32, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 0.2233701882447202, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:00:46,701] Trial 26 finished with value: 0.6193071485090077 and parameters: {'n_estimators': 138, 'max_depth': 19, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 0.4081880430843394, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:01:48,664] Trial 27 finished with value: 0.6207722599103139 and parameters: {'n_estimators': 183, 'max_depth': 43, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 0.1701774044455046, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:02:31,838] Trial 28 finished with value: 0.5145342856693935 and parameters: {'n_estimators': 320, 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_features': 0.26232862505245286, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:03:02,304] Trial 29 finished with value: 0.617761328607521 and parameters: {'n_estimators': 105, 'max_depth': 22, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_features': 0.1463596534311391, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:03:18,441] Trial 30 finished with value: 0.37728155739401004 and parameters: {'n_estimators': 214, 'max_depth': 2, 'min_samples_split': 6, 'min_samples_leaf': 6, 'max_features': 0.6003319764705982, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:04:30,570] Trial 31 finished with value: 0.6287378330131724 and parameters: {'n_estimators': 135, 'max_depth': 47, 'min_samples_split': 9, 'min_samples_leaf': 1, 'max_features': 0.2591201828432443, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:06:34,186] Trial 32 finished with value: 0.6287976223354494 and parameters: {'n_estimators': 177, 'max_depth': 50, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 0.3510496114039422, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:07:37,898] Trial 33 finished with value: 0.6263939166020324 and parameters: {'n_estimators': 79, 'max_depth': 41, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_features': 0.4272311276664821, 'bootstrap': True}. Best is trial 22 with value: 0.6306281709131739.
[I 2025-03-11 01:09:56,543] Trial 34 finished with value: 0.631400526595973 and parameters: {'n_estimators': 235, 'max_depth': 46, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 0.2752805011830782, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:15:39,552] Trial 35 finished with value: 0.6210674281880888 and parameters: {'n_estimators': 226, 'max_depth': 41, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.535771593606995, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:16:30,930] Trial 36 finished with value: 0.5983473889027081 and parameters: {'n_estimators': 258, 'max_depth': 31, 'min_samples_split': 4, 'min_samples_leaf': 8, 'max_features': 0.10548969328214802, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:19:17,486] Trial 37 finished with value: 0.6292937412765831 and parameters: {'n_estimators': 232, 'max_depth': 38, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 0.24980995621160834, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:25:58,383] Trial 38 finished with value: 0.6214887032262908 and parameters: {'n_estimators': 306, 'max_depth': 47, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 0.8055679774916544, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:36:02,106] Trial 39 finished with value: 0.624385851020383 and parameters: {'n_estimators': 465, 'max_depth': 43, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.4414729163447213, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:38:23,840] Trial 40 finished with value: 0.6195409913431552 and parameters: {'n_estimators': 250, 'max_depth': 27, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': 0.35050087194361945, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:40:00,516] Trial 41 finished with value: 0.628719134388162 and parameters: {'n_estimators': 167, 'max_depth': 48, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 0.28432512676161, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:40:46,093] Trial 42 finished with value: 0.6291091407430672 and parameters: {'n_estimators': 125, 'max_depth': 45, 'min_samples_split': 6, 'min_samples_leaf': 2, 'max_features': 0.16568490395033006, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:43:05,037] Trial 43 finished with value: 0.6293830461740036 and parameters: {'n_estimators': 212, 'max_depth': 48, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 0.32611510085475826, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:46:32,587] Trial 44 finished with value: 0.6248722314532263 and parameters: {'n_estimators': 219, 'max_depth': 50, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.5137625022486628, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:48:38,702] Trial 45 finished with value: 0.62924655746126 and parameters: {'n_estimators': 276, 'max_depth': 42, 'min_samples_split': 9, 'min_samples_leaf': 2, 'max_features': 0.23481283830198188, 'bootstrap': True}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:50:23,384] Trial 46 finished with value: 0.6300610462507478 and parameters: {'n_estimators': 183, 'max_depth': 47, 'min_samples_split': 16, 'min_samples_leaf': 3, 'max_features': 0.197869493345238, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:52:04,102] Trial 47 finished with value: 0.6294937253557323 and parameters: {'n_estimators': 186, 'max_depth': 45, 'min_samples_split': 17, 'min_samples_leaf': 3, 'max_features': 0.1841744896334152, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:53:52,883] Trial 48 finished with value: 0.6258000537196128 and parameters: {'n_estimators': 248, 'max_depth': 39, 'min_samples_split': 17, 'min_samples_leaf': 4, 'max_features': 0.14667922166059807, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:55:33,568] Trial 49 finished with value: 0.6285641891978309 and parameters: {'n_estimators': 186, 'max_depth': 36, 'min_samples_split': 19, 'min_samples_leaf': 3, 'max_features': 0.1890485724927273, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:57:15,256] Trial 50 finished with value: 0.6242256686541773 and parameters: {'n_estimators': 142, 'max_depth': 45, 'min_samples_split': 16, 'min_samples_leaf': 5, 'max_features': 0.27747718175743863, 'bootstrap': False}. Best is trial 34 with value: 0.631400526595973.
[I 2025-03-11 01:58:51,789] Trial 51 finished with value: 0.6328540454427964 and parameters: {'n_estimators': 170, 'max_depth': 46, 'min_samples_split': 15, 'min_samples_leaf': 2, 'max_features': 0.18367607408558165, 'bootstrap': False}. Best is trial 51 with value: 0.6328540454427964.
[I 2025-03-11 01:59:52,183] Trial 52 finished with value: 0.6254534080746676 and parameters: {'n_estimators': 177, 'max_depth': 46, 'min_samples_split': 16, 'min_samples_leaf': 3, 'max_features': 0.10032101275305855, 'bootstrap': False}. Best is trial 51 with value: 0.6328540454427964.
[I 2025-03-11 02:01:58,428] Trial 53 finished with value: 0.6327863382947452 and parameters: {'n_estimators': 203, 'max_depth': 43, 'min_samples_split': 15, 'min_samples_leaf': 2, 'max_features': 0.20572475066350096, 'bootstrap': False}. Best is trial 51 with value: 0.6328540454427964.
[I 2025-03-11 02:04:38,720] Trial 54 finished with value: 0.6324284196635994 and parameters: {'n_estimators': 234, 'max_depth': 43, 'min_samples_split': 15, 'min_samples_leaf': 2, 'max_features': 0.23237632473191605, 'bootstrap': False}. Best is trial 51 with value: 0.6328540454427964.
[I 2025-03-11 02:07:28,366] Trial 55 finished with value: 0.6324316217067338 and parameters: {'n_estimators': 238, 'max_depth': 42, 'min_samples_split': 14, 'min_samples_leaf': 2, 'max_features': 0.24661642469182407, 'bootstrap': False}. Best is trial 51 with value: 0.6328540454427964.
[I 2025-03-11 02:10:44,436] Trial 56 finished with value: 0.6329799902772387 and parameters: {'n_estimators': 286, 'max_depth': 40, 'min_samples_split': 14, 'min_samples_leaf': 2, 'max_features': 0.23269997529305414, 'bootstrap': False}. Best is trial 56 with value: 0.6329799902772387.
[I 2025-03-11 02:15:13,947] Trial 57 finished with value: 0.6290653457480498 and parameters: {'n_estimators': 286, 'max_depth': 33, 'min_samples_split': 14, 'min_samples_leaf': 1, 'max_features': 0.32235047740493883, 'bootstrap': False}. Best is trial 56 with value: 0.6329799902772387.
[I 2025-03-11 02:19:03,527] Trial 58 finished with value: 0.6329516925479804 and parameters: {'n_estimators': 352, 'max_depth': 39, 'min_samples_split': 15, 'min_samples_leaf': 2, 'max_features': 0.22033750329631271, 'bootstrap': False}. Best is trial 56 with value: 0.6329799902772387.
[I 2025-03-11 02:21:41,924] Trial 59 finished with value: 0.6320417680606027 and parameters: {'n_estimators': 367, 'max_depth': 30, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 0.13385570498453642, 'bootstrap': False}. Best is trial 56 with value: 0.6329799902772387.
[I 2025-03-11 02:25:07,954] Trial 60 finished with value: 0.6281462210751747 and parameters: {'n_estimators': 343, 'max_depth': 38, 'min_samples_split': 15, 'min_samples_leaf': 4, 'max_features': 0.21774932094785932, 'bootstrap': False}. Best is trial 56 with value: 0.6329799902772387.
[I 2025-03-11 02:28:11,241] Trial 61 finished with value: 0.6343009565611479 and parameters: {'n_estimators': 395, 'max_depth': 40, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 0.14008246806883484, 'bootstrap': False}. Best is trial 61 with value: 0.6343009565611479.
[I 2025-03-11 02:31:14,899] Trial 62 finished with value: 0.6336207660078312 and parameters: {'n_estimators': 380, 'max_depth': 40, 'min_samples_split': 15, 'min_samples_leaf': 2, 'max_features': 0.1489516885006435, 'bootstrap': False}. Best is trial 61 with value: 0.6343009565611479.
[I 2025-03-11 02:34:20,729] Trial 63 finished with value: 0.6343302177679375 and parameters: {'n_estimators': 414, 'max_depth': 40, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 0.13389213360490346, 'bootstrap': False}. Best is trial 63 with value: 0.6343302177679375.
[I 2025-03-11 02:37:26,664] Trial 64 finished with value: 0.6346493331090748 and parameters: {'n_estimators': 418, 'max_depth': 40, 'min_samples_split': 12, 'min_samples_leaf': 2, 'max_features': 0.13081652169523608, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 02:40:16,444] Trial 65 finished with value: 0.6299855277873545 and parameters: {'n_estimators': 417, 'max_depth': 34, 'min_samples_split': 12, 'min_samples_leaf': 3, 'max_features': 0.12591284808201877, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 02:43:53,064] Trial 66 finished with value: 0.6315519638505693 and parameters: {'n_estimators': 434, 'max_depth': 40, 'min_samples_split': 13, 'min_samples_leaf': 3, 'max_features': 0.16193082078862503, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 02:46:40,312] Trial 67 finished with value: 0.6336802267378727 and parameters: {'n_estimators': 402, 'max_depth': 36, 'min_samples_split': 12, 'min_samples_leaf': 2, 'max_features': 0.11985321438956228, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 02:49:10,777] Trial 68 finished with value: 0.6191734534926359 and parameters: {'n_estimators': 401, 'max_depth': 35, 'min_samples_split': 12, 'min_samples_leaf': 6, 'max_features': 0.1265855849614169, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 02:51:35,281] Trial 69 finished with value: 0.6105068047646077 and parameters: {'n_estimators': 370, 'max_depth': 37, 'min_samples_split': 11, 'min_samples_leaf': 10, 'max_features': 0.14970775534527145, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 02:54:46,985] Trial 70 finished with value: 0.6338025224458157 and parameters: {'n_estimators': 447, 'max_depth': 39, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 0.12506972359519677, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 02:57:24,386] Trial 71 finished with value: 0.6315236624334964 and parameters: {'n_estimators': 439, 'max_depth': 38, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 0.10058977216836264, 'bootstrap': False}. Best is trial 64 with value: 0.6346493331090748.
[I 2025-03-11 03:01:02,660] Trial 72 finished with value: 0.6351041237375576 and parameters: {'n_estimators': 500, 'max_depth': 40, 'min_samples_split': 11, 'min_samples_leaf': 2, 'max_features': 0.12564503192698598, 'bootstrap': False}. Best is trial 72 with value: 0.6351041237375576.
[I 2025-03-11 03:04:10,589] Trial 73 finished with value: 0.6167439960207428 and parameters: {'n_estimators': 497, 'max_depth': 40, 'min_samples_split': 11, 'min_samples_leaf': 7, 'max_features': 0.13079295177392752, 'bootstrap': False}. Best is trial 72 with value: 0.6351041237375576.
[I 2025-03-11 03:08:30,426] Trial 74 finished with value: 0.6362203571724588 and parameters: {'n_estimators': 475, 'max_depth': 36, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_features': 0.16044800538332105, 'bootstrap': False}. Best is trial 74 with value: 0.6362203571724588.
[I 2025-03-11 03:12:58,311] Trial 75 finished with value: 0.6361669361569048 and parameters: {'n_estimators': 473, 'max_depth': 36, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_features': 0.16819670872472556, 'bootstrap': False}. Best is trial 74 with value: 0.6362203571724588.
[I 2025-03-11 03:16:30,139] Trial 76 finished with value: 0.6364749019888654 and parameters: {'n_estimators': 478, 'max_depth': 36, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_features': 0.1211956238422473, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 03:21:01,961] Trial 77 finished with value: 0.6356793203951749 and parameters: {'n_estimators': 476, 'max_depth': 32, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.1732865897416445, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 03:25:37,001] Trial 78 finished with value: 0.6361547582929876 and parameters: {'n_estimators': 478, 'max_depth': 33, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.17333893759516966, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 03:37:55,550] Trial 79 finished with value: 0.6119217637253384 and parameters: {'n_estimators': 476, 'max_depth': 24, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.6094232571962, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 03:42:23,116] Trial 80 finished with value: 0.6337783548222016 and parameters: {'n_estimators': 477, 'max_depth': 28, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.17807790666474232, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 03:46:31,507] Trial 81 finished with value: 0.6362479803011923 and parameters: {'n_estimators': 462, 'max_depth': 33, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.15762696586019037, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 03:50:43,662] Trial 82 finished with value: 0.6360381729814143 and parameters: {'n_estimators': 457, 'max_depth': 32, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.16512325996953972, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 03:54:59,430] Trial 83 finished with value: 0.6358330571679274 and parameters: {'n_estimators': 461, 'max_depth': 32, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.16864559695133644, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 04:15:49,213] Trial 84 finished with value: 0.4196137642621592 and parameters: {'n_estimators': 459, 'max_depth': 32, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.992326836179392, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 04:20:19,350] Trial 85 finished with value: 0.6351399965327552 and parameters: {'n_estimators': 485, 'max_depth': 30, 'min_samples_split': 11, 'min_samples_leaf': 1, 'max_features': 0.1699772118654213, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 04:24:34,751] Trial 86 finished with value: 0.635739341830653 and parameters: {'n_estimators': 468, 'max_depth': 30, 'min_samples_split': 9, 'min_samples_leaf': 1, 'max_features': 0.1625859945347049, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 04:29:34,009] Trial 87 finished with value: 0.6354742948869152 and parameters: {'n_estimators': 457, 'max_depth': 33, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.20211856681699064, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 04:47:14,590] Trial 88 finished with value: 0.5807564703934153 and parameters: {'n_estimators': 470, 'max_depth': 29, 'min_samples_split': 9, 'min_samples_leaf': 1, 'max_features': 0.8377878907955474, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 04:51:19,629] Trial 89 finished with value: 0.6324295393829659 and parameters: {'n_estimators': 487, 'max_depth': 25, 'min_samples_split': 9, 'min_samples_leaf': 1, 'max_features': 0.1619196640589218, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 04:56:01,881] Trial 90 finished with value: 0.6350082346868751 and parameters: {'n_estimators': 448, 'max_depth': 31, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.19514253201931625, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 05:01:02,923] Trial 91 finished with value: 0.6354475528432477 and parameters: {'n_estimators': 458, 'max_depth': 33, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.20459249205825183, 'bootstrap': False}. Best is trial 76 with value: 0.6364749019888654.
[I 2025-03-11 05:04:55,796] Trial 92 finished with value: 0.6370103520033256 and parameters: {'n_estimators': 427, 'max_depth': 34, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.16097934211279252, 'bootstrap': False}. Best is trial 92 with value: 0.6370103520033256.
[I 2025-03-11 05:08:58,254] Trial 93 finished with value: 0.6380229062474505 and parameters: {'n_estimators': 433, 'max_depth': 34, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 0.1598767054073213, 'bootstrap': False}. Best is trial 93 with value: 0.6380229062474505.
[I 2025-03-11 05:15:31,900] Trial 94 finished with value: 0.6321843665303758 and parameters: {'n_estimators': 430, 'max_depth': 34, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 0.3008832102895299, 'bootstrap': False}. Best is trial 93 with value: 0.6380229062474505.
[I 2025-03-11 05:19:43,176] Trial 95 finished with value: 0.637864147910455 and parameters: {'n_estimators': 448, 'max_depth': 36, 'min_samples_split': 9, 'min_samples_leaf': 1, 'max_features': 0.15946882287881017, 'bootstrap': False}. Best is trial 93 with value: 0.6380229062474505.
[I 2025-03-11 05:25:44,844] Trial 96 finished with value: 0.6334084800866305 and parameters: {'n_estimators': 449, 'max_depth': 36, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_features': 0.2515601766919521, 'bootstrap': False}. Best is trial 93 with value: 0.6380229062474505.
[I 2025-03-11 05:29:41,544] Trial 97 finished with value: 0.6378299215719864 and parameters: {'n_estimators': 426, 'max_depth': 34, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 0.15805613811502128, 'bootstrap': False}. Best is trial 93 with value: 0.6380229062474505.
[I 2025-03-11 05:32:42,051] Trial 98 finished with value: 0.6375081021406674 and parameters: {'n_estimators': 424, 'max_depth': 35, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 0.10623581281112382, 'bootstrap': False}. Best is trial 93 with value: 0.6380229062474505.
[I 2025-03-11 05:35:30,799] Trial 99 finished with value: 0.6371231041480161 and parameters: {'n_estimators': 424, 'max_depth': 35, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 0.10022368675079485, 'bootstrap': False}. Best is trial 93 with value: 0.6380229062474505."""
