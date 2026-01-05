import optuna
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_regression
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
    # Define the search space for hyperparameters
    max_depth = trial.suggest_int('max_depth', 30, 60)  # Extended range based on your findings
    n_estimators = trial.suggest_int('n_estimators', 100, 300)  # Allowing a wider range

    # Create the model with the current hyperparameters
    base_estimator = DecisionTreeRegressor(max_depth=max_depth)
    model = AdaBoostRegressor(estimator=base_estimator, n_estimators=n_estimators)

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
best_model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=best_params['max_depth']),
    n_estimators=best_params['n_estimators']
)
best_model.fit(X_train, y_train)

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print("Test R^2 Score:", test_score)

"""
[I 2025-03-10 14:02:08,288] A new study created in memory with name: no-name-22aad897-4617-48ab-bcd3-dfea1f03fbe6
[I 2025-03-10 14:04:48,381] Trial 0 finished with value: 0.46470699101064195 and parameters: {'max_depth': 8, 'n_estimators': 48}. Best is trial 0 with value: 0.46470699101064195.
[I 2025-03-10 14:11:20,788] Trial 1 finished with value: 0.5627453398040873 and parameters: {'max_depth': 12, 'n_estimators': 97}. Best is trial 1 with value: 0.5627453398040873.
[I 2025-03-10 14:16:30,158] Trial 2 finished with value: 0.6136225736976281 and parameters: {'max_depth': 58, 'n_estimators': 77}. Best is trial 2 with value: 0.6136225736976281.
[I 2025-03-10 14:21:41,626] Trial 3 finished with value: 0.616424705929524 and parameters: {'max_depth': 82, 'n_estimators': 75}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:23:53,571] Trial 4 finished with value: 0.6048827232997029 and parameters: {'max_depth': 49, 'n_estimators': 32}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:25:40,860] Trial 5 finished with value: 0.6037735746471202 and parameters: {'max_depth': 43, 'n_estimators': 26}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:25:52,877] Trial 6 finished with value: 0.515872570630488 and parameters: {'max_depth': 25, 'n_estimators': 3}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:31:45,541] Trial 7 finished with value: 0.5876169549903475 and parameters: {'max_depth': 14, 'n_estimators': 85}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:32:35,784] Trial 8 finished with value: 0.5814497758554931 and parameters: {'max_depth': 87, 'n_estimators': 13}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:36:34,641] Trial 9 finished with value: 0.6137852413455743 and parameters: {'max_depth': 52, 'n_estimators': 57}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:40:53,943] Trial 10 finished with value: 0.613131540930517 and parameters: {'max_depth': 99, 'n_estimators': 66}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:44:29,041] Trial 11 finished with value: 0.6141132226496697 and parameters: {'max_depth': 72, 'n_estimators': 56}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:47:22,251] Trial 12 finished with value: 0.6110508364920959 and parameters: {'max_depth': 74, 'n_estimators': 45}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:51:45,402] Trial 13 finished with value: 0.6147299761043592 and parameters: {'max_depth': 75, 'n_estimators': 69}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 14:56:32,651] Trial 14 finished with value: 0.6158313960712982 and parameters: {'max_depth': 75, 'n_estimators': 75}. Best is trial 3 with value: 0.616424705929524.
[I 2025-03-10 15:03:33,040] Trial 15 finished with value: 0.6197141600763477 and parameters: {'max_depth': 96, 'n_estimators': 99}. Best is trial 15 with value: 0.6197141600763477.
[I 2025-03-10 15:10:13,298] Trial 16 finished with value: 0.6166354336631074 and parameters: {'max_depth': 99, 'n_estimators': 95}. Best is trial 15 with value: 0.6197141600763477.
[I 2025-03-10 15:16:55,965] Trial 17 finished with value: 0.6178218698456608 and parameters: {'max_depth': 100, 'n_estimators': 100}. Best is trial 15 with value: 0.6197141600763477.
[I 2025-03-10 15:23:38,049] Trial 18 finished with value: 0.6174858402072141 and parameters: {'max_depth': 90, 'n_estimators': 100}. Best is trial 15 with value: 0.6197141600763477.
[I 2025-03-10 15:29:35,555] Trial 19 finished with value: 0.6168911778018049 and parameters: {'max_depth': 63, 'n_estimators': 89}. Best is trial 15 with value: 0.6197141600763477."""


"""
[I 2025-03-10 17:18:07,743] A new study created in memory with name: no-name-00483946-fb3c-4e7e-a9be-d9f383c2e555
[I 2025-03-10 17:19:09,426] Trial 0 finished with value: 0.5907890666838169 and parameters: {'max_depth': 16, 'n_estimators': 17}. Best is trial 0 with value: 0.5907890666838169.        
[I 2025-03-10 17:21:07,161] Trial 1 finished with value: 0.6112707180453512 and parameters: {'max_depth': 27, 'n_estimators': 34}. Best is trial 1 with value: 0.6112707180453512.        
[I 2025-03-10 17:22:59,742] Trial 2 finished with value: 0.5876236090427034 and parameters: {'max_depth': 14, 'n_estimators': 32}. Best is trial 1 with value: 0.6112707180453512.        
[I 2025-03-10 17:23:04,412] Trial 3 finished with value: 0.3239395165511015 and parameters: {'max_depth': 1, 'n_estimators': 4}. Best is trial 1 with value: 0.6112707180453512.
[I 2025-03-10 17:26:04,263] Trial 4 finished with value: 0.6143069716730635 and parameters: {'max_depth': 22, 'n_estimators': 52}. Best is trial 4 with value: 0.6143069716730635.        
[I 2025-03-10 17:26:59,373] Trial 5 finished with value: 0.5966724579455319 and parameters: {'max_depth': 30, 'n_estimators': 15}. Best is trial 4 with value: 0.6143069716730635.        
[I 2025-03-10 17:29:53,294] Trial 6 finished with value: 0.3860653727163789 and parameters: {'max_depth': 7, 'n_estimators': 66}. Best is trial 4 with value: 0.6143069716730635.
[I 2025-03-10 17:30:42,027] Trial 7 finished with value: 0.5523360192119326 and parameters: {'max_depth': 10, 'n_estimators': 16}. Best is trial 4 with value: 0.6143069716730635.        
[I 2025-03-10 17:32:30,604] Trial 8 finished with value: 0.6118650840600401 and parameters: {'max_depth': 23, 'n_estimators': 30}. Best is trial 4 with value: 0.6143069716730635.        
[I 2025-03-10 17:35:01,323] Trial 9 finished with value: 0.46793833992377537 and parameters: {'max_depth': 8, 'n_estimators': 53}. Best is trial 4 with value: 0.6143069716730635.        
[I 2025-03-10 17:40:03,352] Trial 10 finished with value: 0.613839774324601 and parameters: {'max_depth': 20, 'n_estimators': 89}. Best is trial 4 with value: 0.6143069716730635.        
[I 2025-03-10 17:44:56,707] Trial 11 finished with value: 0.6144456539548264 and parameters: {'max_depth': 21, 'n_estimators': 89}. Best is trial 11 with value: 0.6144456539548264.      
[I 2025-03-10 17:50:18,914] Trial 12 finished with value: 0.6193786436890315 and parameters: {'max_depth': 23, 'n_estimators': 98}. Best is trial 12 with value: 0.6193786436890315.      
[I 2025-03-10 17:56:08,016] Trial 13 finished with value: 0.6079893338224233 and parameters: {'max_depth': 18, 'n_estimators': 100}. Best is trial 12 with value: 0.6193786436890315.     
[I 2025-03-10 18:00:21,560] Trial 14 finished with value: 0.6178730261877632 and parameters: {'max_depth': 26, 'n_estimators': 75}. Best is trial 12 with value: 0.6193786436890315.      
[I 2025-03-10 18:04:36,828] Trial 15 finished with value: 0.6186313349491553 and parameters: {'max_depth': 26, 'n_estimators': 75}. Best is trial 12 with value: 0.6193786436890315.      
[I 2025-03-10 18:08:45,821] Trial 16 finished with value: 0.6181984299459014 and parameters: {'max_depth': 26, 'n_estimators': 72}. Best is trial 12 with value: 0.6193786436890315.      
[I 2025-03-10 18:13:40,814] Trial 17 finished with value: 0.6191198720221331 and parameters: {'max_depth': 30, 'n_estimators': 85}. Best is trial 12 with value: 0.6193786436890315.      
[I 2025-03-10 18:19:29,751] Trial 18 finished with value: 0.6193332141954365 and parameters: {'max_depth': 30, 'n_estimators': 100}. Best is trial 12 with value: 0.6193786436890315.     
[I 2025-03-10 18:25:09,832] Trial 19 finished with value: 0.6199301589486523 and parameters: {'max_depth': 30, 'n_estimators': 99}. Best is trial 19 with value: 0.6199301589486523.      
[I 2025-03-10 18:28:44,889] Trial 20 finished with value: 0.6173086588720841 and parameters: {'max_depth': 25, 'n_estimators': 62}. Best is trial 19 with value: 0.6199301589486523.      
[I 2025-03-10 18:34:30,579] Trial 21 finished with value: 0.620398905258608 and parameters: {'max_depth': 30, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 18:39:45,294] Trial 22 finished with value: 0.6196593244848895 and parameters: {'max_depth': 28, 'n_estimators': 93}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 18:44:27,071] Trial 23 finished with value: 0.6185384951874793 and parameters: {'max_depth': 29, 'n_estimators': 82}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 18:49:33,697] Trial 24 finished with value: 0.6179583879131998 and parameters: {'max_depth': 27, 'n_estimators': 93}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 18:54:14,713] Trial 25 finished with value: 0.6195772603211027 and parameters: {'max_depth': 28, 'n_estimators': 81}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 18:59:22,598] Trial 26 finished with value: 0.6180710485117592 and parameters: {'max_depth': 25, 'n_estimators': 94}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:02:48,637] Trial 27 finished with value: 0.6167757087365382 and parameters: {'max_depth': 24, 'n_estimators': 62}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:05:21,019] Trial 28 finished with value: 0.6157298971909737 and parameters: {'max_depth': 28, 'n_estimators': 43}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:10:56,038] Trial 29 finished with value: 0.5873786458276867 and parameters: {'max_depth': 14, 'n_estimators': 90}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:15:43,289] Trial 30 finished with value: 0.6117251906276897 and parameters: {'max_depth': 19, 'n_estimators': 81}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:20:19,773] Trial 31 finished with value: 0.6200307777283692 and parameters: {'max_depth': 28, 'n_estimators': 84}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:25:43,350] Trial 32 finished with value: 0.6186460283018891 and parameters: {'max_depth': 28, 'n_estimators': 94}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:30:48,231] Trial 33 finished with value: 0.6197473179686114 and parameters: {'max_depth': 30, 'n_estimators': 86}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:36:04,105] Trial 34 finished with value: 0.6045022711299526 and parameters: {'max_depth': 17, 'n_estimators': 85}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:40:03,008] Trial 35 finished with value: 0.6191995255108088 and parameters: {'max_depth': 30, 'n_estimators': 70}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:44:53,992] Trial 36 finished with value: 0.5845997675489938 and parameters: {'max_depth': 14, 'n_estimators': 78}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 19:45:47,391] Trial 37 finished with value: 0.3201309159781821 and parameters: {'max_depth': 3, 'n_estimators': 87}. Best is trial 21 with value: 0.620398905258608.        
[I 2025-03-10 19:51:31,269] Trial 38 finished with value: 0.6200594738032991 and parameters: {'max_depth': 30, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608.      
[I 2025-03-10 19:56:48,899] Trial 39 finished with value: 0.6163393486542894 and parameters: {'max_depth': 22, 'n_estimators': 96}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:02:26,752] Trial 40 finished with value: 0.6176909304687366 and parameters: {'max_depth': 27, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608.      
[I 2025-03-10 20:07:41,909] Trial 41 finished with value: 0.6182858476843585 and parameters: {'max_depth': 30, 'n_estimators': 92}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:12:31,290] Trial 42 finished with value: 0.6179067270958619 and parameters: {'max_depth': 29, 'n_estimators': 85}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:12:46,865] Trial 43 finished with value: 0.49871489815503267 and parameters: {'max_depth': 29, 'n_estimators': 4}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:14:22,162] Trial 44 finished with value: 0.6107265329217582 and parameters: {'max_depth': 25, 'n_estimators': 27}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:19:44,551] Trial 45 finished with value: 0.6193993179834011 and parameters: {'max_depth': 27, 'n_estimators': 96}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:24:31,651] Trial 46 finished with value: 0.6170404912637476 and parameters: {'max_depth': 23, 'n_estimators': 89}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:27:38,976] Trial 47 finished with value: 0.5493453778290021 and parameters: {'max_depth': 11, 'n_estimators': 56}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:29:47,999] Trial 48 finished with value: 0.614472520099031 and parameters: {'max_depth': 30, 'n_estimators': 37}. Best is trial 21 with value: 0.620398905258608.        
[I 2025-03-10 20:35:27,714] Trial 49 finished with value: 0.6189694301260756 and parameters: {'max_depth': 27, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608.      
[I 2025-03-10 20:39:38,650] Trial 50 finished with value: 0.6147332822485467 and parameters: {'max_depth': 21, 'n_estimators': 76}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:44:47,949] Trial 51 finished with value: 0.6190162142523148 and parameters: {'max_depth': 28, 'n_estimators': 92}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:50:10,644] Trial 52 finished with value: 0.6193541586433433 and parameters: {'max_depth': 29, 'n_estimators': 96}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:55:09,032] Trial 53 finished with value: 0.6197650680357645 and parameters: {'max_depth': 26, 'n_estimators': 89}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 20:59:59,290] Trial 54 finished with value: 0.6183521429230322 and parameters: {'max_depth': 26, 'n_estimators': 88}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:03:46,774] Trial 55 finished with value: 0.6166249087660403 and parameters: {'max_depth': 24, 'n_estimators': 69}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:08:27,467] Trial 56 finished with value: 0.6179597684754676 and parameters: {'max_depth': 30, 'n_estimators': 82}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:12:57,895] Trial 57 finished with value: 0.6176823185310658 and parameters: {'max_depth': 29, 'n_estimators': 79}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:18:22,284] Trial 58 finished with value: 0.6192348072675486 and parameters: {'max_depth': 26, 'n_estimators': 96}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:23:42,863] Trial 59 finished with value: 0.6195606974418327 and parameters: {'max_depth': 24, 'n_estimators': 97}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:28:28,638] Trial 60 finished with value: 0.6187142301279913 and parameters: {'max_depth': 28, 'n_estimators': 85}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:33:41,423] Trial 61 finished with value: 0.6191426545209711 and parameters: {'max_depth': 28, 'n_estimators': 91}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:39:12,768] Trial 62 finished with value: 0.6198728154780827 and parameters: {'max_depth': 29, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608.      
[I 2025-03-10 21:44:38,422] Trial 63 finished with value: 0.6180115339769452 and parameters: {'max_depth': 29, 'n_estimators': 97}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:50:06,911] Trial 64 finished with value: 0.6202106862287629 and parameters: {'max_depth': 27, 'n_estimators': 99}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 21:55:25,467] Trial 65 finished with value: 0.6183677709819786 and parameters: {'max_depth': 26, 'n_estimators': 98}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:00:36,937] Trial 66 finished with value: 0.6193256302392882 and parameters: {'max_depth': 27, 'n_estimators': 93}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:06:06,481] Trial 67 finished with value: 0.6168054000672635 and parameters: {'max_depth': 25, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608.      
[I 2025-03-10 22:11:09,275] Trial 68 finished with value: 0.6180114734540009 and parameters: {'max_depth': 29, 'n_estimators': 90}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:16:29,571] Trial 69 finished with value: 0.6190208798585115 and parameters: {'max_depth': 28, 'n_estimators': 94}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:19:31,157] Trial 70 finished with value: 0.23263493269569907 and parameters: {'max_depth': 5, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608.      
[I 2025-03-10 22:24:34,103] Trial 71 finished with value: 0.6194783499197126 and parameters: {'max_depth': 30, 'n_estimators': 88}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:29:18,701] Trial 72 finished with value: 0.6183382281021903 and parameters: {'max_depth': 30, 'n_estimators': 84}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:34:38,377] Trial 73 finished with value: 0.6198085496868766 and parameters: {'max_depth': 27, 'n_estimators': 95}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:39:54,146] Trial 74 finished with value: 0.6192797272348647 and parameters: {'max_depth': 27, 'n_estimators': 94}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:45:24,315] Trial 75 finished with value: 0.6181339607830225 and parameters: {'max_depth': 26, 'n_estimators': 98}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:50:44,718] Trial 76 finished with value: 0.5628736017814374 and parameters: {'max_depth': 12, 'n_estimators': 91}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 22:56:30,953] Trial 77 finished with value: 0.6171735339924019 and parameters: {'max_depth': 25, 'n_estimators': 95}. Best is trial 21 with value: 0.620398905258608.       
[I 2025-03-10 23:02:46,977] Trial 78 finished with value: 0.6188495388277785 and parameters: {'max_depth': 29, 'n_estimators': 100}. Best is trial 21 with value: 0.620398905258608."""