import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold

# ---------------- CONFIG -----------------
DATA_CSV       = "data.csv"   # full dataset with booleans
LOG_ALPHA_MIN  = np.log10(1.2e-5)
LOG_ALPHA_MAX  = np.log10(1)
N_POINTS       = 50           # number of alphas on the grid
CV_FOLDS       = 5
RANDOM_STATE   = 42

# 1. Load frozen split
from data_utils import load_split
X_train, X_test, y_train, y_test = load_split(DATA_CSV)

# 2. Build preprocessor (same as HPO setup)
num_cols  = X_train.select_dtypes(include="number").columns
bool_cols = X_train.columns.difference(num_cols)

num_pipe = [("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())]
preprocessor = ColumnTransformer(
    [("num",  Pipeline(num_pipe), num_cols),
     ("bool", "passthrough",      bool_cols)]
)

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# 3. Evaluate grid of alphas
alpha_grid = np.logspace(LOG_ALPHA_MIN, LOG_ALPHA_MAX, N_POINTS)
mse_vals = []
mse_stds = []
nz_vals  = []

for alpha in alpha_grid:
    model = Lasso(alpha=alpha, max_iter=10_000, random_state=RANDOM_STATE)
    pipe  = Pipeline([("prep", preprocessor), ("model", model)])
    
    # CV MSE (negative -> make positive)
    scores = cross_val_score(pipe, X_train, y_train,
                             scoring="neg_mean_squared_error",
                             cv=cv, n_jobs=-1)
    mse_cv = -scores.mean()
    mse_std = scores.std()
    mse_vals.append(mse_cv)
    mse_stds.append(mse_std)
    
    # Fit once on full train to count non‑zero coefficients
    pipe.fit(X_train, y_train)
    nz = (np.abs(pipe.named_steps["model"].coef_) > 1e-6).sum()
    nz_vals.append(nz)

alpha_grid = np.array(alpha_grid)
mse_vals   = np.array(mse_vals)
mse_stds   = np.array(mse_stds)
nz_vals    = np.array(nz_vals)

# 1-SE rule: find smallest alpha with MSE <= min(MSE) + std at min(MSE)
min_idx = np.argmin(mse_vals)
mse_1se = mse_vals[min_idx] + mse_stds[min_idx]
one_se_idx = np.where(mse_vals <= mse_1se)[0][-1]  # largest alpha satisfying the rule
alpha_1se = alpha_grid[one_se_idx]
print(f"1-SE alpha: {alpha_1se:.4f} (index {one_se_idx})")
print(f"Non-zero coefficients at 1-SE alpha: {nz_vals[one_se_idx]}")
print(f"Lowest MSE alpha: {alpha_grid[min_idx]:.4f} (index {min_idx})")
print(f"Non-zero coefficients at lowest MSE alpha: {nz_vals[min_idx]}")
print(f"CV MSE at 1-SE alpha: {mse_vals[one_se_idx]:.4f}")
print(f"CV MSE at lowest MSE alpha: {mse_vals[min_idx]:.4f}")

# 4. Plot dual‑axis chart
fig, ax1 = plt.subplots(figsize=(7,4))

color1 = "tab:blue"
ax1.set_xscale("log")
ax1.set_xlabel("Alpha parameter")
ax1.set_ylabel("Non‑zero coefficients", color=color1)
ax1.plot(alpha_grid, nz_vals, marker="o", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

# Add vertical line for 1-SE rule
ax1.axvline(alpha_1se, color="green", linestyle="--", label="1-SE alpha")

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("CV MSE", color=color2)
ax2.plot(alpha_grid, mse_vals, marker="o", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Sparsity vs CV MSE for Lasso Regression")
fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.85))
plt.tight_layout()
plt.show()
