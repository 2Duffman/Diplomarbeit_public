import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
train_idx, test_idx = train_test_split(
    df.index, test_size=0.20, random_state=42, shuffle=True
)
train_ids = df.loc[train_idx, "bgg_id"].to_numpy()
test_ids  = df.loc[test_idx , "bgg_id"].to_numpy()

np.savetxt("train_ids.txt", train_ids, fmt="%d")   # one integer per line
np.savetxt("test_ids.txt" , test_ids , fmt="%d")
