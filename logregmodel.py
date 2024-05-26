import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
col_names = [i for i in range(0, 1520)]
df = pd.read_csv('data.csv')
X = df.loc[:, df.columns != 'OUT']
y = df['OUT']
model = LogisticRegression(random_state=16)
model.fit(X, y)
from pickle import dump
with open("filename.pkl", "wb") as f:
    dump(model, f, protocol=5)