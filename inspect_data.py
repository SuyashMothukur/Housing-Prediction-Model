import pandas as pd

df = pd.read_csv("data.txt", nrows=5)
print(df.columns.tolist())
