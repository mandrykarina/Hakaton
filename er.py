import pandas as pd
df = pd.read_parquet("data/train.parquet")
df.to_csv("train.csv", index=False)
da = pd.read_parquet("data/test.parquet")
da.to_csv("test.csv", index=False)

