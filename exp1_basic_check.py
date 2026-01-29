import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Dataset loaded successfully")
print(df.head())
print("Dataset shape:", df.shape)
raise ValueError("Simulated dataset error for CI testing")

