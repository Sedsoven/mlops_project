import pandas as pd
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()

# Convert to DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Add target column
df["class"] = wine.target

print("Wine dataset loaded successfully")
print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDataset shape:", df.shape)
