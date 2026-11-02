import pandas as pd

# Load the dataset
df = pd.read_csv("dataset.csv")

# Display the first five rows
print("First five rows:")
print(df.head())

# Display dataset information
print("\nDataset information:")
print(df.info())

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display dataset dimensions
print("\nDataset shape:")
print(df.shape)
