import pandas as pd

df = pd.read_csv("dataset/DATASET SELF-EFFICACY_WELL-BEING.csv", sep=';')

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nStatistics:")
print(df.describe())