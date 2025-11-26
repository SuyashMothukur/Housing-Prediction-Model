import pandas as pd
import numpy as np

df = pd.read_csv('realtor-data.zip.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())

print("\n" + "=" * 50)
print("DATA TYPES & INFO")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("PRICE RANGE")
print("=" * 50)
print(f"Min price: ${df['price'].min():,.0f}")
print(f"Max price: ${df['price'].max():,.0f}")
print(f"Median price: ${df['price'].median():,.0f}")
print(f"Mean price: ${df['price'].mean():,.0f}")
