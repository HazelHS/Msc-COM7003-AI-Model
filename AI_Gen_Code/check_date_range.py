import pandas as pd
import os

# Set path to combined dataset
combined_dataset_path = os.path.join('..', 'datasets', 'combined_dataset', 'combined_dataset_latest.csv')

# Read the combined dataset
print(f"Reading dataset from: {combined_dataset_path}")
df = pd.read_csv(combined_dataset_path)

# Check date range
print(f"Total rows: {len(df)}")
print(f"Date range: from {df['Date'].min()} to {df['Date'].max()}")

# Count rows by year
year_counts = df['Date'].str[:4].value_counts().sort_index()
print("\nRows by year:")
for year, count in year_counts.items():
    print(f"  {year}: {count} rows") 