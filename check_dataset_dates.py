import pandas as pd
import os

# Paths to datasets
filtered_path = os.path.join('datasets', 'combined_dataset', 'filtered_dataset_20250304_160818.csv')
combined_path = os.path.join('datasets', 'combined_dataset', 'combined_dataset_20250304_160812.csv')

print("Checking date ranges in datasets...")

# Check filtered dataset
if os.path.exists(filtered_path):
    print(f"\nFiltered dataset: {filtered_path}")
    df_filtered = pd.read_csv(filtered_path)
    print(f"Shape: {df_filtered.shape}")
    print(f"Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
    print("Last 10 dates:")
    last_dates = df_filtered['Date'].sort_values().tail(10).values
    for date in last_dates:
        print(f"  - {date}")
else:
    print(f"Filtered dataset not found: {filtered_path}")

# Check combined dataset
if os.path.exists(combined_path):
    print(f"\nCombined dataset: {combined_path}")
    df_combined = pd.read_csv(combined_path)
    print(f"Shape: {df_combined.shape}")
    print(f"Date range: {df_combined['Date'].min()} to {df_combined['Date'].max()}")
    print("Last 10 dates:")
    last_dates = df_combined['Date'].sort_values().tail(10).values
    for date in last_dates:
        print(f"  - {date}")
else:
    print(f"Combined dataset not found: {combined_path}")

# List other files
print("\nOther files in combined_dataset directory:")
combined_dir = os.path.join('datasets', 'combined_dataset')
for file in os.listdir(combined_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(combined_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {file} ({file_size:.1f} KB)") 