import pandas as pd
import os
import glob

print("Checking date ranges in input datasets...")

# Check the processed_exchanges data
exchange_dir = os.path.join('datasets', 'processed_exchanges')
averaged_file = os.path.join(exchange_dir, "averaged_exchanges.csv")

if os.path.exists(averaged_file):
    print(f"\nAveraged exchanges: {averaged_file}")
    df = pd.read_csv(averaged_file)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique dates: {df['Date'].nunique()}")
else:
    print(f"Averaged exchanges file not found: {averaged_file}")

# Check additional features
features_dir = os.path.join('datasets', 'additional_features')
feature_files = glob.glob(os.path.join(features_dir, '*.csv'))

for file_path in sorted(feature_files):
    file_name = os.path.basename(file_path)
    print(f"\nFeature file: {file_name}")
    
    try:
        # Skip comment lines
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        skiprows = 0
        if first_line.startswith('#'):
            skiprows = 1
            print(f"Skipping comment line in {file_name}")
        
        df = pd.read_csv(file_path, skiprows=skiprows)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            print(f"Shape: {df.shape}")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Unique dates: {df['Date'].nunique()}")
            
            # Show distribution of dates by year
            years = pd.to_datetime(df['Date']).dt.year
            year_counts = years.value_counts().sort_index()
            print("Date distribution by year:")
            for year, count in year_counts.items():
                print(f"  - {year}: {count} rows")
        else:
            print(f"No Date column found in {file_name}")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# Check the final datasets
print("\nFinal datasets:")
combined_dir = os.path.join('datasets', 'combined_dataset')
for file in ['combined_dataset_latest.csv', 'filtered_dataset_20250304_160818.csv']:
    file_path = os.path.join(combined_dir, file)
    if os.path.exists(file_path):
        print(f"\nDataset: {file}")
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Show distribution of dates by year
        years = pd.to_datetime(df['Date']).dt.year
        year_counts = years.value_counts().sort_index()
        print("Date distribution by year:")
        for year, count in year_counts.items():
            print(f"  - {year}: {count} rows")
    else:
        print(f"File not found: {file}") 