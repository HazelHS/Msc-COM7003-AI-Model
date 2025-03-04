import pandas as pd
import os
import glob
from datetime import datetime

def check_dataset(pattern, title):
    print(f"\nChecking {title}...")
    base_dir = os.path.join(os.getcwd(), 'datasets', 'combined_dataset')
    
    # Find all files matching the pattern
    search_pattern = os.path.join(base_dir, pattern)
    print(f"Looking for files matching: {search_pattern}")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found matching pattern '{pattern}'")
        return
    
    print(f"Found {len(files)} files matching pattern '{pattern}'")
    
    # Sort files by last modified time
    files.sort(key=os.path.getmtime, reverse=True)
    
    print("All found files (sorted by date):")
    for file in files[:5]:  # Show top 5 files
        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"  {file} - {mod_time}")
    
    print(f"\nNewest {title}: {files[0]}")
    
    # Read the dataset
    try:
        df = pd.read_csv(files[0])
        print(f"Shape: {df.shape}")
        
        # Process date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            date_min = df['Date'].min().strftime('%Y-%m-%d')
            date_max = df['Date'].max().strftime('%Y-%m-%d')
            print(f"Date range: {date_min} to {date_max}")
            
            # Check if data is sorted by date
            is_sorted = df['Date'].is_monotonic_increasing
            print(f"Data is sorted by date (oldest to newest): {is_sorted}")
            
            # Display distribution of dates by year
            years = df['Date'].dt.year.unique()
            print("Date distribution by year:")
            for year in sorted(years):
                year_count = len(df[df['Date'].dt.year == year])
                print(f"  - {year}: {year_count} rows")
            
            # Verify fear and greed index has been removed
            if 'fear FearGreedValue' in df.columns:
                print("\nWARNING: Fear/Greed data is still present in the dataset")
            else:
                print("\nConfirmed: Fear/Greed data has been removed from the dataset")
    except Exception as e:
        print(f"Error reading or processing file: {e}")

def main():
    print("Starting check of latest datasets...")
    
    # Check combined dataset
    check_dataset("combined_dataset_*.csv", "combined dataset")
    
    # Check filtered dataset
    check_dataset("filtered_dataset_*.csv", "filtered dataset")

if __name__ == "__main__":
    main() 