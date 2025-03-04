"""
Sort Combined Dataset in Ascending Chronological Order

This script takes the combined dataset and ensures it's sorted with the
oldest dates first (2012 to 2025 in ascending order).

Usage:
    python sort_combined_dataset.py
"""

import os
import pandas as pd
from datetime import datetime

def main():
    print("Starting to sort combined dataset in ascending chronological order...")
    
    # Base path for datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasets_path = os.path.join(base_dir, "datasets")
    combined_dir = os.path.join(datasets_path, "combined_dataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(combined_dir, exist_ok=True)
    print(f"Using dataset directory: {combined_dir}")
    
    # Path to the latest combined dataset
    combined_dataset_path = os.path.join(combined_dir, "combined_dataset_latest.csv")
    print(f"Reading combined dataset from: {combined_dataset_path}")
    
    # Read the combined dataset
    df = pd.read_csv(combined_dataset_path)
    
    # Print original date order
    print(f"Original first date: {df['Date'].iloc[0]}")
    print(f"Original last date: {df['Date'].iloc[-1]}")
    
    # Convert Date to datetime for proper sorting
    df['DateObj'] = pd.to_datetime(df['Date'])
    
    # Sort by date in ascending order (oldest first)
    df_sorted = df.sort_values('DateObj', ascending=True)
    
    # Remove the temporary DateObj column
    df_sorted.drop(columns=['DateObj'], inplace=True)
    
    # Print new date order
    print(f"Sorted first date (oldest): {df_sorted['Date'].iloc[0]}")
    print(f"Sorted last date (newest): {df_sorted['Date'].iloc[-1]}")
    
    # Save the sorted dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sorted_path = os.path.join(combined_dir, f"combined_dataset_chronological_{timestamp}.csv")
    sorted_static_path = os.path.join(combined_dir, "combined_dataset_latest.csv")
    
    # Save both timestamped and latest versions
    df_sorted.to_csv(sorted_path, index=False)
    df_sorted.to_csv(sorted_static_path, index=False)
    
    print(f"\nDataset sorted successfully!")
    print(f"Rows: {len(df_sorted)}")
    print(f"Date range: from {df_sorted['Date'].min()} to {df_sorted['Date'].max()} (ordered oldest first)")
    print(f"Output saved to:\n- {sorted_path}\n- {sorted_static_path}")

if __name__ == "__main__":
    main() 