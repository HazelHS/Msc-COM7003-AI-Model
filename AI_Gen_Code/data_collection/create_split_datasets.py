"""
Split Combined Dataset

This script splits the combined dataset into two parts:
1. Pre-2022 dataset (2012-2021)
2. Post-2021 dataset (2022-2025)

Usage:
    python split_datasets.py
"""

import os
import pandas as pd
from datetime import datetime

def main():
    print("Starting splitting of combined dataset...")
    
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
    print(f"Total rows: {len(df)}")
    print(f"Date range: from {df['Date'].min()} to {df['Date'].max()}")
    
    # Count rows by year
    year_counts = df['Date'].str[:4].value_counts().sort_index()
    print("\nRows by year:")
    for year, count in year_counts.items():
        print(f"  {year}: {count} rows")
    
    # Split dataset by year
    df['Year'] = df['Date'].str[:4].astype(int)
    
    # Pre-2022 dataset (2012-2021)
    pre_2022_df = df[df['Year'] <= 2021].copy()
    pre_2022_df.drop(columns=['Year'], inplace=True)
    
    # Post-2021 dataset (2022-2025)
    post_2021_df = df[df['Year'] >= 2022].copy()
    post_2021_df.drop(columns=['Year'], inplace=True)
    
    # Save the split datasets
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Pre-2022 dataset
    pre_2022_path = os.path.join(combined_dir, f"combined_dataset_pre_2022_{timestamp}.csv")
    pre_2022_static_path = os.path.join(combined_dir, "combined_dataset_pre_2022.csv")
    pre_2022_df.to_csv(pre_2022_path, index=False)
    pre_2022_df.to_csv(pre_2022_static_path, index=False)
    
    # Post-2021 dataset
    post_2021_path = os.path.join(combined_dir, f"combined_dataset_post_2021_{timestamp}.csv")
    post_2021_static_path = os.path.join(combined_dir, "combined_dataset_post_2021.csv")
    post_2021_df.to_csv(post_2021_path, index=False)
    post_2021_df.to_csv(post_2021_static_path, index=False)
    
    print(f"\nDatasets split successfully!")
    print(f"Pre-2022 dataset (2012-2021):")
    print(f"  Rows: {len(pre_2022_df)}")
    print(f"  Date range: from {pre_2022_df['Date'].min()} to {pre_2022_df['Date'].max()}")
    print(f"  Output saved to:\n  - {pre_2022_path}\n  - {pre_2022_static_path}")
    
    print(f"\nPost-2021 dataset (2022-2025):")
    print(f"  Rows: {len(post_2021_df)}")
    print(f"  Date range: from {post_2021_df['Date'].min()} to {post_2021_df['Date'].max()}")
    print(f"  Output saved to:\n  - {post_2021_path}\n  - {post_2021_static_path}")

if __name__ == "__main__":
    main() 