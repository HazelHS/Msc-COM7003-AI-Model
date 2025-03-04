"""
Filter Combined Dataset

This script filters the combined dataset to only include specified columns.
It reads the combined_dataset_latest.csv file and creates a new version with only the selected columns.
It also removes the 'AVG' prefix from column names.

Usage:
    python filter_combined_dataset.py
"""

import os
import pandas as pd
import numpy as np
import shutil
from datetime import datetime
import time
import sys

def main():
    print("Starting filtering of combined dataset...")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Define paths
    combined_dir = os.path.join(project_root, "datasets", "combined_dataset")
    combined_file = os.path.join(combined_dir, "combined_dataset_latest.csv")
    
    if not os.path.exists(combined_file):
        print(f"Error: Combined dataset file not found at: {combined_file}")
        print("Please run create_combined_dataset.py first.")
        return 1
    
    print(f"Reading combined dataset from: {combined_file}")
    try:
        # Read the combined dataset
        df = pd.read_csv(combined_file)
        original_columns = list(df.columns)
        original_rows = len(df)
        
        print(f"Original dataset: {original_rows} rows, {len(original_columns)} columns")
        
        # Process the date column and ensure it's in the correct format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # List of columns to keep
        columns_to_keep = [
            "Date",
            "AVG Volume",
            "AVG CloseUSD",
            "currency US Dollar Index",
            "currency Gold Futures",
            "BTC/USD",
            "Gold/BTC Ratio",
            "onchain Active Addresses",
            "onchain Transaction Count",
            "onchain Mempool Size",
            "onchain Hash Rate (GH/s)",
            "onchain Mining Difficulty",
            "onchain Transaction Fees (BTC)",
            "onchain Median Confirmation Time (min)"
        ]
        
        # Remove fear and greed from the columns list
        columns_to_keep = [col for col in columns_to_keep if "fear" not in col.lower()]
        
        # Verify which columns exist in the dataset
        existing_columns = []
        missing_columns = []
        
        for col in columns_to_keep:
            if col in df.columns:
                existing_columns.append(col)
            else:
                missing_columns.append(col)
        
        if missing_columns:
            print("\nWarning: The following columns were not found in the dataset:")
            for col in missing_columns:
                print(f"  - {col}")
        
        print(f"\nKeeping {len(existing_columns)} columns out of {len(original_columns)} total")
        
        # Filter to keep only the specified columns
        filtered_df = df[existing_columns].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Remove 'AVG' prefix from column names
        rename_dict = {}
        for col in filtered_df.columns:
            if col.startswith('AVG '):
                new_col = col.replace('AVG ', '')
                rename_dict[col] = new_col
                print(f"Renaming: {col} -> {new_col}")
        
        # Apply the renaming
        if rename_dict:
            filtered_df.rename(columns=rename_dict, inplace=True)  # Use inplace=True for direct modification
            print(f"Removed 'AVG' prefix from {len(rename_dict)} columns")
        
        # Print renamed columns to verify
        print("\nColumns after renaming:")
        for col in filtered_df.columns:
            print(f"  - {col}")
        
        # Ensure data is sorted by date (oldest to newest)
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        filtered_df = filtered_df.sort_values('Date', ascending=True)
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Generate output file paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(combined_dir, f"filtered_dataset_{timestamp}.csv")
        latest_path = os.path.join(combined_dir, f"filtered_dataset_latest.csv")
        backup_path = os.path.join(combined_dir, f"filtered_dataset_backup.csv")
        temp_path = os.path.join(combined_dir, f"filtered_dataset_temp.csv")
        
        # Save the timestamped file
        filtered_df.to_csv(output_path, index=False)
        print(f"Saved filtered dataset to: {output_path}")
        
        # Implement safe file handling for the latest version
        try:
            # Save to a temporary file first
            filtered_df.to_csv(temp_path, index=False)
            
            # If the original file exists, create a backup
            if os.path.exists(latest_path):
                try:
                    shutil.copy2(latest_path, backup_path)
                    print(f"Created backup of original dataset")
                except Exception as e:
                    print(f"Warning: Could not create backup file: {e}")
            
            # Try a few times to replace the existing file
            max_attempts = 5
            current_attempt = 0
            success = False
            
            while current_attempt < max_attempts and not success:
                try:
                    # On Windows, we may need to remove the destination file first
                    if os.path.exists(latest_path):
                        try:
                            os.remove(latest_path)
                        except Exception as e:
                            print(f"Warning: Could not remove existing file: {e}")
                    
                    # Rename the temp file to the latest file
                    os.rename(temp_path, latest_path)
                    success = True
                    print(f"Updated combined_dataset_latest.csv with filtered columns")
                except Exception as e:
                    current_attempt += 1
                    if current_attempt < max_attempts:
                        print(f"Attempt {current_attempt} failed: {e}. Retrying in 1 second...")
                        time.sleep(1)
                    else:
                        print(f"Error: Could not update filtered_dataset_latest.csv after {max_attempts} attempts: {e}")
            
            if not success:
                print(f"Using original output file only: {output_path}")
                
        except Exception as e:
            print(f"Error updating latest file: {e}")
        
        # Print year distribution for verification
        print(f"\nFiltered dataset summary:")
        print(f"Rows: {len(filtered_df)}")
        print(f"Columns: {len(filtered_df.columns)}")
        print(f"Date range: from {filtered_df['Date'].min()} to {filtered_df['Date'].max()}")
        
        # Print column summary
        print("\nColumns included:")
        for col in filtered_df.columns:
            non_null_count = filtered_df[col].count()
            print(f"  - {col}: {non_null_count} non-null values")
                
        return 0
        
    except Exception as e:
        print(f"Error filtering combined dataset: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 