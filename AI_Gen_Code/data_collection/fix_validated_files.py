"""
Fix Missing Data in Validated CSV Files

This script identifies and fixes missing data in 'validated' CSV files
by copying data from the original files while preserving the validation.

Usage:
    python fix_validated_files.py
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

def main():
    print("Starting validation fix process...")
    
    # Base path for datasets
    datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
    
    # Create backup directory
    backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(datasets_path, f"backup_validated_{backup_timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Created backup directory: {backup_dir}")
    
    # Find all validated CSV files
    validated_files = []
    for root, dirs, files in os.walk(datasets_path):
        for file in files:
            if '_validated.csv' in file.lower():
                validated_files.append(os.path.join(root, file))
    
    print(f"Found {len(validated_files)} validated CSV files")
    
    # Process each validated file
    for val_file_path in validated_files:
        # Find the corresponding non-validated file
        base_name = os.path.basename(val_file_path).replace('_validated.csv', '.csv')
        base_dir = os.path.dirname(val_file_path)
        orig_file_path = os.path.join(base_dir, base_name)
        
        if not os.path.exists(orig_file_path):
            print(f"Warning: Original file {orig_file_path} not found for {val_file_path}")
            continue
            
        print(f"\nProcessing: {os.path.basename(val_file_path)}")
        
        try:
            # Load both files
            val_df = pd.read_csv(val_file_path, parse_dates=['Date'], index_col='Date')
            orig_df = pd.read_csv(orig_file_path, parse_dates=['Date'], index_col='Date')
            
            # Count missing values before fix
            missing_before = val_df.isna().sum().sum()
            print(f"Missing values before fix: {missing_before}")
            
            # Create backup of validated file
            backup_file = os.path.join(backup_dir, os.path.basename(val_file_path))
            val_df.to_csv(backup_file)
            print(f"Backed up to: {backup_file}")
            
            # Preserve columns that exist only in validated file
            val_only_cols = [col for col in val_df.columns if col not in orig_df.columns]
            val_only_data = val_df[val_only_cols] if val_only_cols else None
            
            # Get all columns from original file
            common_cols = [col for col in orig_df.columns if col in val_df.columns]
            
            # For common columns, use original data where validated data is missing
            for col in common_cols:
                # Where validated data is NaN but original is not, use original
                mask = val_df[col].isna() & ~orig_df[col].isna()
                if mask.any():
                    val_df.loc[mask, col] = orig_df.loc[mask, col]
            
            # Fill remaining NaNs using interpolation where possible
            val_df = val_df.interpolate(method='time')
            
            # Count missing values after fix
            missing_after = val_df.isna().sum().sum()
            print(f"Missing values after fix: {missing_after}")
            print(f"Fixed {missing_before - missing_after} missing values")
            
            # Save the fixed file
            val_df.to_csv(val_file_path)
            print(f"Saved fixed file: {val_file_path}")
            
        except Exception as e:
            print(f"Error processing {val_file_path}: {e}")
    
    print("\nValidation fix process complete!")

if __name__ == "__main__":
    main() 