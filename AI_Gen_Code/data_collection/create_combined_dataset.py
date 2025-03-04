"""
Create Combined Dataset For AI Training

This script combines all data sources (averaged exchange data, additional features) 
into a single aligned CSV file for AI training with the exact format requested.

Format example:
Date,AVG Open,AVG High,AVG Low,AVG Close,AVG Adj Close,AVG Volume,AVG CloseUSD,BTC/USD,Gold/BTC Ratio...

Usage:
    python create_combined_dataset.py
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import shutil
import time

def main():
    print("Starting creation of combined dataset for AI training...")
    
    # Base path for datasets - use environment variables if available
    if 'DATASETS_DIR' in os.environ and 'COMBINED_DATASET_DIR' in os.environ:
        base_dir = os.environ.get('DATASETS_DIR')
        combined_dir = os.environ.get('COMBINED_DATASET_DIR')
        print(f"Using environment variables for paths:")
        print(f"  DATASETS_DIR: {base_dir}")
        print(f"  COMBINED_DATASET_DIR: {combined_dir}")
    else:
        # Fall back to default paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        datasets_path = os.path.join(base_dir, "datasets")
        base_dir = datasets_path
        combined_dir = os.path.join(datasets_path, "combined_dataset")
        print(f"Using default paths:")
        print(f"  Base directory: {base_dir}")
        print(f"  Combined directory: {combined_dir}")
    
    # Create combined dataset directory if it doesn't exist
    os.makedirs(combined_dir, exist_ok=True)
    print(f"Ensured output directory exists: {combined_dir}")
    
    # Process averaged exchange data to establish the base date range
    exchange_dir = os.path.join(base_dir, "processed_exchanges")
    averaged_file = os.path.join(exchange_dir, "averaged_exchanges.csv")
    
    if os.path.exists(averaged_file):
        print("\nProcessing averaged exchange data...")
        try:
            # Read the averaged exchange file
            df = pd.read_csv(averaged_file)
            
            # Ensure Date column is properly formatted
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                print(f"  Exchange data date range: {df['Date'].min()} to {df['Date'].max()}")
                print(f"  Exchange data has {len(df)} rows")
            
            # Create a new DataFrame with date column for averaged exchanges
            exchange_df = pd.DataFrame()
            exchange_df['Date'] = df['Date']
            
            # Add each of the standard columns with 'AVG' prefix
            standard_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'CloseUSD']
            for col in standard_columns:
                if col in df.columns:
                    # Add column with AVG prefix
                    column_name = f"AVG {col}"
                    exchange_df[column_name] = df[col]
                elif col == 'Adj Close' and 'Close' in df.columns:
                    # If Adj Close is missing but Close exists, use Close value as Adj Close
                    column_name = f"AVG {col}"
                    exchange_df[column_name] = df['Close']
                    print(f"  Using Close as {column_name} (Adj Close not in source data)")
                else:
                    # Add empty column with AVG prefix
                    column_name = f"AVG {col}"
                    exchange_df[column_name] = np.nan
            
            # Use this as the base DataFrame with date as index for proper merging
            exchange_df.set_index('Date', inplace=True)
            
            # Create final_df with the exchange data
            final_df = exchange_df.copy()
            print(f"  Added {len(standard_columns)} averaged exchange columns")
            
        except Exception as e:
            print(f"Error processing averaged exchanges: {e}")
            return
    else:
        print("Warning: averaged_exchanges.csv not found. Please run average_exchanges.py first.")
        return
    
    # Process additional features
    additional_features_files = []
    features_dir = os.path.join(base_dir, "additional_features")
    if os.path.exists(features_dir):
        # Use non-validated files for additional features since they contain the Date column
        additional_features_files = [f for f in glob.glob(os.path.join(features_dir, "*.csv")) 
                                   if "validated" not in f]
    print(f"Found {len(additional_features_files)} additional feature files to process")
    
    print("\nProcessing additional features files...")
    for file_path in additional_features_files:
        file_name = os.path.basename(file_path)
        
        # Skip fear_greed_index.csv file
        if 'fear_greed_index' in file_name.lower():
            print(f"Skipping {file_name} as requested")
            continue
        
        print(f"Processing {file_name}...")
        
        try:
            # Extract feature type from filename
            feature_type = file_name.split('_')[0] if '_' in file_name else file_name.split('.')[0]
            
            # Try to read the file
            try:
                # Skip files that start with # or have comments at the beginning
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                
                # Skip comment lines in CSV read
                skiprows = 0
                if first_line.startswith('#'):
                    skiprows = 1
                    print(f"  Skipping comment line in {file_name}")
                
                # Read the file
                df = pd.read_csv(file_path, skiprows=skiprows)
                
                # Check if there is a Date column
                if 'Date' in df.columns:
                    # Ensure Date column is properly formatted
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    print(f"  {file_name} date range: {df['Date'].min()} to {df['Date'].max()}")
                    
                    # Create a new DataFrame with date column for this feature
                    feature_df = pd.DataFrame()
                    feature_df['Date'] = df['Date']
                    
                    # Add each column with feature prefix
                    for col in df.columns:
                        if col != 'Date':  # Skip Date as it's already added
                            # Skip "Miner Revenue (USD)" column
                            if col == "Miner Revenue (USD)":
                                print(f"  Skipping {col} column as requested")
                                continue
                                
                            # Special case for BTC/USD and Gold/BTC Ratio columns to not modify their names
                            if 'currency_metrics.csv' in file_path.lower() and (col == 'BTC/USD' or col == 'Gold/BTC Ratio'):
                                # Use original column name without feature prefix
                                column_name = col
                                print(f"  Adding special column {col} without prefix")
                            else:
                                # Add column with feature prefix
                                column_name = f"{feature_type} {col}"
                            feature_df[column_name] = df[col]
                    
                    # Set Date as index for proper merging
                    feature_df.set_index('Date', inplace=True)
                    
                    # Merge with the current output using outer join to preserve all dates
                    final_df = pd.concat([final_df, feature_df], axis=1, join='outer')
                    
                    # Report on merge results
                    print(f"  Added {len(df.columns) - 1} columns from {file_name}")
                    print(f"  Current merged dataset has {len(final_df)} rows")
                else:
                    print(f"  No Date column found in {file_name}, skipping.")
            except Exception as e:
                print(f"  Error reading {file_name}: {e}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Reset index to make Date a regular column again
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Sort data by Date (oldest to newest)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df.sort_values('Date', ascending=True, inplace=True)
    final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Check for truncated data
    if not final_df.empty:
        print(f"\nData date range check:")
        print(f"  Earliest date: {final_df['Date'].min()}")
        print(f"  Latest date: {final_df['Date'].max()}")
        print(f"  Total date range spans {len(final_df['Date'].unique())} unique days")
    
    # Fill missing values
    print("\nFilling missing values...")
    # Forward fill then backward fill to handle gaps
    final_df = final_df.ffill()
    final_df = final_df.bfill()
    
    # Count any remaining missing values
    missing_values = final_df.isna().sum().sum()
    print(f"Missing values after filling: {missing_values}")
    
    # If there are still missing values, print which columns have them
    if missing_values > 0:
        print("Columns with missing values:")
        for col in final_df.columns:
            null_count = final_df[col].isna().sum()
            if null_count > 0:
                print(f"  - {col}: {null_count} missing values")
    
    # Save the output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(combined_dir, f"combined_dataset_{timestamp}.csv")
    
    # Remove columns containing "Currency" before saving
    print("\nRemoving Currency columns...")
    currency_columns = [col for col in final_df.columns if 'Currency' in col]
    if currency_columns:
        print(f"  Removing {len(currency_columns)} Currency columns: {', '.join(currency_columns)}")
        final_df = final_df.drop(columns=currency_columns)
    
    # Print year distribution for verification
    years = pd.to_datetime(final_df['Date']).dt.year
    year_counts = years.value_counts().sort_index()
    print("\nDate distribution by year:")
    for year, count in year_counts.items():
        print(f"  - {year}: {count} rows")
    
    # Save the timestamped file
    final_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to: {output_path}")
    
    # Also save as static version with safe file handling
    static_path = os.path.join(combined_dir, "combined_dataset_latest.csv")
    backup_file = os.path.join(combined_dir, "combined_dataset_backup.csv")
    temp_file = os.path.join(combined_dir, "combined_dataset_temp.csv")
    
    try:
        # Save to a temporary file first
        final_df.to_csv(temp_file, index=False)
        
        # If the original file exists, try to create a backup
        if os.path.exists(static_path):
            try:
                # Make a backup of the original file
                shutil.copy2(static_path, backup_file)
                print(f"Created backup of original dataset")
            except Exception as e:
                print(f"Warning: Could not create backup of the original file: {e}")
        
        # Try to rename the temporary file to the final destination
        max_attempts = 5
        current_attempt = 0
        success = False
        
        while current_attempt < max_attempts and not success:
            try:
                # On Windows, we may need to remove the destination file first
                if os.path.exists(static_path):
                    try:
                        os.remove(static_path)
                    except Exception as e:
                        print(f"Warning: Could not remove existing file: {e}")
                        # Continue anyway to try the rename
                
                os.rename(temp_file, static_path)
                success = True
                print(f"Updated combined_dataset_latest.csv successfully")
            except Exception as e:
                current_attempt += 1
                if current_attempt < max_attempts:
                    print(f"Attempt {current_attempt} failed: {e}. Retrying in 1 second...")
                    time.sleep(1)
                else:
                    print(f"Error: Could not update combined_dataset_latest.csv after {max_attempts} attempts: {e}")
                    print("Using original output file only.")
                    # In case of failure, we still have the timestamped file
    except Exception as e:
        print(f"Error saving latest version: {e}")
    
    print("\nCombined dataset created successfully!")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")
    print(f"Date range: from {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"Output saved to:")
    print(f"- {output_path}")
    print(f"- {static_path}")

if __name__ == "__main__":
    main() 