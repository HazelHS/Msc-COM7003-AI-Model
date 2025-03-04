"""
Create Combined Dataset For AI Training

This script combines all data sources (exchange data, additional features) 
into a single aligned CSV file for AI training with the exact format requested.

Format example:
Date,GDAXI Open,GDAXI High,GDAXI Low,GDAXI Close,GDAXI Adj Close,GDAXI Volume,GDAXI CloseUSD,GDAXI Currency,IXIC Open...

Usage:
    python create_combined_dataset.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import re

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
    
    # Create the output DataFrame with Date column
    final_df = pd.DataFrame()
    
    # Process exchange data files
    exchange_files = []
    exchange_dir = os.path.join(base_dir, "processed_exchanges")
    if os.path.exists(exchange_dir):
        exchange_files = [f for f in glob.glob(os.path.join(exchange_dir, "*.csv")) 
                        if ("processed" in f or "validated" in f) and not "all_indices" in f]  # Include processed files, exclude all_indices
    print(f"Found {len(exchange_files)} exchange files to process")
    
    print("\nProcessing exchange data files...")
    for file_path in exchange_files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        try:
            # Read the exchange file
            df = pd.read_csv(file_path)
            
            # Determine the exchange symbol
            exchange_symbol = None
            if 'Index' in df.columns:
                # Get unique non-NaN values in Index column
                unique_indices = df['Index'].dropna().unique()
                if len(unique_indices) > 0:
                    exchange_symbol = unique_indices[0]
                    print(f"Found exchange symbol: {exchange_symbol}")
            
            # If no symbol found in Index column, use the filename
            if not exchange_symbol:
                exchange_symbol = file_name.split('_')[0]
            
            # Create a new DataFrame with date column for this exchange
            exchange_df = pd.DataFrame()
            exchange_df['Date'] = df['Date']
            
            # Add each of the standard columns with prefixed names
            standard_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'CloseUSD', 'Currency']
            for col in standard_columns:
                if col in df.columns:
                    # Add column with exchange prefix
                    column_name = f"{exchange_symbol} {col}"
                    exchange_df[column_name] = df[col]
                elif col == 'Adj Close' and 'Close' in df.columns:
                    # If Adj Close is missing but Close exists, use Close value as Adj Close
                    column_name = f"{exchange_symbol} {col}"
                    exchange_df[column_name] = df['Close']
                    print(f"  Using Close as {column_name} (Adj Close not in source data)")
                else:
                    # Add empty column with exchange prefix
                    column_name = f"{exchange_symbol} {col}"
                    exchange_df[column_name] = np.nan
            
            # If this is the first file, use it as the base
            if final_df.empty:
                final_df = exchange_df
            else:
                # Merge with the current output
                # First convert both to use Date as index
                final_df.set_index('Date', inplace=True)
                exchange_df.set_index('Date', inplace=True)
                
                # Merge the dataframes
                final_df = pd.concat([final_df, exchange_df], axis=1)
                
                # Reset index to make Date a regular column again
                final_df.reset_index(inplace=True)
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
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
                
                # Commented out: We want to keep the Gold/BTC Ratio column
                # For currency_metrics.csv specifically, remove Gold/BTC Ratio and BTC/Gold Ratio columns
                # if 'currency_metrics.csv' in file_path.lower():
                #     columns_to_remove = ['Gold/BTC Ratio', 'BTC/Gold Ratio']
                #     for col in columns_to_remove:
                #         if col in df.columns:
                #             df = df.drop(columns=[col])
                #             print(f"  Removed {col} column from {file_name}")
                
                # Check if there is a Date column
                if 'Date' in df.columns:
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
                    
                    # Merge with the current output
                    if not final_df.empty:
                        # First convert both to use Date as index
                        final_df.set_index('Date', inplace=True)
                        feature_df.set_index('Date', inplace=True)
                        
                        # Merge the dataframes
                        final_df = pd.concat([final_df, feature_df], axis=1)
                        
                        # Reset index to make Date a regular column again
                        final_df.reset_index(inplace=True)
                    else:
                        final_df = feature_df
                    
                    print(f"  Added {len(df.columns) - 1} columns from {file_name}")
                else:
                    print(f"  No Date column found in {file_name}, skipping.")
            except Exception as e:
                print(f"  Error reading {file_name}: {e}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Fill missing values
    print("\nFilling missing values...")
    final_df = final_df.ffill()
    final_df = final_df.bfill()
    
    # Count any remaining missing values
    missing_values = final_df.isna().sum().sum()
    print(f"Missing values after filling: {missing_values}")
    
    # Save the output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(combined_dir, f"combined_dataset_{timestamp}.csv")
    
    # Remove columns containing "Currency" before saving
    print("\nRemoving Currency columns...")
    currency_columns = [col for col in final_df.columns if 'Currency' in col]
    if currency_columns:
        print(f"  Removing {len(currency_columns)} Currency columns: {', '.join(currency_columns)}")
        final_df = final_df.drop(columns=currency_columns)
    
    final_df.to_csv(output_path, index=False)
    
    # Also save a static version
    static_path = os.path.join(combined_dir, "combined_dataset_latest.csv")
    final_df.to_csv(static_path, index=False)
    
    print(f"\nCombined dataset created successfully!")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")
    print(f"Date range: from {min(final_df['Date'])} to {max(final_df['Date'])}")
    print(f"Output saved to:\n- {output_path}\n- {static_path}")

if __name__ == "__main__":
    main() 