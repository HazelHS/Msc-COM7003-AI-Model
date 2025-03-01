import os
import pandas as pd
import numpy as np
from datetime import datetime

# Define paths
INPUT_DIR = 'additional_features'
OUTPUT_FILE = os.path.join(INPUT_DIR, 'combined_features_fixed.csv')

def load_and_fix_csv(filename):
    """Load a CSV file, fixing the date column and handling special cases"""
    filepath = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        # Skip comment lines starting with #
        df = pd.read_csv(filepath, comment='#')
        
        # Handle different CSV structures
        if 'Ticker' in df.columns:
            # This is likely the volatility_indices.csv with extra header rows
            # Find the actual data rows
            if 'Date' in df.columns:
                # Remove any rows where Date is empty or non-date
                df = df[pd.notna(df['Date'])]
                # Convert Date to datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # Drop rows with invalid dates
                df = df[pd.notna(df['Date'])]
        else:
            # Standard CSV structure
            if 'Date' in df.columns:
                # Convert Date to datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # Drop rows with invalid dates
                df = df[pd.notna(df['Date'])]
        
        # Set Date as index if it exists
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        # Add file source to column names to avoid overlaps
        file_prefix = filename.replace('.csv', '')
        df = df.add_prefix(f'{file_prefix}_')
        
        # Get the number of rows and columns
        num_rows = len(df)
        num_cols = len(df.columns)
        
        print(f"Loaded {filename}: {num_rows} rows, {num_cols} columns")
        
        return df
    
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def main():
    """Load and combine all datasets with proper handling"""
    # List of files to process
    files = [
        'fear_greed_index.csv',
        'volatility_indices.csv',
        'onchain_metrics.csv',
        'mining_difficulty.csv',
        'miner_revenue.csv',
        'currency_metrics.csv'
    ]
    
    # Create an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()
    
    # Load and process each file
    for filename in files:
        df = load_and_fix_csv(filename)
        
        if df is not None and not df.empty:
            # For the first DataFrame, just copy it
            if combined_df.empty:
                combined_df = df.copy()
            else:
                # Join with existing data, handling any missing dates
                combined_df = combined_df.join(df, how='outer')
    
    # Sort by date
    combined_df = combined_df.sort_index()
    
    # Handle any remaining NaN values
    # Forward fill first
    combined_df = combined_df.fillna(method='ffill')
    # Then backward fill for any remaining NaNs at the beginning
    combined_df = combined_df.fillna(method='bfill')
    
    # Save the combined data
    if not combined_df.empty:
        combined_df.to_csv(OUTPUT_FILE)
        print(f"Combined data saved to {OUTPUT_FILE}")
        print(f"Total shape: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        
        # Create a feature description file
        features_df = pd.DataFrame({
            'Feature': combined_df.columns.tolist(),
            'Source': [col.split('_')[0] for col in combined_df.columns],
            'Description': ['Feature from ' + col.split('_')[0] for col in combined_df.columns]
        })
        features_df.to_csv(os.path.join(INPUT_DIR, 'feature_descriptions_fixed.csv'), index=False)
        print(f"Feature descriptions saved to {os.path.join(INPUT_DIR, 'feature_descriptions_fixed.csv')}")
    else:
        print("No data to combine")

if __name__ == "__main__":
    main() 