import os
import pandas as pd
import numpy as np
from datetime import datetime

# Define paths
INPUT_DIR = 'additional_features'
OUTPUT_FILE = os.path.join(INPUT_DIR, 'volatility_currency_combined.csv')

def clean_csv_for_merge(filepath):
    """Clean CSV file to prepare for merging"""
    try:
        # Read the data, skipping comment lines
        df = pd.read_csv(filepath, comment='#')
        
        # Clean up problematic rows if needed
        if 'Ticker' in df.columns:
            df = df.drop(['Ticker', 'Price'], errors='ignore')
        
        # Handle Date column if it exists
        if 'Date' in df.columns:
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Drop rows with invalid dates
            df = df[pd.notna(df['Date'])]
            # Set as index
            df.set_index('Date', inplace=True)
        
        # Simplify column names by removing prefixes/suffixes if needed
        rename_dict = {}
        for col in df.columns:
            if col.startswith('^') or '=' in col:
                simplified = col.replace('^', '').replace('=F', '')
                rename_dict[col] = simplified
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        return df
    
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")
        return None

def main():
    """Combine volatility and currency data"""
    # Define files to combine
    volatility_file = os.path.join(INPUT_DIR, 'volatility_indices.csv')
    currency_file = os.path.join(INPUT_DIR, 'currency_metrics.csv')
    
    print("Processing volatility indices data...")
    volatility_df = clean_csv_for_merge(volatility_file)
    
    print("Processing currency metrics data...")
    currency_df = clean_csv_for_merge(currency_file)
    
    if volatility_df is not None and currency_df is not None:
        # Combine the data
        print("Combining datasets...")
        
        # Start with the first dataset and get its index
        combined_df = pd.DataFrame(index=volatility_df.index)
        
        # Add volatility columns
        for col in volatility_df.columns:
            if col not in ['Ticker', 'Date']:
                combined_df[f'Volatility_{col}'] = volatility_df[col]
        
        # Add currency columns
        for col in currency_df.columns:
            if col not in ['Ticker', 'Date']:
                combined_df[f'Currency_{col}'] = currency_df[col]
        
        # Fill NaN values
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        # Save to CSV
        combined_df.to_csv(OUTPUT_FILE)
        print(f"Combined data saved to {OUTPUT_FILE}")
        print(f"Shape: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    else:
        print("Could not combine data due to errors")

if __name__ == "__main__":
    main() 