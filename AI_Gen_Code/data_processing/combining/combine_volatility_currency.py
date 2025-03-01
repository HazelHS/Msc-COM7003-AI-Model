import os
import pandas as pd
import numpy as np
from datetime import datetime

# Define paths
INPUT_DIR = '../datasets/additional_features'
OUTPUT_FILE = os.path.join(INPUT_DIR, 'volatility_currency_combined.csv')

def clean_csv_for_merge(filepath):
    """Clean CSV file to prepare for merging"""
    try:
        print(f"Processing: {filepath}")
        
        # Read the data, skipping comment lines
        # Read first 10 rows to analyze structure
        header_rows = pd.read_csv(filepath, nrows=10, comment='#')
        
        # Check if we need to handle special structure (date in rows)
        if 'Date' not in header_rows.columns and any('Date' in str(val) for val in header_rows.values.flatten()):
            print("  Special file structure detected, fixing...")
            
            # Find the row with 'Date' value
            date_row_idx = None
            for idx, row in header_rows.iterrows():
                if 'Date' in row.values:
                    date_row_idx = idx
                    break
            
            if date_row_idx is not None:
                # Read the file again, skipping to after the header rows
                df = pd.read_csv(filepath, comment='#', skiprows=date_row_idx+1)
                
                # Set column names from the date row
                header_row = pd.read_csv(filepath, comment='#', skiprows=date_row_idx, nrows=1)
                df.columns = header_row.columns
                
                # Make sure 'Date' is in the columns now
                if 'Date' in df.columns:
                    print("  Successfully extracted Date column")
                else:
                    print("  Warning: Could not extract Date column properly")
            else:
                # Default read method
                df = pd.read_csv(filepath, comment='#')
        else:
            # Standard file format
            df = pd.read_csv(filepath, comment='#')
        
        # Clean up problematic rows if needed
        if 'Ticker' in df.columns:
            df = df.drop(['Ticker'], errors='ignore')
        if 'Price' in df.columns:
            df = df.drop(['Price'], errors='ignore')
        
        # Handle Date column if it exists
        if 'Date' in df.columns:
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Drop rows with invalid dates
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                print(f"  Dropping {invalid_dates} rows with invalid dates")
                df = df[pd.notna(df['Date'])]
            # Set as index
            df.set_index('Date', inplace=True)
        else:
            print(f"  Warning: No Date column found in {filepath}")
            return None
        
        # Simplify column names by removing prefixes/suffixes if needed
        rename_dict = {}
        for col in df.columns:
            if col.startswith('^') or '=' in col:
                simplified = col.replace('^', '').replace('=F', '')
                rename_dict[col] = simplified
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        print(f"  Processed {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")
        return None

def main():
    """Combine volatility and currency data"""
    # Define files to combine
    volatility_file = os.path.join(INPUT_DIR, 'volatility_indices.csv')
    currency_file = os.path.join(INPUT_DIR, 'currency_metrics.csv')
    
    # Check if fixed versions exist first
    if os.path.exists(volatility_file.replace('.csv', '_fixed.csv')):
        volatility_file = volatility_file.replace('.csv', '_fixed.csv')
        print(f"Using fixed volatility file: {volatility_file}")
    
    if os.path.exists(currency_file.replace('.csv', '_fixed.csv')):
        currency_file = currency_file.replace('.csv', '_fixed.csv')
        print(f"Using fixed currency file: {currency_file}")
        
    print("\nProcessing volatility indices data...")
    volatility_df = clean_csv_for_merge(volatility_file)
    
    print("\nProcessing currency metrics data...")
    currency_df = clean_csv_for_merge(currency_file)
    
    if volatility_df is not None and currency_df is not None:
        # Combine the data
        print("\nCombining datasets...")
        
        # Align date ranges
        common_dates = volatility_df.index.intersection(currency_df.index)
        print(f"Common date range: {min(common_dates)} to {max(common_dates)}")
        print(f"Number of common dates: {len(common_dates)}")
        
        # Filter both dataframes to the common dates
        volatility_df = volatility_df.loc[common_dates]
        currency_df = currency_df.loc[common_dates]
        
        # Start with the first dataset and get its index
        combined_df = pd.DataFrame(index=common_dates)
        
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
        
        # Check for duplicate dates
        if combined_df.index.duplicated().any():
            print("Warning: Found duplicate dates in the combined dataframe")
            print("Removing duplicates...")
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        # Reset index to make Date a column again
        combined_df.reset_index(inplace=True)
        
        # Make sure date is properly formatted
        combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        output_file = OUTPUT_FILE.replace('.csv', '_fixed.csv')
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")
        print(f"Shape: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    else:
        print("Could not combine data due to errors")

if __name__ == "__main__":
    main() 