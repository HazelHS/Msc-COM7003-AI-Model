"""
Data Cleaner Script

This script fixes issues with the date structures in various CSV files:
1. Fixes missing Date columns in currency_metrics.csv and volatility_currency_combined.csv
2. Removes duplicate dates in all_indices_processed.csv
3. Standardizes date formats across all datasets
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Define paths to dataset directories
DIRS = [
    '../datasets/additional_features',
    '../datasets/processed_exchanges'
]

def fix_currency_metrics():
    """
    Fix the currency_metrics.csv file which has Date as a row header instead of a column
    """
    filepath = '../datasets/additional_features/currency_metrics.csv'
    print(f"\nFixing: {filepath}")
    
    try:
        # Read the file with the first few rows to identify structure
        raw_data = pd.read_csv(filepath, nrows=10)
        
        # Check if the issue is that "Date" is in the header row (row 2)
        if 'Date' not in raw_data.columns and 'Ticker' in raw_data.columns:
            print(f"  Date column missing, fixing structure...")
            
            # Read the raw data again, skipping the actual header rows
            df = pd.read_csv(filepath, comment='#')
            
            # Find the row that contains "Date" (typically row 2)
            date_row_idx = None
            for idx, row in df.iterrows():
                if 'Date' in row.values:
                    date_row_idx = idx
                    break
            
            if date_row_idx is not None:
                # Extract column headers from this row
                headers = df.iloc[date_row_idx].values
                
                # Get data starting from the next row
                data = df.iloc[date_row_idx + 1:].reset_index(drop=True)
                
                # Set the new column names
                data.columns = headers
                
                # Clean the Date column
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                
                # Remove any rows with invalid dates
                data = data[pd.notna(data['Date'])]
                
                # Save the fixed file
                fixed_filepath = filepath.replace('.csv', '_fixed.csv')
                
                # Save with source info preserved
                with open(fixed_filepath, 'w') as f:
                    f.write("# Data Source: Yahoo Finance API (https://finance.yahoo.com/)\n")
                data.to_csv(fixed_filepath, index=False, mode='a')
                
                print(f"  Fixed data saved to {fixed_filepath}")
                return True
            else:
                print(f"  Could not find Date row in the file")
                return False
        else:
            print(f"  File structure appears different than expected, no fix applied")
            return False
    
    except Exception as e:
        print(f"  Error fixing {filepath}: {e}")
        return False

def fix_volatility_currency_combined():
    """
    Fix the volatility_currency_combined.csv file which has numeric row indices
    """
    filepath = '../datasets/additional_features/volatility_currency_combined.csv'
    print(f"\nFixing: {filepath}")
    
    try:
        # Read the file
        df = pd.read_csv(filepath)
        
        # Check if we have rows with numeric indices and date column is missing
        if 'Date' not in df.columns and df.columns[0] == '':
            print(f"  Date column missing, fixing structure...")
            
            # Get the first few rows to identify the structure
            first_rows = df.iloc[:5]
            
            # Look for rows containing "Date" and "Ticker"
            date_row_idx = None
            for idx, row in first_rows.iterrows():
                values = row.values
                if any('Date' in str(val) for val in values):
                    date_row_idx = idx
                    break
            
            if date_row_idx is not None:
                # Extract the row with column names
                headers = df.iloc[date_row_idx]
                
                # Find where the date columns are
                date_indices = []
                for i, val in enumerate(headers):
                    if val == 'Date':
                        date_indices.append(i)
                
                if date_indices:
                    # Get actual data (skip header rows)
                    data = df.iloc[date_row_idx + 1:].reset_index(drop=True)
                    
                    # Create a new DataFrame with proper date column
                    # First, extract date column from one of the sections
                    date_idx = date_indices[0]
                    dates = data.iloc[:, date_idx]
                    
                    # Convert dates to datetime
                    dates = pd.to_datetime(dates, errors='coerce')
                    
                    # Create new DataFrame with Date as the first column
                    new_df = pd.DataFrame()
                    new_df['Date'] = dates
                    
                    # Add volatility columns
                    volatility_cols = [col for col in df.columns if col.startswith('Volatility_') and '_Date' not in col and '_Ticker' not in col]
                    for col in volatility_cols:
                        col_idx = df.columns.get_loc(col)
                        new_df[col] = data.iloc[:, col_idx]
                    
                    # Add currency columns
                    currency_cols = [col for col in df.columns if col.startswith('Currency_') and '_Date' not in col and '_Ticker' not in col]
                    for col in currency_cols:
                        col_idx = df.columns.get_loc(col)
                        new_df[col] = data.iloc[:, col_idx]
                    
                    # Drop rows with invalid dates
                    new_df = new_df[pd.notna(new_df['Date'])]
                    
                    # Save the fixed file
                    fixed_filepath = filepath.replace('.csv', '_fixed.csv')
                    new_df.to_csv(fixed_filepath, index=False)
                    print(f"  Fixed data saved to {fixed_filepath}")
                    return True
                else:
                    print(f"  Could not find Date column in header row")
                    return False
            else:
                print(f"  Could not find header row")
                return False
        else:
            print(f"  File structure appears different than expected, no fix applied")
            return False
    
    except Exception as e:
        print(f"  Error fixing {filepath}: {e}")
        return False

def fix_duplicate_dates():
    """
    Fix duplicate dates in all_indices_processed.csv
    """
    filepath = '../datasets/processed_exchanges/all_indices_processed.csv'
    print(f"\nFixing: {filepath}")
    
    try:
        # Read the file in chunks to handle large size
        chunk_size = 10000
        chunks = pd.read_csv(filepath, chunksize=chunk_size)
        
        # Process the first chunk to check structure
        first_chunk = next(chunks)
        
        # Reset the chunks iterator
        chunks = pd.read_csv(filepath, chunksize=chunk_size)
        
        if 'Date' in first_chunk.columns:
            print(f"  Found Date column, checking for duplicates...")
            
            # Combine all chunks
            all_data = pd.concat(chunks)
            
            # Check for duplicate dates
            if 'Index' in all_data.columns:
                # Count duplicates by combining Date and Index
                duplicates = all_data.duplicated(subset=['Date', 'Index']).sum()
                if duplicates > 0:
                    print(f"  Found {duplicates} duplicate entries (Date+Index combinations)")
                    
                    # Remove duplicates
                    all_data = all_data.drop_duplicates(subset=['Date', 'Index'])
                    
                    # Save the fixed file
                    fixed_filepath = filepath.replace('.csv', '_fixed.csv')
                    all_data.to_csv(fixed_filepath, index=False)
                    print(f"  Fixed data saved to {fixed_filepath}")
                    return True
                else:
                    print(f"  No duplicates found with Date+Index as key")
            else:
                # Just check Date column
                duplicates = all_data.duplicated(subset=['Date']).sum()
                if duplicates > 0:
                    print(f"  Found {duplicates} duplicate dates")
                    
                    # Remove duplicates, keeping the first occurrence
                    all_data = all_data.drop_duplicates(subset=['Date'], keep='first')
                    
                    # Save the fixed file
                    fixed_filepath = filepath.replace('.csv', '_fixed.csv')
                    all_data.to_csv(fixed_filepath, index=False)
                    print(f"  Fixed data saved to {fixed_filepath}")
                    return True
                else:
                    print(f"  No duplicate dates found")
            
            return False
        else:
            print(f"  No Date column found, cannot fix duplicates")
            return False
    
    except Exception as e:
        print(f"  Error fixing {filepath}: {e}")
        return False

def standardize_date_format(filepath):
    """
    Standardize date format in a CSV file
    """
    print(f"\nStandardizing dates in: {filepath}")
    
    try:
        # Read the file
        df = pd.read_csv(filepath)
        
        # Check if Date column exists
        if 'Date' in df.columns:
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Drop rows with invalid dates
            invalid_count = df['Date'].isna().sum()
            if invalid_count > 0:
                print(f"  Dropping {invalid_count} rows with invalid dates")
                df = df[pd.notna(df['Date'])]
            
            # Format date consistently as YYYY-MM-DD
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            # Save the standardized file
            df.to_csv(filepath, index=False)
            print(f"  Date format standardized")
            return True
        else:
            print(f"  No Date column found, cannot standardize")
            return False
    
    except Exception as e:
        print(f"  Error standardizing dates in {filepath}: {e}")
        return False

def main():
    """Main function to fix all issues"""
    print("Starting data cleaning process...")
    
    # Fix specific files with known issues
    fix_currency_metrics()
    fix_volatility_currency_combined()
    fix_duplicate_dates()
    
    # Find all CSV files to standardize
    for dir_path in DIRS:
        if os.path.exists(dir_path):
            print(f"\nProcessing directory: {dir_path}")
            for file in os.listdir(dir_path):
                if file.endswith('.csv') and not file.endswith('_fixed.csv'):
                    filepath = os.path.join(dir_path, file)
                    standardize_date_format(filepath)
    
    print("\nData cleaning process completed!")

if __name__ == "__main__":
    main() 