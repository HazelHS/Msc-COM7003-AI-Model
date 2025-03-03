"""
Fix CloseUSD Values in Exchange Data Files

This script updates all existing CSV files in the processed_exchanges directory 
to fill in missing CloseUSD values using improved currency conversion.

Usage:
    python fix_closeusd_values.py

Author: AI Assistant
Created: March 2025
"""

import os
import pandas as pd
import glob
import time
from datetime import datetime

def load_index_info():
    """Load index information from indexInfo.csv"""
    # Find the indexInfo.csv file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_info_path = os.path.join(script_dir, 'indexInfo.csv')
    
    if not os.path.exists(index_info_path):
        print(f"Warning: indexInfo.csv not found at {index_info_path}")
        return None
    
    try:
        # Read the CSV file
        index_info = pd.read_csv(index_info_path)
        print(f"Loaded index information for {len(index_info)} indices")
        
        # Create a dictionary mapping index symbols to currencies
        index_currency_map = {}
        for _, row in index_info.iterrows():
            index_symbol = row['Index']
            currency = row['Currency']
            index_currency_map[index_symbol] = currency
            
            # Also add the symbol without the .XX suffix if present
            if '.' in index_symbol:
                base_symbol = index_symbol.split('.')[0]
                index_currency_map[base_symbol] = currency
        
        return index_currency_map
    except Exception as e:
        print(f"Error loading indexInfo.csv: {str(e)}")
        return None

def apply_currency_conversion(data, index_symbol=None, index_currency_map=None, verbose=True):
    """Apply improved currency conversion to calculate CloseUSD values"""
    if 'Close' not in data.columns:
        if verbose:
            print(f"  WARNING: 'Close' column not found in data")
        return data
    
    # Add or fix the Currency column using the index_info if available
    if index_currency_map and index_symbol and index_symbol in index_currency_map:
        currency = index_currency_map[index_symbol]
        if 'Currency' not in data.columns:
            data['Currency'] = currency
            if verbose:
                print(f"  Added Currency column with value {currency} for {index_symbol}")
        else:
            # Check if there are multiple currencies in the data
            unique_currencies = data['Currency'].unique()
            if len(unique_currencies) > 1:
                if verbose:
                    print(f"  Multiple currencies found in data: {unique_currencies}. Using index_info value: {currency}")
                data['Currency'] = currency
            elif unique_currencies[0] != currency:
                if verbose:
                    print(f"  Updating Currency from {unique_currencies[0]} to {currency} based on index_info")
                data['Currency'] = currency
    
    if 'Currency' not in data.columns:
        if verbose:
            print(f"  WARNING: No currency information - assuming USD")
        data['Currency'] = 'USD'
        
    # If CloseUSD column doesn't exist, create it
    if 'CloseUSD' not in data.columns:
        if verbose:
            print(f"  Creating CloseUSD column")
        data['CloseUSD'] = float('nan')  # Initialize with NaN values
    
    # Ensure Close column has numeric values
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    # Get unique currencies in the data
    currencies = data['Currency'].unique()
    
    for currency in currencies:
        # Create a mask for rows with this currency
        mask = data['Currency'] == currency
        
        if currency == 'USD':
            # For USD indices, CloseUSD is the same as Close
            data.loc[mask, 'CloseUSD'] = data.loc[mask, 'Close']
            if verbose:
                print(f"  Direct copy to CloseUSD for USD data")
        else:
            # Enhanced currency conversion rates
            conversion_rates = {
                'EUR': 1.08,    # 1 EUR = 1.08 USD
                'GBP': 1.26,    # 1 GBP = 1.26 USD
                'JPY': 0.0067,  # 1 JPY = 0.0067 USD
                'CNY': 0.14,    # 1 CNY = 0.14 USD
                'INR': 0.012,   # 1 INR = 0.012 USD
                'CAD': 0.74,    # 1 CAD = 0.74 USD
                'CHF': 1.13,    # 1 CHF = 1.13 USD
                'KRW': 0.00075, # 1 KRW = 0.00075 USD
                'TWD': 0.032,   # 1 TWD = 0.032 USD
                'AUD': 0.67,    # 1 AUD = 0.67 USD
                'HKD': 0.128,   # 1 HKD = 0.128 USD
                'SGD': 0.74,    # 1 SGD = 0.74 USD
                'NZD': 0.61,    # 1 NZD = 0.61 USD
                'MXN': 0.059,   # 1 MXN = 0.059 USD
                'BRL': 0.18,    # 1 BRL = 0.18 USD
                'ZAR': 0.055,   # 1 ZAR = 0.055 USD
                'RUB': 0.011,   # 1 RUB = 0.011 USD
                'TRY': 0.031,   # 1 TRY = 0.031 USD
                'SEK': 0.096,   # 1 SEK = 0.096 USD
                'NOK': 0.095,   # 1 NOK = 0.095 USD
                'DKK': 0.145,   # 1 DKK = 0.145 USD
                'PLN': 0.25,    # 1 PLN = 0.25 USD
                'ILS': 0.27,    # 1 ILS = 0.27 USD
                'THB': 0.028,   # 1 THB = 0.028 USD
                'IDR': 0.000064,# 1 IDR = 0.000064 USD
                'MYR': 0.21,    # 1 MYR = 0.21 USD
                'PHP': 0.018,   # 1 PHP = 0.018 USD
            }
            
            # Apply conversion if rate exists, otherwise use a reasonable estimate
            if currency in conversion_rates:
                rate = conversion_rates[currency]
                # Apply conversion to numeric values only
                data.loc[mask, 'CloseUSD'] = data.loc[mask, 'Close'] * rate
                if verbose:
                    print(f"  Converted {currency} to USD with rate: {rate}")
            else:
                # For unknown currencies, use a fallback method
                if verbose:
                    print(f"  WARNING: No conversion rate for {currency}. Using fallback method.")
                
                # Get average close for this currency
                avg_close = data.loc[mask, 'Close'].mean()
                
                # Heuristic: If average close is very large, it's likely a currency with low USD value
                if avg_close > 10000:
                    estimated_rate = 0.001  # Very small conversion rate
                elif avg_close > 1000:
                    estimated_rate = 0.01   # Small conversion rate
                elif avg_close > 100:
                    estimated_rate = 0.1    # Medium-small conversion rate
                elif avg_close > 10:
                    estimated_rate = 0.5    # Medium conversion rate
                elif avg_close > 1:
                    estimated_rate = 1.0    # 1:1 conversion rate
                else:
                    estimated_rate = 10.0   # Large conversion rate for very small values
                
                data.loc[mask, 'CloseUSD'] = data.loc[mask, 'Close'] * estimated_rate
                if verbose:
                    print(f"  Using estimated conversion rate: {estimated_rate} for {currency}")
    
    # Ensure no NaN values in CloseUSD
    if data['CloseUSD'].isna().any():
        nan_count = data['CloseUSD'].isna().sum()
        if verbose:
            print(f"  WARNING: Found {nan_count} NaN values in CloseUSD column. Filling with interpolated values.")
        
        # Try to interpolate missing values
        data['CloseUSD'] = data['CloseUSD'].interpolate(method='linear')
        
        # If any NaN values remain (at the beginning/end), fill with the closest valid value
        if data['CloseUSD'].isna().any():
            data['CloseUSD'] = data['CloseUSD'].ffill().bfill()
            
        # Final check - if any NaN values still remain, use Close values as a last resort
        if data['CloseUSD'].isna().any():
            if verbose:
                print("  WARNING: Unable to interpolate all NaN values. Using Close values as fallback.")
            data.loc[data['CloseUSD'].isna(), 'CloseUSD'] = data.loc[data['CloseUSD'].isna(), 'Close']
            
    # Fill any remaining NaN values in other columns (while we're at it)
    for col in data.columns:
        if col != 'CloseUSD' and pd.api.types.is_numeric_dtype(data[col]) and data[col].isna().any():
            nan_count = data[col].isna().sum()
            if verbose:
                print(f"  WARNING: Found {nan_count} NaN values in {col} column. Filling with interpolated values.")
            # Try to interpolate missing values
            data[col] = data[col].interpolate(method='linear')
            # If any NaN values remain, fill with the closest valid value
            if data[col].isna().any():
                data[col] = data[col].ffill().bfill()
                
    # Sort by Date if it's an index
    try:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
    except:
        pass
        
    return data

def fix_all_files():
    """Process all CSV files in the processed_exchanges directory"""
    # Load index information
    index_currency_map = load_index_info()
    
    # Path to the processed exchanges directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'processed_exchanges'))
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    # Create backup directory
    backup_dir = os.path.join(data_dir, 'backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Created backup directory: {backup_dir}")
    
    # Process each file
    total_files = len(csv_files)
    fixed_files = 0
    
    print(f"Found {total_files} CSV files to process")
    
    for i, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        print(f"\nProcessing [{i+1}/{total_files}]: {file_name}")
        
        try:
            # Extract index symbol from filename
            index_symbol = None
            if '_processed' in file_name:
                index_symbol = file_name.split('_processed')[0]
            
            # Backup the original file
            backup_path = os.path.join(backup_dir, file_name)
            print(f"  Creating backup at {backup_path}")
            with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            original_rows = len(df)
            print(f"  Loaded {original_rows} rows from {file_name}")
            
            # Count missing CloseUSD values before fix
            missing_before = df['CloseUSD'].isna().sum() if 'CloseUSD' in df.columns else "N/A"
            print(f"  Missing CloseUSD values before fix: {missing_before}")
            
            # Apply currency conversion
            df = apply_currency_conversion(df, index_symbol, index_currency_map)
            
            # Count missing CloseUSD values after fix
            missing_after = df['CloseUSD'].isna().sum()
            print(f"  Missing CloseUSD values after fix: {missing_after}")
            
            # Save the updated file
            df.to_csv(file_path, index=False)
            print(f"  Updated {file_name} successfully")
            fixed_files += 1
            
        except Exception as e:
            print(f"  Error processing {file_name}: {str(e)}")
    
    print(f"\nProcessing complete! Successfully fixed {fixed_files} out of {total_files} files.")

if __name__ == "__main__":
    start_time = time.time()
    print("Starting fix for CloseUSD values...")
    fix_all_files()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds") 