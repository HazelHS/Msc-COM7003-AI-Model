import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime

# Define the correct Yahoo Finance ticker symbols for major indices
INDEX_SYMBOLS = {
    'NYA': '^NYA',         # NYSE Composite
    'IXIC': '^IXIC',       # NASDAQ Composite
    'HSI': '^HSI',         # Hang Seng Index
    '000001.SS': '^SSEC',  # Shanghai Composite
    'N225': '^N225',       # Nikkei 225
    'N100': '^N100',       # Euronext 100
    '399001.SZ': '399001.SZ',  # Shenzhen Component
    'GSPTSE': '^GSPTSE',   # S&P/TSX Composite
    'NSEI': '^NSEI',       # NIFTY 50
    'GDAXI': '^GDAXI',     # DAX
    'KS11': '^KS11',       # KOSPI
    'SSMI': '^SSMI',       # Swiss Market Index
    'TWII': '^TWII',       # Taiwan Weighted
    'J203.JO': 'J203.JO'   # JSE All Share
}

def get_exchange_data(index_symbol, start_date, end_date=None, currency='USD', max_retries=3):
    """
    Fetch historical data for a given index from Yahoo Finance
    
    Args:
        index_symbol (str): The symbol for the index (e.g., 'NYA' for NYSE)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to current date.
        currency (str, optional): Currency of the index. Defaults to 'USD'.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        
    Returns:
        pandas.DataFrame or None: DataFrame with historical data or None if fetching fails
    """
    # Add ^ prefix for indices if not already present
    if not index_symbol.startswith('^'):
        ticker = f"^{index_symbol}"
    else:
        ticker = index_symbol
    
    original_symbol = index_symbol
    
    print(f"Fetching data for {ticker} (original: {original_symbol})...")
    
    # Try to fetch data with exponential backoff
    for attempt in range(1, max_retries + 1):
        try:
            # Use yfinance to download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Check if data is empty or has only Date column
            if data.empty or (len(data.columns) <= 1):
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  Error: Empty data or insufficient columns, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Failed after {max_retries} attempts: Empty data or insufficient columns")
                    return None
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Check for required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"  Error: Missing columns {missing_columns}, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Failed after {max_retries} attempts: Missing columns {missing_columns}")
                    return None
            
            # Convert Date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Format Date as string in YYYY-MM-DD format
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
            
            # Check for duplicate dates
            if data['Date'].duplicated().any():
                print(f"  Warning: Found {data['Date'].duplicated().sum()} duplicate dates, keeping first occurrence")
                data = data.drop_duplicates(subset=['Date'], keep='first')
            
            # Fill NaN values in numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    # Use forward fill then backward fill to handle NaN values
                    # Updated to use ffill() and bfill() instead of fillna(method=...)
                    data[col] = data[col].ffill().bfill()
            
            # Add Index and Currency columns
            data['Index'] = original_symbol
            data['Currency'] = currency
            
            print(f"  Successfully fetched {len(data)} rows of data")
            return data
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"  Error: {str(e)}, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {str(e)}")
                return None
    
    return None

def process_exchange_data(data, index_symbol):
    """
    Process the exchange data, adding metadata and cleaning it for ML
    
    Args:
        data (pd.DataFrame): Raw exchange data from Yahoo Finance
        index_symbol (str): Symbol for the index being processed
        
    Returns:
        pd.DataFrame: Processed data ready for saving
    """
    if data is None or data.empty:
        print(f"  No data to process for {index_symbol}")
        return None
    
    try:
        # Calculate returns
        if 'Adj Close' in data.columns:
            price_col = 'Adj Close'
        else:
            price_col = 'Close'
        
        # Convert Date to datetime for sorting
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by date
        data = data.sort_values('Date')
        
        # Calculate daily returns
        data['Daily_Return'] = data[price_col].pct_change()
        
        # Calculate moving averages
        data['MA_5'] = data[price_col].rolling(window=5).mean()
        data['MA_20'] = data[price_col].rolling(window=20).mean()
        data['MA_50'] = data[price_col].rolling(window=50).mean()
        data['MA_200'] = data[price_col].rolling(window=200).mean()
        
        # Calculate volatility (rolling standard deviation)
        data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std()
        
        # Add CloseUSD column with proper currency conversion
        if 'Currency' in data.columns:
            # For USD indices, CloseUSD is the same as Close
            if data['Currency'].iloc[0] == 'USD':
                data['CloseUSD'] = data['Close']
                print(f"  {index_symbol} is in USD - direct copy to CloseUSD")
            else:
                # For non-USD indices, we need to convert
                # Since we don't have real forex data, we'll use this enhanced approach
                currency = data['Currency'].iloc[0]
                
                # Enhanced currency conversion rates with more currencies and updated values
                # In a production environment, these would be fetched from a forex API
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
                    data['CloseUSD'] = data['Close'] * rate
                    print(f"  {index_symbol} in {currency} converted to USD (rate: {rate})")
                else:
                    # For unknown currencies, log a warning and use a fallback method
                    print(f"  WARNING: No conversion rate available for {currency}. Using fallback method.")
                    
                    # Fallback method: Try to estimate based on the magnitude of the values
                    # This is a heuristic approach and should be replaced with actual rates
                    avg_close = data['Close'].mean()
                    
                    # Heuristic: If average close is very large (>1000), it's likely a currency with low USD value
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
                    
                    data['CloseUSD'] = data['Close'] * estimated_rate
                    print(f"  {index_symbol} in {currency} - using estimated conversion rate: {estimated_rate}")
        else:
            # If no currency column, assume USD but log a warning
            print(f"  WARNING: No currency information for {index_symbol} - assuming USD")
            data['CloseUSD'] = data['Close']
            
        # Ensure no NaN values in CloseUSD
        if data['CloseUSD'].isna().any():
            nan_count = data['CloseUSD'].isna().sum()
            print(f"  WARNING: Found {nan_count} NaN values in CloseUSD column. Filling with interpolated values.")
            
            # Try to interpolate missing values
            data['CloseUSD'] = data['CloseUSD'].interpolate(method='linear')
            
            # If any NaN values remain (at the beginning/end), fill with the closest valid value
            if data['CloseUSD'].isna().any():
                data['CloseUSD'] = data['CloseUSD'].ffill().bfill()
                
            # Final check - if any NaN values still remain, use Close values as a last resort
            if data['CloseUSD'].isna().any():
                print("  WARNING: Unable to interpolate all NaN values. Using Close values as fallback.")
                data.loc[data['CloseUSD'].isna(), 'CloseUSD'] = data.loc[data['CloseUSD'].isna(), 'Close']
        
        # Convert Date back to string for export
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        
        # Convert volume to float for consistency
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].astype(float)
        
        return data
    
    except Exception as e:
        print(f"Error processing data for {index_symbol}: {e}")
        return None

def process_all_exchanges():
    """
    Process all exchanges in indexInfo.csv and save individual CSV files
    for each index from 2012 to present
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir = '../datasets/processed_exchanges'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Read indexInfo.csv
    try:
        # First try in the current directory
        if os.path.exists('indexInfo.csv'):
            index_info = pd.read_csv('indexInfo.csv')
            print(f"Using indexInfo.csv from current directory")
        else:
            # Try in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            info_path = os.path.join(script_dir, 'indexInfo.csv')
            if os.path.exists(info_path):
                index_info = pd.read_csv(info_path)
                print(f"Using indexInfo.csv from script directory: {info_path}")
            else:
                raise FileNotFoundError(f"Could not find indexInfo.csv in either current directory or {script_dir}")
    except Exception as e:
        print(f"Error reading indexInfo.csv: {e}")
        # Create a minimal default if file not found
        index_info = pd.DataFrame({
            'Region': ['North America', 'North America', 'Asia', 'Europe'],
            'Exchange': ['NYSE', 'NASDAQ', 'Japan', 'Germany'],
            'Index': ['NYA', 'IXIC', 'N225', 'GDAXI'],
            'Currency': ['USD', 'USD', 'JPY', 'EUR']
        })
        print("Using default indices list")
    
    # Process each index
    all_data = []
    successful_count = 0
    total_count = len(index_info)
    
    for i, row in index_info.iterrows():
        region = row['Region']
        exchange = row['Exchange']
        index_symbol = row['Index']
        currency = row['Currency']
        
        print(f"\nProcessing [{i+1}/{total_count}]: {region} - {exchange} ({index_symbol})")
        
        # Get data from 2012 to present
        data = get_exchange_data(
            index_symbol=index_symbol,
            start_date="2012-01-01",
            end_date=None,  # None will get data up to current date
            currency=currency
        )
        
        if data is not None and not data.empty:
            # Process the data
            processed_data = process_exchange_data(data, index_symbol)
            
            if processed_data is not None and not processed_data.empty:
                # Clean index symbol for filename
                clean_symbol = index_symbol.replace('^', '').replace('.', '_')
                
                # Save individual CSV file
                output_file = f"{output_dir}/{clean_symbol}_processed.csv"
                processed_data.to_csv(output_file, index=False)
                print(f"Data saved to {output_file} ({len(processed_data)} rows)")
                
                # Add to combined dataset
                all_data.append(processed_data)
                successful_count += 1
            else:
                print(f"Skipping {index_symbol} - processing failed")
        else:
            print(f"Skipping {index_symbol} - no data available")
        
        # Add a short delay between requests to avoid rate limiting
        time.sleep(1)
    
    # Combine all data into one file
    if all_data:
        # Concatenate all dataframes
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Check for duplicate date/index combinations
        duplicates = combined_data.duplicated(subset=['Date', 'Index']).sum()
        if duplicates > 0:
            print(f"\nWarning: Found {duplicates} duplicate date/index combinations in combined data")
            print("Removing duplicates...")
            combined_data = combined_data.drop_duplicates(subset=['Date', 'Index'], keep='first')
        
        # Save the combined file
        combined_file = f"{output_dir}/all_indices_processed.csv"
        combined_data.to_csv(combined_file, index=False)
        print(f"\nCombined data saved to {combined_file} ({len(combined_data)} rows, {combined_data.shape[1]} columns)")
        
        # Create a cleaned version without duplicates for analysis
        # Group by Index and Date to create a pivot table
        pivot_columns = ['Index', 'Currency', 'Close', 'Daily_Return', 'Volatility_20']
        if all(col in combined_data.columns for col in pivot_columns):
            pivot_data = combined_data[pivot_columns].copy()
            # Create feature names by combining Index and metrics
            pivot_data['Feature'] = pivot_data['Index'] + '_' + 'Close'
            
            # Create a pivot table with Date as index and Feature as columns
            date_pivot = pivot_data.pivot_table(
                index='Date',
                columns='Feature',
                values='Close',
                aggfunc='first'
            )
            
            # Save the pivot table
            pivot_file = f"{output_dir}/all_indices_pivot.csv"
            date_pivot.to_csv(pivot_file)
            print(f"Pivot table saved to {pivot_file} ({date_pivot.shape[0]} rows, {date_pivot.shape[1]} columns)")
    
    end_time = time.time()
    print(f"\nProcessing complete! Successfully processed {successful_count} out of {total_count} indices.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        process_all_exchanges()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Partial data may have been saved.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 