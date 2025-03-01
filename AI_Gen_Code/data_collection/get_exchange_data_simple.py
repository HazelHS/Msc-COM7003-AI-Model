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

def get_exchange_data(index_symbol, start_date, end_date, currency, max_retries=3):
    """
    Fetch historical data for a given index from Yahoo Finance with retry logic
    
    Args:
        index_symbol (str): Yahoo Finance ticker symbol for the index
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format, None for current date
        currency (str): Currency code for the index
        max_retries (int): Maximum number of retries for API calls
        
    Returns:
        pd.DataFrame: DataFrame containing the historical data
    """
    # Get the correct Yahoo Finance ticker
    ticker = INDEX_SYMBOLS.get(index_symbol, index_symbol)
    print(f"Fetching data for {ticker} (original: {index_symbol})...")
    
    for retry in range(max_retries):
        try:
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                if retry < max_retries - 1:
                    sleep_time = (2 ** retry) + 1
                    print(f"  No data received, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"  Failed to retrieve data after {max_retries} attempts")
                    return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Add metadata columns
            data['Index'] = index_symbol
            data['Currency'] = currency
            
            # Clean up the data
            # Handle NaN values in price columns
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                if col in data.columns:
                    # Forward fill first, then backward fill
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            
            # Handle NaN values in Volume column if it exists
            if 'Volume' in data.columns:
                data['Volume'] = data['Volume'].fillna(0)
            
            # Check for duplicate dates
            duplicates = data.duplicated(subset=['Date']).sum()
            if duplicates > 0:
                print(f"  Warning: Found {duplicates} duplicate dates, removing...")
                data = data.drop_duplicates(subset=['Date'], keep='first')
            
            # Convert Date to consistent string format
            if pd.api.types.is_datetime64_any_dtype(data['Date']):
                data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
            
            return data
            
        except Exception as e:
            if retry < max_retries - 1:
                sleep_time = (2 ** retry) + 1
                print(f"  Error: {e}, retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
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
    
    # Read indexInfo.csv
    try:
        index_info = pd.read_csv('indexInfo.csv')
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