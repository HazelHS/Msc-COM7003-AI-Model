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
                    print(f"No data found for {ticker}, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"No data found for {ticker} after {max_retries} attempts")
                    return None
            else:
                break  # If we got data, break out of the retry loop
        
        except Exception as e:
            if retry < max_retries - 1:
                sleep_time = (2 ** retry) + 1
                print(f"Error fetching data for {ticker}: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to fetch data for {ticker} after {max_retries} attempts: {e}")
                return None
    
    if data.empty:
        print(f"No data found for {ticker}")
        return None
    
    try:
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Make sure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                print(f"Warning: {col} column missing from data, adding it with zeros")
                data[col] = 0.0
        
        # Handle Adj Close separately - newer versions of yfinance might use different name
        if 'Adj Close' not in data.columns:
            if 'Adjusted Close' in data.columns:
                data.rename(columns={'Adjusted Close': 'Adj Close'}, inplace=True)
            else:
                print(f"Warning: Adj Close column missing from data, using Close values")
                data['Adj Close'] = data['Close']
        
        # Add Index column with the symbol (use the original symbol for consistency)
        data['Index'] = index_symbol
        
        # Add CloseUSD column - for simplicity, we just use a 1:1 conversion here
        # In a real application, you'd want to use proper currency exchange rates
        data['CloseUSD'] = data['Close']
        
        # Reorder columns to match indexProcessed.csv format
        data = data[['Index', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'CloseUSD']]
        
        # Convert volume to float for consistency
        data['Volume'] = data['Volume'].astype(float)
        
        # Format Date as string for consistency (if it's not already)
        if not pd.api.types.is_string_dtype(data['Date']):
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        
        return data
    
    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")
        return None

def process_all_exchanges():
    """
    Process all exchanges in indexInfo.csv and save individual CSV files
    for each index from 2012 to present
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir = 'processed_exchanges'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read indexInfo.csv
    try:
        index_info = pd.read_csv('indexInfo.csv')
    except Exception as e:
        print(f"Error reading indexInfo.csv: {e}")
        return
    
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
            # Clean index symbol for filename
            clean_symbol = index_symbol.replace('^', '').replace('.', '_')
            
            # Save individual CSV file
            output_file = f"{output_dir}/{clean_symbol}_processed.csv"
            data.to_csv(output_file, index=False)
            print(f"Data saved to {output_file} ({len(data)} rows)")
            
            # Add to combined dataset
            all_data.append(data)
            successful_count += 1
            
            # Add a short delay between requests to avoid rate limiting
            time.sleep(1)
        else:
            print(f"Skipping {index_symbol} - no data available")
    
    # Combine all data into one file
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv(f"{output_dir}/all_indices_processed.csv", index=False)
        print(f"\nCombined data saved to {output_dir}/all_indices_processed.csv")
    
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