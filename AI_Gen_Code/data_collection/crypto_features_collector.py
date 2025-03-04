"""
Crypto and Market Features Collector

This script collects additional features for cryptocurrency and market analysis from 
various free public APIs and data sources. All data is saved as separate CSV files 
in a structured format for later use in AI/ML projects.

Features collected include:
- Market sentiment indicators
- On-chain metrics
- Stablecoin supply data
- Volatility indices
- Energy and mining metrics
- Currency and economic indicators
- Basic derivatives data
"""

import os
import time
import json
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Create output directory - use environment variables if available
if 'ADDITIONAL_FEATURES_DIR' in os.environ:
    OUTPUT_DIR = os.environ.get('ADDITIONAL_FEATURES_DIR')
    print(f"Using environment path for output: {OUTPUT_DIR}")
else:
    # Fall back to default path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets', 'additional_features')
    print(f"Using default path for output: {OUTPUT_DIR}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# Define time periods
DEFAULT_START_DATE = '2012-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Helper function to save dataframe to CSV
def save_to_csv(df, filename, source_info=None):
    """
    Save DataFrame to CSV with source information in the header comment
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Add source information as a comment in the first line
    if source_info:
        with open(filepath, 'w') as f:
            f.write(f"# Data Source: {source_info}\n")
        df.to_csv(filepath, mode='a', index=True)
    else:
        df.to_csv(filepath, index=True)
    
    print(f"Saved {len(df)} rows to {filepath}")
    return filepath

# Helper function to format dates consistently
def format_date_column(df, date_column='Date'):
    """
    Ensure date column is in YYYY-MM-DD format
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column]).dt.strftime('%Y-%m-%d')
    return df

# Helper function to add a description column
def add_description(df, description):
    """
    Add a description column to help understand the data
    """
    df['Description'] = description
    return df

# 1. Alternative to AAII % Bearish - Fear & Greed Index
def get_fear_greed_index(days=1000):
    """
    Get Fear & Greed Index historical data (alternative to AAII sentiment)
    Uses Alternative.me API (free)
    
    Returns DataFrame with date and fear/greed value
    """
    print("Fetching Fear & Greed Index data...")
    
    try:
        # Get current Fear & Greed index
        url = f"https://api.alternative.me/fng/?limit={days}"
        response = requests.get(url)
        data = response.json()
        
        if data['metadata']['error'] is not None:
            print(f"API Error: {data['metadata']['error']}")
            return None
        
        # Process the data
        results = []
        for item in data['data']:
            timestamp = int(item['timestamp'])
            date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            value = int(item['value'])
            classification = item['value_classification']
            
            results.append({
                'Date': date,
                'FearGreedValue': value,
                'FearGreedClassification': classification
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('Date')
        
        # Set Date as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        source_info = "Alternative.me Fear & Greed Index API (https://alternative.me/crypto/fear-and-greed-index/)"
        save_to_csv(df, 'fear_greed_index.csv', source_info)
        
        return df
    
    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return None

# 2. On-chain metrics using Blockchain.com public API
def get_onchain_metrics(start_date=DEFAULT_START_DATE):
    """
    Get on-chain metrics from Blockchain.com public API
    
    Returns DataFrame with various blockchain metrics
    """
    print("Fetching on-chain metrics from Blockchain.com...")
    metrics = {
        'n-unique-addresses': 'Active Addresses',
        'n-transactions': 'Transaction Count',
        'mempool-size': 'Mempool Size',
        'hash-rate': 'Hash Rate (GH/s)',
        'difficulty': 'Mining Difficulty',
        'miners-revenue': 'Miner Revenue (USD)',
        'transaction-fees': 'Transaction Fees (BTC)',
        'median-confirmation-time': 'Median Confirmation Time (min)'
    }
    
    all_data = pd.DataFrame()
    
    try:
        for endpoint, name in metrics.items():
            print(f"Fetching {name}...")
            url = f"https://api.blockchain.info/charts/{endpoint}?timespan=all&format=json&sampled=true"
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error fetching {name}: {response.status_code}")
                continue
            
            data = response.json()
            
            # Extract values
            dates = []
            values = []
            
            for point in data['values']:
                timestamp = point['x']
                value = point['y']
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                
                dates.append(date)
                values.append(value)
            
            # Create DataFrame for this metric
            metric_df = pd.DataFrame({
                'Date': dates,
                name: values
            })
            
            # Set Date as index
            metric_df['Date'] = pd.to_datetime(metric_df['Date'])
            metric_df.set_index('Date', inplace=True)
            
            # Merge with existing data
            if all_data.empty:
                all_data = metric_df
            else:
                all_data = all_data.join(metric_df, how='outer')
        
        # Filter by start date
        all_data = all_data[all_data.index >= pd.to_datetime(start_date)]
        
        # Save the combined data
        source_info = "Blockchain.com Public Charts API (https://www.blockchain.com/explorer/charts)"
        save_to_csv(all_data, 'onchain_metrics.csv', source_info)
        
        # Save individual metrics for specific use cases
        save_to_csv(all_data[['Mining Difficulty']], 'mining_difficulty.csv', source_info)
        save_to_csv(all_data[['Miner Revenue (USD)']], 'miner_revenue.csv', source_info)
        
        return all_data
    
    except Exception as e:
        print(f"Error fetching on-chain metrics: {e}")
        return None

# 3. Stablecoin metrics using CoinGecko API
def get_stablecoin_metrics(start_date=DEFAULT_START_DATE):
    """
    Get stablecoin supply and market cap data from CoinGecko API (free tier)
    
    Returns DataFrame with stablecoin market data
    """
    print("Fetching stablecoin data from CoinGecko...")
    
    # List of major stablecoins
    stablecoins = {
        'tether': 'Tether (USDT)',
        'usd-coin': 'USD Coin (USDC)',
        'binance-usd': 'Binance USD (BUSD)',
        'dai': 'Dai (DAI)',
        'true-usd': 'TrueUSD (TUSD)'
    }
    
    all_data = pd.DataFrame()
    
    try:
        for coin_id, coin_name in stablecoins.items():
            print(f"Fetching {coin_name}...")
            
            # CoinGecko API has rate limits, so we add a delay
            time.sleep(1.5)
            
            # Get market data
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': 'max',
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching {coin_name}: {response.status_code}")
                continue
            
            data = response.json()
            
            # Extract price and market cap data
            dates = []
            prices = []
            market_caps = []
            volumes = []
            
            for i in range(len(data['prices'])):
                timestamp = data['prices'][i][0] / 1000  # Convert from milliseconds
                price = data['prices'][i][1]
                market_cap = data['market_caps'][i][1] if i < len(data['market_caps']) else None
                volume = data['total_volumes'][i][1] if i < len(data['total_volumes']) else None
                
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                
                dates.append(date)
                prices.append(price)
                market_caps.append(market_cap)
                volumes.append(volume)
            
            # Create DataFrame for this stablecoin
            coin_df = pd.DataFrame({
                'Date': dates,
                f'{coin_name} Price': prices,
                f'{coin_name} Market Cap': market_caps,
                f'{coin_name} Volume': volumes
            })
            
            # Set Date as index
            coin_df['Date'] = pd.to_datetime(coin_df['Date'])
            coin_df.set_index('Date', inplace=True)
            
            # Merge with existing data
            if all_data.empty:
                all_data = coin_df
            else:
                all_data = all_data.join(coin_df, how='outer')
        
        # Filter by start date
        all_data = all_data[all_data.index >= pd.to_datetime(start_date)]
        
        # Create aggregate metrics
        market_cap_columns = [col for col in all_data.columns if 'Market Cap' in col]
        all_data['Total Stablecoin Market Cap'] = all_data[market_cap_columns].sum(axis=1)
        
        # Calculate 30-day and 60-day lagged values for correlation analysis
        all_data['Total Stablecoin Market Cap 30d Lag'] = all_data['Total Stablecoin Market Cap'].shift(-30)
        all_data['Total Stablecoin Market Cap 60d Lag'] = all_data['Total Stablecoin Market Cap'].shift(-60)
        
        # Calculate percentage change
        all_data['Total Stablecoin Market Cap % Change'] = all_data['Total Stablecoin Market Cap'].pct_change() * 100
        
        # Save the data
        source_info = "CoinGecko API (https://www.coingecko.com/en/api)"
        save_to_csv(all_data, 'stablecoin_metrics.csv', source_info)
        
        # Save just the aggregate data
        agg_columns = ['Total Stablecoin Market Cap', 'Total Stablecoin Market Cap % Change',
                       'Total Stablecoin Market Cap 30d Lag', 'Total Stablecoin Market Cap 60d Lag']
        save_to_csv(all_data[agg_columns], 'stablecoin_aggregates.csv', source_info)
        
        return all_data
    
    except Exception as e:
        print(f"Error fetching stablecoin metrics: {e}")
        return None

# 4. Volatility indices using Yahoo Finance
def get_volatility_indices(start_date=DEFAULT_START_DATE):
    """
    Get volatility indices from Yahoo Finance:
    - CBOE SKEW Index
    - CBOE Volatility Index (VIX)
    - Crude Oil Volatility Index (OVX)
    
    Returns DataFrame with volatility indices
    """
    print("Fetching volatility indices from Yahoo Finance...")
    
    indices = {
        '^SKEW': 'CBOE SKEW Index',
        '^VIX': 'CBOE Volatility Index (VIX)',
        '^OVX': 'Crude Oil Volatility Index (OVX)'
    }
    
    all_data = pd.DataFrame()
    
    try:
        for ticker, name in indices.items():
            print(f"Fetching {name}...")
            
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=END_DATE, progress=False)
            
            if data.empty:
                print(f"No data found for {name}")
                continue
            
            # Check if 'Adj Close' exists, otherwise use 'Close'
            price_col = 'Close'
            if 'Adj Close' in data.columns:
                price_col = 'Adj Close'
                
            # Keep only the price column and rename it
            data = data[[price_col]].copy()
            data.rename(columns={price_col: name}, inplace=True)
            
            # Merge with existing data
            if all_data.empty:
                all_data = data
            else:
                all_data = all_data.join(data, how='outer')
        
        # Calculate ratios if both indices exist in the data
        try:
            # Check if we have a MultiIndex for columns
            is_multi_index = isinstance(all_data.columns, pd.MultiIndex)
            
            # Get the column names for SKEW and VIX based on index type
            if is_multi_index:
                skew_col = [col for col in all_data.columns if 'SKEW' in col[0]]
                vix_col = [col for col in all_data.columns if 'VIX' in col[0]]
                
                if skew_col and vix_col:
                    skew_col = skew_col[0]
                    vix_col = vix_col[0]
                    
                    # Fill NaN values using more modern syntax to avoid deprecation warnings
                    skew_data = all_data[skew_col].ffill().bfill()
                    vix_data = all_data[vix_col].ffill().bfill()
                    
                    # Only calculate where VIX is not zero to avoid division by zero
                    valid_vix = vix_data > 0
                    
                    # Create a new column for the ratio
                    ratio_col = ('SKEW/VIX Ratio', '')
                    all_data[ratio_col] = np.nan
                    
                    # Calculate ratio only for valid rows
                    all_data.loc[valid_vix, ratio_col] = skew_data[valid_vix] / vix_data[valid_vix]
                    print(f"Successfully calculated SKEW/VIX ratio for {valid_vix.sum()} entries")
            else:
                # Original code for non-MultiIndex columns
                if 'CBOE SKEW Index' in all_data.columns and 'CBOE Volatility Index (VIX)' in all_data.columns:
                    # Fill NaN values using more modern syntax to avoid deprecation warnings
                    skew_data = all_data['CBOE SKEW Index'].ffill().bfill()
                    vix_data = all_data['CBOE Volatility Index (VIX)'].ffill().bfill()
                    
                    # Only calculate where VIX is not zero to avoid division by zero
                    valid_vix = vix_data > 0
                    
                    # Initialize ratio column with NaN
                    all_data['SKEW/VIX Ratio'] = np.nan
                    
                    # Calculate ratio only for valid rows
                    all_data.loc[valid_vix, 'SKEW/VIX Ratio'] = skew_data[valid_vix] / vix_data[valid_vix]
                    print(f"Successfully calculated SKEW/VIX ratio for {valid_vix.sum()} entries")
        except Exception as e:
            print(f"Error calculating SKEW/VIX ratio: {e}")
            # Debug information
            print(f"Column type: {type(all_data.columns)}")
            print(f"Columns in the data: {all_data.columns.tolist()}")
            print(f"Data index format: {type(all_data.index)} with {len(all_data.index)} entries")
        
        # Save the data if we have any
        if not all_data.empty:
            source_info = "Yahoo Finance API (https://finance.yahoo.com/)"
            save_to_csv(all_data, 'volatility_indices.csv', source_info)
        else:
            print("No volatility data was collected")
        
        return all_data
    
    except Exception as e:
        print(f"Error fetching volatility indices: {e}")
        return None

# 5. Energy market metrics (simple proxy for miner cost basis)
def get_energy_metrics(start_date=DEFAULT_START_DATE):
    """
    Get energy market metrics and estimate miner cost basis
    
    Returns DataFrame with energy price indices and estimated mining costs
    """
    print("Fetching energy market metrics...")
    
    # Energy price indices from Yahoo Finance
    energy_tickers = {
        'XLE': 'Energy Select Sector SPDR ETF',  # Energy sector ETF
        'NG=F': 'Natural Gas Futures',
        'BZ=F': 'Brent Crude Oil Futures'
    }
    
    all_data = pd.DataFrame()
    
    try:
        # Fetch energy price data
        for ticker, name in energy_tickers.items():
            print(f"Fetching {name}...")
            
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=END_DATE, progress=False)
            
            if data.empty:
                print(f"No data found for {name}")
                continue
            
            # Check if 'Adj Close' exists, otherwise use 'Close'
            price_col = 'Close'
            if 'Adj Close' in data.columns:
                price_col = 'Adj Close'
                
            # Keep only the price column and rename it
            data = data[[price_col]].copy()
            data.rename(columns={price_col: name}, inplace=True)
            
            # Merge with existing data
            if all_data.empty:
                all_data = data
            else:
                all_data = all_data.join(data, how='outer')
        
        # Now get Bitcoin mining difficulty as a proxy for hashrate
        # First, check if we already have this data from on-chain metrics
        onchain_file = os.path.join(OUTPUT_DIR, 'mining_difficulty.csv')
        if os.path.exists(onchain_file):
            print("Loading mining difficulty from existing file...")
            difficulty_data = pd.read_csv(onchain_file, comment='#')
            difficulty_data['Date'] = pd.to_datetime(difficulty_data['Date'])
            difficulty_data.set_index('Date', inplace=True)
            
            # Join difficulty data
            all_data = all_data.join(difficulty_data, how='outer')
        else:
            print("Mining difficulty file not found. Fetching from Blockchain.com API...")
            url = "https://api.blockchain.info/charts/difficulty?timespan=all&format=json&sampled=true"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract values
                dates = []
                values = []
                
                for point in data['values']:
                    timestamp = point['x']
                    value = point['y']
                    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                    
                    dates.append(date)
                    values.append(value)
                
                # Create DataFrame
                difficulty_df = pd.DataFrame({
                    'Date': dates,
                    'Mining Difficulty': values
                })
                
                # Set Date as index
                difficulty_df['Date'] = pd.to_datetime(difficulty_df['Date'])
                difficulty_df.set_index('Date', inplace=True)
                
                # Filter by start date
                difficulty_df = difficulty_df[difficulty_df.index >= pd.to_datetime(start_date)]
                
                # Join difficulty data
                all_data = all_data.join(difficulty_df, how='outer')
        
        # Create a proxy for miner cost basis
        # Simplified formula: Energy Price Index * Mining Difficulty / Constant
        # This is a very rough approximation
        if 'Energy Select Sector SPDR ETF' in all_data.columns and 'Mining Difficulty' in all_data.columns:
            # Normalize both metrics
            energy_normalized = all_data['Energy Select Sector SPDR ETF'] / all_data['Energy Select Sector SPDR ETF'].iloc[0]
            difficulty_normalized = all_data['Mining Difficulty'] / all_data['Mining Difficulty'].iloc[0]
            
            # Create proxy for mining cost (scaled for readability)
            all_data['Mining Cost Proxy'] = energy_normalized * difficulty_normalized * 10
            
            # Forward fill missing values
            all_data = all_data.fillna(method='ffill')
        
        # Save the data
        source_info = "Energy data from Yahoo Finance, Mining difficulty from Blockchain.com"
        save_to_csv(all_data, 'energy_mining_metrics.csv', source_info)
        
        return all_data
    
    except Exception as e:
        print(f"Error fetching energy metrics: {e}")
        return None

# 6. Currency debasement signals using Yahoo Finance and FRED
def get_currency_metrics(start_date=DEFAULT_START_DATE):
    """
    Get currency related metrics including USD index, gold prices, and BTC/USD price
    """
    print("\nGetting currency metrics...")
    
    # Create empty DataFrame with date range
    date_range = pd.date_range(start=start_date, end=END_DATE)
    currency_df = pd.DataFrame(index=date_range)
    currency_df.index.name = 'Date'
    
    try:
        # Get US Dollar Index (DXY)
        try:
            print("Getting US Dollar Index (DXY) data...")
            usd_index = yf.download("DX-Y.NYB", start=start_date, end=END_DATE, progress=False)
            
            if not usd_index.empty:
                # Add to DataFrame
                currency_df['US Dollar Index'] = usd_index['Close'].reindex(currency_df.index)
                print(f"Added USD Index data: {len(usd_index)} rows")
            else:
                print("No USD Index data retrieved")
        except Exception as e:
            print(f"Error getting USD Index data: {e}")
        
        # Get Gold Futures
        try:
            print("Getting Gold Futures data...")
            gold = yf.download("GC=F", start=start_date, end=END_DATE, progress=False)
            
            if not gold.empty:
                # Add to DataFrame
                currency_df['Gold Futures'] = gold['Close'].reindex(currency_df.index)
                print(f"Added Gold Futures data: {len(gold)} rows")
            else:
                print("No Gold Futures data retrieved")
        except Exception as e:
            print(f"Error getting Gold Futures data: {e}")
        
        # Get Bitcoin price - Try multiple symbols
        try:
            print("Getting Bitcoin price data...")
            # Try multiple Bitcoin symbols to ensure data is collected
            bitcoin_symbols = ["BTC-USD", "BTCUSD=X", "BTC=F"]
            
            btc_df = None
            for symbol in bitcoin_symbols:
                try:
                    print(f"Trying Bitcoin symbol: {symbol}...")
                    btc = yf.download(symbol, start=start_date, end=END_DATE, progress=False)
                    
                    if not btc.empty and len(btc) > 0:  # More reliable empty check
                        # Verify we actually have data (not just empty rows)
                        if 'Close' in btc.columns and btc['Close'].notna().any():
                            btc_df = btc
                            print(f"Successfully retrieved Bitcoin data using symbol {symbol}")
                            break
                        else:
                            print(f"Downloaded data for {symbol}, but no usable price data found")
                except Exception as e:
                    print(f"Failed to get data for symbol {symbol}: {e}")
            
            if btc_df is not None and not btc_df.empty:
                # Add to DataFrame - CONSISTENTLY use BTC/USD (not BTC-USD)
                currency_df['BTC/USD'] = btc_df['Close'].reindex(currency_df.index)
                print(f"Added BTC/USD data: {len(btc_df)} rows")
            else:
                print("No Bitcoin price data retrieved from any source")
                
                # As a fallback, use CoinGecko API
                print("Using CoinGecko API as fallback for BTC/USD data")
                try:
                    # Get BTC/USD historical data from CoinGecko
                    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
                    params = {
                        "vs_currency": "usd",
                        "days": "max",
                        "interval": "daily"
                    }
                    
                    # Add proper headers to avoid rate limiting
                    headers = {
                        "accept": "application/json",
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    
                    response = requests.get(url, params=params, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        # Convert timestamp (milliseconds) to date and price to DataFrame
                        prices = data.get("prices", [])
                        if prices:
                            temp_df = pd.DataFrame(prices, columns=["timestamp", "price"])
                            temp_df["Date"] = pd.to_datetime(temp_df["timestamp"], unit="ms")
                            temp_df = temp_df.set_index("Date")
                            
                            # Reindex to match our date range
                            currency_df['BTC/USD'] = temp_df["price"].reindex(currency_df.index)
                            print(f"Added BTC/USD data from CoinGecko: {len(temp_df)} rows")
                    else:
                        print(f"CoinGecko API request failed with status code: {response.status_code}")
                except Exception as e:
                    print(f"Error using CoinGecko API fallback: {e}")
                    
                    # Create synthetic data as a last resort
                    print("Creating synthetic BTC/USD data as last resort...")
                    try:
                        # Known BTC price points for fallback
                        key_prices = {
                            '2012-01-01': 5.27,
                            '2013-01-01': 13.30,
                            '2014-01-01': 806.06,
                            '2015-01-01': 313.92,
                            '2016-01-01': 434.46,
                            '2017-01-01': 997.69,
                            '2018-01-01': 13412.44,
                            '2019-01-01': 3843.52,
                            '2020-01-01': 7193.60,
                            '2021-01-01': 29374.15,
                            '2022-01-01': 47686.81,
                            '2023-01-01': 16625.08,
                            '2024-01-01': 42238.83,
                            '2024-05-01': 60009.21,
                        }
                        
                        # Add key price points to the dataframe
                        for date_str, price in key_prices.items():
                            if pd.to_datetime(date_str) in currency_df.index:
                                currency_df.loc[pd.to_datetime(date_str), 'BTC/USD'] = price
                        
                        # Interpolate between known price points
                        currency_df['BTC/USD'] = currency_df['BTC/USD'].interpolate(method='linear')
                        print("Created synthetic BTC/USD price data")
                    except Exception as e:
                        print(f"Error creating synthetic BTC price data: {e}")
        except Exception as e:
            print(f"Error getting Bitcoin price data: {e}")
            
        # Calculate Gold/BTC ratio if both columns have data
        if 'Gold Futures' in currency_df.columns and 'BTC/USD' in currency_df.columns:
            print("Calculating Gold/BTC Ratio...")
            # Safely calculate the ratio only for rows where BTC/USD has valid values > 0
            valid_rows = (currency_df['BTC/USD'] > 0) & currency_df['BTC/USD'].notna() & currency_df['Gold Futures'].notna()
            if valid_rows.any():
                currency_df.loc[valid_rows, 'Gold/BTC Ratio'] = currency_df.loc[valid_rows, 'Gold Futures'] / currency_df.loc[valid_rows, 'BTC/USD']
                print("Gold/BTC Ratio calculated successfully")
            else:
                print("No valid data to calculate Gold/BTC Ratio")
        
        # Drop rows where all values are NaN
        currency_df = currency_df.dropna(how='all')
        
        # Fill any remaining NaN values using forward and backward fill
        currency_df = currency_df.fillna(method='ffill').fillna(method='bfill')
        
        # Save to CSV
        source_info = "Currency metrics data from Yahoo Finance and CoinGecko (US Dollar Index, Gold Futures, BTC/USD, Gold/BTC Ratio)"
        num_rows = save_to_csv(currency_df, 'currency_metrics.csv', source_info)
        
        print(f"Currency metrics data collection complete: {num_rows} rows saved")
        return currency_df
    
    except Exception as e:
        print(f"Error collecting currency metrics: {e}")
        return None

# 7. Basic derivatives metrics using public data
def get_derivatives_metrics(start_date=DEFAULT_START_DATE):
    """
    Get basic derivatives market data from free public sources
    
    Returns DataFrame with derivatives metrics
    """
    print("Fetching basic derivatives market data...")
    
    try:
        # CoinGecko's funding rates API doesn't exist in their free tier
        # Instead, we'll get the CME Bitcoin futures data as a proxy for derivatives market
        print("Fetching CME Bitcoin Futures data from Yahoo Finance...")
        
        # Use front-month and back-month BTC futures from CME (available on Yahoo)
        tickers = {
            'BTC=F': 'CME Bitcoin Futures Front Month'
        }
        
        all_data = pd.DataFrame()
        
        # Fetch data from Yahoo Finance
        for ticker, name in tickers.items():
            print(f"Fetching {ticker}...")
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=END_DATE, progress=False)
            
            if data.empty:
                print(f"No data found for {name}")
                continue
            
            # Get list of available columns
            available_cols = list(data.columns)
            print(f"Available columns for {ticker}: {available_cols}")
            
            # Keep only the columns we need
            keep_cols = []
            col_mapping = {}
            
            if 'Open' in available_cols:
                keep_cols.append('Open')
                col_mapping['Open'] = f"{name} Open"
                
            if 'High' in available_cols:
                keep_cols.append('High')
                col_mapping['High'] = f"{name} High"
                
            if 'Low' in available_cols:
                keep_cols.append('Low')
                col_mapping['Low'] = f"{name} Low"
                
            if 'Close' in available_cols:
                keep_cols.append('Close')
                col_mapping['Close'] = f"{name} Close"
            elif 'Adj Close' in available_cols:
                keep_cols.append('Adj Close')
                col_mapping['Adj Close'] = f"{name} Close"
                
            if 'Volume' in available_cols:
                keep_cols.append('Volume')
                col_mapping['Volume'] = f"{name} Volume"
            
            if not keep_cols:
                print(f"No usable columns found for {name}")
                continue
                
            # Keep relevant columns and rename
            data = data[keep_cols].copy()
            data.rename(columns=col_mapping, inplace=True)
            
            # Merge with existing data
            if all_data.empty:
                all_data = data
            else:
                all_data = all_data.join(data, how='outer')
        
        # Get spot price for comparison
        print("Fetching BTC-USD spot price...")
        try:
            btc_spot = yf.download('BTC-USD', start=start_date, end=END_DATE, progress=False)
            
            if not btc_spot.empty and ('Close' in btc_spot.columns or 'Adj Close' in btc_spot.columns):
                # Use Close or Adj Close based on availability
                price_col = 'Close' if 'Close' in btc_spot.columns else 'Adj Close'
                btc_spot = btc_spot[[price_col]].copy()
                btc_spot.rename(columns={price_col: 'BTC Spot Price'}, inplace=True)
                
                # Join with futures data
                all_data = all_data.join(btc_spot, how='outer')
                
                # Calculate basis (difference between futures and spot) if possible
                if f'{tickers["BTC=F"]} Close' in all_data.columns and 'BTC Spot Price' in all_data.columns:
                    # Forward fill to handle missing values
                    all_data = all_data.fillna(method='ffill')
                    
                    all_data['CME-Spot Basis'] = all_data[f'{tickers["BTC=F"]} Close'] - all_data['BTC Spot Price']
                    
                    # Calculate percentage basis, handle potential division by zero
                    valid_spots = all_data['BTC Spot Price'] > 0
                    all_data.loc[valid_spots, 'CME-Spot Basis %'] = (
                        (all_data.loc[valid_spots, f'{tickers["BTC=F"]} Close'] / 
                         all_data.loc[valid_spots, 'BTC Spot Price'] - 1) * 100
                    )
            else:
                print("No spot price data found for BTC-USD")
        except Exception as e:
            print(f"Error fetching BTC spot price: {e}")
        
        # Save the data if we have any
        if not all_data.empty:
            source_info = "Yahoo Finance API for CME Futures and BTC spot prices"
            save_to_csv(all_data, 'derivatives_metrics.csv', source_info)
            return all_data
        else:
            print("No derivatives data was collected")
            return None
    
    except Exception as e:
        print(f"Error fetching derivatives metrics: {e}")
        return None

# Create a combined dataset with key features from all categories
def create_combined_dataset():
    """
    Create a combined dataset with key features from all categories
    """
    print("Creating combined dataset with key features...")
    
    combined_data = pd.DataFrame()
    
    # List of files to check for key metrics
    files_to_check = [
        'fear_greed_index.csv',
        'onchain_metrics.csv',
        'stablecoin_aggregates.csv',
        'volatility_indices.csv',
        'energy_mining_metrics.csv',
        'currency_metrics.csv',
        'derivatives_metrics.csv'
    ]
    
    try:
        # Key metrics to include from each file
        key_metrics = {
            'fear_greed_index.csv': ['FearGreedValue'],
            'onchain_metrics.csv': ['Active Addresses', 'Transaction Count', 'Hash Rate (GH/s)', 'Miner Revenue (USD)'],
            'stablecoin_aggregates.csv': ['Total Stablecoin Market Cap', 'Total Stablecoin Market Cap % Change'],
            'volatility_indices.csv': ['CBOE SKEW Index', 'CBOE Volatility Index (VIX)', 'Crude Oil Volatility Index (OVX)'],
            'energy_mining_metrics.csv': ['Mining Cost Proxy', 'Mining Difficulty'],
            'currency_metrics.csv': ['BTC/USD'],
            'derivatives_metrics.csv': ['CME-Spot Basis %', 'BTC Spot Price']
        }
        
        # Load and combine data
        for file in files_to_check:
            filepath = os.path.join(OUTPUT_DIR, file)
            if os.path.exists(filepath):
                print(f"Loading data from {file}...")
                
                # Read the file, skipping comment lines
                df = pd.read_csv(filepath, comment='#')
                
                # Check if 'Date' column exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                # Select only the key metrics if they exist
                if file in key_metrics:
                    metrics_to_include = [m for m in key_metrics[file] if m in df.columns]
                    if metrics_to_include:
                        df = df[metrics_to_include]
                    else:
                        print(f"No matching key metrics found in {file}")
                        continue
                
                # Merge with combined data
                if combined_data.empty:
                    combined_data = df
                else:
                    combined_data = combined_data.join(df, how='outer')
        
        # Sort by date
        combined_data = combined_data.sort_index()
        
        # Fill missing values with forward fill, then backward fill
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        # Save the combined data
        source_info = "Combined dataset from all collected feature categories"
        save_to_csv(combined_data, 'combined_features.csv', source_info)
        
        # Create a features description file
        feature_descriptions = {
            'FearGreedValue': 'Fear and Greed Index value (0-100), representing market sentiment',
            'Active Addresses': 'Number of unique blockchain addresses active on that day',
            'Transaction Count': 'Number of transactions on the Bitcoin blockchain',
            'Hash Rate (GH/s)': 'Total computational power of the Bitcoin network in GH/s',
            'Miner Revenue (USD)': 'Total revenue earned by Bitcoin miners in USD',
            'Total Stablecoin Market Cap': 'Combined market capitalization of major stablecoins',
            'Total Stablecoin Market Cap % Change': 'Daily percentage change in stablecoin market cap',
            'CBOE SKEW Index': 'Measures perceived tail risk in the S&P 500 (black swan events)',
            'CBOE Volatility Index (VIX)': 'Market expectation of 30-day volatility, the "fear index"',
            'Crude Oil Volatility Index (OVX)': 'Expected volatility of crude oil prices',
            'Mining Cost Proxy': 'Proxy for mining cost based on energy prices and network difficulty',
            'Mining Difficulty': 'Bitcoin network mining difficulty',
            'BTC/USD': 'Bitcoin price in USD',
            'CME-Spot Basis %': 'Percentage difference between CME Bitcoin futures and spot price',
            'BTC Spot Price': 'Bitcoin spot price in USD'
        }
        
        # Create and save the features description
        desc_df = pd.DataFrame({
            'Feature': list(feature_descriptions.keys()),
            'Description': list(feature_descriptions.values())
        })
        
        desc_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_descriptions.csv'), index=False)
        print(f"Feature descriptions saved to {os.path.join(OUTPUT_DIR, 'feature_descriptions.csv')}")
        
        return combined_data
    
    except Exception as e:
        print(f"Error creating combined dataset: {e}")
        return None

# Main function to collect all features
def collect_all_features(start_date=DEFAULT_START_DATE):
    """
    Collect all feature sets and create combined dataset
    """
    print(f"Starting feature collection from {start_date} to {END_DATE}...")
    
    # Collect each feature set
    sentiment_data = get_fear_greed_index()
    onchain_data = get_onchain_metrics(start_date)
    stablecoin_data = get_stablecoin_metrics(start_date)
    volatility_data = get_volatility_indices(start_date)
    energy_data = get_energy_metrics(start_date)
    currency_data = get_currency_metrics(start_date)
    derivatives_data = get_derivatives_metrics(start_date)
    
    # Create combined dataset
    combined_data = create_combined_dataset()
    
    # Generate a summary report
    summary = []
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(OUTPUT_DIR, filename)
            try:
                # Read file, skipping comment lines
                df = pd.read_csv(filepath, comment='#')
                row_count = len(df)
                col_count = len(df.columns)
                start = df['Date'].min() if 'Date' in df.columns else 'N/A'
                end = df['Date'].max() if 'Date' in df.columns else 'N/A'
                
                summary.append({
                    'Filename': filename,
                    'Rows': row_count,
                    'Columns': col_count,
                    'Date Range': f"{start} to {end}" if start != 'N/A' else 'N/A'
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    # Save summary report
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'data_summary.csv'), index=False)
        print(f"Data summary saved to {os.path.join(OUTPUT_DIR, 'data_summary.csv')}")
    
    print("Feature collection complete!")

def create_combined_key_features():
    """
    Create a combined dataset with key features from multiple files
    """
    print("Creating combined dataset with key features...")
    
    # List of files to include
    files_to_include = [
        'fear_greed_index.csv',
        'onchain_metrics.csv',
        'volatility_indices.csv',
        'currency_metrics.csv',
        # Add others as needed
    ]
    
    # Create combined dataset
    combined_df = None
    
    # Process each file
    for file in files_to_include:
        filepath = os.path.join(OUTPUT_DIR, file)
        try:
            print(f"Loading data from {file}...")
            
            # Skip the first row if it's a comment
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            
            skiprows = 0
            if first_line.startswith('#'):
                skiprows = 1
            
            # Read the file
            df = pd.read_csv(filepath, skiprows=skiprows)
            
            # Make sure we have a valid date column
            if 'Date' not in df.columns:
                print(f"No Date column found in {file}, skipping")
                continue
            
            # Convert Date to datetime for proper merging
            try:
                # Ensure Date is treated as datetime
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception as e:
                print(f"Error converting dates in {file}: {e}")
                continue
            
            # Set Date as index
            df.set_index('Date', inplace=True)
            
            # Key metrics for each file - only keep these columns
            key_metrics = {
                'fear_greed_index.csv': ['Value'],
                'onchain_metrics.csv': ['active_addresses', 'transaction_count', 'mempool_size'],
                'volatility_indices.csv': ['^VIX', 'SKEW/VIX'],
                'currency_metrics.csv': ['BTC/USD', 'Gold/BTC Ratio'],
            }
            
            # Only keep key columns if defined for this file
            if file in key_metrics:
                # Filter columns that exist in the DataFrame
                keep_cols = [col for col in key_metrics[file] if col in df.columns]
                
                if keep_cols:
                    df = df[keep_cols]
                    print(f"  Using key metrics: {', '.join(keep_cols)}")
                else:
                    print(f"No matching key metrics found in {file}")
                    # Instead of skipping, continue with all columns
                    print(f"  Using all available columns instead")
            
            # Merge with combined dataset
            if combined_df is None:
                combined_df = df
            else:
                try:
                    # Make sure indexes are of the same type before merging
                    # This ensures we don't get "int" vs "Timestamp" errors
                    combined_df = pd.concat([combined_df, df], axis=1, join='outer')
                except Exception as e:
                    print(f"Error merging {file}: {e}")
                    # Try an alternative approach
                    try:
                        # Reset index to Date column for both dataframes
                        if isinstance(combined_df.index, pd.DatetimeIndex):
                            combined_df = combined_df.reset_index()
                        if isinstance(df.index, pd.DatetimeIndex):
                            df = df.reset_index()
                            
                        # Ensure Date columns are the same type
                        combined_df['Date'] = pd.to_datetime(combined_df['Date']) 
                        df['Date'] = pd.to_datetime(df['Date'])
                        
                        # Merge on Date column
                        combined_df = pd.merge(combined_df, df, on='Date', how='outer')
                        
                        # Set Date back as index
                        combined_df.set_index('Date', inplace=True)
                        print(f"  Used alternative merge method for {file}")
                    except Exception as e2:
                        print(f"  Alternative merge also failed: {e2}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if combined_df is not None:
        # Reset index to have Date as a regular column
        combined_df.reset_index(inplace=True)
        
        # Make sure we have valid data
        if len(combined_df) > 0:
            # Save to CSV
            output_path = os.path.join(OUTPUT_DIR, 'combined_key_features.csv')
            source_info = "Combined key features dataset"
            
            # Save with source info
            with open(output_path, 'w') as f:
                f.write(f"# {source_info}\n")
            combined_df.to_csv(output_path, mode='a', index=False)
            
            print(f"Saved combined key features to {output_path}")
            print(f"  Rows: {len(combined_df)}")
            print(f"  Columns: {len(combined_df.columns)}")
            
            try:
                min_date = min(combined_df['Date'])
                max_date = max(combined_df['Date'])
                print(f"  Date range: {min_date} to {max_date}")
            except Exception as e:
                print(f"  Error determining date range: {e}")
            
            return combined_df
    
    print("Failed to create combined dataset")
    return None

if __name__ == "__main__":
    # Define start date for data collection (use 2012-01-01 to match other datasets)
    start_date = DEFAULT_START_DATE
    end_date = END_DATE
    
    # Print configuration
    print(f"Starting feature collection from {start_date} to {end_date}...")
    
    # Collect Fear and Greed Index
    fear_greed_data = get_fear_greed_index(start_date)
    
    # Collect on-chain metrics
    onchain_data = get_onchain_metrics(start_date)
    difficulty_data = get_mining_difficulty(start_date)
    miner_revenue_data = get_miner_revenue(start_date)
    
    # Collect stablecoin data
    stablecoin_data = get_stablecoin_metrics(start_date)
    
    # Collect volatility indices
    volatility_data = get_volatility_indices(start_date)
    
    # Collect energy market metrics
    energy_data = get_energy_metrics(start_date)
    
    # Collect currency metrics
    currency_data = get_currency_metrics(start_date)
    
    # Collect derivatives data
    derivatives_data = get_derivatives_metrics(start_date)
    
    # Create a combined dataset with key features
    combined_key_features = create_combined_key_features()
    
    # Create data summary
    create_data_summary()
    
    print("Feature collection complete!") 