import yfinance as yf
import pandas as pd

def get_historical_data(ticker, start_date, end_date, filename):
    # Fetch data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Reformat to match your CSV structure
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    # Select and rename columns to match required format
    formatted_data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save to CSV
    formatted_data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Modified example usage for 2008 to current
get_historical_data(
    ticker="^N225",
    start_date="2012-01-01",  # Changed start year
    end_date=None,  # None will get data up to current date
    filename="N225_2012_present.csv"
) 