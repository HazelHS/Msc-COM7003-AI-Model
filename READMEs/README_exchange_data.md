# Exchange Data Processing Script

## Overview
This script retrieves historical stock market index data from Yahoo Finance, processes it to match a specific format, and saves the data in a designated directory. It is designed to download daily price data for major global stock market indices from 2012 to the present.

## Features
- Downloads historical stock market index data from Yahoo Finance
- Processes data for multiple exchanges listed in `indexInfo.csv`
- Handles missing data and API errors with retry logic
- Creates individual CSV files for each index
- Combines all data into a single comprehensive CSV file
- Includes a simplified USD conversion for closing prices

## Requirements
- Python 3.6 or higher
- Required Python libraries:
  - pandas
  - yfinance
  - os
  - time
  - datetime

## Usage
1. Ensure you have the required Python libraries installed:
   ```
   pip install pandas yfinance
   ```

2. Make sure you have an `indexInfo.csv` file in the same directory with the following columns:
   - Region: Geographic region of the exchange
   - Exchange: Name of the stock exchange
   - Index: Index symbol (used for Yahoo Finance queries)
   - Currency: Currency code for the index (e.g., USD, EUR, JPY)

3. Run the script:
   ```
   python get_exchange_data_simple.py
   ```

4. The script will:
   - Create a directory called `processed_exchanges` (if it doesn't exist)
   - Download data for each index listed in `indexInfo.csv`
   - Save individual CSV files for each index
   - Create a combined CSV file with all data

## Output Format
The script generates CSV files with the following columns:
- Index: The index symbol
- Date: Date in YYYY-MM-DD format
- Open: Opening price
- High: Highest price during the day
- Low: Lowest price during the day
- Close: Closing price
- Adj Close: Adjusted closing price
- Volume: Trading volume
- CloseUSD: Closing price converted to USD (simplified conversion)

## Notes on Currency Conversion
The current implementation uses a simplified approach for currency conversion:
- For indices in USD, the CloseUSD value is the same as the Close value
- For non-USD indices, a 1:1 conversion is used as a placeholder

For accurate currency conversion, you would need to:
1. Fetch historical forex data for each currency pair
2. Apply the appropriate exchange rate for each date
3. Handle cases where forex data might be missing for specific dates

## Limitations
- Some indices may not be available or may have changed their ticker symbols
- The script may encounter rate limiting from Yahoo Finance API
- Currency conversion is simplified and does not reflect actual exchange rates
- Data quality and availability depend on Yahoo Finance's data sources

## Troubleshooting
- If you encounter rate limiting issues, try increasing the delay between requests
- For indices that fail to download, check if the ticker symbol is correct
- If the script crashes, it will attempt to save any data processed up to that point

## License
This script is provided for educational purposes. Please respect Yahoo Finance's terms of service when using their data. 