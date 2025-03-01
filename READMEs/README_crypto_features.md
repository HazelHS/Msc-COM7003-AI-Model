# Crypto and Market Features Collector

## Overview
This script collects additional crypto and market features from various free public APIs and data sources for use in AI/ML projects. It creates a set of CSV files in a structured format with proper source attribution for later reference and analysis.

## Features Collected

### Market Sentiment
- **Fear & Greed Index** (Alternative.me API)
  - Daily market sentiment indicator (0-100)
  - Classification (Extreme Fear to Extreme Greed)

### On-chain Metrics (Blockchain.com)
- **Active Addresses** - Unique blockchain addresses active daily
- **Transaction Count** - Number of Bitcoin transactions
- **Mempool Size** - Pending transaction pool size
- **Hash Rate** - Network computational power
- **Mining Difficulty** - Bitcoin mining difficulty
- **Miner Revenue** - Total miner earnings in USD
- **Transaction Fees** - Average fees in BTC
- **Confirmation Time** - Median confirmation time

### Stablecoin Metrics (CoinGecko)
- Individual metrics for major stablecoins (USDT, USDC, BUSD, DAI, TUSD)
  - Price, Market Cap, Volume
- Aggregate metrics
  - Total stablecoin market cap
  - 30-day and 60-day lagged values for correlation analysis
  - Percentage changes

### Volatility Indices (Yahoo Finance)
- **CBOE SKEW Index** - Black swan event pricing
- **VIX** - Market volatility expectations
- **OVX** - Crude Oil Volatility Index
- SKEW/VIX ratio for additional insights

### Energy Market Dynamics
- Energy price indices (Yahoo Finance)
  - Energy Select Sector SPDR ETF
  - Natural Gas Futures
  - Brent Crude Oil Futures
- Mining cost proxy (calculated from energy prices and difficulty)

### Currency and Economic Metrics
- **Gold Futures** price
- **Bitcoin USD** price
- **US Dollar Index**
- Gold/BTC ratio and BTC/Gold ratio

### Derivatives Market Metrics
- CME Bitcoin futures data
- Spot price comparison
- Basis (difference between futures and spot)
- Percentage basis

## Combined Features Dataset
The script creates a combined dataset with key features from all categories, making it easy to use in AI/ML models.

## Requirements
- Python 3.6 or higher
- Required libraries:
  - pandas
  - numpy
  - requests
  - yfinance
  - matplotlib
  - dateutil

## Installation
```bash
pip install pandas numpy requests yfinance matplotlib python-dateutil
```

## Usage
Simply run the script to collect all the features:
```bash
python crypto_features_collector.py
```

The script will:
1. Create an `additional_features` directory to store all outputs
2. Collect data for each feature category from free public APIs
3. Save individual CSV files for each feature set
4. Create a combined features dataset
5. Generate a data summary report

## Output Files
- `fear_greed_index.csv` - Market sentiment indicators
- `onchain_metrics.csv` - Comprehensive on-chain metrics
- `mining_difficulty.csv` - Bitcoin mining difficulty (subset of on-chain metrics)
- `miner_revenue.csv` - Bitcoin miner revenue (subset of on-chain metrics)
- `stablecoin_metrics.csv` - Detailed stablecoin data
- `stablecoin_aggregates.csv` - Aggregate stablecoin metrics
- `volatility_indices.csv` - Market volatility indicators
- `energy_mining_metrics.csv` - Energy market data and mining costs
- `currency_metrics.csv` - Currency and economic indicators
- `derivatives_metrics.csv` - Derivatives market data
- `combined_features.csv` - Selected key features from all categories
- `feature_descriptions.csv` - Detailed descriptions of each feature
- `data_summary.csv` - Summary statistics for all generated files

## Data Sources
All data is collected from free public APIs with proper attribution:
- **Alternative.me** - Fear & Greed Index
- **Blockchain.com** - On-chain metrics
- **CoinGecko** - Stablecoin and crypto market data
- **Yahoo Finance** - Volatility indices, energy prices, currency metrics

Each CSV file includes a header comment with the source information.

## Notes
- The script handles rate limiting and API errors gracefully
- Missing data is handled using forward/backward filling
- Data collection starts from 2012-01-01 by default to match existing datasets
- All dates are standardized to YYYY-MM-DD format
- The combined features dataset is ready for immediate use in ML models

## Limitations
- Free APIs may have rate limits or incomplete historical data
- Some advanced metrics are approximated due to data availability constraints
- Economic data may have gaps over weekends and holidays
- Some derived metrics (like mining cost) are simplified proxies

## Future Improvements
- Add more advanced metrics if/when free APIs become available
- Implement more sophisticated lag and lead indicators for predictive models
- Add automatic correlation analysis between features
- Support for scheduled updates to keep datasets current 