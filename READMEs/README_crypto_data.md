# Crypto and Market Features Collection Results

## Overview
This document summarizes the additional crypto and market features collected from various free public APIs and data sources. The data can be used for AI/ML projects and analysis of cryptocurrency market trends and correlations.

## Data Files Collected

### Market Sentiment
- **fear_greed_index.csv**: Daily Fear & Greed Index (0-100) with classification indicating market sentiment
  - Source: Alternative.me API
  - Time Range: ~1000 days of recent data
  - Features: FearGreedValue, FearGreedClassification

### On-chain Metrics
- **onchain_metrics.csv**: Comprehensive blockchain metrics from the Bitcoin network
  - Source: Blockchain.com Public API
  - Time Range: 2012-01-02 to present
  - Features: Active Addresses, Transaction Count, Mempool Size, Hash Rate, Mining Difficulty, etc.

- **mining_difficulty.csv**: Bitcoin mining difficulty over time (subset of onchain metrics)
  - Source: Blockchain.com Public API
  - Time Range: 2012-01-04 to present
  - Features: Mining Difficulty

- **miner_revenue.csv**: Bitcoin miner revenue in USD (subset of onchain metrics)
  - Source: Blockchain.com Public API
  - Time Range: 2012-01-02 to present
  - Features: Miner Revenue (USD)

### Volatility Indices
- **volatility_indices.csv**: Market volatility indicators from traditional markets
  - Source: Yahoo Finance API
  - Time Range: 2012-01-03 to present
  - Features: CBOE SKEW Index, CBOE Volatility Index (VIX), Crude Oil Volatility Index (OVX)

### Currency Metrics
- **currency_metrics.csv**: Currency and economic indicators, including Gold, Bitcoin, and USD Index
  - Source: Yahoo Finance API
  - Time Range: 2012-01-03 to present
  - Features: Gold Futures, Bitcoin USD, US Dollar Index, Gold/BTC Ratio

### Combined Datasets
- **volatility_currency_combined.csv**: Combined volatility indices and currency metrics
  - Sources: Yahoo Finance API (both volatility and currency data)
  - Time Range: 2012-01-03 to present
  - Features: 
    - Volatility_CBOE SKEW Index
    - Volatility_CBOE Volatility Index (VIX)
    - Volatility_Crude Oil Volatility Index (OVX)
    - Currency_Gold Futures
    - Currency_Bitcoin USD
    - Currency_US Dollar Index

## Usage Notes
1. All CSV files include header comments with source information
2. Files use standard date format for the index
3. Some files may have missing values for certain dates
4. Data can be imported using pandas:
   ```python
   import pandas as pd
   df = pd.read_csv('additional_features/file_name.csv', comment='#')
   df['Date'] = pd.to_datetime(df['Date'])
   df.set_index('Date', inplace=True)
   ```

## Feature Highlights

### Market Sentiment Indicators
The Fear & Greed Index serves as a valuable contrarian indicator, often showing extreme fear at market bottoms and extreme greed at tops.

### On-chain Health Metrics
Active addresses and transaction counts provide insight into network usage and adoption trends, while mining difficulty reflects the security and competitiveness of the network.

### Volatility Measures
The VIX index (often called the "fear index") measures market expectations of near-term volatility, while the SKEW index indicates the perceived probability of market tail risk (black swan events).

### Currency and Economic Context
Gold prices and the USD index provide context for Bitcoin's performance as both a risk asset and a potential inflation hedge or store of value.

## Collection Challenges
During the data collection process, some challenges were encountered:

1. API rate limits and occasional timeouts
2. Different date formats between data sources
3. Structural differences in CSV files making direct combination difficult
4. Missing data for some metrics on weekends or holidays
5. Limited historical depth for some data sources

## Future Improvements
Potential improvements for the data collection process:

1. Implement automatic scheduled updates
2. Develop more robust error handling for API downtime
3. Create standardized preprocessing pipeline for all data sources
4. Add more advanced economic indicators if free API sources become available
5. Implement statistical correlation analysis between features

## Data Cleaning Issues
Several issues were encountered when attempting to combine all datasets:

1. Inconsistent index formats between datasets (some using timestamps, others using integers)
2. Differences in date ranges and frequencies
3. Column name overlaps between datasets
4. Extra header rows in some files (like Ticker and Price rows)

These issues were partially resolved in the combined volatility and currency dataset, but a complete combination of all datasets would require more extensive preprocessing. 