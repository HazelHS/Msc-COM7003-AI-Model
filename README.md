# MSc COM7003 AI Module Project

This repository contains code for collecting, processing, and analyzing financial data for AI predictions.

## Project Structure

- **datasets/**: Main directory for all datasets
  - **additional_features/**: Contains additional feature files like currency metrics, fear and greed index, etc.
  - **processed_exchanges/**: Contains processed exchange data files for various stock indices
  - **combined_dataset/**: Contains the combined dataset for AI training

## Dataset Format

The combined dataset is stored in CSV format with the following characteristics:

- **File paths:**
  - Combined dataset (sorted oldest to newest): `datasets/combined_dataset/combined_dataset_latest.csv`

- **Date Range:** 
  - Full dataset: 2012-01-02 to 2025-03-03
  - Complete coverage for all years: 2012-2025 (including 2020-2021)
  - **Sorting order**: Ascending chronological order (oldest dates first, from 2012 to 2025)

- **Column Format:**
  - `Date`: Date in YYYY-MM-DD format
  - Exchange columns follow the format: `[Exchange Symbol] [Metric]`
    - Example: `IXIC Open`, `GDAXI Close`, `NYA Adj Close`
  - Special columns without prefixes:
    - `BTC/USD`: Bitcoin price in USD
    - `Gold/BTC Ratio`: Ratio of Gold price to Bitcoin price

- **Metrics included:**
  - Standard metrics for each exchange: Open, High, Low, Close, Adj Close, Volume, CloseUSD
  - Additional metrics from specialized datasets (fear and greed index, mining difficulty, etc.)

- **Excluded columns:**
  - Currency columns (e.g., `IXIC Currency`)
  - Miner Revenue columns

## Data Collection Scripts

The data collection process is managed by several key scripts:

- **main.py**: Central script that orchestrates the data collection workflow
- **data_collection/update_currency_metrics.py**: Updates currency metrics data
- **data_collection/get_exchange_data_simple.py**: Collects stock exchange data
- **data_collection/crypto_features_collector.py**: Collects cryptocurrency-related features
- **data_collection/create_combined_dataset.py**: Combines all data sources into a single aligned dataset
- **data_collection/sort_combined_dataset.py**: Sorts the combined dataset in chronological order

## Visualization Tools

The repository includes visualization tools for analyzing the collected data:

- **visualization/market_btc_comparison.py**: Tool for comparing stock market indices with BTC-USD price
  - Supports multiple stock indices in the same plot
  - Can display normalized percentage change or absolute values
  - Allows filtering by date range
  - Example usage: `python AI_Gen_Code/visualization/market_btc_comparison.py --indices IXIC GDAXI NYA`

## Usage

To run the full data collection pipeline:

```bash
python AI_Gen_Code/main.py
```

To generate the combined dataset only:

```bash
python AI_Gen_Code/data_collection/create_combined_dataset.py
```

To sort the combined dataset in ascending chronological order (oldest dates first):

```bash
python AI_Gen_Code/data_collection/sort_combined_dataset.py
```

To visualize BTC-USD price compared with stock market indices:

```bash
python AI_Gen_Code/visualization/market_btc_comparison.py --indices IXIC GDAXI NYA