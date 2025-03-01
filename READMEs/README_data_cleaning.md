# Data Cleaning and Standardization

This directory contains scripts to fix and standardize the date structures in the cryptocurrency and market data CSV files.

## Issues Identified

The `analyze_dates.py` script identified several issues in the datasets:

1. **Missing Date Columns**:
   - `currency_metrics.csv`: Date column structure incorrect (date in row rather than column)
   - `volatility_currency_combined.csv`: Missing proper Date column

2. **Duplicate Dates**:
   - `all_indices_processed.csv`: Contains over 35,000 duplicate dates

3. **Different Time Frequencies**:
   - Exchange data: "Daily (business days)" pattern (weekdays only)
   - Blockchain data: Complete "Daily" pattern (including weekends)

4. **Limited Common Date Range**:
   - Common date range: 2022-06-05 to 2025-02-27 (limiting factor: fear_greed_index.csv)

## Solution Scripts

### 1. `data_cleaner.py`

Fixes specific issues with date structures:
- Resolves the missing Date column in `currency_metrics.csv`
- Fixes the structure of `volatility_currency_combined.csv`
- Removes duplicate dates from `all_indices_processed.csv`
- Standardizes date formats across all CSV files

### 2. `combine_volatility_currency.py` (Updated)

Improved version that properly combines volatility and currency data:
- Detects and handles special file structures automatically
- Aligns date ranges properly
- Removes duplicate dates
- Creates a proper combined file with consistent date format

### 3. `get_exchange_data_simple.py` (Updated)

Enhanced version that prevents duplicate dates in the combined index file:
- Improves the exchange data processing
- Adds additional features (moving averages, volatility metrics)
- Creates a pivot table version for easier analysis
- Prevents duplicate date issues in the combined file

### 4. `fix_and_standardize_data.py`

Wrapper script that runs all the data cleaning functions in the proper sequence:
1. Fixes specific issues with `data_cleaner.py`
2. Creates a properly combined volatility/currency file
3. Checks the results and provides a summary

## Usage

Run the full data cleaning process:

```bash
cd AI_Gen_Code
python fix_and_standardize_data.py
```

This will:
1. Fix all identified issues
2. Create _fixed.csv versions of the problematic files
3. Generate a summary of the results

## Output Files

The scripts create fixed versions of the problematic files:

- `datasets/additional_features/currency_metrics_fixed.csv`
- `datasets/additional_features/volatility_currency_combined_fixed.csv`
- `datasets/processed_exchanges/all_indices_processed_fixed.csv`

## Verification

After running the cleaning scripts, run the analyze_dates.py script again to verify the issues have been resolved:

```bash
python analyze_dates.py
```

## Next Steps

1. Verify the fixed files have proper date formatting
2. Use the _fixed.csv versions of the files for future analysis
3. Consider renaming the fixed files to the original names if all tests pass
4. Update data collection processes to use the improved scripts going forward 