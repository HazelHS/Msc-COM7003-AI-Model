# Visualization Scripts

This directory contains various scripts for visualizing cryptocurrency data. Each script focuses on a specific aspect of the data and generates visualizations to help understand the dataset.

## Available Scripts

- **data_quality_viz.py**: Assesses data quality by visualizing missing values, distributions, and summary statistics.
  - Generates missing value visualizations
  - Displays data distributions
  - Creates time series overviews

- **time_series_analysis.py**: Analyzes the structure of time series data.
  - Checks date continuity and completeness
  - Detects gaps in the time series
  - Analyzes sampling frequency
  - Examines weekend data presence

- **outlier_detection.py**: Identifies and visualizes outliers in the dataset.
  - Creates box plots for numeric features
  - Uses z-scores to identify statistical outliers
  - Analyzes extreme values
  - Examines local outliers in their temporal context

- **stationarity_analysis.py**: Tests stationarity properties of the time series.
  - Generates rolling statistics visualizations
  - Analyzes time series transformations (first difference, log, etc.)
  - Visualizes autocorrelation functions
  - Provides simple stationarity assessments

- **feature_relationships.py**: Explores relationships between features in the dataset.
  - Creates correlation matrices
  - Generates scatter plots for highly correlated pairs
  - Shows time-aligned comparisons of related features
  - Visualizes feature distributions

- **temporal_patterns.py**: Analyzes temporal patterns in the data.
  - Examines daily, monthly, and yearly patterns
  - Visualizes volatility patterns
  - Detects periodicities
  - Shows year-over-year comparisons

## Usage

All scripts follow a similar usage pattern. They can be run directly from the command line, passing the path to a CSV file:

```bash
python temporal_patterns.py ../datasets/processed_exchanges/BTC_USD.csv
```

## Output

All visualizations are saved to a `visualization_output` directory, which is created automatically if it doesn't exist. Each plot is saved with a timestamp in the filename to avoid overwriting previous results.

## Dependencies

These scripts are designed to be lightweight and have minimal dependencies:

- **pandas**: For data manipulation and analysis
- **matplotlib**: For all visualizations
- **numpy**: For numerical operations

All other functionality is implemented directly within the scripts without relying on additional libraries.

## Important Note

As of the latest update, all visualization scripts have been simplified to use only pandas and matplotlib for visualizations. All functionality previously dependent on seaborn, scipy, and statsmodels has been reimplemented using these core libraries to ensure compatibility across different environments.

## Installation

You can install the required packages with:

```bash
pip install -r requirements.txt
``` 