# Cryptocurrency Data Visualizations

This directory contains scripts for generating visualizations from cryptocurrency data. Each script focuses on a different aspect of data visualization and analysis.

## Available Scripts

1. **data_quality_viz.py** - Assesses data quality through visualizations:
   - Heatmaps of missing values
   - Data distribution histograms
   - Summary statistics tables
   - Time series overview plots

2. **time_series_analysis.py** - Analyzes time series data structure:
   - Time series completeness checks
   - Gap analysis visualizations
   - Sampling frequency pattern detection
   - Weekend coverage analysis

3. **outlier_detection.py** - Identifies and visualizes outliers:
   - Box plots for numeric features
   - Z-score analysis and visualization
   - Extreme value detection
   - Local outlier analysis

4. **stationarity_analysis.py** - Tests and visualizes stationarity properties:
   - Rolling statistics visualizations
   - Time series decomposition plots
   - Stationarity test results
   - Transformation comparisons

5. **feature_relationships.py** - Explores relationships between features:
   - Correlation matrices and clustermaps
   - Scatter plots with regression lines
   - Joint plots for distribution analysis
   - Price relationship analysis

6. **temporal_patterns.py** - Analyzes temporal patterns:
   - Day of week patterns
   - Monthly and seasonal trends
   - Yearly comparison visualizations
   - Time of day analysis (for hourly data)

## Usage

To run any of the visualization scripts, use the following command:

```bash
python <script_name.py> <path_to_csv_file>
```

Example:
```bash
python temporal_patterns.py ../datasets/processed_exchanges/BTC_USD.csv
```

You can also use our convenient GUI tool to run visualizations:
```bash
python ../visualization_gui.py
```

## Output

All visualizations are saved to the `visualization_output` directory, which will be created automatically if it doesn't exist. Each script generates multiple visualizations based on the dataset provided.

## Dependencies

These scripts utilize the following Python packages for high-quality visualizations:

- **pandas** - For data manipulation and analysis
- **matplotlib** - For creating static visualizations
- **seaborn** - For enhanced statistical visualizations with better aesthetics
- **numpy** - For numerical computations

All required dependencies can be installed using:

```bash
pip install -r ../requirements.txt
```

The visualizations created with these scripts provide valuable insights into cryptocurrency data patterns, quality issues, and statistical properties. 