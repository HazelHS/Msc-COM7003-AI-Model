# Cryptocurrency Market Analysis & Visualization Toolkit

## Project Purpose

This project provides a comprehensive suite of Python-based tools for analyzing and visualizing cryptocurrency market data, with a specific focus on Bitcoin. The toolkit enables researchers, traders, and analysts to explore various aspects of cryptocurrency price dynamics, including market patterns, anomalies, correlations, and valuation metrics.

The primary goal is to transform raw cryptocurrency market data into actionable insights through advanced visualization techniques, statistical analysis, and time series exploration. This framework is designed to be integrated with machine learning models (specifically TensorFlow) for predictive analytics.

## Key Features

### Data Collection Modules
- Support for various data sources including public APIs and local historical datasets

### Visualization Modules
- `data_quality_viz.py`: Explores data quality aspects including missing values, outliers, and distributions
- `outlier_detection.py`: Implements multiple methods to detect and visualize anomalies in trading data
- `time_series_analysis.py`: Analyzes sampling frequency, weekend patterns, and temporal coverage
- `feature_relationships.py`: Visualizes correlations and relationships between different market metrics
- `stationarity_analysis.py`: Tests and visualizes time series stationarity using various methods
- `temporal_patterns.py`: Identifies and visualizes seasonal and cyclical patterns

### User Interface
- `visualization_gui.py`: Provides a simple GUI for selecting and running visualization scripts on CSV data files

## Data Requirements

All visualization modules expect data in CSV format with the following structure:
1. An 'Index' column identifying the asset (e.g., 'BTC')
2. A 'Date' column in YYYY-MM-DD format
3. Various metrics columns (depending on the specific dataset)

Standard datasets include:
- Price data: Contains Open, High, Low, Close, Volume columns
- Market indicators: Contains specialized market metrics
- Volatility and currency data: Contains various volatility metrics and currency-related features

## Technical Requirements

### Dependencies
- Python 3.8+
- pandas: Data manipulation and analysis
- matplotlib: Visualization foundation
- seaborn: Enhanced statistical visualizations
- numpy: Numerical computations
- tkinter: GUI implementation

### Data Sources
- Processed exchange data from various cryptocurrency exchanges
- Additional features including volatility and currency metrics

## Integration Points

This visualization toolkit is designed to:
1. Pre-process and validate data for use in TensorFlow AI models
2. Provide feature engineering insights for model development
3. Visualize model results and market predictions

## Data Source Documentation

1. **Exchange Data**
   - Located in `datasets/processed_exchanges/`
   - Sources include major cryptocurrency exchanges
   - Processing includes validation, normalization, and outlier handling

2. **Bitcoin Metrics**
   - Located in `datasets/bitcoin_metrics/`
   - Various on-chain and market metrics for Bitcoin
   - Each metric is documented with its specific source

3. **Additional Features**
   - Located in `datasets/additional_features/`
   - Includes volatility indices and currency-related metrics
   - Each feature set includes source documentation

## Usage Notes

All visualization modules can be run individually on specific datasets or accessed through the GUI. The visualization output is saved to the `visualization_output` directory for further analysis and reporting.

When using this project in AI agent conversations, refer to this document for understanding the project architecture, data requirements, and integration capabilities. 