# Project Structure

This README describes the organization of the MSc-COM7003-AI-Model project, which contains tools for data collection, processing, validation, and visualization of cryptocurrency datasets.

## Directory Structure

The project is organized into the following main directories:

### AI_Gen_Code

This directory contains all the Python code for the project, organized into subdirectories:

#### data_collection

Scripts for collecting data from various sources:

- `crypto_features_collector.py`: Collects cryptocurrency features from multiple sources
- `collect_individual_features.py`: For collecting specific features
- `get_exchange_data_simple.py`: Retrieves exchange rate data
- `get_historical_data.py`: Retrieves historical cryptocurrency data
- `validation.py`: Data validation utilities
- `data_collection_workflow.py`: Integrated workflow for data collection and validation
- `validate_collection_scripts.py`: Validates the collection scripts
- `rename_validated_files.py`: Utility for renaming validated CSV files

#### data_processing

Data processing scripts organized into sub-categories:

##### combining

- `combine_datasets.py`: Combines multiple datasets into a single dataset
- `combine_volatility_currency.py`: Specifically combines volatility and currency data

##### cleaning

- `data_cleaner.py`: Cleans data by removing outliers, handling missing values, etc.
- `fix_and_standardize_data.py`: Standardizes data formats and fixes common issues

##### processing

- `data_processor.py`: Processes raw data into a format suitable for analysis

#### visualizations

Scripts for creating various visualizations:

- `data_quality_viz.py`: Visualizations for data quality assessment
- `time_series_analysis.py`: Visualizations for time series structure analysis
- `outlier_detection.py`: Visualizations for detecting and analyzing outliers
- `stationarity_analysis.py`: Visualizations for stationarity analysis
- `feature_relationships.py`: Visualizations for exploring feature relationships
- `temporal_patterns.py`: Visualizations for temporal pattern analysis (daily, weekly, monthly, etc.)

#### utilities

Miscellaneous utility scripts:

- `dependancies.py`: Lists required package dependencies

### datasets

This directory contains the raw and processed datasets used by the project.

### READMEs

Documentation for various aspects of the project:

- `README_validation.md`: Documentation on the data validation process
- `README_data_collection_best_practices.md`: Best practices for data collection
- `README_project_structure.md`: This file, explaining the project structure

### notes

Contains miscellaneous notes and documentation.

## Usage

Each of the scripts within the AI_Gen_Code directory can be run individually, or some can be used as part of a workflow. See the individual script documentation for specific usage instructions.

For data collection, start with the `data_collection_workflow.py` script, which orchestrates the collection process:

```bash
python AI_Gen_Code/data_collection/data_collection_workflow.py
```

For visualization, the various visualization scripts in the `visualizations` directory can be run with a CSV file as an argument:

```bash
python AI_Gen_Code/visualizations/data_quality_viz.py datasets/your_file.csv
```

## Data Flow

The typical data flow in this project follows these steps:

1. Data collection: Using scripts in `data_collection` to gather data
2. Data validation: Validating the collected data using `validation.py`
3. Data processing: Processing the validated data using scripts in `data_processing`
4. Data visualization: Creating visualizations using scripts in `visualizations`

## Maintenance

When adding new scripts to the project, please follow the established directory structure:

- Data collection scripts should go in `AI_Gen_Code/data_collection`
- Data processing scripts should go in the appropriate subdirectory of `AI_Gen_Code/data_processing`
- Visualization scripts should go in `AI_Gen_Code/visualizations`
- Utility scripts should go in `AI_Gen_Code/utilities` 