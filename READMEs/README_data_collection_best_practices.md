# Best Practices for Data Collection

This document outlines best practices for collecting and processing data to ensure consistency and prevent issues with date structures.

## General Principles

1. **Consistency is Key**:
   - Use the same date format across all datasets
   - Standardize column naming conventions
   - Follow the same file structure for similar data types

2. **Data Quality First**:
   - Validate data at each step of the collection process
   - Handle missing values and invalid entries gracefully
   - Document data sources and any known limitations

3. **Error Handling**:
   - Implement robust error handling in all scripts
   - Log errors and warnings for later analysis
   - Fail gracefully when issues are encountered

## Date Formatting Guidelines

### Standard Date Format

Always use the ISO 8601 format (`YYYY-MM-DD`) for dates:

```python
# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Format consistently
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
```

### Date Column Requirements

1. Every dataset should have a 'Date' column
2. The 'Date' column should be properly parsed as a datetime object
3. Invalid dates should be handled appropriately (either removed or fixed)

### Date Handling Best Practices

```python
# Correct way to handle dates
import pandas as pd

def process_dates(df):
    # Ensure Date column exists
    if 'Date' not in df.columns:
        raise ValueError("Dataset missing required 'Date' column")
    
    # Convert to datetime with error handling
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Remove rows with invalid dates
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        print(f"Removing {invalid_dates} rows with invalid dates")
        df = df[~df['Date'].isna()]
    
    # Standardize format
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Check for duplicate dates
    duplicates = df['Date'].duplicated().sum()
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate dates found")
        # Handle duplicates (e.g., keep first occurrence)
        df = df.drop_duplicates(subset=['Date'], keep='first')
    
    return df
```

## Data Collection Template

When creating a new data collection script, follow this template:

```python
"""
Data Collection Script for [Data Source]

This script collects [type of data] from [source] and saves it as a CSV file.

Usage:
    python script_name.py [options]
"""

import os
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_data():
    """Collect data from source"""
    try:
        # Your data collection code here
        # ...
        
        # Create a DataFrame
        df = pd.DataFrame(...)
        
        # Ensure Date column exists and is properly formatted
        df = process_dates(df)
        
        return df
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return None

def process_dates(df):
    """Process and standardize dates"""
    try:
        # Ensure Date column exists
        if 'Date' not in df.columns:
            logger.error("Dataset missing required 'Date' column")
            raise ValueError("Dataset missing required 'Date' column")
        
        # Convert to datetime with error handling
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removing {invalid_dates} rows with invalid dates")
            df = df[~df['Date'].isna()]
        
        # Standardize format
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Check for duplicate dates
        duplicates = df['Date'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{duplicates} duplicate dates found, removing duplicates")
            df = df.drop_duplicates(subset=['Date'], keep='first')
        
        return df
    except Exception as e:
        logger.error(f"Error processing dates: {e}")
        raise

def save_data(df, output_path):
    """Save data to CSV file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting data collection")
    
    # Collect data
    df = collect_data()
    if df is None:
        logger.error("Data collection failed")
        return
    
    # Generate output path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"datasets/source_name/data_{timestamp}.csv"
    
    # Save data
    save_data(df, output_path)
    
    logger.info("Data collection completed")

if __name__ == "__main__":
    main()
```

## Validation Process

After collecting data, always validate it using the validation script:

```bash
python AI_Gen_Code/validation.py your_new_data.csv
```

This will:
1. Check the structure and content of your data
2. Validate the date formatting
3. Create a validated version if issues are found
4. Provide a detailed report of any problems

## Preventing Common Issues

### Missing Date Columns

Always include a proper Date column in your datasets:

```python
# If your data has a date-like column with a different name
if 'date' in df.columns and 'Date' not in df.columns:
    df['Date'] = df['date']
    
# If your data has a timestamp column
if 'Timestamp' in df.columns and 'Date' not in df.columns:
    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
```

### Duplicate Dates

Handle duplicate dates explicitly:

```python
# Check for duplicates
duplicates = df['Date'].duplicated().sum()
if duplicates > 0:
    logger.warning(f"Found {duplicates} duplicate dates")
    
    # Option 1: Keep first occurrence
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    # Option 2: Aggregate values
    # df = df.groupby('Date').mean().reset_index()
```

### Inconsistent Date Formats

Always standardize date formats:

```python
# Even if you think dates are already formatted correctly, 
# always run this standardization step
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
```

## Integrating with the Workflow

New data collection scripts should be integrated with the data collection workflow:
1. Add your script to the appropriate section in `data_collection_workflow.py`
2. Validate your script using `validate_collection_scripts.py`
3. Run the full workflow to test integration

This ensures all data is collected, validated, and standardized in a consistent manner. 