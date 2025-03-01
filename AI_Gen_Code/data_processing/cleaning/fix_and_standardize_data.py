"""
Data Cleaner and Standardizer

This script runs the data cleaning processes to fix all identified issues:
1. Fix missing or improperly formatted Date columns
2. Remove duplicate dates
3. Standardize date formats across all CSV files
4. Regenerate the combined datasets with proper date handling

Run this script from the AI_Gen_Code directory.
"""

import os
import sys
import subprocess
import time
import pandas as pd
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and return its success status"""
    print(f"\n{'='*80}")
    print(f"Running: {script_name} - {description}")
    print(f"{'='*80}")
    
    try:
        process = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(process.stdout)
        if process.stderr:
            print(f"Warnings/Errors:\n{process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(f"Error output:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"Exception running {script_name}: {e}")
        return False

def check_fixed_files():
    """Check if the fixed files were created successfully"""
    fixed_files = [
        '../datasets/additional_features/currency_metrics_fixed.csv',
        '../datasets/additional_features/volatility_currency_combined_fixed.csv',
        '../datasets/processed_exchanges/all_indices_processed_fixed.csv',
    ]
    
    results = {}
    for file in fixed_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                results[file] = {
                    'exists': True,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'date_column': 'Date' in df.columns
                }
            except Exception as e:
                results[file] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            results[file] = {
                'exists': False
            }
    
    return results

def print_summary(results):
    """Print a summary of the results"""
    print("\n\n" + "="*80)
    print("SUMMARY OF DATA CLEANING OPERATIONS")
    print("="*80)
    
    for file, info in results.items():
        print(f"\nFile: {os.path.basename(file)}")
        if info['exists']:
            if 'error' in info:
                print(f"  Status: Created but has errors - {info['error']}")
            else:
                print(f"  Status: Successfully created")
                print(f"  Rows:   {info['rows']}")
                print(f"  Columns: {info['columns']}")
                print(f"  Has Date column: {'Yes' if info['date_column'] else 'No'}")
        else:
            print(f"  Status: Not created")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("1. Verify the fixed files have proper date formatting")
    print("2. Use the _fixed.csv versions of the files for future analysis")
    print("3. Consider renaming the fixed files to the original names if all tests pass")
    print("4. Run the analyze_dates.py script again to confirm the issues are resolved")
    print("="*80)

def main():
    """Main function to run the data cleaning process"""
    print("Starting data cleaning and standardization process...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Run the data_cleaner script to fix specific issues
    print("\nStep 1: Running data_cleaner.py to fix date structures...")
    clean_success = run_script('data_cleaner.py', "Fix date structures in CSV files")
    
    # Add a short delay
    time.sleep(1)
    
    # Step 2: Run updated combine_volatility_currency.py to create a proper combined file
    print("\nStep 2: Running combine_volatility_currency.py to create proper combined file...")
    combine_volatility_success = run_script('combine_volatility_currency.py', "Combine volatility and currency data")
    
    # Step 3: Check the results
    print("\nStep 3: Checking results...")
    results = check_fixed_files()
    
    # Step 4: Print summary
    print_summary(results)
    
    print("\nData cleaning and standardization process completed!")

if __name__ == "__main__":
    main() 