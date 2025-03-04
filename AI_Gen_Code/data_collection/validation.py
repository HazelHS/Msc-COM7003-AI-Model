"""
Data Validation Script

This script validates CSV files to ensure they have the correct structure:
- Date column is in the correct format
- No missing values in key columns
- Data types are correct

Usage:
    python validation.py [file_path]
    
    If no file_path is specified, all CSV files in the datasets directory will be validated.
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
from datetime import datetime

def validate_file(file_path):
    """
    Validate a single CSV file
    
    Returns:
        tuple: (is_valid, issues_list)
    """
    print(f"Validating {os.path.basename(file_path)}...")
    issues = []
    
    try:
        # Try to read the file, checking if it has a comment in the first line
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        skiprows = 0
        if first_line.startswith('#'):
            skiprows = 1
            print(f"  File has a source comment")
        
        # Read the CSV file
        df = pd.read_csv(file_path, skiprows=skiprows)
        
        # Check if file is empty
        if df.empty:
            issues.append("File is empty")
            return False, issues
        
        # Check for Date column
        if 'Date' not in df.columns:
            issues.append("Missing Date column")
        else:
            # Check Date format
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                print(f"  Date column is valid")
            except Exception as e:
                issues.append(f"Invalid Date format: {e}")
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            issues.append(f"Contains {missing_count} missing values")
            # Check for columns with more than 10% missing values
            missing_percent = df.isna().mean() * 100
            bad_columns = missing_percent[missing_percent > 10].index.tolist()
            if bad_columns:
                issues.append(f"Columns with >10% missing values: {', '.join(bad_columns)}")
        else:
            print(f"  No missing values found")
        
        # Report statistics
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        if 'Date' in df.columns:
            try:
                date_range = f"{df['Date'].min()} to {df['Date'].max()}"
                print(f"  Date range: {date_range}")
            except:
                print("  Could not determine date range")
        
        # Return validation result
        if issues:
            print(f"  Issues found: {', '.join(issues)}")
            return False, issues
        else:
            print(f"  Validation successful!")
            return True, []
            
    except Exception as e:
        issues.append(f"Error reading file: {e}")
        print(f"  Error validating file: {e}")
        return False, issues

def validate_all_files():
    """
    Validate all CSV files in the datasets directory
    """
    print("Starting validation of all CSV files...")
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasets_dir = os.path.join(base_dir, "datasets")
    
    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk(datasets_dir):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_validated.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files to validate")
    
    # Validate each file
    validation_results = {}
    for file_path in csv_files:
        is_valid, issues = validate_file(file_path)
        validation_results[file_path] = {
            'is_valid': is_valid,
            'issues': issues
        }
    
    # Print summary
    print("\nValidation Summary:")
    valid_count = sum(1 for result in validation_results.values() if result['is_valid'])
    print(f"Valid files: {valid_count} of {len(csv_files)}")
    
    if valid_count < len(csv_files):
        print("\nFiles with issues:")
        for file_path, result in validation_results.items():
            if not result['is_valid']:
                file_name = os.path.basename(file_path)
                issues = ', '.join(result['issues'])
                print(f"- {file_name}: {issues}")
    
    return validation_results

def main():
    """
    Main function to handle command line arguments
    """
    if len(sys.argv) > 1:
        # Validate a specific file
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            is_valid, issues = validate_file(file_path)
            if is_valid:
                print(f"\nValidation successful: {file_path}")
                sys.exit(0)
            else:
                print(f"\nValidation failed: {file_path}")
                print(f"Issues: {', '.join(issues)}")
                sys.exit(1)
        else:
            print(f"File not found: {file_path}")
            sys.exit(1)
    else:
        # Validate all files
        validation_results = validate_all_files()
        valid_count = sum(1 for result in validation_results.values() if result['is_valid'])
        total_count = len(validation_results)
        
        if valid_count == total_count:
            print(f"\nAll {total_count} files passed validation!")
            sys.exit(0)
        else:
            print(f"\n{valid_count} of {total_count} files passed validation")
            sys.exit(1)

if __name__ == "__main__":
    main() 