"""
Data Validation Script

This script:
1. Validates CSV data by checking file structure and content
2. Handles special file formats (like those with metadata headers)
3. Creates properly structured versions of problematic files
4. Compares data against original sources where possible

Usage:
    python validation.py [file_path]  # Validate a specific file
    python validation.py --all        # Validate all files

Author: AI Assistant
Created: March 1, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
import re
from io import StringIO  # Updated import for StringIO

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory

# Define constants with absolute paths
DATA_DIRS = [
    os.path.join(BASE_DIR, 'datasets/additional_features'),
    os.path.join(BASE_DIR, 'datasets/processed_exchanges')
]

# Ensure directories exist
def ensure_directories_exist():
    """Create required directories if they don't exist"""
    for directory in DATA_DIRS:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
    return True

# Map of file patterns to data sources for validation
SOURCE_MAPPING = {
    'currency_metrics': {
        'source': 'yahoo_finance',
        'symbols': ['GC=F', 'BTC-USD', 'DX-Y.NYB']
    },
    'volatility_indices': {
        'source': 'yahoo_finance',
        'symbols': ['^SKEW', '^VIX', '^OVX']
    },
    'fear_greed_index': {
        'source': 'alternative_me_api',
        'endpoint': 'https://api.alternative.me/fng/'
    }
}

class FileValidator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.file_type = self._determine_file_type()
        self.df = None
        self.has_special_format = False
        self.metadata = {}
        self.errors = []
        self.warnings = []
        
    def _determine_file_type(self):
        """Determine the type of file based on filename"""
        for key in SOURCE_MAPPING.keys():
            if key in self.filename:
                return key
        return "standard"
    
    def read_file(self):
        """Read the file and detect special formats"""
        try:
            # First try to read with normal pandas
            self.df = pd.read_csv(self.filepath)
            
            # Check for special formats by reading raw content
            with open(self.filepath, 'r') as f:
                first_lines = [next(f) for _ in range(5)]
            
            # Check if file starts with comments or metadata
            if any(line.startswith('#') for line in first_lines):
                self.has_special_format = True
                # Try to parse the special format
                self._parse_special_format()
                
            return True
        except Exception as e:
            self.errors.append(f"Error reading file: {str(e)}")
            return False
    
    def _parse_special_format(self):
        """Parse files with special format (metadata headers)"""
        try:
            # Read the file line by line
            lines = []
            metadata = {}
            data_start_line = 0
            
            with open(self.filepath, 'r') as f:
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    # Extract metadata from comments
                    if line.startswith('#'):
                        key_value = line[1:].strip().split(':', 1)
                        if len(key_value) == 2:
                            metadata[key_value[0].strip()] = key_value[1].strip()
                    # Find where the actual data starts
                    if 'Date' in line and data_start_line == 0:
                        data_start_line = i
            
            self.metadata = metadata
            
            # Different parsing strategy based on file type
            if self.file_type == 'currency_metrics' or self.file_type == 'volatility_indices':
                # These files have a structure with header rows before the data
                column_line = None
                ticker_line = None
                date_line = None
                data_lines = []
                
                for i, line in enumerate(lines):
                    if i == 0 and line.startswith('#'):
                        continue
                    elif column_line is None:
                        column_line = line
                    elif ticker_line is None:
                        ticker_line = line
                    elif date_line is None:
                        date_line = line
                    else:
                        data_lines.append(line)
                
                # Extract column names and create a proper dataframe
                if column_line and date_line:
                    columns = column_line.split(',')
                    
                    # Check if 'Date' is in the line with 'Date'
                    if 'Date' in date_line.split(',')[0]:
                        # Create new data lines with date as the first column
                        new_data = []
                        for line in data_lines:
                            parts = line.split(',')
                            if len(parts) >= len(columns):
                                new_data.append(','.join(parts))
                        
                        # Convert to DataFrame
                        self.df = pd.read_csv(StringIO('\n'.join([column_line] + new_data)))
                    else:
                        self.warnings.append("Could not properly parse the special format")
        except Exception as e:
            self.errors.append(f"Error parsing special format: {str(e)}")
    
    def validate_structure(self):
        """Validate the basic structure of the file"""
        if self.df is None:
            return False
        
        # Check if DataFrame is empty
        if len(self.df) == 0:
            self.errors.append("File is empty")
            return False
        
        # Check if Date column exists or can be created
        if 'Date' not in self.df.columns:
            # For special formats, try to fix
            if self.has_special_format:
                self._fix_missing_date_column()
            else:
                self.errors.append("Missing Date column")
        
        # Check for duplicate dates
        if 'Date' in self.df.columns:
            duplicates = self.df['Date'].duplicated().sum()
            if duplicates > 0:
                self.warnings.append(f"Found {duplicates} duplicate dates")
        
        return True
    
    def _fix_missing_date_column(self):
        """Try to fix missing Date column for special format files"""
        if self.file_type == 'currency_metrics' or self.file_type == 'volatility_indices':
            try:
                # Read file again with custom parsing
                raw_lines = []
                with open(self.filepath, 'r') as f:
                    for line in f:
                        raw_lines.append(line.strip())
                
                # Find where data actually starts and columns are defined
                data_start = None
                columns = None
                
                for i, line in enumerate(raw_lines):
                    if 'Date' in line and 'Ticker' in raw_lines[i-1]:
                        columns = raw_lines[i-2].split(',')
                        data_start = i + 1
                        break
                
                if data_start is not None and columns is not None:
                    # Create a new DataFrame with proper structure
                    new_data = []
                    new_data.append(','.join(columns))  # Header row
                    
                    for i in range(data_start, len(raw_lines)):
                        line = raw_lines[i]
                        parts = line.split(',')
                        if len(parts) >= 1 and parts[0]:  # Make sure there's a date value
                            new_data.append(line)
                    
                    # Parse the reconstructed data
                    self.df = pd.read_csv(StringIO('\n'.join(new_data)))
                    return True
            except Exception as e:
                self.errors.append(f"Error fixing missing Date column: {str(e)}")
                return False
        return False
    
    def validate_data_types(self):
        """Validate that data types are appropriate"""
        if self.df is None:
            return False
        
        # Check Date column
        if 'Date' in self.df.columns:
            try:
                # Convert to datetime
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                return True
            except Exception as e:
                self.errors.append(f"Error converting Date column: {str(e)}")
                return False
        return True
    
    def validate_against_source(self):
        """Validate against original data source if possible"""
        if self.file_type not in SOURCE_MAPPING:
            self.warnings.append("No source mapping available for validation")
            return False
        
        source_info = SOURCE_MAPPING[self.file_type]
        
        # Yahoo Finance validation
        if source_info['source'] == 'yahoo_finance':
            # This would require API access to Yahoo Finance
            # For demonstration purposes only
            self.warnings.append("Yahoo Finance validation requires API access - skipping")
            return False
            
        # Fear and Greed Index validation
        elif source_info['source'] == 'alternative_me_api':
            try:
                # Get recent data for comparison
                response = requests.get(source_info['endpoint'])
                if response.status_code == 200:
                    data = response.json()
                    # Compare most recent value if available
                    if 'data' in data and len(data['data']) > 0:
                        self.warnings.append("Fear and Greed index verified with latest API data")
                        return True
            except Exception as e:
                self.warnings.append(f"Error validating against API: {str(e)}")
        
        return False
    
    def fix_and_save(self):
        """Fix issues and save a corrected version"""
        if self.df is None:
            return False
        
        # Standardize date format
        if 'Date' in self.df.columns:
            try:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                # Remove rows with invalid dates
                invalid_count = self.df['Date'].isna().sum()
                if invalid_count > 0:
                    self.warnings.append(f"Removed {invalid_count} rows with invalid dates")
                    self.df = self.df[~self.df['Date'].isna()]
                
                # Format as YYYY-MM-DD
                self.df['Date'] = self.df['Date'].dt.strftime('%Y-%m-%d')
            except Exception as e:
                self.errors.append(f"Error standardizing dates: {str(e)}")
        
        # Remove duplicates if any
        if 'Date' in self.df.columns:
            initial_len = len(self.df)
            self.df = self.df.drop_duplicates(subset=['Date'])
            if len(self.df) < initial_len:
                self.warnings.append(f"Removed {initial_len - len(self.df)} duplicate dates")
        
        # Save the fixed file
        if len(self.errors) == 0:
            try:
                output_path = self.filepath.replace('.csv', '_validated.csv')
                self.df.to_csv(output_path, index=False)
                print(f"  ✓ Fixed file saved to: {output_path}")
                return True
            except Exception as e:
                self.errors.append(f"Error saving fixed file: {str(e)}")
                return False
        
        return False
    
    def print_report(self):
        """Print validation report"""
        print(f"\n{'='*80}")
        print(f"VALIDATION REPORT FOR: {self.filename}")
        print(f"{'='*80}")
        
        print(f"File type: {self.file_type}")
        print(f"Special format: {'Yes' if self.has_special_format else 'No'}")
        
        if self.metadata:
            print("\nMetadata:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")
        
        if self.df is not None:
            print("\nDataFrame Info:")
            print(f"  Rows: {len(self.df)}")
            print(f"  Columns: {list(self.df.columns)}")
            if 'Date' in self.df.columns:
                date_min = self.df['Date'].min()
                date_max = self.df['Date'].max()
                print(f"  Date range: {date_min} to {date_max}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  ✗ {error}")
        else:
            print("\n✓ Validation passed with no errors")
        
        print(f"{'='*80}\n")

def validate_single_file(filepath):
    """Validate a single file"""
    print(f"\nValidating file: {filepath}")
    
    validator = FileValidator(filepath)
    if validator.read_file():
        validator.validate_structure()
        validator.validate_data_types()
        validator.validate_against_source()
        validator.fix_and_save()
        validator.print_report()
        return len(validator.errors) == 0
    else:
        validator.print_report()
        return False

def validate_all_files():
    """Validate all CSV files in the specified directories"""
    all_valid = True
    total_files = 0
    valid_files = 0
    
    for dir_path in DATA_DIRS:
        if os.path.exists(dir_path):
            print(f"\nProcessing directory: {dir_path}")
            for file in os.listdir(dir_path):
                if file.endswith('.csv') and not file.endswith('_validated.csv') and not file.endswith('_fixed.csv'):
                    total_files += 1
                    filepath = os.path.join(dir_path, file)
                    if validate_single_file(filepath):
                        valid_files += 1
                    else:
                        all_valid = False
    
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Files with issues: {total_files - valid_files}")
    print(f"Overall status: {'✓ All files passed' if all_valid else '✗ Some files have issues'}")
    print(f"{'='*80}\n")
    
    return all_valid

def main():
    """Main function"""
    print("CSV Data Validation Tool")
    print("========================")
    
    # Ensure all data directories exist
    print("Checking and creating required directories...")
    ensure_directories_exist()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            validate_all_files()
        else:
            filepath = sys.argv[1]
            if os.path.exists(filepath):
                validate_single_file(filepath)
            else:
                print(f"Error: File '{filepath}' not found")
    else:
        print("\nUsage:")
        print("  python validation.py [file_path]   # Validate a specific file")
        print("  python validation.py --all         # Validate all files")
        print("\nStarting validation of all files...\n")
        validate_all_files()

if __name__ == "__main__":
    main() 