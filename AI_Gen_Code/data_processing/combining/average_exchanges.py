"""
Script to average exchange data from processed_exchanges folder.

This script reads all CSV files from the processed_exchanges folder,
averages their values, and saves the result as averaged_exchanges.csv.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_exchange_data(exchanges_dir):
    """Load all exchange data from the processed_exchanges directory"""
    exchange_files = []
    exchange_data = {}
    
    # Get all CSV files from processed_exchanges directory
    for file in os.listdir(exchanges_dir):
        if file.endswith('.csv') and not file.startswith('averaged_exchanges'):
            file_path = os.path.join(exchanges_dir, file)
            exchange_name = os.path.splitext(file)[0]  # Get filename without extension
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Check if Date column exists
                if 'Date' not in df.columns:
                    print(f"Warning: {file} doesn't have a Date column, skipping...")
                    continue
                
                # Convert Date to datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Check for duplicate dates and handle them
                duplicate_dates = df['Date'].duplicated().sum()
                if duplicate_dates > 0:
                    print(f"Warning: {file} has {duplicate_dates} duplicate dates. Aggregating by date...")
                    # Group by date and calculate mean for numeric columns
                    df = df.groupby('Date', as_index=False).mean(numeric_only=True)
                
                # Drop any rows with NaT dates
                df = df.dropna(subset=['Date'])
                
                # Convert numeric columns to float, excluding non-numeric columns
                for col in df.columns:
                    if col != 'Date':
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            print(f"Warning: Could not convert column {col} in {file} to numeric. Dropping column.")
                            df = df.drop(columns=[col])
                
                # Only keep the file if it has data and numeric columns
                if df.shape[0] > 0 and df.shape[1] > 1:  # More than just the Date column
                    exchange_data[exchange_name] = df
                    exchange_files.append(file)
                    print(f"Loaded {exchange_name} with {df.shape[0]} rows and {df.shape[1]} columns")
                else:
                    print(f"Warning: {file} has no usable data after processing, skipping...")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return exchange_data, exchange_files

def average_exchanges(exchange_data):
    """Average the values from all exchanges"""
    if not exchange_data:
        raise ValueError("No exchange data provided")
    
    # Get all unique dates across all exchanges
    all_dates = set()
    for df in exchange_data.values():
        all_dates.update(df['Date'].tolist())
    
    # Sort dates chronologically
    all_dates = sorted(all_dates)
    
    # Create a DataFrame with all dates
    result_df = pd.DataFrame({'Date': all_dates})
    
    # Track columns and their counts
    column_sums = {}
    column_counts = {}
    
    # Get all column names (except Date) from all exchanges
    all_columns = set()
    for df in exchange_data.values():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        all_columns.update([col for col in numeric_cols if col != 'Date'])
    
    # Initialize columns in result dataframe
    for col in all_columns:
        result_df[col] = 0.0
        column_sums[col] = np.zeros(len(all_dates))
        column_counts[col] = np.zeros(len(all_dates))
    
    # Process each exchange
    print("Starting to average exchanges...")
    date_to_idx = {date: i for i, date in enumerate(all_dates)}
    
    for exchange_name, df in exchange_data.items():
        print(f"Processing {exchange_name}...")
        
        # Find columns that are in both the exchange and our result columns
        common_cols = [col for col in df.columns if col in all_columns]
        
        # For each row in the exchange data
        for _, row in df.iterrows():
            date = row['Date']
            if date in date_to_idx:
                idx = date_to_idx[date]
                # For each numeric column
                for col in common_cols:
                    if col != 'Date' and not pd.isna(row[col]):
                        try:
                            value = float(row[col])
                            column_sums[col][idx] += value
                            column_counts[col][idx] += 1
                        except:
                            pass  # Skip if we can't convert to float
    
    # Calculate averages
    for col in all_columns:
        for i in range(len(all_dates)):
            if column_counts[col][i] > 0:
                result_df.loc[i, col] = column_sums[col][i] / column_counts[col][i]
            else:
                result_df.loc[i, col] = np.nan
    
    return result_df

def save_averaged_data(df, output_dir):
    """Save the averaged exchange data"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"averaged_exchanges_{timestamp}.csv")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved averaged exchange data to {output_file}")
    
    # Also save without timestamp for pipeline use
    latest_file = os.path.join(output_dir, "averaged_exchanges.csv")
    df.to_csv(latest_file, index=False)
    print(f"Saved latest version to {latest_file}")
    
    return output_file, latest_file

def main():
    """Main function to run the averaging process"""
    try:
        # Get the project root directory (two levels up from this script)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Define paths
        exchanges_dir = os.path.join(project_root, "datasets", "processed_exchanges")
        output_dir = os.path.join(project_root, "datasets", "processed_exchanges")
        
        print("Starting exchange averaging process...")
        print(f"Reading from directory: {exchanges_dir}")
        
        # Load exchange data
        exchange_data, exchange_files = load_exchange_data(exchanges_dir)
        
        if not exchange_data:
            print("No exchange data found to process")
            return
        
        print(f"\nProcessing {len(exchange_files)} exchange files:")
        for file in exchange_files:
            print(f"- {file}")
        
        # Average the exchanges
        print("Calculating averages...")
        averaged_df = average_exchanges(exchange_data)
        
        # Save the results
        output_file, latest_file = save_averaged_data(averaged_df, output_dir)
        
        print("\nAveraging process completed successfully!")
        print(f"Output saved to: {output_file}")
        print(f"Latest version saved to: {latest_file}")
        
        # Print summary statistics
        print("\nSummary of averaged data:")
        print(f"Date range: {min(averaged_df['Date'])} to {max(averaged_df['Date'])}")
        print(f"Number of columns: {len(averaged_df.columns)}")
        print(f"Number of rows: {len(averaged_df)}")
        print("\nColumns included:")
        for col in averaged_df.columns:
            if col != "Date":
                non_null_count = averaged_df[col].count()
                print(f"- {col}: {non_null_count} non-null values")
        
        return 0
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main() 