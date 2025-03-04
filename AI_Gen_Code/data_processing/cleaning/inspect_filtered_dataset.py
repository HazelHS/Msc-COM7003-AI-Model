import pandas as pd
import os
import glob

# Get the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

# Define the path to the combined dataset directory
dataset_dir = os.path.join('datasets', 'combined_dataset')
print(f"Looking for files in: {os.path.abspath(dataset_dir)}")

try:
    # Find the most recent filtered dataset file
    filtered_files = glob.glob(os.path.join(dataset_dir, 'filtered_dataset_*.csv'))
    
    if filtered_files:
        # Sort by modification time (newest first)
        filtered_files.sort(key=os.path.getmtime, reverse=True)
        filtered_dataset_path = filtered_files[0]
        print(f"Using most recent filtered dataset: {filtered_dataset_path}")
    else:
        # If no filtered dataset files found, try using the combined_dataset_latest.csv
        filtered_dataset_path = os.path.join(dataset_dir, 'combined_dataset_latest.csv')
        print(f"No filtered datasets found, using latest combined dataset: {filtered_dataset_path}")
    
    # Check if the file exists
    if os.path.exists(filtered_dataset_path):
        print(f"File exists: {filtered_dataset_path}")
        
        # Read the CSV file
        df = pd.read_csv(filtered_dataset_path)
        
        # Print basic information
        print("\nDataset Information:")
        print(f"Shape: {df.shape} (rows, columns)")
        
        # Print column names
        print("\nColumns:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        
        # Print date range
        if 'Date' in df.columns:
            print(f"\nDate Range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Number of unique dates: {df['Date'].nunique()}")
        
        # Print the first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
    else:
        print(f"File does not exist: {filtered_dataset_path}")
        
        # List files in the directory
        print(f"\nFiles in {dataset_dir}:")
        for file in os.listdir(dataset_dir):
            print(f"  - {file}")
            
except Exception as e:
    print(f"Error: {str(e)}") 