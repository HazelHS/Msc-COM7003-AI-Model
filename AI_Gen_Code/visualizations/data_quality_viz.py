"""
Data Quality Visualizations

This script provides visualizations for assessing data quality in cryptocurrency datasets,
including missing values, data distributions, and basic statistics.

Usage:
    python data_quality_viz.py [csv_file_path]

Author: AI Assistant
Created: March 1, 2025
Modified: Current date - Simplified to use only pandas and matplotlib
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_data(file_path):
    """Load data from a CSV file"""
    try:
        # Try to parse the date column automatically
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Check if Date is actually parsed as datetime
        if not pd.api.types.is_datetime64_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        # Set Date as index if it exists
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        
        # If date parsing failed, try without parsing
        try:
            df = pd.read_csv(file_path)
            print("Loaded without date parsing")
            return df
        except:
            print(f"Failed to load {file_path}")
            return None

def missing_values_heatmap(df):
    """Create a heatmap of missing values in the dataset"""
    # Calculate missing values
    missing = df.isnull().transpose()
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure
    plt.figure(figsize=(12, max(6, len(df.columns) * 0.4)))
    
    # Plot missing values (light for present, dark for missing)
    plt.imshow(missing, cmap='Blues', aspect='auto')
    
    # Add labels and title
    plt.yticks(range(len(df.columns)), df.columns)
    plt.xticks([])
    plt.ylabel('Features')
    plt.title(f'Missing Values in Dataset ({missing.sum().sum()} total missing values)')
    
    # Add text annotation with the percentage of missing values for each feature
    missing_percentage = df.isnull().mean() * 100
    for i, col in enumerate(df.columns):
        plt.text(
            len(df) + 5, i, 
            f"{missing_percentage[col]:.1f}% missing", 
            va='center'
        )
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/missing_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def data_distribution(df):
    """Create histograms for each numeric feature"""
    # Get numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for distribution analysis.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create histograms for each numeric feature
    for col in numeric_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        plt.hist(numeric_df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add labels and title
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add basic statistics
        stats_text = (
            f"Mean: {numeric_df[col].mean():.2f}\n"
            f"Median: {numeric_df[col].median():.2f}\n"
            f"Std Dev: {numeric_df[col].std():.2f}\n"
            f"Min: {numeric_df[col].min():.2f}\n"
            f"Max: {numeric_df[col].max():.2f}"
        )
        plt.figtext(0.95, 0.7, stats_text, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.8), ha='right')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distribution_{col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

def data_summary_table(df):
    """Create a summary table of the dataset"""
    # Get basic statistics
    summary = df.describe(include='all').transpose()
    
    # Add missing values information
    summary['missing'] = df.isnull().sum()
    summary['missing_pct'] = df.isnull().mean() * 100
    
    # Add data types
    summary['dtype'] = df.dtypes
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    summary.to_csv(f"{output_dir}/data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Visualize the summary as a table
    fig, ax = plt.subplots(figsize=(12, max(6, len(df.columns) * 0.4)))
    ax.axis('off')
    ax.axis('tight')
    
    # Format the summary for display
    display_cols = ['count', 'missing', 'missing_pct', 'mean', 'std', 'min', 'max']
    display_summary = summary.copy()
    for col in display_cols:
        if col in display_summary:
            if col == 'missing_pct':
                display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1f}%")
            elif col in ['mean', 'std', 'min', 'max']:
                display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    display_data = display_summary[display_cols].reset_index()
    display_data.columns = ['Feature', 'Count', 'Missing', 'Missing %', 'Mean', 'Std Dev', 'Min', 'Max']
    
    # Create the table
    table = ax.table(cellText=display_data.values, colLabels=display_data.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Set title
    plt.title('Dataset Summary Statistics', fontsize=14, pad=20)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_summary_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def time_series_overview(df):
    """Plot a time series overview of key numeric columns"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for time series overview.")
        return
    
    # Get numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for time series overview.")
        return
    
    # Choose key features (up to 4)
    price_keywords = ['price', 'close', 'value', 'btc', 'usd']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        if any(keyword in col.lower() for keyword in price_keywords) and len(key_features) < 4:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 4 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 4:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot time series
    plt.figure(figsize=(14, 8))
    
    for col in key_features:
        plt.plot(df.index, numeric_df[col], label=col)
    
    plt.title('Time Series Overview')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def run_all_visualizations(df, file_name=""):
    """Run all data quality visualizations on the dataframe"""
    print(f"\nGenerating data quality visualizations for: {file_name}\n")
    
    # Run all visualizations
    missing_values_heatmap(df)
    data_distribution(df)
    data_summary_table(df)
    time_series_overview(df)
    
    print(f"\nVisualizations saved to the 'visualization_output' directory.")

def main():
    """Main function to parse arguments and run visualizations"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            df = load_data(file_path)
            if df is not None:
                run_all_visualizations(df, os.path.basename(file_path))
            else:
                print(f"Failed to load data from {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Usage: python data_quality_viz.py [csv_file_path]")
        print("Example: python data_quality_viz.py ../datasets/processed_exchanges/BTC_USD.csv")

if __name__ == "__main__":
    main() 