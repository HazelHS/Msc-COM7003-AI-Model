"""
Data Quality Visualizations

This script provides visualizations for assessing data quality in cryptocurrency datasets,
including missing values, data distributions, and basic statistics.

Usage:
    python data_quality_viz.py [csv_file_path] [--output_dir OUTPUT_DIR]

Author: AI Assistant
Created: March 1, 2025
Modified: Current date - Enhanced with seaborn for better visualizations
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import argparse

# Set the seaborn style for better visualizations
sns.set(style="whitegrid")

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

def missing_values_heatmap(df, output_dir):
    """Create a heatmap of missing values using seaborn"""
    print("Generating missing values heatmap...")
    
    # Create a boolean mask for missing values
    missing_data = df.isna()
    
    # Calculate percentage of missing values for each column
    missing_percent = missing_data.mean().round(4) * 100
    
    plt.figure(figsize=(10, 8))
    # Create a heatmap of missing values
    ax = sns.heatmap(missing_data, 
                 cmap='Blues', 
                 cbar_kws={'label': 'Missing Values'},
                 yticklabels=False)
    
    # Add percentage annotations on top of the heatmap
    for i, col in enumerate(df.columns):
        plt.text(i + 0.5, -0.25, f"{missing_percent[col]:.1f}%", 
                 ha='center', fontsize=9, color='black')
    
    plt.title('Missing Values Heatmap', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Records', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'missing_values_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Missing values heatmap saved to {output_path}")
    
    # Additionally, create a bar plot of missing values percentage
    plt.figure(figsize=(12, 6))
    missing_bar = sns.barplot(x=missing_percent.index, y=missing_percent.values)
    
    # Add percentage labels on top of bars
    for i, p in enumerate(missing_bar.patches):
        missing_bar.annotate(f"{missing_percent.values[i]:.1f}%", 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', fontsize=9)
    
    plt.title('Percentage of Missing Values by Feature', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Missing Values (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'missing_values_percentage.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Missing values percentage plot saved to {output_path}")

def data_distribution(df, output_dir):
    """Create distribution plots for each numeric feature"""
    print("Generating data distribution visualizations...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create distributions for each numeric column
    for i, col in enumerate(numeric_cols):
        plt.figure(figsize=(12, 6))
        
        # Create a subplot with 1 row and 2 columns
        grid = plt.GridSpec(1, 2, width_ratios=[3, 1])
        
        # Plot 1: Distribution with KDE
        ax0 = plt.subplot(grid[0])
        sns.histplot(df[col].dropna(), kde=True, ax=ax0)
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Plot 2: Box plot
        ax1 = plt.subplot(grid[1])
        sns.boxplot(y=df[col].dropna(), ax=ax1)
        plt.title('Box Plot', fontsize=14)
        plt.ylabel(col, fontsize=12)
        
        # Add statistics text
        stats = df[col].describe()
        stats_text = (f"Mean: {stats['mean']:.2f}\n"
                     f"Median: {stats['50%']:.2f}\n"
                     f"Std Dev: {stats['std']:.2f}\n"
                     f"Min: {stats['min']:.2f}\n"
                     f"Max: {stats['max']:.2f}")
        
        plt.figtext(0.92, 0.5, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'distribution_{col}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Distribution visualizations saved to {output_dir}")

def data_summary_table(df, output_dir=None):
    """Create a summary table of data statistics"""
    print("Generating data summary table...")
    
    # Calculate basic statistics
    summary = df.describe().T
    
    # Add additional statistics
    summary['missing'] = df.isnull().sum()
    summary['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
    summary['unique'] = df.nunique()
    
    # Print summary table
    print("\nData Summary:")
    pd.set_option('display.max_columns', None)
    print(summary)
    
    # Save to CSV if output_dir is provided
    if output_dir:
        summary_path = os.path.join(output_dir, 'data_summary.csv')
        summary.to_csv(summary_path)
        print(f"Summary table saved to {summary_path}")

def time_series_overview(df, output_dir):
    """Plot time series overview of key numeric columns"""
    print("Generating time series overview...")
    
    # Ensure DataFrame has Date as index
    if 'Date' in df.columns:
        df_ts = df.set_index('Date')
    else:
        df_ts = df.copy()
    
    # Get numeric columns (exclude Date if it exists)
    numeric_cols = df_ts.select_dtypes(include=['number']).columns.tolist()
    
    # Limit to top 4 most important columns if there are too many
    if len(numeric_cols) > 4:
        numeric_cols = numeric_cols[:4]
    
    # Create a time series plot with seaborn
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(numeric_cols):
        plt.subplot(len(numeric_cols), 1, i+1)
        sns.lineplot(data=df_ts, x=df_ts.index, y=col, linewidth=1.5)
        plt.title(f'Time Series: {col}', fontsize=12)
        plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'time_series_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series overview saved to {output_path}")
    
    # Create a pairplot for correlations between numeric features
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        correlation = df_ts[numeric_cols].corr()
        
        # Create a heatmap of correlations
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f",
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation heatmap saved to {output_path}")

def run_all_visualizations(df, output_dir, file_name=""):
    """Run all data quality visualizations on the dataframe"""
    print(f"\nGenerating data quality visualizations for: {file_name}\n")
    print(f"Output directory: {output_dir}")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Run all visualizations
    missing_values_heatmap(df, output_dir)
    data_distribution(df, output_dir)
    data_summary_table(df, output_dir)
    time_series_overview(df, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    # Create a simple HTML index file to view all visualizations
    create_visualization_index(output_dir, file_name)

def create_visualization_index(output_dir, file_name):
    """Create a simple HTML index file to view all visualizations"""
    print("Creating visualization index...")
    
    # Get all image files
    image_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg'))]
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Quality Visualizations - {file_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .image-container {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #777; }}
        </style>
    </head>
    <body>
        <h1>Data Quality Visualizations</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for {file_name}</p>
    """
    
    # Add each image
    for img_file in sorted(image_files):
        img_title = ' '.join(img_file.replace('.png', '').replace('.jpg', '').replace('_', ' ').title().split())
        html_content += f"""
        <div class="image-container">
            <h2>{img_title}</h2>
            <img src="{img_file}" alt="{img_title}">
        </div>
        """
    
    # Close HTML
    html_content += """
        <div class="footer">
            Generated by Data Quality Visualization Tool
        </div>
    </body>
    </html>
    """
    
    # Write to file
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization index created at: {index_path}")

def main():
    """Main function to parse arguments and run visualizations"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate data quality visualizations for a CSV file')
    parser.add_argument('file_path', help='Path to the CSV file')
    parser.add_argument('--output_dir', default='visualization_output', 
                        help='Directory to save visualizations (default: visualization_output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if os.path.exists(args.file_path):
        print(f"Loading data from: {args.file_path}")
        df = load_data(args.file_path)
        if df is not None:
            print(f"Data loaded successfully. Shape: {df.shape}")
            run_all_visualizations(df, args.output_dir, os.path.basename(args.file_path))
        else:
            print(f"Failed to load data from {args.file_path}")
    else:
        print(f"File not found: {args.file_path}")

if __name__ == "__main__":
    main() 