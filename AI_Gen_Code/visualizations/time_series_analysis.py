"""
Time Series Analysis Visualizations

This script provides visualizations for analyzing time series data structure,
focusing on trends, seasonality, and cyclic patterns.

Usage:
    python time_series_analysis.py [csv_file_path] [--output_dir OUTPUT_DIR]

Author: AI Assistant
Created: March 1, 2025
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import argparse
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import re  # Add import for regex

# Set the seaborn style for better visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

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

def check_date_completeness(df, output_dir):
    """Check for gaps in time series data"""
    print("Checking date completeness in time series data...")
    
    # Ensure DataFrame has Date column
    if 'Date' not in df.columns:
        print("Error: DataFrame must have 'Date' column for time series analysis")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy with Date as index
    df_ts = df.copy()
    if not isinstance(df_ts.index, pd.DatetimeIndex):
        df_ts = df_ts.set_index('Date')
    
    # Get the date range
    start_date = df_ts.index.min()
    end_date = df_ts.index.max()
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Check for missing dates
    missing_dates = date_range.difference(df_ts.index)
    completeness = 100 - (len(missing_dates) / len(date_range) * 100)
    
    # Create a plot with seaborn
    plt.figure(figsize=(14, 8))
    
    # Plot existing data points
    key_col = df_ts.select_dtypes(include=['number']).columns[0]  # Use first numeric column
    sns.lineplot(x=df_ts.index, y=df_ts[key_col], label=f'{key_col}', marker='o', markersize=4)
    
    # Highlight gaps if there are any
    if len(missing_dates) > 0:
        # Create a Series with NaN values for missing dates
        missing_series = pd.Series(index=missing_dates, data=[np.nan] * len(missing_dates))
        
        # Plot the gaps as red points
        if len(missing_dates) < 100:  # Only highlight individual gaps if there aren't too many
            for date in missing_dates:
                plt.axvline(x=date, color='red', alpha=0.2, linestyle='--')
        else:
            # For many gaps, just highlight problematic areas
            plt.fill_between(df_ts.index, df_ts[key_col].min(), df_ts[key_col].max(), 
                            where=df_ts.index.isin(missing_dates), color='red', alpha=0.1)
    
    plt.title(f'Time Series Completeness: {completeness:.2f}% Complete\n({len(missing_dates)} missing dates out of {len(date_range)} total dates)', 
            fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'date_completeness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Date completeness visualization saved to {output_path}")
    
    return completeness, missing_dates

def analyze_gaps(df, output_dir):
    """Analyze gaps in time series data"""
    print("Analyzing gaps in time series data...")
    
    # Ensure DataFrame has Date column
    if 'Date' not in df.columns:
        print("Error: DataFrame must have 'Date' column for gap analysis")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure Date is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Calculate the time difference between consecutive dates
    df['days_diff'] = df['Date'].diff().dt.days
    
    # Create a DataFrame for gap analysis
    gaps = df['days_diff'].dropna()
    
    # Create a histogram of gaps with seaborn
    plt.figure(figsize=(12, 8))
    
    sns.histplot(gaps, kde=True, bins=20, color='skyblue')
    
    plt.title('Distribution of Gaps Between Consecutive Records', fontsize=14)
    plt.xlabel('Gap Size (Days)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add vertical line for most common gap
    most_common_gap = gaps.mode().iloc[0] if not gaps.empty else 0
    plt.axvline(x=most_common_gap, color='red', linestyle='--', 
               label=f'Most Common Gap: {most_common_gap} days')
    
    # Add statistics as text
    stats_text = (
        f"Mean Gap: {gaps.mean():.2f} days\n"
        f"Median Gap: {gaps.median():.2f} days\n"
        f"Max Gap: {gaps.max():.0f} days\n"
        f"Min Gap: {gaps.min():.0f} days\n"
        f"Std Dev: {gaps.std():.2f} days"
    )
    plt.figtext(0.75, 0.75, stats_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'gap_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gap distribution saved to {output_path}")
    
    # Also create a time series plot of gaps over time
    plt.figure(figsize=(14, 6))
    
    sns.scatterplot(x=df['Date'][1:], y=df['days_diff'], hue=df['days_diff'] > most_common_gap, 
                   palette={False: 'blue', True: 'red'}, legend=False)
    
    plt.title('Gaps Between Records Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Gap Size (Days)', fontsize=12)
    plt.axhline(y=most_common_gap, color='green', linestyle='--', 
               label=f'Most Common Gap: {most_common_gap} days')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'gaps_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gaps over time visualization saved to {output_path}")
    
    return gaps

def analyze_sampling_frequency(df, output_dir):
    """Analyze sampling frequency patterns"""
    print("Analyzing sampling frequency patterns...")
    
    # Ensure DataFrame has Date column
    if 'Date' not in df.columns:
        print("Error: DataFrame must have 'Date' column for sampling frequency analysis")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure Date is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Extract time components
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.dayofweek
    df['hour'] = df['Date'].dt.hour
    
    # Create a heatmap of samples by month and year using seaborn
    # Aggregate samples by month and year
    samples_by_month_year = df.groupby(['year', 'month']).size().unstack()
    
    # Create a heatmap
    plt.figure(figsize=(14, 8))
    
    ax = sns.heatmap(samples_by_month_year, cmap='YlGnBu', annot=True, fmt='g', 
                     cbar_kws={'label': 'Number of Records'})
    
    plt.title('Number of Records by Month and Year', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'sampling_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sampling frequency heatmap saved to {output_path}")
    
    # Create a line plot of records per month over time
    plt.figure(figsize=(14, 6))
    
    # Extract year-month as a datetime
    monthly_counts = df.groupby(pd.Grouper(key='Date', freq='M')).size()
    
    # Plot with seaborn
    sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o')
    
    plt.title('Records per Month Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Records', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'records_per_month.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Records per month visualization saved to {output_path}")
    
    return monthly_counts

def check_weekend_data(df, output_dir):
    """Check if dataset contains weekend data"""
    print("Checking for weekend data...")
    
    # Ensure DataFrame has Date column
    if 'Date' not in df.columns:
        print("Error: DataFrame must have 'Date' column for weekend analysis")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure Date is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Extract weekday
    df['weekday'] = df['Date'].dt.dayofweek
    df['weekday_name'] = df['Date'].dt.day_name()
    df['is_weekend'] = df['weekday'].isin([5, 6])  # 5=Saturday, 6=Sunday
    
    # Count records by day of week
    weekday_counts = df.groupby('weekday_name').size()
    
    # Sort by day of week (Monday first)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = weekday_counts.reindex(days_order)
    
    # Create a bar plot of records by day of week using seaborn
    plt.figure(figsize=(12, 6))
    
    weekday_plot = sns.barplot(x=weekday_counts.index, y=weekday_counts.values, 
                              palette=['skyblue' if day not in ['Saturday', 'Sunday'] else 'lightcoral' 
                                       for day in weekday_counts.index])
    
    # Add count labels on top of bars
    for i, p in enumerate(weekday_plot.patches):
        weekday_plot.annotate(f"{weekday_counts.values[i]}", 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', fontsize=10)
    
    plt.title('Number of Records by Day of Week', fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Number of Records', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'records_by_weekday.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Records by weekday visualization saved to {output_path}")
    
    # Calculate weekend statistics
    weekend_records = df['is_weekend'].sum()
    total_records = len(df)
    weekend_percent = (weekend_records / total_records) * 100 if total_records > 0 else 0
    
    # Create a pie chart of weekend vs. weekday records
    plt.figure(figsize=(10, 8))
    
    # Use seaborn colors
    colors = sns.color_palette('pastel')[0:2]
    
    plt.pie([total_records - weekend_records, weekend_records], 
           labels=['Weekday', 'Weekend'], 
           autopct='%1.1f%%',
           colors=colors,
           startangle=90,
           explode=(0, 0.1))
    
    plt.title('Weekend vs. Weekday Records', fontsize=14)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the figure
    output_path = os.path.join(output_dir, 'weekend_vs_weekday.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Weekend vs. weekday visualization saved to {output_path}")
    
    return weekend_percent

def visualize_coverage(df):
    """Visualize data coverage over different time periods"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for coverage visualization.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create month-year labels
    df_coverage = df.copy()
    df_coverage['year'] = df_coverage.index.year
    df_coverage['month'] = df_coverage.index.month
    df_coverage['day'] = df_coverage.index.day
    
    # Plotting by month-year
    month_year_counts = df_coverage.groupby(['year', 'month']).size()
    if len(month_year_counts) > 0:
        # Create a pivot table for the heatmap
        pivot_data = month_year_counts.reset_index()
        pivot_data.columns = ['Year', 'Month', 'Count']
        pivot_table = pivot_data.pivot(index='Month', columns='Year', values='Count')
        
        # Fill NaN with zeros
        pivot_table = pivot_table.fillna(0)
        
        # Create a figure for the heatmap
        plt.figure(figsize=(12, 8))
        
        # Create the heatmap manually with matplotlib
        im = plt.imshow(pivot_table.values, cmap='YlGnBu')
        
        # Add labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.yticks(range(len(pivot_table.index)), [month_labels[i-1] for i in pivot_table.index])
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        
        # Add a colorbar
        plt.colorbar(im, label='Record Count')
        
        # Add title and labels
        plt.title('Data Coverage by Month-Year')
        plt.xlabel('Year')
        plt.ylabel('Month')
        
        # Add text annotations with the counts
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                count = pivot_table.values[i, j]
                if count > 0:
                    text_color = 'black' if count < pivot_table.values.max() * 0.7 else 'white'
                    plt.text(j, i, f"{int(count)}", ha='center', va='center', color=text_color)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
    
    # Create a figure for the year coverage
    year_counts = df_coverage.groupby('year').size()
    if len(year_counts) > 0:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(year_counts.index, year_counts.values, color='skyblue')
        plt.title('Record Count by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Records')
        
        # Add value labels on bars
        for i, (year, count) in enumerate(year_counts.items()):
            plt.text(year, count + max(year_counts) * 0.01, str(count), ha='center')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/yearly_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

def run_all_analyses(df, output_dir, file_name=""):
    """Run all time series structure analyses on the dataframe"""
    print(f"\nAnalyzing time series structure for: {file_name}\n")
    
    # Run all analyses
    completeness, missing_dates = check_date_completeness(df, output_dir)
    print(f"Date completeness: {completeness:.1f}%")
    
    gaps = analyze_gaps(df, output_dir)
    analyze_sampling_frequency(df, output_dir)
    
    has_weekend_data = check_weekend_data(df, output_dir)
    print(f"Dataset has weekend data: {has_weekend_data:.1f}%")
    
    visualize_coverage(df)
    
    print(f"\nAnalyses saved to the '{output_dir}' directory.")

def run_all_visualizations(df, output_dir, file_name=""):
    """Run all time series analysis visualizations on the dataframe"""
    print(f"\nGenerating time series analysis visualizations for: {file_name}\n")
    print(f"Output directory: {output_dir}")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found for analysis")
        return
    
    # Run all visualizations (pass the output directory)
    visualize_time_series(df, output_dir)
    visualize_seasonal_decomposition(df, output_dir)
    visualize_autocorrelation(df, output_dir)
    visualize_seasonal_patterns(df, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    # Create a simple HTML index file
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
        <title>Time Series Analysis - {file_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .image-container {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #777; }}
        </style>
    </head>
    <body>
        <h1>Time Series Analysis Visualizations</h1>
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
            Generated by Time Series Analysis Tool
        </div>
    </body>
    </html>
    """
    
    # Write to file
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization index created at: {index_path}")

def visualize_time_series(df, output_dir):
    """Visualize the time series data"""
    print("Generating time series visualization...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Only plot numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found for time series visualization")
        return
    
    # Limit to first 5 columns if there are too many
    if len(numeric_df.columns) > 5:
        print(f"Too many columns ({len(numeric_df.columns)}), limiting to first 5...")
        plot_cols = numeric_df.columns[:5]
    else:
        plot_cols = numeric_df.columns
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot each column
    for col in plot_cols:
        plt.plot(numeric_df.index, numeric_df[col], label=col)
    
    plt.title('Time Series Visualization', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Add data source information
    plt.figtext(0.5, 0.01, f"Source: {os.path.basename(df.name) if hasattr(df, 'name') else 'Unknown'}", 
                ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save the figure
    output_path = os.path.join(output_dir, 'time_series_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series visualization saved to {output_path}")

def sanitize_filename(name):
    """Convert column name to a safe filename by replacing invalid characters"""
    # Replace special characters that are invalid in filenames
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return safe_name

def visualize_seasonal_decomposition(df, output_dir):
    """Visualize the seasonal decomposition of time series"""
    print("Generating seasonal decomposition visualizations...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the dataframe has a date column
    if 'Date' not in df.columns:
        print("Warning: No 'Date' column found. Skipping seasonal decomposition.")
        return
    
    # Set Date as index if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Filter for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        try:
            # Get a complete time series without missing values
            series = df[col].dropna()
            
            # Skip if too few data points
            if len(series) < 14:
                print(f"  Skipping {col}: insufficient data points")
                continue
                
            # Check if the series has a uniform frequency
            if not pd.infer_freq(series.index):
                print(f"  Skipping {col}: no clear frequency detected")
                continue
                
            # Ensure the index is uniform by resampling if needed
            uniform_series = series.asfreq('D')
                
            # Perform seasonal decomposition
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Try to decompose with sensible frequency values
            try:
                # Default to 7 days for daily data (weekly seasonality)
                result = seasonal_decompose(uniform_series, model='additive', period=7)
                
                # Create the plot
                fig, axes = plt.subplots(4, 1, figsize=(14, 16))
                
                # Original data
                axes[0].plot(result.observed)
                axes[0].set_title(f'Original Time Series: {col}', fontsize=14)
                axes[0].grid(True)
                
                # Trend component
                axes[1].plot(result.trend)
                axes[1].set_title('Trend Component', fontsize=14)
                axes[1].grid(True)
                
                # Seasonal component
                axes[2].plot(result.seasonal)
                axes[2].set_title('Seasonal Component', fontsize=14)
                axes[2].grid(True)
                
                # Residual component
                axes[3].plot(result.resid)
                axes[3].set_title('Residual Component', fontsize=14)
                axes[3].grid(True)
                
                plt.tight_layout()
                
                # Sanitize column name for filename
                safe_col_name = sanitize_filename(col)
                
                # Save the figure
                output_path = os.path.join(output_dir, f'seasonal_decomposition_{safe_col_name}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"  Error analyzing {col}: {str(e)}")
                
        except Exception as e:
            print(f"  Error processing {col}: {str(e)}")
    
    print(f"Seasonal decomposition visualizations saved to {output_dir}")

def visualize_autocorrelation(df, output_dir):
    """Visualize autocorrelation and partial autocorrelation of time series"""
    print("Generating autocorrelation visualizations...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the dataframe has a date column
    if 'Date' not in df.columns:
        print("Warning: No 'Date' column found. Skipping autocorrelation.")
        return
    
    # Set Date as index if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Filter for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    for col in numeric_cols:
        try:
            # Get a complete time series without missing values
            series = df[col].dropna()
            
            # Skip if too few data points
            if len(series) < 30:
                print(f"  Skipping {col}: insufficient data points")
                continue
            
            # Create the plot
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot ACF
            plot_acf(series, ax=axes[0], lags=40, alpha=0.05)
            axes[0].set_title(f'Autocorrelation Function (ACF): {col}', fontsize=14)
            axes[0].grid(True)
            
            # Plot PACF
            plot_pacf(series, ax=axes[1], lags=40, alpha=0.05)
            axes[1].set_title(f'Partial Autocorrelation Function (PACF): {col}', fontsize=14)
            axes[1].grid(True)
            
            plt.tight_layout()
            
            # Sanitize column name for filename
            safe_col_name = sanitize_filename(col)
            
            # Save the figure
            output_path = os.path.join(output_dir, f'autocorrelation_{safe_col_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Error analyzing {col}: {str(e)}")
    
    print(f"Autocorrelation visualizations saved to {output_dir}")

def visualize_seasonal_patterns(df, output_dir):
    """Visualize seasonal patterns in time series by different time periods"""
    print("Generating seasonal pattern visualizations...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the dataframe has a date column
    if 'Date' not in df.columns:
        print("Warning: No 'Date' column found. Skipping seasonal patterns.")
        return
    
    # Set Date as index if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Filter for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        try:
            # Get a complete time series without missing values
            ts_df = df[[col]].dropna()
            
            # Skip if too few data points
            if len(ts_df) < 365:  # Need at least a year of data
                print(f"  Skipping {col}: insufficient data points")
                continue
            
            # Create a copy of the data for analysis
            ts_df = ts_df.copy()
            
            # Extract features from the date
            ts_df['Year'] = ts_df.index.year
            ts_df['Month'] = ts_df.index.month
            ts_df['Day'] = ts_df.index.day
            ts_df['DayOfWeek'] = ts_df.index.dayofweek
            ts_df['Quarter'] = ts_df.index.quarter
            
            # Create the plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot by month
            monthly_avg = ts_df.groupby('Month')[col].mean()
            axes[0, 0].plot(monthly_avg.index, monthly_avg.values, marker='o')
            axes[0, 0].set_title(f'Monthly Pattern: {col}', fontsize=14)
            axes[0, 0].set_xticks(range(1, 13))
            axes[0, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            axes[0, 0].grid(True)
            
            # Plot by day of week
            dow_avg = ts_df.groupby('DayOfWeek')[col].mean()
            axes[0, 1].plot(dow_avg.index, dow_avg.values, marker='o')
            axes[0, 1].set_title(f'Day of Week Pattern: {col}', fontsize=14)
            axes[0, 1].set_xticks(range(0, 7))
            axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            axes[0, 1].grid(True)
            
            # Plot by quarter
            quarter_avg = ts_df.groupby('Quarter')[col].mean()
            axes[1, 0].plot(quarter_avg.index, quarter_avg.values, marker='o')
            axes[1, 0].set_title(f'Quarterly Pattern: {col}', fontsize=14)
            axes[1, 0].set_xticks(range(1, 5))
            axes[1, 0].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
            axes[1, 0].grid(True)
            
            # Plot yearly trend
            yearly_avg = ts_df.groupby('Year')[col].mean()
            axes[1, 1].plot(yearly_avg.index, yearly_avg.values, marker='o')
            axes[1, 1].set_title(f'Yearly Trend: {col}', fontsize=14)
            axes[1, 1].set_xticks(yearly_avg.index)
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Sanitize column name for filename
            safe_col_name = sanitize_filename(col)
            
            # Save the figure
            output_path = os.path.join(output_dir, f'seasonal_patterns_{safe_col_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Error analyzing {col}: {str(e)}")
    
    print(f"Seasonal pattern visualizations saved to {output_dir}")

def main():
    """Main function to parse arguments and run visualizations"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate time series analysis visualizations')
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