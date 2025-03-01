"""
Time Series Structure Analysis

This script provides visualizations for analyzing the structure of time series data,
including completeness checks, gap analysis, and sampling frequency analysis.

Usage:
    python time_series_analysis.py [csv_file_path]

Author: AI Assistant
Created: March 1, 2025
Modified: Current date - Simplified to use only pandas and matplotlib
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

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

def check_date_completeness(df):
    """Plot the time series with focus on gaps and discontinuities"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for date completeness check.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get date range information
    min_date = df.index.min()
    max_date = df.index.max()
    date_range_days = (max_date - min_date).days
    
    # Choose an appropriate unit based on the range
    if date_range_days <= 60:  # Less than 2 months
        unit = 'days'
        ideal_freq = 'D'
    elif date_range_days <= 365 * 2:  # Less than 2 years
        unit = 'weeks'
        ideal_freq = 'W'
    else:
        unit = 'months'
        ideal_freq = 'M'
    
    # Create an ideal date range
    ideal_dates = pd.date_range(start=min_date, end=max_date, freq=ideal_freq)
    
    # Check which dates from the ideal range exist in the actual data
    exists = pd.Series(False, index=ideal_dates)
    for date in df.index:
        closest_ideal = ideal_dates[ideal_dates.get_indexer([date], method='nearest')[0]]
        exists[closest_ideal] = True
    
    # Create a mask to highlight missing dates
    has_data = exists.astype(int)
    missing_dates = (~exists).sum()
    
    # Plot the date completeness
    plt.figure(figsize=(14, 6))
    plt.plot(ideal_dates, has_data, marker='o', linestyle='-', markersize=4)
    plt.title(f'Date Completeness ({missing_dates} missing {unit} out of {len(ideal_dates)})')
    plt.ylabel('Has Data (1 = Yes, 0 = No)')
    plt.yticks([0, 1])
    plt.grid(True, alpha=0.3)
    
    # Format x-axis based on date range
    if unit == 'days':
        plt.xlabel('Day')
    elif unit == 'weeks':
        plt.xlabel('Week')
    else:
        plt.xlabel('Month')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/date_completeness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Return the percent completeness
    return (1 - missing_dates / len(ideal_dates)) * 100

def analyze_gaps(df):
    """Analyze gaps in the time series data"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for gap analysis.")
        return
    
    # Sort the index to ensure chronological order
    df = df.sort_index()
    
    # Calculate time differences between consecutive dates
    time_diffs = df.index.to_series().diff()
    
    # Create a gap analysis dataframe
    gap_analysis = pd.DataFrame({
        'time_diff_days': time_diffs.dt.total_seconds() / (24 * 3600)
    })
    
    # Identify gaps (where time difference is greater than the median difference)
    median_diff = gap_analysis['time_diff_days'].median()
    gap_analysis['is_gap'] = gap_analysis['time_diff_days'] > 2 * median_diff
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histogram of time differences
    plt.figure(figsize=(12, 6))
    plt.hist(gap_analysis['time_diff_days'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=median_diff, color='red', linestyle='--', label=f'Median: {median_diff:.2f} days')
    plt.title('Distribution of Time Gaps Between Consecutive Records')
    plt.xlabel('Gap Length (days)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gap_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Create a table of significant gaps
    significant_gaps = gap_analysis[gap_analysis['is_gap']].copy()
    if len(significant_gaps) > 0:
        # Add the date information
        significant_gaps['start_date'] = df.index[:-1][significant_gaps.index - 1]
        significant_gaps['end_date'] = df.index[1:][significant_gaps.index - 1]
        
        # Sort by gap size
        significant_gaps = significant_gaps.sort_values('time_diff_days', ascending=False)
        
        # Plot the top gaps
        top_n = min(10, len(significant_gaps))
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(top_n), significant_gaps['time_diff_days'].iloc[:top_n], color='salmon')
        plt.title(f'Top {top_n} Largest Gaps in the Dataset')
        plt.xlabel('Gap Rank')
        plt.ylabel('Gap Length (days)')
        plt.xticks(range(top_n))
        
        # Add gap period labels
        for i, (_, row) in enumerate(significant_gaps.iloc[:top_n].iterrows()):
            gap_label = f"{row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}"
            plt.text(i, row['time_diff_days'] * 0.5, gap_label, ha='center', rotation=90, color='black', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Save the gap analysis to CSV
        significant_gaps[['start_date', 'end_date', 'time_diff_days']].to_csv(
            f"{output_dir}/significant_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    else:
        print("No significant gaps found in the dataset.")

def analyze_sampling_frequency(df):
    """Analyze the sampling frequency patterns in the dataset"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for sampling frequency analysis.")
        return
    
    # Sort the index to ensure chronological order
    df = df.sort_index()
    
    # Calculate time differences between consecutive dates in hours
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600
    
    # Create a frequency analysis dataframe
    freq_analysis = pd.DataFrame({
        'time_diff_hours': time_diffs
    })
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histogram of sampling frequencies
    plt.figure(figsize=(12, 6))
    plt.hist(freq_analysis['time_diff_hours'].dropna(), bins=50, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Sampling Intervals')
    plt.xlabel('Interval (hours)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Identify common sampling intervals
    common_intervals = freq_analysis['time_diff_hours'].value_counts().nlargest(5)
    
    # Add text box with common intervals
    intervals_text = "Common intervals (hours):\n"
    for interval, count in common_intervals.items():
        percent = count / len(freq_analysis) * 100
        intervals_text += f"{interval:.1f}: {count} occurrences ({percent:.1f}%)\n"
    
    plt.figtext(0.7, 0.7, intervals_text, fontsize=10, 
              bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sampling_frequency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Plot sampling frequency change over time
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[1:], freq_analysis['time_diff_hours'], marker='.', linestyle='-', alpha=0.5)
    plt.title('Sampling Interval Change Over Time')
    plt.xlabel('Date')
    plt.ylabel('Interval (hours)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sampling_frequency_overtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def check_weekend_data(df):
    """Check if the dataset contains weekend data"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for weekend data check.")
        return
    
    # Create weekday information
    weekdays = df.index.weekday
    is_weekend = (weekdays == 5) | (weekdays == 6)  # 5=Saturday, 6=Sunday
    
    # Count records by day of week
    weekday_counts = pd.Series(weekdays).value_counts().sort_index()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts.index = [days[i] for i in weekday_counts.index]
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot records by day of week
    plt.figure(figsize=(12, 6))
    bars = plt.bar(weekday_counts.index, weekday_counts.values, color='skyblue')
    
    # Highlight weekends
    weekend_indices = [i for i, day in enumerate(weekday_counts.index) if day in ['Saturday', 'Sunday']]
    for idx in weekend_indices:
        if idx < len(bars):
            bars[idx].set_color('salmon')
    
    plt.title('Record Count by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Records')
    
    # Add value labels on bars
    for i, v in enumerate(weekday_counts.values):
        plt.text(i, v + 1, str(v), ha='center')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weekday_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Calculate weekend statistics
    weekend_count = sum(is_weekend)
    weekday_count = len(df) - weekend_count
    weekend_percent = weekend_count / len(df) * 100 if len(df) > 0 else 0
    
    # Create a summary figure
    plt.figure(figsize=(8, 8))
    plt.pie(
        [weekday_count, weekend_count], 
        labels=['Weekdays', 'Weekends'], 
        autopct='%1.1f%%', 
        colors=['skyblue', 'salmon'],
        explode=(0, 0.1)
    )
    plt.title('Weekday vs Weekend Distribution')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weekend_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    return weekend_percent > 5  # Consider the dataset has weekend data if >5% of records are on weekends

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

def run_all_analyses(df, file_name=""):
    """Run all time series structure analyses on the dataframe"""
    print(f"\nAnalyzing time series structure for: {file_name}\n")
    
    # Run all analyses
    completeness = check_date_completeness(df)
    print(f"Date completeness: {completeness:.1f}%")
    
    analyze_gaps(df)
    analyze_sampling_frequency(df)
    
    has_weekend_data = check_weekend_data(df)
    print(f"Dataset has weekend data: {has_weekend_data}")
    
    visualize_coverage(df)
    
    print(f"\nAnalyses saved to the 'visualization_output' directory.")

def main():
    """Main function to parse arguments and run analyses"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            df = load_data(file_path)
            if df is not None:
                run_all_analyses(df, os.path.basename(file_path))
            else:
                print(f"Failed to load data from {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Usage: python time_series_analysis.py [csv_file_path]")
        print("Example: python time_series_analysis.py ../datasets/processed_exchanges/BTC_USD.csv")

if __name__ == "__main__":
    main() 