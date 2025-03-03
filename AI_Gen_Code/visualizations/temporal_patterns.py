"""
Temporal Patterns Visualizations

This script provides visualizations for analyzing temporal patterns in cryptocurrency time series data,
including seasonality, periodicity, trend analysis, and day-of-week effects.

Usage:
    python temporal_patterns.py [csv_file_path]

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
import calendar

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

def analyze_daily_patterns(df):
    """Analyze patterns in daily data"""
    print("Analyzing daily patterns...")
    
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for daily pattern analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for daily pattern analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'btc', 'usd', 'value']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # For each key feature, analyze day-of-week pattern
    for feature in key_features:
        # Create day of week feature
        feature_data = numeric_df[feature].copy()
        feature_data = pd.DataFrame({feature: feature_data})
        feature_data['day_of_week'] = feature_data.index.dayofweek
        feature_data['day_name'] = feature_data.index.day_name()
        
        # Calculate daily statistics
        daily_stats = feature_data.groupby('day_of_week').agg({
            feature: ['mean', 'median', 'std', 'min', 'max', 'count']
        })
        
        # Order by day of week (Monday=0, Sunday=6)
        daily_stats = daily_stats.reindex(range(7))
        
        # Add day names
        day_names = [calendar.day_name[i] for i in range(7)]
        
        # Plot daily pattern
        plt.figure(figsize=(12, 7))
        
        # Calculate daily average
        daily_mean = daily_stats[(feature, 'mean')]
        daily_std = daily_stats[(feature, 'std')]
        
        # Plot bar chart of daily means
        plt.bar(day_names, daily_mean, yerr=daily_std, alpha=0.7, capsize=10, color='skyblue')
        
        plt.title(f'Average {feature} by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel(f'Average {feature}')
        plt.grid(True, alpha=0.3)
        
        # Add data labels
        for i, (mean, std) in enumerate(zip(daily_mean, daily_std)):
            if not np.isnan(mean):
                plt.text(i, mean + std + (max(daily_mean) * 0.02), 
                         f'{mean:.2f}', ha='center', va='bottom')
        
        # Rotate x-labels for better readability
        plt.xticks(rotation=45)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_pattern_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Also create volume distribution by day of week (if 'volume' is in the column name)
        if 'volume' in feature.lower():
            volume_by_day = feature_data.boxplot(column=feature, by='day_name', 
                                                 figsize=(12, 7), grid=False, 
                                                 return_type='axes')
            
            plt.title(f'{feature} Distribution by Day of Week')
            plt.suptitle('')  # Remove default title
            plt.xlabel('Day of Week')
            plt.ylabel(feature)
            plt.grid(True, alpha=0.3)
            
            # Rotate x-labels for better readability
            plt.xticks(rotation=45)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f"{output_dir}/daily_volume_dist_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.close()
    
    print(f"Daily pattern analysis saved to {output_dir}")

def analyze_monthly_patterns(df):
    """Analyze patterns in monthly data"""
    print("Analyzing monthly patterns...")
    
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for monthly pattern analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for monthly pattern analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'btc', 'usd', 'value']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # For each key feature, analyze monthly pattern
    for feature in key_features:
        # Create month feature
        feature_data = numeric_df[feature].copy()
        feature_data = pd.DataFrame({feature: feature_data})
        feature_data['month'] = feature_data.index.month
        feature_data['month_name'] = feature_data.index.month_name()
        
        # Calculate monthly statistics
        monthly_stats = feature_data.groupby('month').agg({
            feature: ['mean', 'median', 'std', 'min', 'max', 'count']
        })
        
        # Order by month (1=Jan, 12=Dec)
        monthly_stats = monthly_stats.reindex(range(1, 13))
        
        # Add month names
        month_names = [calendar.month_name[i] for i in range(1, 13)]
        
        # Plot monthly pattern
        plt.figure(figsize=(14, 7))
        
        # Calculate monthly average
        monthly_mean = monthly_stats[(feature, 'mean')]
        monthly_std = monthly_stats[(feature, 'std')]
        
        # Plot bar chart of monthly means
        plt.bar(month_names, monthly_mean, yerr=monthly_std, alpha=0.7, capsize=10, color='skyblue')
        
        plt.title(f'Average {feature} by Month')
        plt.xlabel('Month')
        plt.ylabel(f'Average {feature}')
        plt.grid(True, alpha=0.3)
        
        # Add data labels
        for i, (mean, std) in enumerate(zip(monthly_mean, monthly_std)):
            if not np.isnan(mean):
                plt.text(i, mean + std + (max(monthly_mean) * 0.02), 
                         f'{mean:.2f}', ha='center', va='bottom')
        
        # Rotate x-labels for better readability
        plt.xticks(rotation=45)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_pattern_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Also create heatmap of monthly patterns across years
        years = feature_data.index.year.unique()
        
        if len(years) >= 2:  # Only if we have at least 2 years of data
            # Create pivot table of monthly averages by year
            monthly_pivot = pd.pivot_table(
                feature_data,
                values=feature,
                index=feature_data.index.year,
                columns=feature_data.index.month,
                aggfunc='mean'
            )
            
            # Rename columns to month names
            monthly_pivot.columns = [calendar.month_abbr[month] for month in monthly_pivot.columns]
            
            # Plot heatmap
            plt.figure(figsize=(14, 8))
            
            # Use imshow for the heatmap
            im = plt.imshow(monthly_pivot, cmap='coolwarm')
            
            # Add colorbar
            plt.colorbar(im, label=f'Average {feature}')
            
            # Set y-ticks (years)
            plt.yticks(range(len(monthly_pivot.index)), monthly_pivot.index)
            
            # Set x-ticks (months)
            plt.xticks(range(len(monthly_pivot.columns)), monthly_pivot.columns)
            
            # Add value annotations
            for i in range(len(monthly_pivot.index)):
                for j in range(len(monthly_pivot.columns)):
                    value = monthly_pivot.iloc[i, j]
                    if not np.isnan(value):
                        plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                color='black' if abs(value) < monthly_pivot.max().max() * 0.7 else 'white',
                                fontsize=8)
            
            plt.title(f'Monthly {feature} Heatmap by Year')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/monthly_heatmap_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.close()
    
    print(f"Monthly pattern analysis saved to {output_dir}")

def analyze_yearly_trends(df):
    """Analyze yearly trends in the data"""
    print("Analyzing yearly trends...")
    
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for yearly trend analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for yearly trend analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'btc', 'usd', 'value']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # For each key feature, analyze yearly trend
    for feature in key_features:
        # Calculate yearly statistics
        yearly_stats = numeric_df.groupby(numeric_df.index.year)[feature].agg(['mean', 'median', 'std', 'min', 'max'])
        
        # Plot yearly trend
        plt.figure(figsize=(14, 7))
        
        # Calculate yearly statistics
        yearly_mean = yearly_stats['mean']
        yearly_std = yearly_stats['std']
        yearly_min = yearly_stats['min']
        yearly_max = yearly_stats['max']
        
        # Plot line chart with error bars
        plt.errorbar(yearly_mean.index, yearly_mean, yerr=yearly_std, 
                    fmt='o-', capsize=10, alpha=0.7, linewidth=2, color='blue',
                    label='Mean ± Std Dev')
        
        # Add min-max range as a filled area
        plt.fill_between(yearly_mean.index, yearly_min, yearly_max, alpha=0.2, color='blue',
                        label='Min-Max Range')
        
        plt.title(f'Yearly Trend of {feature}')
        plt.xlabel('Year')
        plt.ylabel(feature)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add data labels for the means
        for i, (year, mean) in enumerate(yearly_mean.items()):
            plt.annotate(f'{mean:.2f}', 
                        (year, mean), 
                        textcoords="offset points",
                        xytext=(0, 10), 
                        ha='center')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/yearly_trend_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Also create a year-over-year comparison (overlay months)
        years = sorted(numeric_df.index.year.unique())
        
        if len(years) >= 2:  # Only if we have at least 2 years of data
            plt.figure(figsize=(14, 8))
            
            # Define a colormap for different years
            colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
            
            for i, year in enumerate(years):
                # Extract data for this year
                year_data = numeric_df[numeric_df.index.year == year]
                
                if len(year_data) > 0:
                    # Group by month and calculate mean
                    monthly_data = year_data[feature].groupby(year_data.index.month).mean()
                    
                    # Plot line for this year
                    plt.plot(monthly_data.index, monthly_data, 'o-', 
                            linewidth=2, markersize=6, color=colors[i], 
                            label=str(year))
            
            plt.title(f'Year-over-Year Comparison of Monthly {feature}')
            plt.xlabel('Month')
            plt.ylabel(f'Average {feature}')
            plt.xticks(range(1, 13), [calendar.month_abbr[i] for i in range(1, 13)])
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f"{output_dir}/year_over_year_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.close()
    
    print(f"Yearly trend analysis saved to {output_dir}")

def analyze_volatility_patterns(df):
    """Analyze volatility patterns in the data"""
    print("Analyzing volatility patterns...")
    
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for volatility pattern analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for volatility pattern analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'btc', 'usd', 'value']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # For each key feature, analyze volatility patterns
    for feature in key_features:
        # Calculate returns if we have at least 2 data points
        if len(numeric_df[feature]) >= 2:
            # Calculate percentage change
            returns = numeric_df[feature].pct_change().dropna()
            
            # Calculate rolling volatility (standard deviation of returns)
            window_sizes = [7, 30, 90]  # 1 week, 1 month, 3 months
            
            # Create DataFrame for volatility
            volatility_df = pd.DataFrame(index=returns.index)
            
            for window in window_sizes:
                if len(returns) >= window:
                    volatility_df[f'{window}-day'] = returns.rolling(window=window).std()
            
            # Plot volatility over time
            plt.figure(figsize=(14, 7))
            
            for column in volatility_df.columns:
                plt.plot(volatility_df.index, volatility_df[column], linewidth=2, alpha=0.7, label=column)
            
            plt.title(f'Rolling Volatility of {feature}')
            plt.xlabel('Date')
            plt.ylabel('Volatility (Standard Deviation of Returns)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f"{output_dir}/volatility_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.close()
            
            # Also analyze volatility by month of year
            if len(volatility_df) > 0:
                # Create month feature
                volatility_df['month'] = volatility_df.index.month
                
                # Calculate monthly volatility statistics
                monthly_vol = {}
                
                for window in window_sizes:
                    col_name = f'{window}-day'
                    if col_name in volatility_df.columns:
                        monthly_vol[col_name] = volatility_df.groupby('month')[col_name].mean()
                
                if monthly_vol:  # If we have data
                    plt.figure(figsize=(14, 7))
                    
                    # Plot monthly volatility for each window
                    for window_name, monthly_data in monthly_vol.items():
                        plt.plot(monthly_data.index, monthly_data, 'o-', 
                                linewidth=2, markersize=6, label=window_name)
                    
                    plt.title(f'Average Monthly Volatility of {feature}')
                    plt.xlabel('Month')
                    plt.ylabel('Average Volatility')
                    plt.xticks(range(1, 13), [calendar.month_abbr[i] for i in range(1, 13)])
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save the plot
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/monthly_volatility_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.close()
            
            # Analyze volatility by day of week
            if len(volatility_df) > 0:
                # Create day of week feature
                volatility_df['day_of_week'] = volatility_df.index.dayofweek
                
                # Calculate daily volatility statistics
                daily_vol = {}
                
                for window in window_sizes:
                    col_name = f'{window}-day'
                    if col_name in volatility_df.columns:
                        daily_vol[col_name] = volatility_df.groupby('day_of_week')[col_name].mean()
                
                if daily_vol:  # If we have data
                    plt.figure(figsize=(14, 7))
                    
                    # Plot daily volatility for each window
                    for window_name, daily_data in daily_vol.items():
                        plt.plot(daily_data.index, daily_data, 'o-', 
                                linewidth=2, markersize=6, label=window_name)
                    
                    plt.title(f'Average Daily Volatility of {feature}')
                    plt.xlabel('Day of Week')
                    plt.ylabel('Average Volatility')
                    plt.xticks(range(7), [calendar.day_name[i] for i in range(7)])
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save the plot
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/daily_volatility_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.close()
        else:
            print(f"Not enough data to calculate returns for {feature}.")
    
    print(f"Volatility pattern analysis saved to {output_dir}")

def manual_periodogram(series, periods=None):
    """Calculate a simple periodogram to detect periodicities in time series"""
    # If no periods provided, create default period range
    if periods is None:
        periods = np.arange(2, min(len(series) // 2, 366))  # Up to 1 year for daily data or half the data
    
    # Calculate periodogram
    periodogram = np.zeros(len(periods))
    
    for i, period in enumerate(periods):
        # Create sine and cosine waves of the current period
        t = np.arange(len(series))
        sin_wave = np.sin(2 * np.pi * t / period)
        cos_wave = np.cos(2 * np.pi * t / period)
        
        # Calculate correlation with sine and cosine waves
        sin_corr = np.corrcoef(series, sin_wave)[0, 1]
        cos_corr = np.corrcoef(series, cos_wave)[0, 1]
        
        # Periodogram value is the sum of squared correlations
        periodogram[i] = sin_corr**2 + cos_corr**2
    
    return periods, periodogram

def analyze_periodicities(df):
    """Analyze periodicities in the data"""
    print("Analyzing periodicities...")
    
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for periodicity analysis.")
        return
    
    # Check if data is regularly spaced (daily)
    date_diffs = df.index[1:] - df.index[:-1]
    unique_diffs = pd.unique(date_diffs)
    
    if len(unique_diffs) > 1:
        print("Warning: Data is not regularly spaced. Periodicity analysis may not be accurate.")
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for periodicity analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'btc', 'usd', 'value']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Common periods to highlight in days
    common_periods = {
        7: 'Weekly',
        14: 'Bi-weekly',
        30: 'Monthly',
        91: 'Quarterly',
        182: 'Half-yearly',
        365: 'Yearly'
    }
    
    # For each key feature, analyze periodicities
    for feature in key_features:
        # Check if we have enough data
        if len(numeric_df[feature]) < 60:  # Need at least 60 data points for meaningful analysis
            print(f"Not enough data for periodicity analysis of {feature}. Need at least 60 data points.")
            continue
        
        # Prepare data: detrend and normalize
        data = numeric_df[feature].dropna()
        
        # Simple detrending using first-order differencing
        diff_data = data.diff().dropna()
        
        # Calculate periodogram
        periods, power = manual_periodogram(diff_data.values)
        
        # Plot periodogram
        plt.figure(figsize=(14, 7))
        plt.plot(periods, power, linewidth=2, alpha=0.7)
        
        # Add markers for common periods if they're in our period range
        for period, label in common_periods.items():
            if period in periods:
                period_idx = np.where(periods == period)[0][0]
                plt.axvline(x=period, color='red', linestyle='--', alpha=0.5)
                plt.text(period, 0, label, rotation=90, verticalalignment='bottom')
        
        # Add annotations for the top 3 periods
        top_periods_idx = np.argsort(power)[-3:][::-1]  # Indices of top 3 periods
        
        for idx in top_periods_idx:
            period = periods[idx]
            plt.annotate(f'Period: {period:.1f} days',
                        xy=(period, power[idx]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.title(f'Periodogram of {feature}')
        plt.xlabel('Period (days)')
        plt.ylabel('Power')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/periodogram_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
    
    print(f"Periodicity analysis saved to {output_dir}")

def run_all_temporal_analyses(df, file_name=""):
    """Run all temporal pattern analyses on the dataframe"""
    print(f"\nAnalyzing temporal patterns for: {file_name}\n")
    
    # Run all analyses
    analyze_daily_patterns(df)
    analyze_monthly_patterns(df)
    analyze_yearly_trends(df)
    analyze_volatility_patterns(df)
    analyze_periodicities(df)
    
    print(f"\nTemporal pattern analyses saved to the 'visualization_output' directory.")

def main():
    """Main function to parse arguments and run analyses"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            df = load_data(file_path)
            if df is not None:
                run_all_temporal_analyses(df, os.path.basename(file_path))
            else:
                print(f"Failed to load data from {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Usage: python temporal_patterns.py [csv_file_path]")
        print("Example: python temporal_patterns.py ../datasets/processed_exchanges/BTC_USD.csv")

if __name__ == "__main__":
    main() 