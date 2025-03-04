"""
Stationarity Analysis Visualizations

This script provides visualizations for analyzing the stationarity of cryptocurrency time series data,
including rolling statistics, time series transformations, and autocorrelation analysis.

Usage:
    python stationarity_analysis.py [csv_file_path] [--output_dir OUTPUT_DIR]

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
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import re  # Add import for regex

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

def visualize_rolling_statistics(df, column, window=30, output_dir='visualization_output'):
    """Visualize rolling mean and standard deviation to check for stationarity"""
    print(f"Visualizing rolling statistics for {column}...")
    
    # Create a copy of the data
    data = df.copy()
    
    # Ensure the data has a datetime index
    if 'Date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index('Date')
    
    # Calculate rolling statistics
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot using seaborn
    plt.figure(figsize=(14, 10))
    
    # Create a 2x1 subplot grid
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Plot original series and rolling mean
    ax1 = plt.subplot(gs[0])
    sns.lineplot(x=data.index, y=data[column], label='Original', alpha=0.7, ax=ax1)
    sns.lineplot(x=data.index, y=rolling_mean, label=f'Rolling Mean (window={window})', ax=ax1)
    
    plt.title(f'Rolling Statistics Analysis of {column}', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    
    # Plot rolling standard deviation
    ax2 = plt.subplot(gs[1])
    sns.lineplot(x=data.index, y=rolling_std, color='red', label=f'Rolling Std Dev (window={window})', ax=ax2)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'rolling_stats_{column}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Rolling statistics visualization saved to {output_path}")
    
    # Check if mean and std dev change significantly over time
    mean_change = (rolling_mean.iloc[-1] - rolling_mean.iloc[window]) / rolling_mean.iloc[window] * 100
    std_change = (rolling_std.iloc[-1] - rolling_std.iloc[window]) / rolling_std.iloc[window] * 100
    
    # Interpretation
    is_stationary = abs(mean_change) < 10 and abs(std_change) < 10
    
    # Create a summary plot with histograms of different periods
    plt.figure(figsize=(14, 10))
    
    # Split the series into 3 segments
    n_segments = 3
    segment_size = len(data) // n_segments
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(data)
        segment = data[column].iloc[start_idx:end_idx]
        
        plt.subplot(n_segments, 1, i+1)
        sns.histplot(segment, kde=True, bins=30, 
                    label=f'Period {i+1}: {data.index[start_idx].strftime("%Y-%m-%d")} to {data.index[end_idx-1].strftime("%Y-%m-%d")}')
        
        plt.title(f'Distribution in Period {i+1}', fontsize=12)
        plt.xlabel(column, fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'distribution_periods_{column}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Period distribution comparison saved to {output_path}")
    
    return is_stationary, mean_change, std_change

def visualize_decomposition(df, column, output_dir='visualization_output'):
    """Visualize the decomposition of time series into trend, seasonal, and residual components"""
    print(f"Visualizing time series decomposition for {column}...")
    
    # Create a copy of the data
    data = df.copy()
    
    # Ensure the data has a datetime index
    if 'Date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index('Date')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine frequency based on data
    if len(data) < 100:  # For short time series
        window = min(7, len(data) // 4)
    else:
        window = 30  # Default for longer series
    
    # Calculate trend using rolling mean
    trend = data[column].rolling(window=window, center=True).mean()
    
    # Calculate seasonal decomposition using simple differencing from trend
    seasonal = data[column] - trend
    
    # Create a figure with subplots using seaborn styling
    plt.figure(figsize=(14, 12))
    
    # Original data
    plt.subplot(411)
    sns.lineplot(x=data.index, y=data[column], color='blue')
    plt.title(f'Original Time Series: {column}', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.tick_params(labelbottom=False)
    
    # Trend component
    plt.subplot(412)
    sns.lineplot(x=data.index, y=trend, color='red')
    plt.title(f'Trend Component (Rolling Window = {window})', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.tick_params(labelbottom=False)
    
    # Seasonal component
    plt.subplot(413)
    sns.lineplot(x=data.index, y=seasonal, color='green')
    plt.title('Seasonal & Residual Component', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.tick_params(labelbottom=False)
    
    # Create a simple autocorrelation plot for the seasonal component
    acf_values = [1.0]  # Start with 1 at lag 0
    max_lag = min(50, len(seasonal) // 4)
    
    for lag in range(1, max_lag + 1):
        # Calculate autocorrelation manually
        series = seasonal.dropna()
        if len(series) <= lag:
            continue
            
        mean = np.mean(series)
        variance = np.var(series)
        
        if variance == 0:
            acf_values.append(0)
            continue
            
        # Calculate autocorrelation
        acf = np.mean(((series[:-lag] - mean) * (series[lag:] - mean))) / variance
        acf_values.append(acf)
    
    # Plot ACF
    plt.subplot(414)
    sns.barplot(x=list(range(len(acf_values))), y=acf_values, color='purple')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add confidence intervals (approximately 95%)
    conf_level = 1.96 / np.sqrt(len(series))
    plt.axhline(y=conf_level, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=-conf_level, color='gray', linestyle='--', linewidth=1)
    
    plt.title('Autocorrelation Function of Seasonal & Residual Component', fontsize=14)
    plt.xlabel('Lag', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'decomposition_{column}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series decomposition saved to {output_path}")
    
    # Detect seasonality using autocorrelation
    # Find the first significant peak after lag 1
    significant_lags = []
    for i in range(2, len(acf_values)):
        if acf_values[i] > conf_level and acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1 if i+1 < len(acf_values) else i]:
            significant_lags.append(i)
            if len(significant_lags) >= 3:
                break
    
    # Determine if there's seasonality
    has_seasonality = len(significant_lags) > 0
    
    return has_seasonality, significant_lags, trend, seasonal

def visualize_stationarity_analysis(df, column, output_dir='visualization_output'):
    """Perform and visualize comprehensive stationarity analysis"""
    print(f"Performing comprehensive stationarity analysis for {column}...")
    
    # Create a copy of the data
    data = df.copy()
    
    # Ensure the data has a datetime index
    if 'Date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index('Date')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Conduct ADF test
    result_dict = manual_adf_test(data[column])
    
    # Analyze different transformations to achieve stationarity
    transformations = {
        'Original': data[column],
        'Log': np.log(data[column]) if (data[column] > 0).all() else None,
        'Diff': data[column].diff().dropna(),
        'Log + Diff': np.log(data[column]).diff().dropna() if (data[column] > 0).all() else None,
        'Percent Change': data[column].pct_change().dropna()
    }
    
    # Remove None transformations (log can't be applied to negative values)
    transformations = {k: v for k, v in transformations.items() if v is not None}
    
    # Create a plot with all transformations
    plt.figure(figsize=(15, 10))
    
    for i, (name, series) in enumerate(transformations.items()):
        plt.subplot(len(transformations), 1, i+1)
        sns.lineplot(x=series.index, y=series.values, label=name)
        
        # Add stationarity test result
        test_result = manual_adf_test(series)
        is_stationary = test_result['p-value'] < 0.05
        stationary_text = "Stationary" if is_stationary else "Non-stationary"
        
        plt.title(f'{name} Series: {stationary_text} (p-value: {test_result["p-value"]:.4f})', 
                fontsize=12)
        plt.ylabel('Value', fontsize=10)
        if i < len(transformations) - 1:
            plt.tick_params(labelbottom=False)
    
    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'stationarity_transforms_{column}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stationarity transformation comparison saved to {output_path}")
    
    # Create a plot of value distributions before and after best transformation
    # Find the best transformation (one with lowest p-value)
    best_transform = min(transformations.items(), 
                         key=lambda x: manual_adf_test(x[1])['p-value'] if len(x[1]) > 10 else 1)
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data[column].dropna(), kde=True, bins=30, color='blue')
    plt.title(f'Original {column} Distribution', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.histplot(best_transform[1].dropna(), kde=True, bins=30, color='green')
    plt.title(f'Best Transform ({best_transform[0]}) Distribution', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'distribution_comparison_{column}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution comparison saved to {output_path}")
    
    return result_dict, best_transform[0]

def manual_adf_test(series):
    """
    A simple function to mimic the results of the ADF test without using statsmodels.
    This is a VERY simplified version and does not produce actual ADF statistics.
    It only checks for general characteristics associated with stationarity.
    """
    result = {}
    
    # Simple check for stationarity: variance in the first half vs second half
    half_point = len(series) // 2
    first_half = series[:half_point].dropna()
    second_half = series[half_point:].dropna()
    
    # Check if means are similar
    mean_first = first_half.mean()
    mean_second = second_half.mean()
    mean_diff_pct = abs((mean_second - mean_first) / mean_first) * 100 if mean_first != 0 else float('inf')
    
    # Check if variances are similar
    var_first = first_half.var()
    var_second = second_half.var()
    var_diff_pct = abs((var_second - var_first) / var_first) * 100 if var_first != 0 else float('inf')
    
    # Simple trend check: compare first and last values
    trend_direction = series.iloc[-1] - series.iloc[0]
    
    # Autocorrelation: simple lag-1 correlation
    lag1_corr = series.autocorr(lag=1)
    
    # Make a simple determination based on these checks
    result['is_stationary'] = mean_diff_pct < 20 and var_diff_pct < 50 and abs(lag1_corr) < 0.7
    result['mean_diff_pct'] = mean_diff_pct
    result['var_diff_pct'] = var_diff_pct
    result['trend_direction'] = trend_direction
    result['lag1_correlation'] = lag1_corr
    
    return result

def analyze_stationarity(df):
    """Analyze stationarity of time series data"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for stationarity analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for stationarity analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'value', 'btc', 'usd']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        if any(keyword in col.lower() for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test stationarity for each key feature
    results = {}
    for feature in key_features:
        feature_data = numeric_df[feature].dropna()
        
        if len(feature_data) < 30:
            print(f"Not enough data for {feature} to perform stationarity analysis.")
            continue
        
        # Test original series
        results[feature] = {}
        results[feature]['original'] = manual_adf_test(feature_data)
        
        # Test first difference
        diff = feature_data.diff().dropna()
        results[feature]['diff'] = manual_adf_test(diff)
        
        # Test log transform (if all values are positive)
        if all(feature_data > 0):
            log_data = np.log(feature_data)
            results[feature]['log'] = manual_adf_test(log_data)
            
            # Test log transform + first difference
            log_diff = log_data.diff().dropna()
            results[feature]['log_diff'] = manual_adf_test(log_diff)
    
    # Write stationarity results to file
    with open(f"{output_dir}/stationarity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.write("Stationarity Analysis Results\n")
        f.write("============================\n\n")
        
        for feature, feature_results in results.items():
            f.write(f"{feature}:\n")
            f.write("-" * len(feature) + "\n")
            
            for transform, result in feature_results.items():
                f.write(f"  {transform.capitalize()}:\n")
                f.write(f"    Likely stationary: {result['is_stationary']}\n")
                f.write(f"    Mean difference between halves: {result['mean_diff_pct']:.2f}%\n")
                f.write(f"    Variance difference between halves: {result['var_diff_pct']:.2f}%\n")
                f.write(f"    Trend direction: {'Upward' if result['trend_direction'] > 0 else 'Downward'}\n")
                f.write(f"    Lag-1 autocorrelation: {result['lag1_correlation']:.4f}\n\n")
    
    return results

def analyze_transformations(df):
    """Analyze different transformations for stationarity"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for transformation analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for transformation analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'value', 'btc', 'usd']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        if any(keyword in col.lower() for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze transformations for each key feature
    for feature in key_features:
        feature_data = numeric_df[feature].dropna()
        
        if len(feature_data) < 30:
            print(f"Not enough data for {feature} to perform transformation analysis.")
            continue
        
        # Calculate transformations
        transformations = {
            'Original': feature_data,
            'First Difference': feature_data.diff().dropna()
        }
        
        # Add log transforms if data is positive
        if all(feature_data > 0):
            transformations['Log'] = np.log(feature_data)
            transformations['Log + First Difference'] = np.log(feature_data).diff().dropna()
        
        # Plot transformations
        fig, axes = plt.subplots(len(transformations), 1, figsize=(14, 4 * len(transformations)))
        
        for i, (name, data) in enumerate(transformations.items()):
            ax = axes[i] if len(transformations) > 1 else axes
            ax.plot(data.index, data)
            ax.set_title(f'{name} - {feature}')
            ax.grid(True, alpha=0.3)
            
            # Add stationary/non-stationary label based on simple check
            result = manual_adf_test(data)
            status = "Likely Stationary" if result['is_stationary'] else "Likely Non-Stationary"
            ax.annotate(
                status,
                xy=(0.02, 0.9),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/transformations_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

def manual_acf(series, lags=20):
    """Calculate autocorrelation function manually"""
    acf_values = []
    mean = series.mean()
    variance = series.var()
    
    if variance == 0:
        return [0] * (lags + 1)
    
    # Calculate autocorrelation for each lag
    for lag in range(lags + 1):
        if lag == 0:
            acf_values.append(1.0)  # Autocorrelation at lag 0 is always 1
            continue
            
        # Calculate numerator (covariance)
        numerator = sum((series.iloc[i] - mean) * (series.iloc[i-lag] - mean) for i in range(lag, len(series)))
        
        # Calculate denominator (variance)
        denominator = sum((series.iloc[i] - mean) ** 2 for i in range(len(series)))
        
        if denominator == 0:
            acf_values.append(0)
        else:
            acf_value = numerator / denominator
            acf_values.append(acf_value)
    
    return acf_values

def manual_pacf(series, lags=20):
    """Calculate partial autocorrelation function using Durbin-Levinson recursion (simplified)"""
    # This is a simplified implementation and may not be as accurate as statsmodels pacf
    n = len(series)
    pacf_values = [1.0]  # PACF at lag 0 is always 1
    
    # Calculate ACF values first
    acf_values = manual_acf(series, lags)
    
    # Calculate PACF using simplified Yule-Walker equations
    for k in range(1, lags + 1):
        if k == 1:
            pacf_values.append(acf_values[1])
            continue
            
        # Simple approximation for PACF 
        # Note: This is a simplified approach and not as accurate as the full algorithm
        numerator = acf_values[k]
        for j in range(1, k):
            numerator -= pacf_values[j] * acf_values[k-j]
            
        denominator = 1.0
        for j in range(1, k):
            denominator -= pacf_values[j] * acf_values[j]
            
        if denominator == 0:
            pacf_values.append(0)
        else:
            pacf_values.append(numerator / denominator)
    
    return pacf_values

def analyze_autocorrelation(df):
    """Analyze autocorrelation and partial autocorrelation functions"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for autocorrelation analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for autocorrelation analysis.")
        return
    
    # Choose key features (up to 3)
    price_keywords = ['price', 'close', 'value', 'btc', 'usd']
    key_features = []
    
    # First try to find price-related columns
    for col in numeric_df.columns:
        if any(keyword in col.lower() for keyword in price_keywords) and len(key_features) < 3:
            key_features.append(col)
    
    # If we don't have enough, add other numeric columns
    remaining_slots = 3 - len(key_features)
    for col in numeric_df.columns:
        if col not in key_features and len(key_features) < 3:
            key_features.append(col)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Maximum number of lags to consider
    max_lags = 30
    
    # For each feature, calculate and plot ACF and PACF
    for feature in key_features:
        feature_data = numeric_df[feature].dropna()
        
        if len(feature_data) < max_lags + 10:
            print(f"Not enough data for {feature} to perform autocorrelation analysis.")
            continue
        
        # Calculate autocorrelation and partial autocorrelation
        acf_values = manual_acf(feature_data, max_lags)
        pacf_values = manual_pacf(feature_data, max_lags)
        
        # Plot ACF
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(acf_values)), acf_values, alpha=0.7, color='skyblue')
        
        # Add confidence intervals (approximately ±2/sqrt(n))
        conf_level = 1.96 / np.sqrt(len(feature_data))
        plt.axhline(y=conf_level, linestyle='--', color='red', alpha=0.7)
        plt.axhline(y=-conf_level, linestyle='--', color='red', alpha=0.7)
        
        plt.title(f'Autocorrelation Function (ACF) for {feature}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        
        # Save ACF plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/acf_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Plot PACF
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(pacf_values)), pacf_values, alpha=0.7, color='lightgreen')
        
        # Add confidence intervals
        plt.axhline(y=conf_level, linestyle='--', color='red', alpha=0.7)
        plt.axhline(y=-conf_level, linestyle='--', color='red', alpha=0.7)
        
        plt.title(f'Partial Autocorrelation Function (PACF) for {feature}')
        plt.xlabel('Lag')
        plt.ylabel('Partial Correlation')
        plt.grid(True, alpha=0.3)
        
        # Save PACF plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pacf_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Do the same for differenced data
        diff_data = feature_data.diff().dropna()
        
        # Calculate autocorrelation and partial autocorrelation for differenced data
        acf_diff = manual_acf(diff_data, max_lags)
        pacf_diff = manual_pacf(diff_data, max_lags)
        
        # Plot ACF for differenced data
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(acf_diff)), acf_diff, alpha=0.7, color='skyblue')
        
        # Add confidence intervals
        conf_level_diff = 1.96 / np.sqrt(len(diff_data))
        plt.axhline(y=conf_level_diff, linestyle='--', color='red', alpha=0.7)
        plt.axhline(y=-conf_level_diff, linestyle='--', color='red', alpha=0.7)
        
        plt.title(f'ACF for First Difference of {feature}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        
        # Save ACF plot for differenced data
        plt.tight_layout()
        plt.savefig(f"{output_dir}/acf_diff_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Plot PACF for differenced data
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(pacf_diff)), pacf_diff, alpha=0.7, color='lightgreen')
        
        # Add confidence intervals
        plt.axhline(y=conf_level_diff, linestyle='--', color='red', alpha=0.7)
        plt.axhline(y=-conf_level_diff, linestyle='--', color='red', alpha=0.7)
        
        plt.title(f'PACF for First Difference of {feature}')
        plt.xlabel('Lag')
        plt.ylabel('Partial Correlation')
        plt.grid(True, alpha=0.3)
        
        # Save PACF plot for differenced data
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pacf_diff_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

def run_all_stationarity_analyses(df, file_name=""):
    """Run all stationarity analysis functions on the dataframe"""
    print(f"\nAnalyzing stationarity for: {file_name}\n")
    
    # Run all analyses
    visualize_rolling_statistics(df)
    analyze_stationarity(df)
    analyze_transformations(df)
    analyze_autocorrelation(df)
    
    print(f"\nStationarity analyses saved to the 'visualization_output' directory.")

def run_all_visualizations(df, output_dir, file_name=""):
    """Run all stationarity analysis visualizations on the dataframe"""
    print(f"\nGenerating stationarity analysis visualizations for: {file_name}\n")
    print(f"Output directory: {output_dir}")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get numeric columns only for analysis
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found for analysis")
        return
    
    # Run all visualizations with updated output directory
    visualize_stationarity_tests(df, output_dir)
    visualize_rolling_statistics(df, output_dir)
    visualize_differencing(df, output_dir)
    visualize_transformations(df, output_dir)
    
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
        <title>Stationarity Analysis - {file_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .image-container {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #777; }}
        </style>
    </head>
    <body>
        <h1>Stationarity Analysis Visualizations</h1>
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
            Generated by Stationarity Analysis Tool
        </div>
    </body>
    </html>
    """
    
    # Write to file
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization index created at: {index_path}")

def visualize_stationarity_tests(df, output_dir):
    """Run and visualize stationarity tests for each numeric column"""
    print("Running stationarity tests on all numeric columns...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found for stationarity tests")
        return
    
    # Create a summary dataframe for test results
    results = pd.DataFrame(
        index=numeric_df.columns,
        columns=['ADF p-value', 'KPSS p-value', 'Is Stationary']
    )
    
    # Run tests for each column
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        
        if len(series) < 10:
            print(f"Not enough data for stationarity tests on {col}")
            results.loc[col] = ['Insufficient data', 'Insufficient data', 'Unknown']
            continue
        
        try:
            # ADF Test (null hypothesis: has unit root, i.e., non-stationary)
            adf_result = adfuller(series, autolag='AIC')
            adf_pvalue = adf_result[1]
            
            # KPSS Test (null hypothesis: is stationary)
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_pvalue = kpss_result[1]
            
            # Determine stationarity
            # Stationary if we can reject ADF null hypothesis and fail to reject KPSS null hypothesis
            is_stationary = 'Yes' if adf_pvalue < 0.05 and kpss_pvalue >= 0.05 else 'No'
            
            results.loc[col] = [adf_pvalue, kpss_pvalue, is_stationary]
            
        except Exception as e:
            print(f"Error in stationarity tests for {col}: {e}")
            results.loc[col] = ['Error', 'Error', 'Unknown']
    
    # Create visualization of results
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    
    # Plot ADF p-values
    ax.bar(
        np.arange(len(results)) - 0.2, 
        results['ADF p-value'].replace('Error', np.nan).replace('Insufficient data', np.nan).astype(float),
        width=0.4,
        label='ADF p-value',
        color='steelblue'
    )
    
    # Plot KPSS p-values
    ax.bar(
        np.arange(len(results)) + 0.2, 
        results['KPSS p-value'].replace('Error', np.nan).replace('Insufficient data', np.nan).astype(float),
        width=0.4,
        label='KPSS p-value',
        color='orange'
    )
    
    # Add significance level line
    ax.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
    
    # Add labels and legend
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results.index, rotation=45, ha='right')
    ax.set_ylabel('p-value')
    ax.set_title('Stationarity Test Results')
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'stationarity_test_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stationarity test results saved to {output_path}")
    
    # Also save the results table as CSV
    results_path = os.path.join(output_dir, 'stationarity_test_results.csv')
    results.to_csv(results_path)
    print(f"Stationarity test results table saved to {results_path}")
    
    return results

def visualize_differencing(df, output_dir):
    """Visualize the effect of differencing on time series data"""
    print("Generating differencing visualizations...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the dataframe has a date column
    if 'Date' not in df.columns:
        print("Warning: No 'Date' column found. Skipping differencing visualization.")
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
            if len(series) < 30:
                print(f"  Skipping {col}: insufficient data points")
                continue
            
            # Create the plot
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            # Original series
            axes[0].plot(series.index, series.values)
            axes[0].set_title(f'Original Series: {col}', fontsize=14)
            axes[0].grid(True)
            
            # First differencing
            diff1 = series.diff().dropna()
            axes[1].plot(diff1.index, diff1.values)
            axes[1].set_title('First Differencing', fontsize=14)
            axes[1].grid(True)
            
            # Second differencing
            diff2 = diff1.diff().dropna()
            axes[2].plot(diff2.index, diff2.values)
            axes[2].set_title('Second Differencing', fontsize=14)
            axes[2].grid(True)
            
            plt.tight_layout()
            
            # Sanitize column name for filename
            safe_col_name = sanitize_filename(col)
            
            # Save the figure
            output_path = os.path.join(output_dir, f'differencing_{safe_col_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create ACF and PACF plots for each level of differencing
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            
            # Original ACF/PACF
            plot_acf(series, ax=axes[0, 0], lags=40, alpha=0.05)
            axes[0, 0].set_title(f'ACF - Original: {col}', fontsize=12)
            
            plot_pacf(series, ax=axes[0, 1], lags=40, alpha=0.05)
            axes[0, 1].set_title(f'PACF - Original: {col}', fontsize=12)
            
            # First differencing ACF/PACF
            plot_acf(diff1, ax=axes[1, 0], lags=40, alpha=0.05)
            axes[1, 0].set_title('ACF - First Differencing', fontsize=12)
            
            plot_pacf(diff1, ax=axes[1, 1], lags=40, alpha=0.05)
            axes[1, 1].set_title('PACF - First Differencing', fontsize=12)
            
            # Second differencing ACF/PACF
            plot_acf(diff2, ax=axes[2, 0], lags=40, alpha=0.05)
            axes[2, 0].set_title('ACF - Second Differencing', fontsize=12)
            
            plot_pacf(diff2, ax=axes[2, 1], lags=40, alpha=0.05)
            axes[2, 1].set_title('PACF - Second Differencing', fontsize=12)
            
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(output_dir, f'differencing_acf_pacf_{safe_col_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Error analyzing {col}: {str(e)}")
    
    print(f"Differencing visualizations saved to {output_dir}")

def visualize_transformations(df, output_dir):
    """Visualize various transformations for stationarity"""
    print("Generating transformations visualizations...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the dataframe has a date column
    if 'Date' not in df.columns:
        print("Warning: No 'Date' column found. Skipping transformations visualization.")
        return
    
    # Set Date as index if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Filter for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    from scipy import stats
    
    for col in numeric_cols:
        try:
            # Get a complete time series without missing values
            series = df[col].dropna()
            
            # Skip if too few data points
            if len(series) < 30:
                print(f"  Skipping {col}: insufficient data points")
                continue
            
            # Skip series with non-positive values
            if (series <= 0).any():
                print(f"  Skipping {col}: contains non-positive values")
                continue
            
            # Create the plot
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            
            # Original series
            axes[0, 0].plot(series.index, series.values)
            axes[0, 0].set_title(f'Original Series: {col}', fontsize=12)
            axes[0, 0].grid(True)
            
            # Log transformation
            log_series = np.log(series)
            axes[0, 1].plot(log_series.index, log_series.values)
            axes[0, 1].set_title('Log Transformation', fontsize=12)
            axes[0, 1].grid(True)
            
            # Square root transformation
            sqrt_series = np.sqrt(series)
            axes[1, 0].plot(sqrt_series.index, sqrt_series.values)
            axes[1, 0].set_title('Square Root Transformation', fontsize=12)
            axes[1, 0].grid(True)
            
            # Box-Cox transformation
            boxcox_series, _ = stats.boxcox(series)
            boxcox_series = pd.Series(boxcox_series, index=series.index)
            axes[1, 1].plot(boxcox_series.index, boxcox_series.values)
            axes[1, 1].set_title('Box-Cox Transformation', fontsize=12)
            axes[1, 1].grid(True)
            
            # Difference of log transformation
            diff_log = log_series.diff().dropna()
            axes[2, 0].plot(diff_log.index, diff_log.values)
            axes[2, 0].set_title('Difference of Log Transformation', fontsize=12)
            axes[2, 0].grid(True)
            
            # Percent change
            pct_change = series.pct_change().dropna()
            axes[2, 1].plot(pct_change.index, pct_change.values)
            axes[2, 1].set_title('Percent Change', fontsize=12)
            axes[2, 1].grid(True)
            
            plt.tight_layout()
            
            # Sanitize column name for filename
            safe_col_name = sanitize_filename(col)
            
            # Save the figure
            output_path = os.path.join(output_dir, f'transformations_{safe_col_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Error analyzing {col}: {str(e)}")
    
    print(f"Transformation visualizations saved to {output_dir}")

def visualize_rolling_statistics(df, output_dir):
    """Visualize rolling mean and standard deviation"""
    print("Generating rolling statistics visualizations...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the dataframe has a date column
    if 'Date' not in df.columns:
        print("Warning: No 'Date' column found. Skipping rolling statistics visualization.")
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
            if len(series) < 60:  # Need enough points for rolling window
                print(f"  Skipping {col}: insufficient data points")
                continue
            
            # Calculate rolling statistics
            rolling_mean = series.rolling(window=30).mean()
            rolling_std = series.rolling(window=30).std()
            
            # Create the plot
            plt.figure(figsize=(14, 8))
            
            # Original series
            plt.plot(series.index, series.values, label='Original', linewidth=1)
            
            # Rolling mean
            plt.plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean (30)', color='red', linewidth=2)
            
            # Rolling standard deviation
            plt.plot(rolling_std.index, rolling_std.values, label='Rolling Std (30)', color='green', linewidth=2)
            
            plt.title(f'Rolling Statistics: {col}', fontsize=14)
            plt.legend(loc='best')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Sanitize column name for filename
            safe_col_name = sanitize_filename(col)
            
            # Save the figure
            output_path = os.path.join(output_dir, f'rolling_statistics_{safe_col_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Error analyzing {col}: {str(e)}")
    
    print(f"Rolling statistics visualizations saved to {output_dir}")

def main():
    """Main function to parse arguments and run visualizations"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate stationarity analysis visualizations')
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