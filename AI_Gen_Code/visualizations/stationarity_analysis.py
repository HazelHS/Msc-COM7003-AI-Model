"""
Stationarity Analysis Visualizations

This script provides visualizations for analyzing the stationarity of cryptocurrency time series data,
including rolling statistics, time series transformations, and autocorrelation analysis.

Usage:
    python stationarity_analysis.py [csv_file_path]

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

def plot_rolling_statistics(df):
    """Plot rolling mean and standard deviation to assess stationarity"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for rolling statistics analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for rolling statistics analysis.")
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
    
    # Calculate rolling windows
    window_sizes = [30, 90]  # 30-day and 90-day rolling windows
    
    for feature in key_features:
        for window in window_sizes:
            if len(numeric_df) <= window:
                print(f"Not enough data for {window}-day rolling window for {feature}.")
                continue
                
            # Calculate rolling statistics
            rolling_mean = numeric_df[feature].rolling(window=window).mean()
            rolling_std = numeric_df[feature].rolling(window=window).std()
            
            # Plot the original data and rolling statistics
            plt.figure(figsize=(14, 7))
            plt.plot(numeric_df.index, numeric_df[feature], label=f'Original {feature}')
            plt.plot(rolling_mean.index, rolling_mean, label=f'{window}-day Rolling Mean')
            plt.plot(rolling_std.index, rolling_std, label=f'{window}-day Rolling Std Dev')
            
            plt.title(f'Rolling Statistics for {feature} (Window Size: {window} days)')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f"{output_dir}/rolling_stats_{feature}_w{window}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.close()

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
    plot_rolling_statistics(df)
    analyze_stationarity(df)
    analyze_transformations(df)
    analyze_autocorrelation(df)
    
    print(f"\nStationarity analyses saved to the 'visualization_output' directory.")

def main():
    """Main function to parse arguments and run analyses"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            df = load_data(file_path)
            if df is not None:
                run_all_stationarity_analyses(df, os.path.basename(file_path))
            else:
                print(f"Failed to load data from {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Usage: python stationarity_analysis.py [csv_file_path]")
        print("Example: python stationarity_analysis.py ../datasets/processed_exchanges/BTC_USD.csv")

if __name__ == "__main__":
    main() 