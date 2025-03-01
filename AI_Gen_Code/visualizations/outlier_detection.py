"""
Outlier Detection Visualizations

This script provides visualizations for detecting and analyzing outliers in cryptocurrency datasets,
including box plots, z-score analysis, and isolation of anomalous data points.

Usage:
    python outlier_detection.py [csv_file_path]

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

def create_boxplots(df):
    """Create box plots for numeric features to visualize outliers"""
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for box plot analysis.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a single figure with all boxplots
    plt.figure(figsize=(12, 8))
    
    # Plot the boxplots
    bp = plt.boxplot(
        [numeric_df[col].dropna() for col in numeric_df.columns],
        labels=numeric_df.columns,
        patch_artist=True,
        vert=True,
        whis=1.5
    )
    
    # Customize boxplots
    for box in bp['boxes']:
        box.set(facecolor='lightblue', alpha=0.8)
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.2, linestyle='--')
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='red', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='red', markersize=4, alpha=0.5)
    
    plt.title('Box Plots for Numeric Features')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Value')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots_all_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # If there are more than 5 features, create individual boxplots
    if len(numeric_df.columns) > 5:
        for col in numeric_df.columns:
            plt.figure(figsize=(8, 6))
            
            bp = plt.boxplot(
                numeric_df[col].dropna(),
                labels=[col],
                patch_artist=True,
                vert=True,
                whis=1.5
            )
            
            # Customize boxplot
            for box in bp['boxes']:
                box.set(facecolor='lightblue', alpha=0.8)
            for whisker in bp['whiskers']:
                whisker.set(color='gray', linewidth=1.2, linestyle='--')
            for cap in bp['caps']:
                cap.set(color='gray', linewidth=1.2)
            for median in bp['medians']:
                median.set(color='red', linewidth=1.5)
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor='red', markersize=4, alpha=0.5)
            
            plt.title(f'Box Plot for {col}')
            plt.ylabel('Value')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add basic statistics
            stats_text = (
                f"Mean: {numeric_df[col].mean():.2f}\n"
                f"Median: {numeric_df[col].median():.2f}\n"
                f"Std Dev: {numeric_df[col].std():.2f}\n"
                f"Min: {numeric_df[col].min():.2f}\n"
                f"Max: {numeric_df[col].max():.2f}"
            )
            plt.figtext(0.65, 0.7, stats_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f"{output_dir}/boxplot_{col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.close()

def analyze_z_scores(df):
    """Analyze z-scores to identify potential outliers"""
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for z-score analysis.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate z-scores for each numeric feature
    z_scores = pd.DataFrame(index=numeric_df.index)
    for col in numeric_df.columns:
        # Calculate mean and std dev, ignoring NaNs
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        
        # Calculate z-score (manually, without scipy)
        if std != 0:  # Avoid division by zero
            z_scores[f'z_{col}'] = (numeric_df[col] - mean) / std
        else:
            z_scores[f'z_{col}'] = 0
    
    # Identify outliers (|z| > 3)
    outliers = pd.DataFrame(index=z_scores.index)
    for col in z_scores.columns:
        outliers[col] = abs(z_scores[col]) > 3
    
    # Count outliers for each feature
    outlier_counts = outliers.sum()
    outlier_percents = (outliers.sum() / outliers.count()) * 100
    
    # Prepare outlier summary
    outlier_summary = pd.DataFrame({
        'feature': [col.replace('z_', '') for col in outlier_counts.index],
        'outlier_count': outlier_counts.values,
        'outlier_percent': outlier_percents.values
    })
    
    # Plot outlier counts
    plt.figure(figsize=(12, 6))
    bars = plt.bar(outlier_summary['feature'], outlier_summary['outlier_count'], color='salmon')
    plt.title('Number of Outliers by Feature (|z-score| > 3)')
    plt.xlabel('Feature')
    plt.ylabel('Outlier Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.1, 
            str(int(bar.get_height())), 
            ha='center'
        )
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/outlier_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Plot outlier percentages
    plt.figure(figsize=(12, 6))
    bars = plt.bar(outlier_summary['feature'], outlier_summary['outlier_percent'], color='lightblue')
    plt.title('Percentage of Outliers by Feature (|z-score| > 3)')
    plt.xlabel('Feature')
    plt.ylabel('Outlier Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.1, 
            f"{bar.get_height():.1f}%", 
            ha='center'
        )
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/outlier_percentages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Write outlier summary to file
    with open(f"{output_dir}/outlier_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.write("Outlier Summary (|z-score| > 3):\n")
        f.write("------------------------------\n\n")
        
        for i, row in outlier_summary.iterrows():
            f.write(f"Feature: {row['feature']}\n")
            f.write(f"Outlier count: {int(row['outlier_count'])}\n")
            f.write(f"Outlier percentage: {row['outlier_percent']:.2f}%\n\n")
    
    return outlier_summary

def find_extreme_values(df):
    """Find and visualize extreme values based on percentiles"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for extreme value analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for extreme value analysis.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Calculate percentiles for each key feature
    percentiles = [1, 5, 95, 99]
    
    extreme_days = {}
    for feature in key_features:
        # Calculate percentiles
        p_values = {}
        for p in percentiles:
            p_values[p] = np.percentile(numeric_df[feature].dropna(), p)
        
        # Find extreme high and low days
        extreme_highs = df[numeric_df[feature] >= p_values[99]]
        extreme_lows = df[numeric_df[feature] <= p_values[1]]
        
        extreme_days[feature] = {
            'highs': extreme_highs,
            'lows': extreme_lows
        }
        
        # Plot feature with percentile lines
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, numeric_df[feature], color='gray', alpha=0.7)
        
        # Plot extreme values
        plt.scatter(
            extreme_highs.index, 
            extreme_highs[feature], 
            color='red', 
            label=f'Extreme Highs (>{p_values[99]:.2f})',
            zorder=5
        )
        plt.scatter(
            extreme_lows.index, 
            extreme_lows[feature], 
            color='blue', 
            label=f'Extreme Lows (<{p_values[1]:.2f})',
            zorder=5
        )
        
        # Add percentile lines
        plt.axhline(y=p_values[99], color='red', linestyle='--', alpha=0.5, label=f'99th percentile')
        plt.axhline(y=p_values[95], color='orange', linestyle='--', alpha=0.5, label=f'95th percentile')
        plt.axhline(y=p_values[5], color='green', linestyle='--', alpha=0.5, label=f'5th percentile')
        plt.axhline(y=p_values[1], color='blue', linestyle='--', alpha=0.5, label=f'1st percentile')
        
        plt.title(f'Extreme Values for {feature}')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/extreme_values_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        
        # Save extreme days to file
        with open(f"{output_dir}/extreme_days_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write(f"Extreme Days for {feature}:\n")
            f.write("========================\n\n")
            
            f.write("Extreme Highs (99th percentile):\n")
            f.write("-------------------------------\n")
            for date, row in extreme_highs.iterrows():
                f.write(f"{date.strftime('%Y-%m-%d')}: {row[feature]:.2f}\n")
            
            f.write("\nExtreme Lows (1st percentile):\n")
            f.write("-----------------------------\n")
            for date, row in extreme_lows.iterrows():
                f.write(f"{date.strftime('%Y-%m-%d')}: {row[feature]:.2f}\n")
    
    return extreme_days

def analyze_local_outliers(df):
    """Analyze outliers relative to their local context"""
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for local outlier analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for local outlier analysis.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Window sizes for rolling calculations
    window_sizes = [7, 14, 30]  # 1 week, 2 weeks, 1 month
    
    # For each feature and window size, calculate local z-scores
    for feature in key_features:
        for window in window_sizes:
            if len(numeric_df) <= window:
                continue  # Skip if window is too large for the dataset
            
            # Calculate rolling mean and std
            rolling_mean = numeric_df[feature].rolling(window=window).mean()
            rolling_std = numeric_df[feature].rolling(window=window).std()
            
            # Calculate local z-scores
            local_z = pd.Series(index=numeric_df.index)
            for i in range(window, len(numeric_df)):
                if rolling_std.iloc[i] != 0:  # Avoid division by zero
                    local_z.iloc[i] = (numeric_df[feature].iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
            
            # Identify local outliers (|z| > 3)
            local_outliers = abs(local_z) > 3
            
            if local_outliers.any():
                # Plot the feature with local outliers highlighted
                plt.figure(figsize=(14, 7))
                
                # Plot original data
                plt.plot(numeric_df.index, numeric_df[feature], color='gray', alpha=0.7, label=feature)
                
                # Highlight local outliers
                outlier_dates = local_outliers[local_outliers].index
                outlier_values = numeric_df.loc[outlier_dates, feature]
                plt.scatter(outlier_dates, outlier_values, color='red', s=50, label='Local Outliers', zorder=5)
                
                plt.title(f'Local Outliers for {feature} ({window}-day window)')
                plt.xlabel('Date')
                plt.ylabel(feature)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(f"{output_dir}/local_outliers_{feature}_w{window}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.close()
                
                # Plot the local z-scores
                plt.figure(figsize=(14, 7))
                plt.plot(local_z.index, local_z, color='blue', alpha=0.7)
                
                # Add threshold lines
                plt.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='z = 3')
                plt.axhline(y=-3, color='red', linestyle='--', alpha=0.5, label='z = -3')
                
                plt.title(f'Local Z-Scores for {feature} ({window}-day window)')
                plt.xlabel('Date')
                plt.ylabel('Z-Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(f"{output_dir}/local_z_scores_{feature}_w{window}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.close()
                
                # Save local outliers to file
                with open(f"{output_dir}/local_outliers_{feature}_w{window}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
                    f.write(f"Local Outliers for {feature} ({window}-day window):\n")
                    f.write("===========================================\n\n")
                    
                    for date in outlier_dates:
                        z_score = local_z.loc[date]
                        f.write(f"{date.strftime('%Y-%m-%d')}: {numeric_df.loc[date, feature]:.2f} (z-score: {z_score:.2f})\n")
            else:
                print(f"No local outliers found for {feature} with {window}-day window.")

def run_all_outlier_analyses(df, file_name=""):
    """Run all outlier detection analyses on the dataframe"""
    print(f"\nAnalyzing outliers for: {file_name}\n")
    
    # Run all analyses
    create_boxplots(df)
    analyze_z_scores(df)
    find_extreme_values(df)
    analyze_local_outliers(df)
    
    print(f"\nOutlier analyses saved to the 'visualization_output' directory.")

def main():
    """Main function to parse arguments and run analyses"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            df = load_data(file_path)
            if df is not None:
                run_all_outlier_analyses(df, os.path.basename(file_path))
            else:
                print(f"Failed to load data from {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Usage: python outlier_detection.py [csv_file_path]")
        print("Example: python outlier_detection.py ../datasets/processed_exchanges/BTC_USD.csv")

if __name__ == "__main__":
    main() 