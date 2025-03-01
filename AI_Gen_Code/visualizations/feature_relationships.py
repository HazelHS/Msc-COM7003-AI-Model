"""
Feature Relationships Visualizations

This script provides visualizations for exploring relationships between features in cryptocurrency datasets,
including correlation analysis, scatter plots, and time-aligned comparisons.

Usage:
    python feature_relationships.py [csv_file_path]

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

def create_correlation_matrix(df):
    """Create and visualize a correlation matrix for numeric features"""
    print("Analyzing feature correlations...")
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for correlation analysis.")
        return
    
    # If there are too many features, select a subset
    if len(numeric_df.columns) > 20:
        print(f"Too many features ({len(numeric_df.columns)}). Selecting first 20 for correlation matrix.")
        numeric_df = numeric_df.iloc[:, :20]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create correlation matrix visualization
    plt.figure(figsize=(12, 10))
    
    # Plot correlation matrix using imshow
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, label='Correlation coefficient')
    
    # Add feature names
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Add correlation values as text
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white",
                           fontsize=8)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_dir}/correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Save highly correlated pairs to a text file
    high_corr_threshold = 0.7
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    # Write to file if there are high correlations
    if high_corr_pairs:
        with open(f"{output_dir}/high_correlations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write("Highly Correlated Feature Pairs (|correlation| >= 0.7)\n")
            f.write("=====================================================\n\n")
            
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"{feat1} -- {feat2}: {corr:.4f}\n")
    
    print(f"Correlation analysis saved to {output_dir}")
    return corr_matrix

def create_scatter_plots(df):
    """Create scatter plots for pairs of features with high correlation"""
    print("Creating scatter plots for highly correlated features...")
    
    # Get correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for scatter plot analysis.")
        return
    
    corr_matrix = numeric_df.corr()
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find highly correlated pairs
    high_corr_threshold = 0.7
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    # Sort by absolute correlation (highest first)
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    # Take top 10 pairs at most
    high_corr_pairs = high_corr_pairs[:min(10, len(high_corr_pairs))]
    
    if not high_corr_pairs:
        print("No highly correlated feature pairs found (threshold: 0.7).")
        
        # If no high correlation pairs, take top 5 pairs instead
        all_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                all_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        high_corr_pairs = sorted(all_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        print(f"Showing top 5 correlated pairs instead.")
    
    # Create scatter plots for each pair
    for feat1, feat2, corr in high_corr_pairs:
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        # Get data and remove rows with NaN values
        data = numeric_df[[feat1, feat2]].dropna()
        
        # Skip if not enough data
        if len(data) < 10:
            print(f"Not enough data for scatter plot between {feat1} and {feat2}.")
            continue
        
        # Plot scatter with transparency
        plt.scatter(data[feat1], data[feat2], alpha=0.6, s=30, c='cornflowerblue')
        
        # Add regression line
        if len(data) > 1:  # Need at least 2 points for regression
            try:
                # Fit a line using numpy's polyfit
                fit = np.polyfit(data[feat1], data[feat2], 1)
                fit_fn = np.poly1d(fit)
                
                # Create a range of x values for the line
                x_range = np.linspace(data[feat1].min(), data[feat1].max(), 100)
                
                # Plot the line
                plt.plot(x_range, fit_fn(x_range), '--r', linewidth=2)
                
                # Add equation text
                equation = f'y = {fit[0]:.4f}x + {fit[1]:.4f}'
                plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            except:
                print(f"Could not calculate regression line for {feat1} vs {feat2}.")
        
        # Add correlation coefficient text
        plt.annotate(f'Correlation: {corr:.4f}', xy=(0.05, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.title(f'Scatter Plot: {feat1} vs {feat2}')
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scatter_{feat1}_vs_{feat2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
    
    print(f"Scatter plots saved to {output_dir}")

def plot_time_aligned_features(df):
    """Plot time-aligned features to visualize relationships over time"""
    print("Analyzing time-aligned feature relationships...")
    
    if df.index.name != 'Date':
        print("DataFrame must have 'Date' as index for time-aligned feature analysis.")
        return
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for time-aligned feature analysis.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find the most interesting feature (usually price or volume)
    price_keywords = ['price', 'close', 'btc', 'usd', 'value']
    volume_keywords = ['volume', 'vol']
    
    # First try to find a price-related column
    primary_feature = None
    
    for col in numeric_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in price_keywords):
            primary_feature = col
            break
    
    # If no price feature found, try volume
    if primary_feature is None:
        for col in numeric_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in volume_keywords):
                primary_feature = col
                break
    
    # If still no feature found, use the first column
    if primary_feature is None and len(numeric_df.columns) > 0:
        primary_feature = numeric_df.columns[0]
    
    if primary_feature is None:
        print("No suitable features found for time-aligned analysis.")
        return
    
    # Find the top 3 most correlated features with the primary feature
    related_features = []
    
    for col in corr_matrix.columns:
        if col != primary_feature:
            related_features.append((col, abs(corr_matrix.loc[primary_feature, col])))
    
    # Sort by absolute correlation (highest first)
    related_features = sorted(related_features, key=lambda x: x[1], reverse=True)
    
    # Take top 3 features
    top_features = [feat for feat, _ in related_features[:min(3, len(related_features))]]
    
    # If fewer than 3 related features available, add more features
    remaining_slots = 3 - len(top_features)
    if remaining_slots > 0:
        for col in numeric_df.columns:
            if col != primary_feature and col not in top_features and len(top_features) < 3:
                top_features.append(col)
    
    # Plot primary feature with each related feature
    for related_feature in top_features:
        # Create time series plot
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot primary feature
        color1 = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(primary_feature, color=color1)
        ax1.plot(numeric_df.index, numeric_df[primary_feature], color=color1, label=primary_feature)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Create second y-axis for related feature
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(related_feature, color=color2)
        ax2.plot(numeric_df.index, numeric_df[related_feature], color=color2, label=related_feature)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add correlation information
        corr_value = corr_matrix.loc[primary_feature, related_feature]
        plt.annotate(f'Correlation: {corr_value:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.title(f'Time Series: {primary_feature} vs {related_feature}')
        plt.grid(True, alpha=0.3)
        
        # Add a legend for both lines
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_aligned_{primary_feature}_vs_{related_feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
    
    # Also create a normalized plot with all features
    features_to_plot = [primary_feature] + top_features
    
    # Normalize all selected features to 0-1 range for comparison
    normalized_df = pd.DataFrame(index=numeric_df.index)
    
    for feat in features_to_plot:
        # Skip if constant or missing values
        if numeric_df[feat].max() == numeric_df[feat].min() or numeric_df[feat].isnull().all():
            print(f"Skipping {feat} in normalized plot due to constant values or missing data.")
            continue
        
        # Min-max normalization
        normalized_df[feat] = (numeric_df[feat] - numeric_df[feat].min()) / (numeric_df[feat].max() - numeric_df[feat].min())
    
    # Create normalized comparison plot
    plt.figure(figsize=(14, 7))
    
    # Plot each normalized feature
    for feat in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[feat], label=feat, linewidth=2, alpha=0.7)
    
    plt.title(f'Normalized Time Series Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (0-1 scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normalized_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    print(f"Time-aligned feature analysis saved to {output_dir}")

def plot_feature_distributions(df):
    """Plot distributions of key features to understand their characteristics"""
    print("Analyzing feature distributions...")
    
    # Select only numeric features
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) == 0:
        print("No numeric features found for distribution analysis.")
        return
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # If there are too many features, select a subset
    if len(numeric_df.columns) > 8:
        # Try to find key features
        price_keywords = ['price', 'close', 'btc', 'usd', 'value']
        volume_keywords = ['volume', 'vol']
        
        key_features = []
        
        # Add price-related columns
        for col in numeric_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in price_keywords) and len(key_features) < 4:
                key_features.append(col)
        
        # Add volume-related columns
        for col in numeric_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in volume_keywords) and len(key_features) < 6:
                key_features.append(col)
        
        # Add other columns if needed
        for col in numeric_df.columns:
            if col not in key_features and len(key_features) < 8:
                key_features.append(col)
        
        numeric_df = numeric_df[key_features]
    
    # Create histogram grid
    num_features = len(numeric_df.columns)
    num_rows = (num_features + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))
    axes = axes.flatten() if num_features > 1 else [axes]
    
    for i, column in enumerate(numeric_df.columns):
        if i < len(axes):
            # Get data without NaN values
            data = numeric_df[column].dropna()
            
            if len(data) == 0:
                axes[i].text(0.5, 0.5, f"No data for {column}", 
                           horizontalalignment='center', verticalalignment='center')
                axes[i].set_title(column)
                continue
            
            # Create histogram with KDE
            axes[i].hist(data, bins=30, alpha=0.7, density=True, color='skyblue')
            
            # Add basic stats
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            # Add vertical lines for mean and median
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='-', linewidth=1, label=f'Median: {median_val:.2f}')
            
            # Add stats text
            stats_text = (f"Mean: {mean_val:.2f}\n"
                         f"Median: {median_val:.2f}\n"
                         f"Std Dev: {std_val:.2f}\n"
                         f"Min: {data.min():.2f}\n"
                         f"Max: {data.max():.2f}")
            
            axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            axes[i].set_title(f'Distribution of {column}')
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    print(f"Feature distribution analysis saved to {output_dir}")

def run_all_analyses(df, file_name=""):
    """Run all feature relationship analyses on the dataframe"""
    print(f"\nAnalyzing feature relationships for: {file_name}\n")
    
    # Run all analyses
    create_correlation_matrix(df)
    create_scatter_plots(df)
    plot_time_aligned_features(df)
    plot_feature_distributions(df)
    
    print(f"\nFeature relationship analyses saved to the 'visualization_output' directory.")

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
        print("Usage: python feature_relationships.py [csv_file_path]")
        print("Example: python feature_relationships.py ../datasets/processed_exchanges/BTC_USD.csv")

if __name__ == "__main__":
    main() 