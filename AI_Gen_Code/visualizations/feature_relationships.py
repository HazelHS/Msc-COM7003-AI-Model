"""
Feature Relationship Visualizations

This script provides visualizations for exploring relationships between features in cryptocurrency datasets,
including correlation analysis, scatter plots, and pair plots.

Usage:
    python feature_relationships.py [csv_file_path]

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

def create_correlation_matrix(df, output_dir):
    """Create a correlation matrix heatmap using seaborn"""
    print("Generating correlation matrix...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Skip if no numeric columns
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for correlation analysis")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 10))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create the heatmap with seaborn
    sns.heatmap(corr_matrix, 
                annot=True,
                mask=mask,
                cmap='coolwarm',
                vmin=-1, vmax=1,
                linewidths=0.5,
                fmt=".2f",
                square=True,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"})
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation matrix saved to {output_path}")
    
    # Create a clustermap for hierarchical clustering of correlations
    plt.figure(figsize=(14, 12))
    
    cluster = sns.clustermap(corr_matrix, 
                           cmap='coolwarm', 
                           vmin=-1, vmax=1,
                           annot=True, 
                           fmt=".2f",
                           linewidths=0.5,
                           figsize=(14, 12),
                           cbar_kws={"label": "Correlation Coefficient"})
    
    plt.suptitle('Hierarchical Clustering of Feature Correlations', fontsize=16, y=1.02)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'correlation_clustermap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation clustermap saved to {output_path}")
    
    return corr_matrix

def create_scatter_plots(df, output_dir, n_top_pairs=5):
    """Create scatter plots for the most correlated feature pairs using seaborn"""
    print("Generating scatter plots for feature relationships...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Skip if no numeric columns
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for scatter plots")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Get the upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find the top n_top_pairs feature pairs
    if not upper.empty:
        # Sort correlations in descending order
        sorted_corrs = upper.unstack().sort_values(kind="quicksort", ascending=False)
        sorted_corrs = sorted_corrs[sorted_corrs > 0]  # Remove zeros
        top_pairs = sorted_corrs[:n_top_pairs]
        
        # Create scatter plots for top pairs
        for i, ((col1, col2), corr_value) in enumerate(top_pairs.items()):
            plt.figure(figsize=(10, 8))
            
            # Create a scatter plot with regression line using seaborn
            sns.regplot(x=df[col1], y=df[col2], 
                      scatter_kws={'alpha':0.5}, 
                      line_kws={'color':'red'})
            
            plt.title(f'Scatter Plot: {col1} vs {col2}\nCorrelation: {corr_value:.3f}', fontsize=14)
            plt.xlabel(col1, fontsize=12)
            plt.ylabel(col2, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create safe filenames by replacing special characters
            safe_col1 = col1.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            safe_col2 = col2.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") 
            
            # Save the figure with safe filenames
            output_path = os.path.join(output_dir, f'scatter_{safe_col1}_vs_{safe_col2}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Scatter plot for {col1} vs {col2} saved to {output_path}")
            
        # Create a pairplot for the features involved in top correlations
        unique_cols = list(set([pair[0] for pair in top_pairs.index] + [pair[1] for pair in top_pairs.index]))
        
        if len(unique_cols) >= 2:
            print("Generating pairplot for top correlated features...")
            # Use seaborn's pairplot function
            pairplot = sns.pairplot(df[unique_cols], height=2.5, diag_kind='kde', plot_kws={'alpha': 0.6})
            pairplot.fig.suptitle('Pairwise Relationships Between Top Correlated Features', 
                                  y=1.02, fontsize=16)
            
            # Save the pairplot
            output_path = os.path.join(output_dir, 'pairplot_top_features.png')
            pairplot.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Pairplot saved to {output_path}")
    else:
        print("No correlations found for scatter plots.")

def analyze_price_relationships(df, output_dir):
    """Analyze relationships between price and other features using seaborn"""
    print("Analyzing price relationships with other features...")
    
    # Try to identify price-related columns
    price_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                  ['price', 'close', 'open', 'high', 'low', 'btc', 'usd'])]
    
    if not price_cols:
        print("No price-related columns found in the dataset")
        return
    
    # Select the first price column
    price_col = price_cols[0]
    
    # Get numeric columns excluding the price column
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != price_col]
    
    # Skip if no numeric columns
    if not numeric_cols:
        print("No numeric columns available for price relationship analysis")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select top correlated features with price
    correlations = df[[price_col] + numeric_cols].corr()[price_col].drop(price_col).abs()
    top_features = correlations.sort_values(ascending=False).head(5).index.tolist()
    
    if not top_features:
        print("No significant correlations with price found")
        return
    
    # Create a figure with subplots for each top feature
    plt.figure(figsize=(15, 12))
    
    for i, feature in enumerate(top_features):
        plt.subplot(len(top_features), 1, i+1)
        
        # Create a dual-axis plot
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot the price
        ax1.plot(df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df)), 
                df[price_col], 'b-', label=price_col)
        
        # Plot the feature
        ax2.plot(df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df)), 
                df[feature], 'r-', label=feature)
        
        # Add labels and legends
        ax1.set_ylabel(price_col, color='b', fontsize=10)
        ax2.set_ylabel(feature, color='r', fontsize=10)
        
        # Add a title with correlation value
        corr_value = df[[price_col, feature]].corr().iloc[0, 1]
        plt.title(f'{price_col} vs {feature} (Correlation: {corr_value:.3f})', fontsize=12)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        if isinstance(df.index, pd.DatetimeIndex):
            plt.xlabel('Date', fontsize=10)
        else:
            plt.xlabel('Index', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'price_relationships.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Price relationship analysis saved to {output_path}")
    
    # Create a joint plot for the feature most correlated with price
    if top_features:
        top_feature = top_features[0]
        
        # Use seaborn's jointplot function
        joint_plot = sns.jointplot(
            data=df, 
            x=price_col, 
            y=top_feature,
            kind="reg",  # with regression line
            scatter_kws={"alpha": 0.5},
            height=10,
            marginal_kws=dict(bins=20, fill=True)
        )
        
        joint_plot.fig.suptitle(f'Relationship between {price_col} and {top_feature}', 
                               y=1.02, fontsize=16)
        
        # Create safe filenames by replacing special characters
        safe_price_col = price_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        safe_top_feature = top_feature.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        
        # Save the figure with safe filenames
        output_path = os.path.join(output_dir, f'jointplot_{safe_price_col}_vs_{safe_top_feature}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Joint plot saved to {output_path}")

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

def run_all_analyses(df, output_dir='visualization_output', file_name=""):
    """Run all visualization analyses"""
    print(f"\nRunning feature relationship analyses on {file_name if file_name else 'data'}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    create_correlation_matrix(df, output_dir)
    create_scatter_plots(df, output_dir)
    
    # Try to identify price column and run price-specific analyses
    analyze_price_relationships(df, output_dir)
    
    print(f"\nAll feature relationship analyses completed for {file_name if file_name else 'data'}")
    print(f"Results saved to: {output_dir}")

def main():
    """Main function to parse arguments and run analyses"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze feature relationships in a dataset')
    parser.add_argument('file_path', help='Path to the CSV file to analyze')
    parser.add_argument('--output_dir', default='visualization_output',
                        help='Directory to save visualization outputs (default: visualization_output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if file exists
    if os.path.exists(args.file_path):
        print(f"\nAnalyzing feature relationships for: {os.path.basename(args.file_path)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        
        # Load the data
        df = load_data(args.file_path)
        if df is not None:
            # Run all analyses with the specified output directory
            run_all_analyses(df, args.output_dir, os.path.basename(args.file_path))
        else:
            print(f"Failed to load data from {args.file_path}")
    else:
        print(f"File not found: {args.file_path}")

if __name__ == "__main__":
    main() 