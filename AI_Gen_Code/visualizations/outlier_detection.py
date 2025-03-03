"""
Outlier Detection Visualizations

This script provides visualizations for detecting and analyzing outliers in cryptocurrency data.

Usage:
    python outlier_detection.py [csv_file_path] [--output_dir OUTPUT_DIR]

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
from sklearn.ensemble import IsolationForest
from scipy import stats

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

def create_boxplots(df, output_dir):
    """Create box plots for numeric columns using seaborn"""
    print("Generating box plots for outlier detection...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create box plots for all numeric columns
    plt.figure(figsize=(14, 10))
    
    # Use seaborn's boxplot function
    ax = sns.boxplot(data=df[numeric_cols], palette="Set3")
    
    # Enhance the plot
    plt.title('Box Plots for Detecting Outliers', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'boxplots_all_features.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved to {output_path}")
    
    # Create individual box plots with strip plots for better visualization
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        
        # Create a box plot with a swarm plot overlay
        ax = sns.boxplot(y=df[col], width=0.4)
        sns.swarmplot(y=df[col], color="0.25", size=3, alpha=0.6)
        
        # Add title and labels
        plt.title(f'Box Plot with Data Points: {col}', fontsize=14)
        plt.ylabel(col, fontsize=12)
        
        # Add statistics
        stats = df[col].describe()
        iqr = stats['75%'] - stats['25%']
        lower_fence = stats['25%'] - 1.5 * iqr
        upper_fence = stats['75%'] + 1.5 * iqr
        outliers = df[col][(df[col] < lower_fence) | (df[col] > upper_fence)]
        
        stats_text = (
            f"IQR: {iqr:.2f}\n"
            f"Lower fence: {lower_fence:.2f}\n"
            f"Upper fence: {upper_fence:.2f}\n"
            f"Outliers: {len(outliers)} ({len(outliers)/len(df[col])*100:.1f}%)"
        )
        
        plt.figtext(0.92, 0.5, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'boxplot_{col}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def analyze_z_scores(df, output_dir, threshold=3.0):
    """Analyze outliers using z-scores and visualize using seaborn"""
    print(f"Analyzing outliers using z-scores (threshold: {threshold})...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate z-scores for each numeric column
    z_scores = pd.DataFrame()
    for col in numeric_cols:
        z_scores[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Count outliers for each column based on z-score threshold
    outlier_counts = {}
    for col in numeric_cols:
        outliers = abs(z_scores[col]) > threshold
        outlier_counts[col] = outliers.sum()
    
    # Create a bar plot of outlier counts
    plt.figure(figsize=(12, 6))
    
    # Use seaborn's barplot
    outlier_bar = sns.barplot(
        x=list(outlier_counts.keys()),
        y=list(outlier_counts.values()),
        palette="Blues_d"
    )
    
    # Add count labels on top of bars
    for i, p in enumerate(outlier_bar.patches):
        outlier_bar.annotate(f"{outlier_counts[numeric_cols[i]]}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'bottom', fontsize=10)
    
    plt.title(f'Number of Outliers (Z-Score > {threshold}) by Feature', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Number of Outliers', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'z_score_outliers_count.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Z-score outlier counts saved to {output_path}")
    
    # Create a heatmap of extreme z-scores
    plt.figure(figsize=(12, 8))
    
    # Clip z-scores for better visualization
    z_scores_clipped = z_scores.copy()
    for col in z_scores.columns:
        z_scores_clipped[col] = z_scores[col].clip(-threshold, threshold)
    
    # Create a heatmap using seaborn
    sns.heatmap(z_scores_clipped.T, cmap='coolwarm', center=0, 
              vmin=-threshold, vmax=threshold, 
              cbar_kws={'label': 'Z-Score'})
    
    plt.title('Z-Score Heatmap (Clipped to +/- {})'.format(threshold), fontsize=14)
    plt.xlabel('Data Point Index', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'z_score_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Z-score heatmap saved to {output_path}")
    
    # For the top column with most outliers, show the distribution with outliers highlighted
    if outlier_counts:
        top_col = max(outlier_counts, key=outlier_counts.get)
        plt.figure(figsize=(12, 6))
        
        # Create distribution plot using seaborn
        sns.histplot(df[top_col], kde=True, color='skyblue')
        
        # Highlight outliers
        outlier_mask = abs(z_scores[top_col]) > threshold
        outlier_values = df.loc[outlier_mask, top_col]
        
        if not outlier_values.empty:
            plt.axvspan(df[top_col].mean() + threshold * df[top_col].std(), 
                      df[top_col].max() * 1.1, 
                      alpha=0.2, color='red', label='Upper Outlier Zone')
            
            plt.axvspan(df[top_col].min() * 1.1, 
                      df[top_col].mean() - threshold * df[top_col].std(), 
                      alpha=0.2, color='red', label='Lower Outlier Zone')
            
            # Add a rug plot for outlier positions
            sns.rugplot(outlier_values, color='red', height=0.1)
        
        plt.title(f'Distribution of {top_col} with Outliers Highlighted', fontsize=14)
        plt.xlabel(top_col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'distribution_outliers_{top_col}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution with outliers highlighted saved to {output_path}")

def find_extreme_values(df, output_dir):
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

def analyze_local_outliers(df, output_dir):
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

def run_all_visualizations(df, output_dir, file_name=""):
    """Run all outlier detection visualizations on the dataframe"""
    print(f"\nGenerating outlier detection visualizations for: {file_name}\n")
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
    create_boxplots(df, output_dir)
    analyze_z_scores(df, output_dir)
    
    # Update these functions to use output_dir
    find_extreme_values(df, output_dir)
    analyze_local_outliers(df, output_dir)
    
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
        <title>Outlier Detection - {file_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .image-container {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #777; }}
        </style>
    </head>
    <body>
        <h1>Outlier Detection Visualizations</h1>
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
            Generated by Outlier Detection Tool
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
    parser = argparse.ArgumentParser(description='Generate outlier detection visualizations')
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