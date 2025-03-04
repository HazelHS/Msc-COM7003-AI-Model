"""
Collect Currency Metrics and Create Combined Dataset

This script collects currency metrics with the fixed BTC/USD data
and then creates a combined dataset.
"""

import os
import sys
from crypto_features_collector import get_currency_metrics
from create_combined_dataset import main as create_combined_dataset

def main():
    print("Starting currency data collection and combined dataset creation...")
    
    # Step 1: Collect currency metrics with fixed BTC/USD values
    print("\nStep 1: Collecting currency metrics...")
    currency_data = get_currency_metrics()
    
    if currency_data is not None:
        print("Currency metrics collection successful.")
    else:
        print("Failed to collect currency metrics.")
        return
    
    # Step 2: Create combined dataset
    print("\nStep 2: Creating combined dataset...")
    create_combined_dataset()
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main() 