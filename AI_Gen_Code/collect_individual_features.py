"""
Individual Feature Collector

This script allows collecting crypto and market features individually from free public APIs.
Run with a specific feature name to collect just that feature.
"""

import os
import sys
from crypto_features_collector import (
    get_fear_greed_index,
    get_onchain_metrics,
    get_stablecoin_metrics,
    get_volatility_indices,
    get_energy_metrics,
    get_currency_metrics,
    get_derivatives_metrics,
    create_combined_dataset,
    DEFAULT_START_DATE
)

def print_usage():
    """Print usage instructions"""
    print("Usage: python collect_individual_features.py [feature_name]")
    print("\nAvailable features:")
    print("  fear_greed      - Crypto Fear & Greed Index")
    print("  onchain         - On-chain metrics from Blockchain.com")
    print("  stablecoins     - Stablecoin market data from CoinGecko")
    print("  volatility      - Volatility indices from Yahoo Finance")
    print("  energy          - Energy market metrics and mining cost proxy")
    print("  currency        - Currency and economic indicators")
    print("  derivatives     - Basic derivatives market data")
    print("  combine         - Create combined dataset from existing files")
    print("  all             - Collect all features (may take a while)")
    print("\nExample: python collect_individual_features.py fear_greed")

def main():
    """Main function to collect specified features"""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    feature = sys.argv[1].lower()
    start_date = DEFAULT_START_DATE
    
    # Allow specifying a different start date
    if len(sys.argv) > 2:
        start_date = sys.argv[2]
    
    print(f"Collecting feature: {feature} (from {start_date})")
    
    # Process based on feature name
    if feature == 'fear_greed':
        get_fear_greed_index()
    elif feature == 'onchain':
        get_onchain_metrics(start_date)
    elif feature == 'stablecoins':
        get_stablecoin_metrics(start_date)
    elif feature == 'volatility':
        get_volatility_indices(start_date)
    elif feature == 'energy':
        get_energy_metrics(start_date)
    elif feature == 'currency':
        get_currency_metrics(start_date)
    elif feature == 'derivatives':
        get_derivatives_metrics(start_date)
    elif feature == 'combine':
        create_combined_dataset()
    elif feature == 'all':
        print("Collecting all features. This may take a while...")
        get_fear_greed_index()
        get_onchain_metrics(start_date)
        get_stablecoin_metrics(start_date)
        get_volatility_indices(start_date)
        get_energy_metrics(start_date)
        get_currency_metrics(start_date)
        get_derivatives_metrics(start_date)
        create_combined_dataset()
    else:
        print(f"Unknown feature: {feature}")
        print_usage()
        return
    
    print(f"\nFeature collection complete: {feature}")

if __name__ == "__main__":
    main() 