import pandas as pd

def add_s2f_feature(df):
    """Add Stock-to-Flow ratio as a feature"""
    # Bitcoin halving dates and block rewards
    halving_dates = {
        '2009-01-03': 50,
        '2012-11-28': 25,
        '2016-07-09': 12.5,
        '2020-05-11': 6.25,
        '2024-04-20': 3.125
    }
    
    # Create daily S2F calculation
    s2f_data = []
    current_reward = 50
    total_supply = 0
    daily_flow = 0
    
    for date in pd.date_range(start='2009-01-03', end=pd.Timestamp.today()):
        if str(date.date()) in halving_dates:
            current_reward = halving_dates[str(date.date())]
        
        # 144 blocks per day * reward
        daily_flow = 144 * current_reward
        total_supply += daily_flow
        annual_flow = daily_flow * 365
        
        s2f = total_supply / annual_flow if annual_flow != 0 else 0
        s2f_data.append({'Date': date, 's2f_ratio': s2f})
    
    s2f_df = pd.DataFrame(s2f_data).set_index('Date')
    return df.merge(s2f_df, left_index=True, right_index=True, how='left')

def align_datasets(stock_df, crypto_df):
    # Convert stock data to daily frequency (including weekends)
    stock_daily = stock_df.resample('D').ffill()
    
    # Merge datasets
    combined_df = pd.merge(crypto_df, stock_daily, 
                          left_index=True, right_index=True,
                          how='left', suffixes=('_btc', '_stock'))
    # Add S2F feature
    combined_df = add_s2f_feature(combined_df)
    return combined_df

def exclude_crypto_weekends(crypto_df):
    # Convert to business day frequency
    return crypto_df[crypto_df.index.dayofweek < 5]

# Usage:
nikkei = pd.read_csv('N225.csv', parse_dates=['Date'], index_col='Date')
btc = pd.read_csv('BTC-USD.csv', parse_dates=['Date'], index_col='Date')

aligned_data = align_datasets(nikkei, btc)

print(f"Earliest data point: {aligned_data['Date'].min()}") 