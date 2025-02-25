import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load and prepare data
df = pd.read_csv('btcusd_1-min_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Timestamp', inplace=True)

# Resample to daily data
daily_df = df.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last'
}).dropna()

# Create single figure
fig, ax = plt.subplots(figsize=(14, 7))

# Candlestick plot
ax.bar(daily_df.index, daily_df['High'] - daily_df['Low'], 
       bottom=daily_df['Low'], color='gray', width=1, label='High-Low')
ax.bar(daily_df.index, daily_df['Close'] - daily_df['Open'], 
       bottom=daily_df['Open'], color=['green' if cl > op else 'red' 
       for cl, op in zip(daily_df.Close, daily_df.Open)], 
       width=0.6, label='Open-Close')

# Formatting
ax.set_title('Bitcoin Daily Candlestick Chart')
ax.set_ylabel('Price (USD)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()