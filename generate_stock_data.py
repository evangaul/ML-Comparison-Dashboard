import numpy as np
import yfinance as yf

symbol = 'AAPL'
start_date = '2018-01-01'
end_date = '2025-05-01'

# Fetch data
data = yf.Ticker(symbol)
df = data.history(period='1d', start=start_date, end=end_date)

# Drop unnecessary columns
df.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')

# Add feature columns
df['Returns'] = df['Close'].pct_change()  # Daily returns
df['MA5'] = df['Close'].rolling(window=5).mean()  # 5 day moving average
df['MA20'] = df['Close'].rolling(window=20).mean()  # 20 day moving average
df['Volatility'] = df['Returns'].rolling(window=20).std()  # 20 day volatility
df['Price Up/Down'] = np.where(df['Close'] > df['Open'], 1, 0)  # Target Column

df.drop(columns=['Close'], inplace=True) # Drop close to avoid leakage
df.dropna(inplace=True)
df.to_csv('aaplnew.csv', index=False)