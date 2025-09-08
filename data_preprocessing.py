import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

ticker = 'GXU.V'
start_date = '2015-01-01'
end_date = '2023-12-31'

df = yf.download(ticker, start=start_date, end=end_date)

df.to_csv("stock_data.csv")

data = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

pd.DataFrame(scaled_data, index=df.index, columns=["Scaled_Close"]).to_csv("scaled_stock_data.csv")
