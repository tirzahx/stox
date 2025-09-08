import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

ticker = 'GXU.V'

df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
data = df['Close']

train_data = data.iloc[:int(len(data)*0.8)]
test_data = data.iloc[int(len(data)*0.8):]

model = auto_arima(train_data, 
                   start_p=1, start_q=1, 
                   max_p=5, max_q=5, 
                   m=1, 
                   d=None, 
                   seasonal=False, 
                   trace=True, 
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

print(model.summary())

raw_predictions = model.predict(n_periods=len(test_data))
predictions = pd.Series(raw_predictions, index=test_data.index)

rmse = math.sqrt(mean_squared_error(test_data, predictions))
mae = mean_absolute_error(test_data, predictions)
print(f"ARIMA RMSE: {rmse:.2f}, MAE: {mae:.2f}")

plt.figure(figsize=(14, 7))
plt.plot(train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Actual Prices', color='blue')
plt.plot(predictions.index, predictions, label='Predicted Prices', color='red', linestyle='--')
plt.title(f'{ticker} Stock Price Prediction (ARIMA Model)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
