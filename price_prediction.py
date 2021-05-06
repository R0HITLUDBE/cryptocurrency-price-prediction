import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

st.title('Cryptocurrency price prediction')

currency = ["Bitcoin", "Dogecoin","Ethereum","Litecoin","Tether","Binance coin","other"]
crypto = st.radio('Select crytocurreny',currency)
 
if crypto == "other":
  val = st.text_input("enter ticker symbol")
elif crypto == "Bitcoin":
  val = "BTC"
elif crypto == "Dogecoin":
  val = "DOGE"
elif crypto == "Ethereum":
  val = "ETH"
elif crypto == "Binance coin":
  val = "LTC"
elif crypto == "Tether":
  val = "USDT"
elif crypto == "Litecoin":
  val = "LTC"
try:
  crypto_currency = val
  against_currency = 'USD'

  start = dt.datetime(2020,1,1)
  end = dt.datetime(2021,5,1)

  data = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo', start, end)
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
  print(data.shape)
  prediction_days = 60

  x_train, y_train = [], []

  for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  mod = Sequential()

  mod.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  mod.add(Dropout(0.2))
  mod.add(LSTM(units=50, return_sequences=True))
  mod.add(Dropout(0.2))
  mod.add(LSTM(units=50))
  mod.add(Dropout(0.2))
  mod.add(Dense(units=1))

  mod.compile(optimizer='adam', loss='mean_squared_error')
  mod.fit(x_train, y_train, epochs=25, batch_size=32)

  test_start = dt.datetime(2020,1,1)
  test_end = dt.datetime.now()

  test_data = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo', test_start, test_end)
  actual_prices = test_data['Close'].values

  total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
  model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
  model_inputs = model_inputs.reshape(-1,1)
  model_inputs = scaler.fit_transform(model_inputs)
  x_test = []

  for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  prediction_prices = mod.predict(x_test)
  prediction_prices = scaler.inverse_transform(prediction_prices)
  st.line_chart(actual_prices)
  st.line_chart(prediction_prices)

  real_data=[model_inputs[len(model_inputs)+ 1 - prediction_days:len(model_inputs) - 1,0]]
  real_data=np.array(real_data)
  real_data=np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

  prediction=mod.predict(real_data)
  prediction=scaler.inverse_transform(prediction)
  prediction_val = "Next day's prediction is " + str(prediction[0][0])
  st.text(prediction_val)

except:
  pass
