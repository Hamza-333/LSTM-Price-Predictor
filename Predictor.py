import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yahoo_fin.stock_info as yf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
import tensorflow as tf
import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import datetime
SEQ_LEN = 30
# Pull data from yahoo finance and save to csv

TICKER = 'AAPL'
# data = yf.get_data(TICKER,start_date = '14/06/2014', end_date='15/06/2021')
# data.to_csv(TICKER + '.csv')


df = pd.read_csv(TICKER + '.csv')
df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
#train on 80% of data, test on 20% 
TRAIN_RANGE = round(len(df.index) * 0.8)

# Plot the adjusted close price
ax = plt.gca()
plt.xticks(rotation=45)

plt.plot(df['Date'], df["close"], label = 'Close (Real)')
x_min, x_max = ax.get_xlim()
ax.set_xticks(np.linspace(x_min, x_max, 10))

# df['Date'] = pd.to_numeric(pd.to_datetime(df['Date']))
filtered_df = pd.DataFrame(index = range(0, len(df)), columns = ['Date', 'Adj_Close'])

df.dropna(subset = ['adjclose'], inplace = True)


def Scale_Data():
 
    scaler = MinMaxScaler(feature_range = (0,1))
    df.drop("Date", axis=1, inplace=True)
    df.drop("ticker", axis=1, inplace=True)
    dataset = df.values
    
    train_data = dataset[0:TRAIN_RANGE, :] 
    test_data = dataset[TRAIN_RANGE:, :]
  
    # Scale the data between 0 and 1
    scaled_data = scaler.fit_transform(dataset)

    # Create sequences of SEQ_LEN
    # Append current sequence to x_train
    # Append current adj close price to y_train
    x_train, y_train = [], []
    for i in range(SEQ_LEN, len(train_data)):
        x_train.append(scaled_data[i-SEQ_LEN:i, :])
        y_train.append(scaled_data[i, 3])
  
    # Convert to numpy array and shape the data
    x_train, y_train = np.array(x_train),np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 6))

    return x_train, y_train, test_data

def Build_Model():
    model = Sequential()
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    x_train, y_train, test_data = Scale_Data()

    model.add(LSTM(75, return_sequences=True, input_shape = (x_train.shape[1], 6)))
    model.add(Dropout(0.2))

    model.add(LSTM(75, input_shape = (x_train.shape[1], 6)))
    model.add(Dropout(0.2))

    model.add(Dense(6))

    input_data = df[len(df) - len(test_data)-SEQ_LEN:].values
    input_data = input_data.reshape(-6,6)
    input_data = scaler.fit_transform(input_data)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt)

    model.fit(x_train, y_train, epochs=30,batch_size = 32)
    

    X_test = []
    for i in range(SEQ_LEN, input_data.shape[0]):
        X_test.append(input_data[i-SEQ_LEN : i, :])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))

    # Predict and convert back to price data
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    model.save('lstm.h5')

    return predicted_price

def Analysis():
    predicted_price = Build_Model()

    train_data = df[:TRAIN_RANGE]
    test_data = df[TRAIN_RANGE:]

    test_data['Predicted'] = predicted_price[:, 3]
    test_data.to_csv('test.csv')
    plt.plot(train_data['close'], "-r", label = 'Train')
    plt.plot(test_data['close'], '-g', label = 'Test')
    plt.plot(test_data['Predicted'], '-b', label = 'Close (Predicted)')
    ax.legend()
    plt.show()


Analysis()


