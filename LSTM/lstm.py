from pyexpat import model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import sys


np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'


def get_data():
    endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=500'
    api_key = '2b967442562fe24c326388ed0d91d7dd9dab4fbbbcbf929bdc71e5c5fe56f9b3'
    res = requests.get(endpoint + '&api_key={your_api_key}')

    global hist, high_col,low_col,open_col,volumefrom_col,volumeto_col,close_col
    hist = pd.DataFrame(json.loads(res.content)['Data']['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')

    high_col = 'high'
    low_col = 'low'
    open_col = 'open'
    volumefrom_col = 'volumefrom'
    volumeto_col = 'volumeto'
    close_col = 'close'
    hist.drop("conversionType", axis=1, inplace=True)
    hist.drop("conversionSymbol", axis=1, inplace=True)

def train_test_split(df, test_size):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


def line_plot(line1, line2, label1='train', label2='test',title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_xlabel('time[year, month]', fontsize=14)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (df.max() - df.min())


def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def extract_window_data_for_prediction(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len + 1):
        tmp = df[idx: (idx + window_len)].copy()
        if  zero_base:
            tmp = normalise_zero_base(tmp)
            #tmp = scaler.fit_transform(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    Y_train = train_data[target_col][window_len:].values
    Y_test = test_data[target_col][window_len:].values
    if zero_base:
        Y_train = Y_train / train_data[target_col][:-window_len].values - 1
        Y_test = Y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, Y_train, Y_test


def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def save_data():
    global train_high, test_high, X_train_high, X_test_high, Y_train_high, Y_test_high 
    global train_low, test_low, X_train_low, X_test_low, Y_train_low, Y_test_low
    global train_open, test_open, X_train_open, X_test_open, Y_train_open, Y_test_open
    global train_volumefrom, test_volumefrom, X_train_volumefrom, X_test_volumefrom, Y_train_volumefrom, Y_test_volumefrom
    global train_volumeto, test_volumeto, X_train_volumeto, X_test_volumeto, Y_train_volumeto, Y_test_volumeto
    global train_close, test_close, X_train_close, X_test_close, Y_train_close, Y_test_close
    
    train_high, test_high, X_train_high, X_test_high, Y_train_high, Y_test_high = prepare_data(
        hist, high_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    train_low, test_low, X_train_low, X_test_low, Y_train_low, Y_test_low = prepare_data(
        hist, low_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    train_open, test_open, X_train_open, X_test_open, Y_train_open, Y_test_open = prepare_data(
        hist, open_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    train_volumefrom, test_volumefrom, X_train_volumefrom, X_test_volumefrom, Y_train_volumefrom, Y_test_volumefrom = prepare_data(
        hist, volumefrom_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    train_volumeto, test_volumeto, X_train_volumeto, X_test_volumeto, Y_train_volumeto, Y_test_volumeto = prepare_data(
        hist, volumeto_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    train_close, test_close, X_train_close, X_test_close, Y_train_close, Y_test_close = prepare_data(
        hist, close_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

def build_and_train_model():
    get_data()
    save_data()
    global model_high, model_low,model_open,model_close,model_volumefrom, model_volumeto
    model_high= build_lstm_model(
        X_train_high, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    model_low = build_lstm_model(
        X_train_low, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    model_open = build_lstm_model(
        X_train_open, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    model_volumefrom = build_lstm_model(
        X_train_volumefrom, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    model_volumeto = build_lstm_model(
        X_train_volumeto, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    model_close = build_lstm_model(
        X_train_close, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)

    #global history_close,history_high,history_low,history_close,history_volumefrom,history_volumeto
    model_high.fit(X_train_high, Y_train_high, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    model_low.fit(X_train_low, Y_train_low, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    model_open.fit(X_train_open, Y_train_open, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    model_volumefrom.fit(X_train_volumefrom, Y_train_volumefrom, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    model_volumeto.fit(X_train_volumeto, Y_train_volumeto, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    model_close.fit(X_train_close, Y_train_close, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

def test_models():
    #test model_high
    preds = model_high.predict(X_test_high).squeeze()
    targets = test_high[high_col][window_len:]
    preds = test_high[high_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual high', 'predicted high', lw=3)

    #test model_low
    preds = model_low.predict(X_test_low).squeeze()
    targets = test_low[low_col][window_len:]
    preds = test_low[low_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual low', 'predicted low', lw=3)

    #test model_open
    preds = model_open.predict(X_test_open).squeeze()
    targets = test_open[open_col][window_len:]
    preds = test_open[open_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual open', 'predicted open', lw=3)

    #test model_volumefrom
    preds = model_volumefrom.predict(X_test_volumefrom).squeeze()
    targets = test_volumefrom[volumefrom_col][window_len:]
    preds = test_volumefrom[volumefrom_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual volumefrom', 'predicted volumefrom', lw=3)

    #test model_volumeto
    preds = model_volumeto.predict(X_test_volumeto).squeeze()
    targets = test_volumeto[volumeto_col][window_len:]
    preds = test_volumeto[volumeto_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual volumeto', 'predicted volumeto', lw=3)

    #test model_close
    preds = model_close.predict(X_test_close).squeeze()
    targets = test_close[close_col][window_len:]
    preds = test_close[close_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual', 'prediction', lw=3)


def predict(days):
    days_count = 0
    price_predictions = []
    while days_count < days:
        #prediction for high
        test1 = test_high[-window_len:]
        test1 = extract_window_data_for_prediction(test1, window_len, True)
        pred = model_high.predict(test1)
        mean = test_high[high_col].values[-5:].squeeze().mean()
        pred1 = mean * (pred + 1)

        #prediction for low
        test1 = test_low[-window_len:]
        test1 = extract_window_data_for_prediction(test1, window_len, True)
        pred = model_low.predict(test1)
        mean = test_low[low_col].values[-5:].squeeze().mean()
        pred2 = mean * (pred + 1)

        #prediction for open
        test1 = test_open[-window_len:]
        test1 = extract_window_data_for_prediction(test1, window_len, True)
        pred = model_open.predict(test1)
        mean = test_open[open_col].values[-5:].squeeze().mean()
        pred3 = mean * (pred + 1)

        #prediction for volumefrom
        test1 = test_volumefrom[-window_len:]
        test1 = extract_window_data_for_prediction(test1, window_len, True)
        pred = model_volumefrom.predict(test1)
        mean = test_volumefrom[volumefrom_col].values[-5:].squeeze().mean()
        pred4 = mean * (pred + 1)

        #prediction for volumeto
        test1 = test_volumeto[-window_len:]
        test1 = extract_window_data_for_prediction(test1, window_len, True)
        pred = model_volumeto.predict(test1)
        mean = test_volumeto[volumeto_col].values[-5:].squeeze().mean()
        pred5 = mean * (pred + 1)

        #prediction for close price
        test1 = test_close[-window_len:]
        test1 = extract_window_data_for_prediction(test1, window_len, True)
        pred = model_close.predict(test1)
        mean = test_close[close_col].values[-5:].squeeze().mean()
        pred6 = mean * (pred + 1)

        price_predictions.append(pred6)

        # extend data frame with these predictions to allow more predictions
        new_day = hist.index[-1]+pd.Timedelta(days = 1)
        df_new = pd.DataFrame([[new_day,pred1[0][0],pred2[0][0],pred3[0][0],pred4[0][0],pred5[0][0],pred6[0][0]]],columns= ['time','high','low','open','volumefrom','volumeto','close'])
        df_new = df_new.set_index('time')
        hist.append(df_new)

        save_data()

        days_count += 1

    price_predictions = (np.array(price_predictions)).squeeze()
    return price_predictions


if __name__ == '__main__':
    build_and_train_model()
    days = 10
    price_predictions = predict(days)
    print(price_predictions)
    line_plot(hist['close'][:-days],hist['close'][-days:], label1='actual prices',label2='predicted prices')


