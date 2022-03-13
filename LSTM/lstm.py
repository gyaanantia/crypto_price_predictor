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

class CryptoCurrencyPricePredictor():
    def __init__(self,window_len=5,test_size=0.2,zero_base=True,lstm_neurons=100,epochs=20,batch_size=32,loss='mse',dropout=0.2,optimizer='adam'):
        np.random.seed(42)
        self.window_len = window_len
        self.test_size = test_size
        self.zero_base = zero_base
        self.lstm_neurons = lstm_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.dropout = dropout
        self.optimizer = optimizer

        #model storage data
        self.hist = None
        self.high_col,self.low_col,self.open_col,self.volumefrom_col,self.volumeto_col,self.close_col = \
            None, None, None, None, None, None
        self.train_high, self.test_high, self.X_train_high, self.X_test_high, self.Y_train_high, self.Y_test_high = \
            None, None, None, None, None, None
        self.train_low, self.test_low, self.X_train_low, self.X_test_low, self.Y_train_low, self.Y_test_low = \
            None, None, None, None, None, None
        self.train_open, self.test_open, self.X_train_open, self.X_test_open, self.Y_train_open, self.Y_test_open = \
            None, None, None, None, None, None
        self.train_volumefrom, self.test_volumefrom, self.X_train_volumefrom, self.X_test_volumefrom, self.Y_train_volumefrom, self.Y_test_volumefrom = \
            None, None, None, None, None, None
        self.train_volumeto, self.test_volumeto, self.X_train_volumeto, self.X_test_volumeto, self.Y_train_volumeto, self.Y_test_volumeto = \
            None, None, None, None, None, None
        self.train_close, self.test_close, self.X_train_close, self.X_test_close, self.Y_train_close, self.Y_test_close = \
            None, None, None, None, None, None
        self.model_high, self.model_low,self.model_open,self.model_close,self.model_volumefrom, self.model_volumeto = \
            None, None, None, None, None, None

    def get_data(self):
        endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=500'
        api_key = '2b967442562fe24c326388ed0d91d7dd9dab4fbbbcbf929bdc71e5c5fe56f9b3'
        res = requests.get(endpoint + '&api_key={your_api_key}')
        self.hist = pd.DataFrame(json.loads(res.content)['Data']['Data'])
        self.hist = self.hist.set_index('time')
        self.hist.index = pd.to_datetime(self.hist.index, unit='s')

        self.high_col,self.low_col,self.open_col,self.volumefrom_col,self.volumeto_col,self.close_col = \
        'high', 'low', 'open', 'volumefrom','volumeto','close'
        self.hist.drop("conversionType", axis=1, inplace=True)
        self.hist.drop("conversionSymbol", axis=1, inplace=True)

    def train_test_split(self,df, test_size):
        split_row = len(df) - int(test_size * len(df))
        train_data = df.iloc[:split_row]
        test_data = df.iloc[split_row:]
        return train_data, test_data


    def line_plot(self,line1, line2, label1='train', label2='test',title='', lw=2):
        fig, ax = plt.subplots(1, figsize=(13, 7))
        ax.plot(line1, label=label1, linewidth=lw)
        ax.plot(line2, label=label2, linewidth=lw)
        ax.set_xlabel('time[year, month]', fontsize=14)
        ax.set_ylabel('price [USD]', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', fontsize=16)

    def normalise_zero_base(self,df):
        return df / df.iloc[0] - 1

    def normalise_min_max(self,df):
        return (df - df.min()) / (df.max() - df.min())


    def extract_window_data(self,df, window_len=5, zero_base=True):
        window_data = []
        for idx in range(len(df) - window_len):
            tmp = df[idx: (idx + window_len)].copy()
            if zero_base:
                tmp = self.normalise_zero_base(tmp)
            window_data.append(tmp.values)
        return np.array(window_data)

    def extract_window_data_for_prediction(self,df, window_len=5, zero_base=True):
        window_data = []
        for idx in range(len(df) - window_len + 1):
            tmp = df[idx: (idx + window_len)].copy()
            if  zero_base:
                tmp = self.normalise_zero_base(tmp)
            window_data.append(tmp.values)
        return np.array(window_data)

    def prepare_data(self,df, target_col, window_len=10, zero_base=True, test_size=0.2):
        train_data, test_data = self.train_test_split(df, test_size=test_size)
        X_train = self.extract_window_data(train_data, window_len, zero_base)
        X_test = self.extract_window_data(test_data, window_len, zero_base)
        Y_train = train_data[target_col][window_len:].values
        Y_test = test_data[target_col][window_len:].values
        if zero_base:
            Y_train = Y_train / train_data[target_col][:-window_len].values - 1
            Y_test = Y_test / test_data[target_col][:-window_len].values - 1

        return train_data, test_data, X_train, X_test, Y_train, Y_test


    def build_lstm_model(self,input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def save_data(self):
        self.train_high, self.test_high, self.X_train_high, self.X_test_high, self.Y_train_high, self.Y_test_high = \
            self.prepare_data(self.hist, self.high_col, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)
        self.train_low, self.test_low, self.X_train_low, self.X_test_low, self.Y_train_low, self.Y_test_low = \
            self.prepare_data(self.hist, self.low_col, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)
        self.train_open, self.test_open, self.X_train_open, self.X_test_open, self.Y_train_open, self.Y_test_open = \
            self.prepare_data(self.hist, self.open_col, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)
        self.train_volumefrom, self.test_volumefrom, self.X_train_volumefrom, self.X_test_volumefrom, self.Y_train_volumefrom, self.Y_test_volumefrom = \
            self.prepare_data(self.hist, self.volumefrom_col, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)
        self.train_volumeto, self.test_volumeto, self.X_train_volumeto, self.X_test_volumeto, self.Y_train_volumeto, self.Y_test_volumeto = \
            self.prepare_data(self.hist, self.volumeto_col, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)
        self.train_close, self.test_close, self.X_train_close, self.X_test_close, self.Y_train_close, self.Y_test_close = \
            self.prepare_data(self.hist, self.close_col, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)

    def build_and_train_model(self):
        self.get_data()
        self.save_data()

        self.model_high= self.build_lstm_model(
            self.X_train_high, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)
        self.model_low = self.build_lstm_model(
            self.X_train_low, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)
        self.model_open = self.build_lstm_model(
            self.X_train_open, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)
        self.model_volumefrom = self.build_lstm_model(
            self.X_train_volumefrom, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)
        self.model_volumeto = self.build_lstm_model(
            self.X_train_volumeto, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)
        self.model_close = self.build_lstm_model(
            self.X_train_close, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)

        #global history_close,history_high,history_low,history_close,history_volumefrom,history_volumeto
        self.model_high.fit(self.X_train_high, self.Y_train_high, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        self.model_low.fit(self.X_train_low, self.Y_train_low, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        self.model_open.fit(self.X_train_open, self.Y_train_open, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        self.model_volumefrom.fit(self.X_train_volumefrom, self.Y_train_volumefrom, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        self.model_volumeto.fit(self.X_train_volumeto, self.Y_train_volumeto, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        self.model_close.fit(self.X_train_close, self.Y_train_close, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)

    def test_models(self):
        #test model_high
        preds = self.model_high.predict(self.X_test_high).squeeze()
        targets = self.test_high[self.high_col][self.window_len:]
        preds = self.test_high[self.high_col].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        self.line_plot(targets, preds, 'actual high', 'predicted high', lw=3)

        #test model_low
        preds = self.model_low.predict(self.X_test_low).squeeze()
        targets = self.test_low[self.low_col][self.window_len:]
        preds = self.test_low[self.low_col].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        self.line_plot(targets, preds, 'actual low', 'predicted low', lw=3)

        #test model_open
        preds = self.model_open.predict(self.X_test_open).squeeze()
        targets = self.test_open[self.open_col][self.window_len:]
        preds = self.test_open[self.open_col].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        self.line_plot(targets, preds, 'actual open', 'predicted open', lw=3)

        #test model_volumefrom
        preds =self.vmodel_volumefrom.predict(self.X_test_volumefrom).squeeze()
        targets = self.test_volumefrom[self.volumefrom_col][self.window_len:]
        preds = self.test_volumefrom[self.volumefrom_col].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        self.line_plot(targets, preds, 'actual volumefrom', 'predicted volumefrom', lw=3)

        #test model_volumeto
        preds = self.model_volumeto.predict(self.X_test_volumeto).squeeze()
        targets = self.test_volumeto[self.volumeto_col][self.window_len:]
        preds = self.test_volumeto[self.volumeto_col].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        self.line_plot(targets, preds, 'actual volumeto', 'predicted volumeto', lw=3)

        #test model_close
        preds = self.model_close.predict(self.X_test_close).squeeze()
        targets = self.test_close[self.close_col][self.window_len:]
        preds = self.test_close[self.close_col].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        self.line_plot(targets, preds, 'actual', 'prediction', lw=3)


    def predict(self,days):
        days_count = 0
        price_predictions = []
        while days_count < days:
            #prediction for high
            test1 = self.test_high[-self.window_len:]
            test1 = self.extract_window_data_for_prediction(test1, self.window_len, True)
            pred = self.model_high.predict(test1)
            mean = self.test_high[self.high_col].values[-5:].squeeze().mean()
            pred1 = mean * (pred + 1)

            #prediction for low
            test1 = self.test_low[-self.window_len:]
            test1 = self.extract_window_data_for_prediction(test1, self.window_len, True)
            pred = self.model_low.predict(test1)
            mean = self.test_low[self.low_col].values[-5:].squeeze().mean()
            pred2 = mean * (pred + 1)

            #prediction for open
            test1 = self.test_open[-self.window_len:]
            test1 = self.extract_window_data_for_prediction(test1, self.window_len, True)
            pred = self.model_open.predict(test1)
            mean = self.test_open[self.open_col].values[-5:].squeeze().mean()
            pred3 = mean * (pred + 1)

            #prediction for volumefrom
            test1 = self.test_volumefrom[-self.window_len:]
            test1 = self.extract_window_data_for_prediction(test1, self.window_len, True)
            pred = self.model_volumefrom.predict(test1)
            mean = self.test_volumefrom[self.volumefrom_col].values[-5:].squeeze().mean()
            pred4 = mean * (pred + 1)

            #prediction for volumeto
            test1 = self.test_volumeto[-self.window_len:]
            test1 = self.extract_window_data_for_prediction(test1, self.window_len, True)
            pred = self.model_volumeto.predict(test1)
            mean = self.test_volumeto[self.volumeto_col].values[-5:].squeeze().mean()
            pred5 = mean * (pred + 1)

            #prediction for close price
            test1 = self.test_close[-self.window_len:]
            test1 = self.extract_window_data_for_prediction(test1, self.window_len, True)
            pred = self.model_close.predict(test1)
            mean = self.test_close[self.close_col].values[-5:].squeeze().mean()
            pred6 = mean * (pred + 1)

            price_predictions.append(pred6)

            # extend data frame with these predictions to allow more predictions
            new_day = self.hist.index[-1]+pd.Timedelta(days = 1)
            df_new = pd.DataFrame([[new_day,pred1[0][0],pred2[0][0],pred3[0][0],pred4[0][0],pred5[0][0],pred6[0][0]]],columns= ['time','high','low','open','volumefrom','volumeto','close'])
            df_new = df_new.set_index('time')
            self.hist = self.hist.append(df_new)

            self.save_data()

            days_count += 1

        price_predictions = (np.array(price_predictions)).squeeze()
        return price_predictions


if __name__ == '__main__':

    CPP = CryptoCurrencyPricePredictor()
    CPP.build_and_train_model()
    days = 10
    price_predictions = CPP.predict(days)
    print(price_predictions)
    CPP.line_plot(CPP.hist['close'][:-days],CPP.hist['close'][-days:], label1='actual prices',label2='predicted prices')


