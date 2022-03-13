import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import io
import base64


class Crypto_Model:

    def __init__(self, df, target_col, window_len=10, zero_base=True, test_size=0.2):
        self.df = df
        self.target_col = target_col
        self.window_len = window_len
        self.zero_base = zero_base
        self.test_size = 0.2
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.prepare_data()
        self.model = build_lstm_model(self.X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)


    def train_test_split(self):
        split_row = len(self.df) - int(self.test_size * len(self.df))
        self.train_data = df.iloc[:split_row]
        self.test_data = df.iloc[split_row:]


    def line_plot(self,line1, line2, label1=None, label2=None, title='', lw=2):
        fig, ax = plt.subplots(1, figsize=(13, 7))
        ax.plot(line1, label=label1, linewidth=lw)
        ax.plot(line2, label=label2, linewidth=lw)
        ax.set_ylabel('price [CAD]', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        datalink = f"data:image/png;base64,{data}"
        return datalink


        def normalise_zero_base(self,df):
            window_df = np.array(df.values[:,0:6])
            max_df = np.array(df.iloc[0].values[0:6])
            return window_df / max_df

        def normalise_min_max(self,df):
            return (df - df.min()) / (df.max() - df.min())

        def extract_window_data(self,df, window_len=5, zero_base=True):
            window_data = []
            for idx in range(len(df) - window_len):
                tmp = df[idx: (idx + window_len)].copy()
                if zero_base:
                    tmp = normalise_zero_base(tmp)
                window_data.append(tmp)
            return np.array(window_data)

        def prepare_data(self):
            self.train_test_split()
            self.X_train = self.extract_window_data(self.train_data, self.window_len, self.zero_base)
            self.X_test = self.extract_window_data(self.test_data, self.window_len, self.zero_base)
            self.y_train = self.train_data[self.target_col][self.window_len:].values
            self.y_test = self.test_data[self.target_col][self.window_len:].values
            if self.zero_base:
                self.y_train = self.y_train / self.train_data[self.target_col][:-self.window_len].values - 1
                self.y_test =self. y_test / self.test_data[self.target_col][:-self.window_len].values - 1
            return

        def build_lstm_model(self,input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
            model.add(Dropout(dropout))
            model.add(Dense(units=output_size))
            model.add(Activation(activ_func))
            model.compile(loss=loss, optimizer=optimizer)
            return model


endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=CAD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'

hist.head(5)

# def train_test_split(df, test_size=0.2):
#     split_row = len(df) - int(test_size * len(df))
#     train_data = df.iloc[:split_row]
#     test_data = df.iloc[split_row:]
#     return train_data, test_data
train, test = train_test_split(hist, test_size=0.2)

# def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
#     fig, ax = plt.subplots(1, figsize=(13, 7))
#     ax.plot(line1, label=label1, linewidth=lw)
#     ax.plot(line2, label=label2, linewidth=lw)
#     ax.set_ylabel('price [CAD]', fontsize=14)
#     ax.set_title(title, fontsize=16)
#     ax.legend(loc='best', fontsize=16)
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     # Embed the result in the html output.
#     data = base64.b64encode(buf.getbuffer()).decode("ascii")
#     datalink = f"data:image/png;base64,{data}"
#     return datalink

line_plot(train[target_col], test[target_col], 'training', 'test', title='')

# def normalise_zero_base(df):
#     window_df = np.array(df.values[:,0:6])
#     max_df = np.array(df.iloc[0].values[0:6])
#     return window_df / max_df
#
# def normalise_min_max(df):
#     return (df - df.min()) / (df.max() - df.min())
#
# def extract_window_data(df, window_len=5, zero_base=True):
#     window_data = []
#     for idx in range(len(df) - window_len):
#         tmp = df[idx: (idx + window_len)].copy()
#         if zero_base:
#             tmp = normalise_zero_base(tmp)
#         window_data.append(tmp)
#     return np.array(window_data)
#
# def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
#     train_data, test_data = train_test_split(df, test_size=test_size)
#     X_train = extract_window_data(train_data, window_len, zero_base)
#     X_test = extract_window_data(test_data, window_len, zero_base)
#     y_train = train_data[target_col][window_len:].values
#     y_test = test_data[target_col][window_len:].values
#     if zero_base:
#         y_train = y_train / train_data[target_col][:-window_len].values - 1
#         y_test = y_test / test_data[target_col][:-window_len].values - 1
#
#     return train_data, test_data, X_train, X_test, y_train, y_test
#
# def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
#     model = Sequential()
#     model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
#     model.add(Dropout(dropout))
#     model.add(Dense(units=output_size))
#     model.add(Activation(activ_func))
#     model.compile(loss=loss, optimizer=optimizer)
#     return model

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

train, test, X_train, X_test, y_train, y_test = prepare_data(hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
history = model.fit(np.asarray(X_train).astype('float32'), np.asarray(y_train).astype('float32'), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

targets = test[target_col][window_len:]
preds = model.predict(np.asarray(X_test).astype('float32')).squeeze()
mean_absolute_error(preds, y_test)

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
# line_plot(targets, preds, 'actual', 'prediction', lw=3)
