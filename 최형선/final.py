import os
import datetime
import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, Sequential
from keras.layers import RNN
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.layers.core import Activation, Dense, Dropout
from keras import layers, models
from sklearn import preprocessing
from keras import datasets
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import model_to_dot
from sklearn import model_selection

print("Predict")
forecasting=20
def kalman_filter(z_meas, x_esti, P):
    A = 1
    H = 1
    Q = 0.00001
    R = 0.001


    """Kalman Filter Algorithm for One Variable."""
    # (1) Prediction.
    x_pred = A * x_esti
    P_pred = A * P * A + Q

    # (2) Kalman Gain.
    K = P_pred * H / (H * P_pred * H + R)

    # (3) Estimation.
    x_esti = x_pred + K * (z_meas - H * x_pred)

    # (4) Error Covariance.
    P = P_pred - K * H * P_pred

    return x_esti, P

def make_dataset(data, label, window_size=forecasting):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size-20):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size+20]))
    return np.array(feature_list), np.array(label_list)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))




final_train_feature=[]
final_train_label=[]
for _ in range(1,40):
  # 학습시킬 Cell 호출
  bring_cell='Bank_cell/'+str(_)+'_cell.csv'
  df = pd.read_csv(bring_cell, encoding='utf8')
  df = df['ResistValue']
  # 칼만 필터 적용하여 학습
  n_samples=len(df)
  x_esti, P = None, None
  esti_save = np.zeros(n_samples)
  x_0 = df[0]
  P_0 = 1
  for i in range(n_samples):
      if i == 0:
          x_esti, P = x_0, P_0
      else:
          x_esti, P = kalman_filter(df[i], x_esti, P)
      esti_save[i] = x_esti
  x_train = esti_save
  y_train = esti_save
  x_train = pd.DataFrame(x_train)
  y_train = pd.DataFrame(y_train)

  # fitting
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  x_train = scaler.fit_transform(x_train)
  y_train = scaler.fit_transform(y_train)
  x_train = pd.DataFrame(x_train)
  y_train = pd.DataFrame(y_train)

  # make_dataset
  lookback = 20
  train_feature, train_label = make_dataset(x_train, y_train, lookback)
  if len(final_train_feature)==0:
    final_train_feature=train_feature
    final_train_label=train_label
  else:
#    final_train_feature=np.array(final_train_feature)
#    final_train_label=np.array(final_train_label)
    final_train_feature = np.concatenate((final_train_feature, train_feature),axis=0)
    final_train_label = np.concatenate((final_train_label, train_label),axis=0)

model = Sequential()
# model.add(LSTM(30,
#                input_shape=(final_train_feature.shape[1], final_train_feature.shape[2]),
#                activation='relu',
#                return_sequences=True)
#           )
# model.add(Dropout(0.1))
# model.add(Dense(1))

for i in range(3):
    model.add(LSTM(30, batch_input_shape=(1, lookback, 1),
              stateful=True, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))
    model.add(Dropout(0.3))
model.add(Dense(lookback))


# 모델 학습
custom_hist = CustomHistory()
custom_hist.init()
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

for i in range(50):
    model.fit(final_train_feature, final_train_label, epochs=1, batch_size=20, shuffle=False,
              callbacks=[custom_hist, early_stopping], validation_data=(final_train_feature, final_train_label))

    model.reset_states()

model.save('./models/finalTest.h5')




'''-------------------------------'''
