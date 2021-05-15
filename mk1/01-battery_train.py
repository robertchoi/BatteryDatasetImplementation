from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

features_num = 3
num = 280

Train_dataset = np.loadtxt('EXP-I_20121008_0006_0009_20120426061623_cell.csv', delimiter=",", skiprows=1)
X_train = Train_dataset[:num, 0:features_num]
Y_train = Train_dataset[:num, features_num]

X_test = Train_dataset[num:331, 0:features_num]
Y_test = Train_dataset[num:331, features_num]




def make_dataset(data, label, window_size=2):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data[i:i+window_size]))
        label_list.append(np.array(label[i+window_size]))
    return np.array(feature_list), np.array(label_list)

# train dataset (실제 예측 해볼 데이터)
train_feature, train_label = make_dataset(X_train, Y_train, 2)


# test dataset (실제 예측 해볼 데이터)
test_feature, test_label = make_dataset(X_test, Y_test, 2)
test_feature.shape, test_label.shape
print(test_feature.shape, test_label.shape)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
print(x_train.shape, x_valid.shape)


model = Sequential()
model.add(LSTM(16,
               input_shape=(train_feature.shape[1], train_feature.shape[2]),
               activation='relu',
               return_sequences=False)
          )
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)
filename = os.path.join('C:/Users/WBH/PycharmProjects/BATTERY', 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train,
                                    epochs=200,
                                    batch_size=5,
                                    validation_data=(x_valid, y_valid),
                                    callbacks=[early_stop, checkpoint])


model.load_weights(filename)
pred = model.predict(test_feature)
print(pred)

model.summary()

plt.figure(figsize=(12, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
