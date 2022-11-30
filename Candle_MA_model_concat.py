import keras.layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras.backend as K
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
candles_data = 60      # 사용할 분봉 데이터
candles_window = 100     # 데이터 프레임에 들어갈 캔들 개수 - 20 선택시 20+20=40개의 input
MA_A = 5    # 첫번째 MA - 5, 10, 20, 60, 120
MA_B = 10   # 두번째 MA - 5, 10, 20, 60, 120

# 엑셀
xl = pd.read_table('./'+str(candles_data)+'candles.txt',
                   names=['date', 'price', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'per'])

ma1 = xl.loc[:, 'ma'+str(MA_A)]
ma1 = np.array(ma1)
ma1 = ma1.reshape(-1, 1)
ma1_mean = ma1.min()
ma1_std = ma1.std()
ma1 = (ma1-ma1_mean)/ma1_std

x_data = []
y_data = []
for i in range(0, len(xl.index)-candles_window):
    x_trainblock = []
    for j in range(i, i+candles_window):
        x_trainblock.append(ma1[i])
    x_data.append(x_trainblock)

    """y_data.append(xl.loc[i+19, 'per'])"""
    if xl.loc[i+19, 'per'] >= 0:
        y_data.append([0, 1])
    elif xl.loc[i+19, 'per'] < 0:
        y_data.append([1, 0])


x_data = np.array(x_data)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2, random_state=777)

print(x_train.shape)
print(x_train)
print(y_train.shape)
print(y_train)

K.clear_session()
model = Sequential() # Sequeatial Model
model.add(LSTM(100, input_shape=(candles_window, 1), return_sequences=True)) # (timestep, feature)
model.add(LSTM(40, input_shape=(100, 1))) # (timestep, feature)
model.add(Dense(2, activation='softmax')) # output = 1
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

"""inputs = tf.keras.Input(shape=(10336, 100,))
x = keras.layers.LSTM(20, input_shape=(candles_window, 1))
output = Dense(1)(x)"""

model.summary()


early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(x_train, y_train, epochs=100,
          batch_size=30, validation_split=0.2, callbacks=[early_stop])
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

y_pred = model.predict(x_test)
y_pred = np.array(y_pred)

#print(y_pred)
for i in y_pred:
    print(np.argmax(i))
    print(i[np.argmax(i)])
print(np.array(x_test).shape)
print(y_pred.shape)
print(y_pred.max())
print(y_pred.min())