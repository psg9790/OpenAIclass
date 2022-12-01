import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
candles_data = 5      # 사용할 분봉 데이터
target_MA_1 = 5
target_MA_2 = 10

# 엑셀
xl = pd.read_table('./'+str(candles_data)+'candles.txt',
                   names=['date', 'price', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'per'])

"""============================================="""
# MA 두개 증감률 계산
MA = []
MA = xl.loc[:,'ma'+str(target_MA_1)]
MA_1 = []
for i in range(len(MA)):
    if i != len(MA)-1:
        tooken = ((MA[i+1] - MA[i])/MA[i])*100
        MA_1.append(tooken)

MA_1 = np.array(MA_1)

MA = xl.loc[:,'ma'+str(target_MA_2)]
MA_2 = []
for i in range(len(MA)):
    if i != len(MA)-1:
        tooken = ((MA[i+1] - MA[i])/MA[i])*100
        MA_2.append(tooken)
"""============================================="""
# data 생성
x_data = []
for i in range(len(MA_1) - 20):
    window = []
    for j in range(i, i+20):
        window.append([MA_1[j], MA_2[j]])
    x_data.append(window)

x_data = np.array(x_data)

per = xl.loc[:,'per']
y_data = []
for i in range(len(MA_1) - 20):
    if per[i] >= 0:
        y_data.append([1])
    else:
        y_data.append([0])

y_data = np.array(y_data)
print('x_data shape: ' + str(x_data.shape))
print('y_data shape: ' + str(y_data.shape))
"""============================================="""
# model
inputs = keras.Input(shape=(20,2))
lstm_layer = tf.keras.layers.LSTM(10)(inputs)
x = tf.keras.layers.Dense(10, activation='relu')(lstm_layer)
x = tf.keras.layers.Dense(10, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
"""============================================="""
# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2, random_state=777)
print('x_train length: ' + str(len(x_train)))
print('x_test length: ' + str(len(x_test)))
print('y_train length: ' + str(len(y_train)))
print('y_test length: ' + str(len(y_test)))
"""============================================="""
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
hist = model.fit(np.array(x_train), np.array(y_train), epochs=60, validation_split=0.2)

# 훈련 과정 시각화 (손실)
plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
plt.xlabel('Epoch')
#plt.show()
plt.savefig('./Model/cross_model'+str(candles_data)+'_'+ str(target_MA_1) + '_' + str(target_MA_2) +'.png',
            dpi=300)

model.save('./Model/cross_model'+str(candles_data)+'_'+ str(target_MA_1) + '_' + str(target_MA_2) +'.h5')
