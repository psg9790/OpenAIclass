# 데이터를 전처리하고, 모델에 넣어 학습까지 시키는 스크립트입니다
# 모델을 재사용할 수 있도록 /Model 디렉토리에 저장합니다

import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
# 분석할 분봉
candles_data =60
# 분석에 사용할 MA 1번
target_MA_1 = 20
# 분석에 사용할 MA 2번
target_MA_2 = 60

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 엑셀로 저장된 데이터를 불러옵니다, 위 파라미터에서 설정한 파라미터를 기반으로 파일을 불러옴
xl = pd.read_table('./'+str(candles_data)+'candles.txt',
                   names=['date', 'price', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'per'])

# """============================================="""
# 데이터 전처리 과정입니다
# MA 값은 가격이라서 편차가 굉장히 크기 때문에 stationary한 값들로 바꿔줘야 합니다
# MA 값들의 변화율로 바꿔줄 것입니다
MA = []
# 첫번째 타깃의 MA 데이터들을 불러옵니다
MA = xl.loc[:,'ma'+str(target_MA_1)]
# 변화율로 바뀐 MA1들이 담길 배열
MA_1 = []
# 현재 MA1 값에서 다음 MA1으로의 변화율을 계산해서 담아줍니다
for i in range(len(MA)):
    if i != len(MA)-1:
        tooken = ((MA[i+1] - MA[i])/MA[i])*100
        MA_1.append(tooken)

# 두번째 타깃의 MA 데이터들을 불러옴
MA = xl.loc[:,'ma'+str(target_MA_2)]
# 변화율로 바뀐 MA2들이 담길 배열
MA_2 = []
# 현재 MA2 값에서 다음 MA2로의 변화율을 계산해서 담아줍니다
for i in range(len(MA)):
    if i != len(MA)-1:
        tooken = ((MA[i+1] - MA[i])/MA[i])*100
        MA_2.append(tooken)
# """============================================="""
# 모델에서는 인풋으로 X를 받으면, Y를 맞추는 방향으로 학습하게 됩니다
# 학습에 사용할 X와 Y를 만듭니다

# X데이터는 (시계열, 20, 2)의 3중배열 크기를 가지게 됩니다
# 이는 우리가 시계열 데이터를 처리하는 RNN 모델 중 하나인 LSTM을 사용하기 때문입니다
# '시계열'의 개수는 우리가 가지고 있는 데이터의 크기와 비례해서 늘어납니다 (1.5년 전까지의 캔들 개수에 비례할 것입니다)
# '20,2'인 이유는 [MA1, MA2]를 20개씩 래핑했기 때문 (window크기를 20캔들로 잡은 것임, 이 window들이 시계열로 나열되어 있음)
x_data = []
for i in range(len(MA_1) - 20):
    window = []
    for j in range(i, i+20):
        window.append([MA_1[j], MA_2[j]])
    x_data.append(window)
x_data = np.array(x_data)

# Y데이터로는 상승시 1, 하락시 0의 값으로 준비해줍니다 (0 또는 1이 나오기 때문에 '분류'문제입니다)
per = xl.loc[:,'per']
y_data = []
for i in range(len(MA_1) - 20):
    if per[i] >= 0:
        y_data.append([1])
    else:
        y_data.append([0])
y_data = np.array(y_data)

# 콘솔 디버그용
print('x_data head5: ' + str(x_data[:5]))
print('y_data head5: ' + str(y_data[:5]))
print('x_data shape: ' + str(x_data.shape))
print('y_data shape: ' + str(y_data.shape))
# """============================================="""
# model의 형태를 만듭니다 (학습은 아직 X)
# 모델에서 (20,2)가 시계열로 이루어진 입력 데이터를 받을 것입니다
inputs = keras.Input(shape=(20,2))
# 1. 입력값에 대해 LSTM 레이어 사용
lstm_layer = tf.keras.layers.LSTM(10)(inputs)
# 2. 10 by 10 레이어를 사용 (relu)
x = tf.keras.layers.Dense(10, activation='relu')(lstm_layer)
# 3. 10 by 10 레이어를 사용 (relu)
x = tf.keras.layers.Dense(10, activation='relu')(x)
# 4. 10 by 1 레이어를 사용 (sigmoid) - 이 마지막 1개의 노드 값이 loss 함수를 통해 Y데이터와 비슷해지는 방향으로 학습합니다
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
# """============================================="""
# 앞에서 재가공한 X,Y 데이터를 학습용과 테스트용으로 쪼갭니다 (80퍼센트는 학습, 20퍼센트는 테스트)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2, random_state=777)
print('x_train length: ' + str(len(x_train)))
print('x_test length: ' + str(len(x_test)))
print('y_train length: ' + str(len(y_train)))
print('y_test length: ' + str(len(y_test)))
# """============================================="""
# 모델 학습을 시작합니다
# 최적화 함수로 adam, loss function으로 분류문제이기 때문에 crossEntropy를 사용합니다
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
hist = model.fit(np.array(x_train), np.array(y_train), epochs=60, validation_split=0.2)

# 훈련이 완료되면 과정을 시각화해서 저장합니다 (/Model 디렉토리)
plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
plt.xlabel('Epoch')
#plt.show()
plt.savefig('./Model/cross_model'+str(candles_data)+'_'+ str(target_MA_1) + '_' + str(target_MA_2) +'.png',
            dpi=300)

# 학습이 완료되면 모델을 저장합니다 (/Model 디렉토리)
model.save('./Model/cross_model'+str(candles_data)+'_'+ str(target_MA_1) + '_' + str(target_MA_2) +'.h5')
