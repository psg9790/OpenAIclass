import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
candles_data = 5      # 사용할 분봉 데이터

# 엑셀
xl = pd.read_table('./'+str(candles_data)+'candles.txt',
                   names=['date', 'price', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'per'])


def makeX(df, i):
    tempX = []
    tempX.append(df['ma5'].iloc[i])
    tempX.append(df['ma10'].iloc[i])
    tempX.append(df['ma20'].iloc[i])
    tempX.append(df['ma60'].iloc[i])
    tempX.append(df['ma120'].iloc[i])

    tempX = np.array(tempX)
    max, min = tempX.max(), tempX.min()
    tempX = (tempX - min) / (max - min)

    return tempX


def makeY(df, i):
    gap = df['per'].iloc[i]
    tempY = 1 if gap > 0 else 0
    return tempY


x = []
y = []

for i in range(len(xl.index)):
    x.append(makeX(xl, i))
    y.append(makeY(xl, i))

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777)



#인풋 데이터 5개 이므로 shape=(5,)를 대입
inputs = tf.keras.Input(shape=(5,))

# h1 ~ h3은 히든 레이어, 층이 깊을 수록 정확도가 높아질 수 있음
# relu, tanh는 활성화 함수의 종류
BN1 = tf.keras.layers.BatchNormalization()(inputs)
h1 = tf.keras.layers.Dense(128, activation='relu')(BN1)
BN2 = tf.keras.layers.BatchNormalization()(h1)
h2 = tf.keras.layers.Dense(128, activation='tanh')(BN2)
BN3 = tf.keras.layers.BatchNormalization()(h2)
h3 = tf.keras.layers.Dense(128, activation='relu')(BN3)

# 값을 0 ~ 1 사이로 표현할 경우 sigmoid 활성화 함수 활용
# 마지막 아웃풋 값은 1개여야 함
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h3)

# 인풋, 아웃풋 설정을 대입하여 모델 생성
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

hist = model.fit(np.array(x_train), np.array(y_train), epochs=2000, validation_split=0.2)

# 훈련 과정 시각화 (손실)
plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.xlabel('Epoch')
plt.show()

model.save('./model'+str(candles_data)+'.h5')
