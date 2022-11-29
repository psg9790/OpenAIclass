import keras.layers
import tensorflow as tf
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 엑셀 불러와서 데이터프레임으로
candles_data = 60      # 사용할 분봉 데이터
candles_window = 20     # 데이터 프레임에 들어갈 캔들 개수 - 20 선택시 20+20=40개의 input
MA_A = 5    # 첫번째 MA - 5, 10, 20, 60, 120
MA_B = 10   # 두번째 MA - 5, 10, 20, 60, 120

xl = pd.read_table('./'+str(candles_data)+'candles.txt', names=['date', 'price', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'per'])

x_datas = []
y_datas = []
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        temp -= 1
        norm_arr.append(temp)
    return norm_arr


for i in range(0, len(xl.index)-candles_window):
    df = xl.loc[i:i+(candles_window-1), ['ma'+str(MA_A), 'ma'+str(MA_B), 'per']]

    x_trainBlock = []
    for j in range(i,i+candles_window):
        x_trainBlock.append(df.loc[j, 'ma'+str(MA_A)])
    #for j in range(i,i+candles_window):
        x_trainBlock.append(df.loc[j, 'ma'+str(MA_B)])
    # x_trainBlock = normalize(x_trainBlock, 0, 2)
    x_datas.append(x_trainBlock)
    y_datas.append(df.loc[i+(candles_window-1), 'per'])

x_datas = np.array(x_datas)
y_datas = np.array(y_datas)


# 해당 데이터를 train, test data로 쪼갬
x_train, x_test, y_train, y_test = train_test_split(x_datas, y_datas, test_size=0.2, random_state=777)


"""for i in range(10):
    plt.plot(x_train[i*100][:19])
    plt.plot(x_train[i*100][20:])
    plt.show()"""


# data를 실제 모델에 넣을 땐 타겟으로 한 MA중 더 큰값으로 normalize해서 반환 (size는 내가 정의? 20)
# BatchNormalization 쓰면 될듯?

# ex) 20+20, 40개의 데이터를 MLP에 넣어줌
inputs = tf.keras.Input(shape=(40,))

x = keras.layers.BatchNormalization()(inputs)
x = tf.keras.layers.Dense(40)(x)

x = keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32, activation='tanh')(x)


x = keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(16, activation='tanh')(x)

x = keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(8, activation='tanh')(x)

x = keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(4, activation='relu')(x)

outputs = Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)
model.summary()

hist = model.fit(x_train, y_train, batch_size=16,
                 epochs=1000, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

# 훈련 과정 시각화 (손실)
plt.plot(hist.history['loss'])
plt.title('loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

"""# 훈련 과정 시각화 (정확도)
#plt.plot(hist.history['accuracy'])
#plt.legend(['loss', 'accuracy'])
#plt.title('accuracy')
plt.xlabel('Epoch')
#plt.ylabel('accuracy')
plt.show()"""

model.save('./mymodel.h5')


for i in range(10):
    testing = np.array([
        x_test[i*10]
    ])
    y_predict = model.predict(testing)
    print(y_predict[0])
    print(y_test[i*10])
