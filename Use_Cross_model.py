import time
import requests
import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
candles_data = 15   # 사용할 분봉 데이터
target_MA_1 = 10
target_MA_2 = 20

# 모델 불러오기
model = keras.models.load_model('./Model/cross_model'+str(candles_data)+'_'+ str(target_MA_1) + '_' + str(target_MA_2) +'.h5')
model.summary()

# 타겟 시간
targetTimeStr = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime((time.time())))
print(targetTimeStr)

# API
url = "https://api.upbit.com/v1/candles/minutes/"+str(candles_data)+"?market=KRW-BTC&to="+targetTimeStr+"%2B09:00&count=200"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers).json()

# 종가 지난 200봉부터 시간 오름차순으로
trade_price = []
for i in range(199, -1, -1):
    trade_price.append(response[i]['trade_price'])
dic = {'price':trade_price}
df = pd.DataFrame(dic)

# MA 추출
means_A = df.loc[:,'price'].rolling(window=target_MA_1).mean()
means_B = df.loc[:,'price'].rolling(window=target_MA_2).mean()

# plt.plot(means_A[179:])
# plt.plot(means_B[179:])
# plt.show()

ma_A = []
ma_B = []
for i in range(1, 200):
    ma_A.append((means_A[i]-means_A[i-1])/means_A[i-1] * 100)
    ma_B.append((means_B[i]-means_B[i-1])/means_B[i-1] * 100)

# ma_A = ma_A[179:]
# ma_B = ma_B[179:]

inputs = []
for i in range(19, -1, -1):
    window = []
    for j in range(20):
        window.append([ma_A[179-i+j], ma_B[179-i+j]])
    inputs.append(window)

inputs = np.array(inputs)
print(inputs)
print(inputs.shape)

y_predict = model.predict(inputs)
print(y_predict)

percentage = round(float(y_predict[19]*100), 1)
print("-----------------------------------------------------------------")
print(targetTimeStr)
print(str(target_MA_1)+","+str(target_MA_2)+"MA를 이용한 "+str(candles_data)+"분봉 다음 캔들 예상은 "+ (str(percentage) if percentage >=50 else str(100-percentage)) +"% 확률로 "+ ("상승" if percentage >= 50 else "하락")+ "합니다.")
# print(y_predict)
