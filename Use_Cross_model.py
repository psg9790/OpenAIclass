# 앞서 학습 후 저장했던 모델을 불러와서 예측에 사용해볼 것입니다

import time
import requests
import keras
import pandas as pd
import numpy as np

# 파라미터
# 사용할 분봉
candles_data = 15
# 타깃 MA1
target_MA_1 = 10
# 타깃 MA2
target_MA_2 = 20

# 모델 불러오기
model = keras.models.load_model('./Model/cross_model'+str(candles_data)+'_'+ str(target_MA_1) + '_' + str(target_MA_2) +'.h5')
model.summary()

# 타겟 시간 - 현재 시간을 기준으로 예측해봅니다
targetTimeStr = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime((time.time())))
print(targetTimeStr)

# API로 모델에 입력할 이전 데이터들을 불러와야 합니다
url = "https://api.upbit.com/v1/candles/minutes/"+str(candles_data)+"?market=KRW-BTC&to="+targetTimeStr+"%2B09:00&count=200"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers).json()

# 역시 시간 오름차순으로 받아옵니다
trade_price = []
for i in range(199, -1, -1):
    trade_price.append(response[i]['trade_price'])
dic = {'price':trade_price}
df = pd.DataFrame(dic)

# MA 추출
means_A = df.loc[:,'price'].rolling(window=target_MA_1).mean()
means_B = df.loc[:,'price'].rolling(window=target_MA_2).mean()

# 추출한 MA는 가격값이기 때문에 학습에서 해줬던 것처럼 변화율로 바꿔줌
ma_A = []
ma_B = []
for i in range(1, 200):
    ma_A.append((means_A[i]-means_A[i-1])/means_A[i-1] * 100)
    ma_B.append((means_B[i]-means_B[i-1])/means_B[i-1] * 100)

# 모델에 넣기 위해 (20,2) window를 20개 생성해줍니다
# 그렇다면 예측에 사용할 input은 (20,20,2)의 3중배열 형태겠죠
inputs = []
for i in range(19, -1, -1):
    window = []
    for j in range(20):
        window.append([ma_A[179-i+j], ma_B[179-i+j]])
    inputs.append(window)

# 모델에 넣으려면 numpy 배열로 바꿔줘야하기 때문에 변환해줍니다
inputs = np.array(inputs)
print(inputs)
print(inputs.shape)

# 모델에 넣어서 예측해줍니다
# 마지막 노드 값을 출력하기 때문에 0~1의 실수값으로 나옵니다
# 0.5 이상이면 1에 가깝기 때문에 상승을 예측한거고, 0.5 미만이 나오면 0에 가깝기 때문에 하락을 예측한 것이겠죠
y_predict = model.predict(inputs)
print(y_predict)

# 모델 출력값이 0~1이기 때문에 백분율로 바꿔줍니다
# 역시 50%이상이면 상승을 예측, 50 미만이면 하락을 예측
percentage = round(float(y_predict[19]*100), 1)
print("-----------------------------------------------------------------")
print(targetTimeStr)
# 모델 예측을 콘솔에 출력해줍니다
# 근데 50% 미만이면 하락을 예측한다고 했으니까, 50% 미만이면 그냥 (100-percentage)해서 하락 기준의 확률로 표시해줬습니다
print(str(target_MA_1)+","+str(target_MA_2)+"MA를 이용한 "+str(candles_data)+"분봉 다음 캔들 예상은 "+ (str(percentage) if percentage >=50 else str(100-percentage)) +"% 확률로 "+ ("상승" if percentage >= 50 else "하락")+ "합니다.")
print(y_predict)
