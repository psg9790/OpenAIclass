# 캔들 데이터를 수집하는 스크립트입니다
# 업비트 API를 사용합니다
# 데이터 수집 후 엑셀파일로 저장하는 작업까지 합니다

import time

import pandas as pd
import requests
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from pandas import Series, DataFrame

# 데이터의 마지막 타임스탬프입니다. 11/28일 09:00:01시 입니다
curTime = 1669593601
targetTime = 1669593601

# 5분봉 데이터를 불러올 것입니다
candleMinute = 5
# 1.5년이전의 데이터부터 불러올 것입니다
dataScale = 1.5
# 1.5년 전의 타임스탬프
minTime = targetTime - dataScale * 60*60*24*365
# 이중 반복문 탈출하기 위해 사용
breakCheck = False

# 엑셀파일 폼 생성
wb = openpyxl.Workbook()
sheet = wb.active
sheet.column_dimensions['A'].width = 25


# 아래 반복문에서 날짜와 가격 정보를 받아옵니다
idx = 0
date=[]
price=[]
while True:
    # 타임스탬프를 date string 형식으로 변환
    targetTimeStr = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime((targetTime)))

    # 분봉을 가져오는 API입니다
    url = "https://api.upbit.com/v1/candles/minutes/"+str(candleMinute)+"?market=KRW-BTC&to="+targetTimeStr+"%2B09:00&count=200"

    # 일봉
    # url = "https://api.upbit.com/v1/candles/days?market=KRW-BTC&to="+targetTimeStr+"%2B09:00&count=200"

    headers = {"accept": "application/json"}

    # API 반환값을 json 형식으로 불러옵니다
    response = requests.get(url, headers=headers).json()

    # API 1회 call의 반환값은 총 200개로, 반복문을 통해 date와 price 배열에 넣어줍니다
    # 최근 날짜부터 이 200회짜리 call을 반복해서 1.5년 전의 데이터까지 점진적으로 접근해갑니다
    for j in range(200):
        stamp = int(str(response[j]['timestamp'])[:10])
        # 초기에 설정한 1.5년 전 이후의 데이터라면 추가, 아니라면 이중 반복문을 탈출합니다
        if stamp > minTime:
            # 날짜와
            date.append(response[j]['candle_date_time_kst'])
            # 종가를 가져옵니다
            price.append(response[j]['trade_price'])
        else:
            breakCheck = True
            break

    # 데이터 가져오는 진행도를 콘솔에서 디버그 하기위한 line
    stamp = int(str(response[199]['timestamp'])[:10])
    print(str(((curTime - stamp) / (curTime - minTime)) * 100) + '%')

    # 이중 반복문 탈출
    if(breakCheck):
        break

    # 다음 API 호출을 위한 날짜 수정 (200캔들만큼의 시간만큼 minus)
    targetTime -= candleMinute * 60 * 200
    # 원활한 API call을 위한 delay
    time.sleep(0.1)

# 데이터가 [최근날짜~오래된날짜]로 내림차순이기에 다시 오름차순으로 정렬하기 위해 reverse 합니다
date.reverse()
price.reverse()

# 데이터의 0번 인덱스부터 다음 인덱스 데이터로의 변화량을 백분율(%)로 계산합니다
percentage = []
for i in range(0, len(price)-1):
    per = price[i+1]-price[i]
    per = per / price[i] * 100
    percentage.append(round(per, 2))
percentage.append(0)

# 데이터 프레임을 생성합니다
# 우선 날짜와 종가부터 추가
dic = {'date' : date,'price':price}
df = pd.DataFrame(dic)

# 오름차순된 데이터에서 MA5 계산
means_5 = df.loc[:,'price'].rolling(window=5).mean()
df['mean5'] = means_5
# 오름차순된 데이터에서 MA10 계산
means_10 = df.loc[:,['price']].rolling(window=10).mean()
df['mean10'] = means_10
# 오름차순된 데이터에서 MA20 계산
means_20= df.loc[:,['price']].rolling(window=20).mean()
df['mean20'] = means_20
# 오름차순된 데이터에서 MA60 계산
means_60= df.loc[:,['price']].rolling(window=60).mean()
df['mean60'] = means_60
# 오름차순된 데이터에서 MA120 계산
means_120= df.loc[:,['price']].rolling(window=120).mean()
df['mean120'] = means_120

df['per'] = pd.DataFrame(percentage)
# 위의 과정을 다 하게 되면 [날짜, 종가, MA5, MA10, MA20, MA60, MA120, per]의 데이터로 이루어진 구조가 완성됩니다

# 데이터들을 엑셀로 추가
for r in dataframe_to_rows(df, index=False, header=False):
    sheet.append(r)
# 엑셀을 저장합니다
new_fileName = "./" + str(candleMinute) + "candles.xlsx"
wb.save(new_fileName)
