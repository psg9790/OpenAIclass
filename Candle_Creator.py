import time

import pandas as pd
import requests
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from pandas import Series, DataFrame

curTime = 1669593601
targetTime = 1669593601
candleMinute = 5       # 분봉
dataScale = 1.5           # 데이터 크기 (year)
minTime = targetTime - dataScale * 60*60*24*365
breakCheck = False


wb = openpyxl.Workbook()
sheet = wb.active
sheet.column_dimensions['A'].width = 25
new_fileName = "./" + str(candleMinute) + "candles.xlsx"

idx = 0
date=[]
price=[]
while True:
    targetTimeStr = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime((targetTime)))
    url = "https://api.upbit.com/v1/candles/minutes/"+str(candleMinute)+"?market=KRW-BTC&to="+targetTimeStr+"%2B09:00&count=200"
    # print(url)

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers).json()

    for j in range(200):
        stamp = int(str(response[j]['timestamp'])[:10])
        if stamp > minTime:
            date.append(response[j]['candle_date_time_kst'])
            price.append(response[j]['trade_price'])
        else:
            breakCheck = True
            break

    stamp = int(str(response[199]['timestamp'])[:10])
    print(str(((curTime - stamp) / (curTime - minTime)) * 100) + '%')

    if(breakCheck):
        break

    targetTime -= candleMinute * 60 * 200
    time.sleep(0.1)


date.reverse()
price.reverse()

percentage = []
for i in range(0, len(price)-1):
    per = price[i+1]-price[i]
    per = per / price[i] * 100
    percentage.append(round(per, 2))
percentage.append(0)

# 분봉
dic = {'date' : date,'price':price}
df = pd.DataFrame(dic)


means_5 = df.loc[:,'price'].rolling(window=5).mean()
df['mean5'] = means_5
means_10 = df.loc[:,['price']].rolling(window=10).mean()
df['mean10'] = means_10
means_20= df.loc[:,['price']].rolling(window=20).mean()
df['mean20'] = means_20
means_60= df.loc[:,['price']].rolling(window=60).mean()
df['mean60'] = means_60
means_120= df.loc[:,['price']].rolling(window=120).mean()
df['mean120'] = means_120

df['per'] = pd.DataFrame(percentage)

for r in dataframe_to_rows(df, index=False, header=False):
    sheet.append(r)

wb.save(new_fileName)
