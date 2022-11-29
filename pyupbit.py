import requests
import pandas as pd


data = pd.read_csv('./dataset.csv',
                   names=['date', 'article', 'label'], encoding='CP949')

print(type(data))
print(data)
print(data.loc[[0], ['date']])

# url = "https://api.upbit.com/v1/candles/minutes/60?market=KRW-BTC&to=2022-11-26T09:00:01%2B09:00&count=2"

# headers = {"accept": "application/json"}

# response = requests.get(url, headers=headers).json()

# print(response)
# print((response[0]['trade_price'] - response[1]
#       ['trade_price'])/response[1]['trade_price'] * 100)
# # print(response[0]['market'])
