import pandas as pd

targetCandle = 5
means = 15

print('./'+str(targetCandle)+'candles.xlsx')
candle_data = pd.read_excel('./'+str(targetCandle)+'candles.xlsx', names=['date', 'price'])
means_15 = candle_data.rolling(window=means).mean()
print(means_15)
