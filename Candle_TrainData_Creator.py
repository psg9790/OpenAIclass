import openpyxl
import pandas as pd

candles_data = 60      # 사용할 분봉 데이터
candles_window = 20     # 데이터 프레임에 들어갈 캔들 개수 - 20 선택시 20+20=40개의 input
MA_A = 5    # 첫번째 MA - 5, 10, 20, 60, 120
MA_B = 10   # 두번째 MA - 5, 10, 20, 60, 120

xl = pd.read_table('./'+str(candles_data)+'candles.txt', names=['date','price', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'per'])

x_train = []
y_train = []

for i in range(0, len(xl.index)-candles_window):
    df = xl.loc[i:i+(candles_window-1), ['ma'+str(MA_A), 'ma'+str(MA_B), 'per']]

    x_trainBlock = []
    for j in range(i,i+candles_window):
        x_trainBlock.append(df.loc[j, 'ma'+str(MA_A)])
    for j in range(i,i+candles_window):
        x_trainBlock.append(df.loc[j, 'ma'+str(MA_B)])
    x_train.append(x_trainBlock)
    y_train.append(df.loc[i+(candles_window-1), 'per'])


print(len(x_train))
print(len(y_train))
print(x_train[0])
print(y_train[0])