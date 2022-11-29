import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
print('import complete')

# 데이터셋 total_data로 가져옴
total_data = pd.read_csv(
    './coinpan.csv', names=['label', 'article'])

print(total_data[:5])



train_data, test_data = train_test_split(
    total_data, test_size=0.25, random_state=42)
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))

# 중복된 값 제외해서 저장
# article 열에서 중복인 내용이 있다면 중복 제거
# 학습데이터 - 한글과 공백을 제외하고 모두 제거
total_data.drop_duplicates(subset=['article'], inplace=True)
train_data['article'] = train_data['article'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
train_data['article'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')
print('전처리 후 학습용 샘플의 개수 :', len(train_data))

# 테스트데이터 - 한글과 공백을 제외하고 모두 제거
test_data.drop_duplicates(subset=['article'], inplace=True)  # 중복 제거
test_data['article'] = test_data['article'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)  # 정규 표현식 수행
test_data['article'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :', len(test_data))

# 불용어 정의
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를',
             '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

# 형태소 분석기 Mecab을 사용해서 토큰화
mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
train_data['tokenized'] = train_data['article'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(
    lambda x: [item for item in x if item not in stopwords])
test_data['tokenized'] = test_data['article'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(
    lambda x: [item for item in x if item not in stopwords])

negative_words = np.hstack(
    train_data[train_data.label == -1]['tokenized'].values)
positive_words = np.hstack(
    train_data[train_data.label == 1]['tokenized'].values)

negative_word_count = Counter(negative_words)
#print(negative_word_count.most_common(20))
positive_word_count = Counter(positive_words)
#print(positive_word_count.most_common(20))

X_train = train_data['tokenized'].values
y_train = train_data['label'].values
X_test = test_data['tokenized'].values
y_test = test_data['label'].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# 단어 집합이 생성됨과 동시에 각 단어에 고유한 정수 부여
# tokenizer.word_index를 출력하여 확인 가능
# 등장 횟수가 1인 단어들은 자연어 처리에서 배제하는지 확인하는 과정
# 이 단어들이 이 데이터에서 얼만큼의 비중을 차지하는지 출력해서 제외해도 되는지 확인
threshold = 2
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거
# 0번 패딩 토큰과 1번 oov 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :', vocab_size)

# 토크나이저가 텍스트 시퀀스를 숫자 시퀀스로 변환
# 변환과정에서 큰 숫자가 부여된 단어들은 OOV로 변환
tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[:5])

# 최대길이 64, 평균길이 15... 60으로 패딩해도 될까?


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if (len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' %
          (max_len, (count / len(nested_list))*100))


# 60길이로 패딩했을 때 적절한지 비율 확인
max_len = 60
below_threshold_len(max_len, X_train)

# 실제로 패딩 적용
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 모델 생성, 학습
import re
from keras.layers import Embedding, Dense, LSTM, Bidirectional
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units))) # Bidirectional LSTM을 사용
model.add(Dense(1, activation='sigmoid'))

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[mc], batch_size=256, validation_split=0.2)

# 학습 완료된 모델 불러오기
loaded_model = load_model('best_model.h5')
print("테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# 예측을 위해서 입력한.. 텍스트에 대해서 학습하기 전에 했던 전처리 과정을 동일하게 해줌
def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = mecab.morphs(new_sentence) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))

sentiment_predict('가격대비 정말 혜자게임인거는 맞지만 중간에 어인족 마을에서 노가다는 정말 짜증났습니다. 마치 메이플스토리 퀘스트 노가다 마냥 계속 왔다리갔다리 시키는거 개극혐 근데 게임은 전체적으로 재밌습니다.')