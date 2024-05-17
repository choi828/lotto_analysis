import requests
import pandas as pd
from tqdm.keras import TqdmCallback
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# # 빈 리스트 생성
# all_lotto_numbers = []

# for num in range(2, 1120):  # 여기서 num 범위를 적절히 설정하세요.
#     url = f'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={num}'
#     response = requests.get(url)
#     data = response.json()
    
#     if data['returnValue'] == 'success':
#         # 데이터 추출
#         lotto_data = {
#             'draw': data['drwNo'],
#             'num1': data['drwtNo1'],
#             'num2': data['drwtNo2'],
#             'num3': data['drwtNo3'],
#             'num4': data['drwtNo4'],
#             'num5': data['drwtNo5'],
#             'num6': data['drwtNo6']
#         }
#         # 리스트에 추가
#         all_lotto_numbers.append(lotto_data)
#     else:
#         print(f'Draw {num}: Data not found')

# # 리스트를 DataFrame으로 변환
# all_lotto_numbers_df = pd.DataFrame(all_lotto_numbers)

# # 결과 출력
# all_lotto_numbers_df.to_csv('/Users/daewoong/Documents/SNU_Python/lotto_analysis/lotto_numbers.csv', index=False)

# 데이터 로드
data = pd.read_csv('/Users/daewoong/Documents/SNU_Python/lotto_analysis/lotto_numbers.csv')

# 입력(X)과 출력(y) 데이터 분리
X = data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values[:-1]  # 전체 데이터에서 하나 뺀 데이터를 행렬로 변환
y = data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values[1:]  # 전체 데이터에서 하나 더한 데이터를 행렬로 변환

print(X)

# 데이터 정규화 (선택사항) 로또 번호가 45개이기 때문에 그렇다.
X = X / 45.0
y = y / 45.0

# 데이터 형태 변환 (LSTM 입력 형태로)
'''
로또 번호 예측의 경우, 각 회차의 로또 번호가 하나의 샘플이 된다. 그러나 우리는 각 회차를 독립적인 시간 스텝으로 간주하지 않고, 
단순히 각 회차의 번호들을 하나의 피처 집합으로 다룬다. 따라서 각 샘플은 하나의 시간 스텝만 가지며, 각 시간 스텝에서 6개의 피처(로또 번호)를 가진다.
'''
X = X.reshape((X.shape[0], 1, X.shape[1]))

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM 모델 설계
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(6))  # 출력 노드 수는 예측할 번호의 개수와 같음

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[TqdmCallback(verbose=1)])

# 모델 평가
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 예측
predictions = model.predict(X_test)
predictions = predictions * 45.0
predictions = np.round(predictions).astype(int)
print(predictions[:5])  # 첫 5개의 예측 결과 출력
