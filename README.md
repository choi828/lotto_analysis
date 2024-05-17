# LSTM을 이용한 로또 번호 예측

이 프로젝트는 장단기 기억(Long Short-Term Memory, LSTM) 신경망을 이용하여 로또 번호를 예측하는 것을 목표로 합니다. 모델은 과거의 로또 데이터를 학습하여 미래의 당첨 번호를 예측합니다.

## 목차
- [소개](#소개)
- [설치](#설치)
- [사용 방법](#사용-방법)
- [코드 설명](#코드-설명)
  - [데이터 수집](#데이터-수집)
  - [데이터 전처리](#데이터-전처리)
  - [모델 학습](#모델-학습)
  - [예측](#예측)
- [결과](#결과)
- [기여](#기여)
- [라이선스](#라이선스)

## 소개
LSTM 네트워크는 긴 시퀀스 데이터를 학습하는 데 유리한 순환 신경망(Recurrent Neural Network, RNN)의 일종입니다. 이 프로젝트에서는 LSTM을 사용하여 로또 번호 예측을 시도합니다.

## 설치
이 프로젝트를 실행하려면 Python과 다음 라이브러리가 필요합니다:
- requests
- pandas
- tqdm
- numpy
- tensorflow
- scikit-learn

필요한 패키지는 다음 명령어를 통해 설치할 수 있습니다:

bash
pip install requests pandas tqdm numpy tensorflow scikit-learn

## 사용방법

데이터 수집: 스크립트에서 데이터 수집 코드를 주석 해제하고 실행하여 로또 데이터를 수집합니다.
모델 학습: 스크립트가 데이터를 로드하고 전처리하여 LSTM 모델을 학습시킵니다.
예측: 학습된 모델을 사용하여 로또 번호를 예측하고 결과를 출력합니다.
스크립트를 실행하려면 다음 명령어를 사용하세요:

python lotto_analysis.py

## 코드 설명
데이터 수집
requests 라이브러리를 사용하여 로또 데이터를 수집하고 CSV 파일로 저장합니다:

```
all_lotto_numbers = []
for num in range(2, 1120):
    url = f'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={num}'
    response = requests.get(url)
    data = response.json()
    if data['returnValue'] == 'success':
        lotto_data = {
            'draw': data['drwNo'],
            'num1': data['drwtNo1'],
            'num2': data['drwtNo2'],
            'num3': data['drwtNo3'],
            'num4': data['drwtNo4'],
            'num5': data['drwtNo5'],
            'num6': data['drwtNo6']
        }
        all_lotto_numbers.append(lotto_data)
    else:
        print(f'Draw {num}: Data not found')
all_lotto_numbers_df = pd.DataFrame(all_lotto_numbers)
all_lotto_numbers_df.to_csv('lotto_numbers.csv', index=False)
```

## 데이터 전처리
데이터를 로드하여 정규화하고 LSTM 입력 형태로 변환합니다:

```
data = pd.read_csv('lotto_numbers.csv')
X = data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values[:-1]
y = data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values[1:]
X = X / 45.0
y = y / 45.0
X = X.reshape((X.shape[0], 1, X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 모델 학습
전처리된 데이터를 사용하여 LSTM 모델을 구성하고 학습합니다:
```
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(6))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[TqdmCallback(verbose=1)])
```

## 예측

학습된 모델을 사용하여 로또 번호를 예측하고 결과를 출력합니다:
```
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
predictions = model.predict(X_test)
predictions = predictions * 45.0
predictions = np.round(predictions).astype(int)
print(predictions[:5])
```

## 결과
```
Test Loss: 0.022802114486694336
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 403us/step
[[ 7 13 20 26 33 40]
 [ 7 13 20 26 33 40]
 [ 7 13 20 26 33 40]
 [ 7 13 20 26 33 40]
 [ 7 13 20 26 33 40]]
```



