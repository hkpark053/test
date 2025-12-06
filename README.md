# 도쿄 23구 집값 예측 AI 프로젝트

AutoML과 딥러닝을 사용하여 도쿄 23구의 집값을 예측하는 AI 모델입니다.

## 프로젝트 구조

```
.
├── requirements.txt                    # 필요한 패키지 목록
├── generate_data.py                    # 합성 데이터 생성 스크립트
├── automl_model.py                     # AutoKeras를 사용한 AutoML 모델
├── custom_deep_learning_model.py       # 직접 구현한 딥러닝 모델들
└── README.md                          # 프로젝트 설명서
```

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 생성

먼저 학습용 데이터를 생성합니다:

```bash
python generate_data.py
```

이 스크립트는 `tokyo_housing_data.csv` 파일을 생성합니다.

### 2. AutoML 모델 학습 (AutoKeras)

AutoKeras를 사용하여 자동으로 최적의 모델을 찾습니다:

```bash
python automl_model.py
```

**주요 특징:**
- AutoKeras가 자동으로 최적의 신경망 아키텍처를 탐색
- 하이퍼파라미터 자동 튜닝
- 최적 모델 자동 선택

**구현된 기능:**
- 데이터 전처리 (라벨 인코딩, 정규화)
- AutoKeras StructuredDataRegressor 사용
- 모델 평가 (MAE, MSE, RMSE, MAPE)
- 최적 모델 아키텍처 출력

### 3. 직접 구현한 딥러닝 모델 학습

다양한 딥러닝 아키텍처를 직접 구현하여 비교합니다:

```bash
python custom_deep_learning_model.py
```

**구현된 모델들:**

1. **Deep MLP (깊은 다층 퍼셉트론)**
   - 4개의 은닉층 (256 → 128 → 64 → 32)
   - BatchNormalization과 Dropout 사용
   - 깊은 네트워크로 복잡한 패턴 학습

2. **Wide MLP (넓은 다층 퍼셉트론)**
   - 넓은 은닉층 (512 → 256)
   - 더 많은 뉴런으로 다양한 특성 동시 학습

3. **Residual MLP (잔차 연결 네트워크)**
   - 잔차 연결(Residual Connection) 사용
   - 그래디언트 소실 문제 완화
   - 더 깊은 네트워크 학습 가능

4. **Ensemble Model (앙상블 모델)**
   - 여러 모델의 예측을 결합
   - Deep, Wide, Residual 모델의 앙상블
   - 더 안정적인 예측 성능

**딥러닝 레이어 상세:**

```python
# 예시: Deep MLP 구조
inputs → Dense(256) + BatchNorm + Dropout(0.3)
      → Dense(128) + BatchNorm + Dropout(0.3)
      → Dense(64) + BatchNorm + Dropout(0.2)
      → Dense(32)
      → Dense(1) [출력]
```

**주요 딥러닝 기법:**
- **BatchNormalization**: 학습 안정화 및 가속화
- **Dropout**: 과적합 방지
- **Residual Connection**: 깊은 네트워크 학습 개선
- **Multi-Head Attention**: 특성 간 관계 학습 (Attention 모델)
- **Early Stopping**: 과적합 방지
- **Learning Rate Scheduling**: 학습률 자동 조정

## 데이터 구조

생성된 데이터는 다음 특성을 포함합니다:

- `ward`: 도쿄 23구 (범주형)
- `area`: 면적 (제곱미터)
- `year_built`: 건축 연도
- `age`: 건물 연령
- `floor`: 층수
- `station_distance`: 역까지 거리 (미터)
- `rooms`: 방 개수
- `price`: 집값 (만엔) - 예측 대상

## 모델 성능 평가 지표

- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **MSE (Mean Squared Error)**: 평균 제곱 오차
- **RMSE (Root Mean Squared Error)**: 평균 제곱근 오차
- **MAPE (Mean Absolute Percentage Error)**: 평균 절대 백분율 오차

## 결과 파일

학습 후 다음 파일들이 생성됩니다:

- `tokyo_housing_data.csv`: 생성된 데이터셋
- `automl_tokyo_housing_model/`: AutoML 모델 저장 폴더
- `best_*.h5`: 각 딥러닝 모델의 최적 가중치
- `*_training_history.png`: 학습 히스토리 그래프

## 코드 예시

### AutoML 모델 사용

```python
from automl_model import load_and_preprocess_data, train_automl_model

# 데이터 로드
X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data()

# 모델 학습
model = train_automl_model(X_train, y_train, max_trials=10, epochs=30)

# 예측
predictions = model.predict(X_test)
```

### 커스텀 딥러닝 모델 사용

```python
from custom_deep_learning_model import (
    load_and_preprocess_data,
    build_mlp_model,
    train_model
)

# 데이터 로드
X_train, X_val, X_test, y_train, y_val, y_test, _, _ = \
    load_and_preprocess_data()

# 모델 생성
model = build_mlp_model(input_dim=X_train.shape[1], architecture='deep')

# 모델 학습
history = train_model(model, X_train, y_train, X_val, y_val)

# 예측
predictions = model.predict(X_test.values)
```

## 참고사항

- 이 프로젝트는 합성 데이터를 사용합니다. 실제 데이터를 사용하려면 `generate_data.py`를 수정하거나 실제 데이터셋을 로드하도록 변경하세요.
- AutoKeras는 최적의 모델을 찾기 위해 여러 시도를 하므로 학습 시간이 오래 걸릴 수 있습니다.
- GPU가 있으면 학습 속도가 크게 향상됩니다.

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.
