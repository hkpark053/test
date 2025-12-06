# 도쿄 23구 주택 가격 예측 AI

AutoML과 딥러닝을 사용하여 도쿄 23구의 주택 가격을 예측하는 AI 모델입니다.

## 📋 프로젝트 개요

이 프로젝트는 두 가지 접근 방식을 사용합니다:
1. **AutoKeras를 사용한 AutoML**: 자동으로 최적의 모델 아키텍처를 탐색
2. **직접 구성한 딥러닝 모델**: 다양한 아키텍처(Deep, Wide&Deep, Residual)를 수동으로 구성

## 🏗️ 모델 아키텍처

### 직접 구성한 딥러닝 모델

#### 1. Deep Neural Network (DNN)
```
입력 레이어 (10 features)
  ↓
Dense(128) + BatchNorm + Dropout(0.3)
  ↓
Dense(256) + BatchNorm + Dropout(0.3)
  ↓
Dense(128) + BatchNorm + Dropout(0.2)
  ↓
Dense(64) + BatchNorm + Dropout(0.2)
  ↓
출력 레이어 (1 - 가격)
```

#### 2. Wide & Deep 모델
- **Wide 부분**: 선형 모델 (단순한 특성 조합 학습)
- **Deep 부분**: 깊은 신경망 (복잡한 비선형 관계 학습)
- 두 부분을 결합하여 예측

#### 3. Residual Network (ResNet)
- 잔차 연결(Residual Connection)을 사용한 깊은 네트워크
- 그래디언트 소실 문제 완화
- 여러 잔차 블록을 쌓아 구성

### AutoML 모델 (AutoKeras)
- AutoKeras가 자동으로 최적의 아키텍처 탐색
- 다양한 레이어 조합과 하이퍼파라미터를 시도
- 최대 10개의 모델을 시도하여 최적 모델 선택

## 📊 데이터 특성

### 입력 특성 (10개)
1. `ward_encoded`: 구 (23개 구 중 하나)
2. `area_sqm`: 면적 (제곱미터)
3. `age_years`: 건물 연령 (년)
4. `floor`: 층수
5. `distance_to_station_m`: 가장 가까운 역까지 거리 (미터)
6. `nearest_station_lines`: 가장 가까운 역의 노선 수
7. `building_type_encoded`: 건물 유형 (맨션, 아파트, 주택)
8. `has_parking`: 주차장 유무 (0/1)
9. `has_balcony`: 발코니 유무 (0/1)
10. `rooms`: 방 개수

### 출력
- `price_yen`: 주택 가격 (엔)

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
python train.py
```

이 스크립트는:
- 샘플 데이터 생성 (또는 CSV 파일 로드)
- 데이터 전처리
- 직접 구성한 딥러닝 모델 학습
- AutoML 모델 학습
- 결과 비교 및 시각화

### 3. 결과 확인

학습 완료 후 `results/` 디렉토리에 다음 파일들이 생성됩니다:
- `prediction_results.png`: 실제 vs 예측 가격 비교 그래프
- `training_history.png`: 학습 과정 그래프
- `best_model_deep.h5`: 최적의 모델 가중치

## 📁 파일 구조

```
.
├── requirements.txt          # 필요한 라이브러리
├── data_loader.py           # 데이터 로드 및 전처리
├── model.py                 # 직접 구성한 딥러닝 모델
├── automl_model.py          # AutoKeras AutoML 모델
├── train.py                 # 메인 학습 스크립트
└── README.md                # 프로젝트 설명
```

## 🔧 주요 코드 설명

### 딥러닝 레이어 구조 (model.py)

```python
# Deep 모델 예시
model = Sequential([
    # 입력 레이어
    Dense(128, activation='relu', input_shape=(10,)),
    BatchNormalization(),  # 배치 정규화
    Dropout(0.3),          # 드롭아웃 (과적합 방지)
    
    # 히든 레이어들
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # 출력 레이어
    Dense(1)  # 회귀 문제이므로 활성화 함수 없음
])
```

### AutoML 사용 (automl_model.py)

```python
# AutoKeras로 자동 모델 탐색
model = ak.StructuredDataRegressor(
    max_trials=10,  # 최대 10개 모델 시도
    overwrite=True
)

# 학습 (자동으로 최적 아키텍처 탐색)
model.fit(X_train, y_train, epochs=50)
```

## 📈 모델 성능 지표

- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **MSE (Mean Squared Error)**: 평균 제곱 오차
- **MAPE (Mean Absolute Percentage Error)**: 평균 절대 백분율 오차
- **R² Score**: 결정 계수 (1에 가까울수록 좋음)

## 💡 사용 팁

1. **실제 데이터 사용**: `data_loader.py`의 `load_data()` 메서드에 CSV 파일 경로를 전달하면 실제 데이터를 사용할 수 있습니다.

2. **하이퍼파라미터 조정**: `model.py`에서 학습률, 레이어 크기, 드롭아웃 비율 등을 조정할 수 있습니다.

3. **모델 아키텍처 변경**: `train.py`에서 `architecture` 파라미터를 변경하여 다른 모델을 학습할 수 있습니다:
   - `'deep'`: 깊은 신경망
   - `'wide_deep'`: Wide & Deep 모델
   - `'residual'`: Residual Network

## 🔍 딥러닝 레이어 상세 설명

### 1. Dense Layer (완전 연결 레이어)
- 모든 입력 뉴런이 모든 출력 뉴런과 연결
- 가중치 행렬을 사용하여 선형 변환 수행

### 2. BatchNormalization
- 배치 단위로 정규화하여 학습 안정화
- 그래디언트 소실/폭발 문제 완화

### 3. Dropout
- 학습 중 일부 뉴런을 무작위로 비활성화
- 과적합 방지

### 4. Activation Functions
- **ReLU**: 음수는 0으로, 양수는 그대로 (가장 많이 사용)
- 출력 레이어는 활성화 함수 없음 (회귀 문제)

### 5. Residual Connection
- 입력을 출력에 직접 더함
- 깊은 네트워크에서 그래디언트 전파 개선

## 📝 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.
