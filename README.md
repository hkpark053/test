# 도쿄 23구 주택 가격 예측 AI 프로젝트

## 🎯 프로젝트 개요

이 프로젝트는 **AutoML**과 **커스텀 딥러닝**을 활용하여 도쿄 23구의 주택 가격을 예측하는 AI 모델입니다.

### 주요 특징
- ✅ **AutoKeras AutoML**: 자동 신경망 구조 탐색
- ✅ **커스텀 딥러닝**: Attention, Residual Block 등 고급 기법 적용
- ✅ **상세한 레이어 구현**: 각 레이어의 동작 원리 주석 포함
- ✅ **실전 데이터 파이프라인**: 전처리, 특성 엔지니어링, 평가
- ✅ **시각화**: 데이터 분포, 상관관계, 예측 결과

## 📁 프로젝트 구조

```
/workspace/
├── data/
│   ├── generate_tokyo_housing_data.py   # 샘플 데이터 생성 스크립트
│   └── tokyo_housing_data.csv           # 생성된 데이터 (5,000 샘플)
│
├── src/
│   ├── data_preprocessing.py            # 데이터 전처리 및 특성 엔지니어링
│   ├── automl_model.py                  # AutoKeras AutoML 모델
│   ├── custom_deep_model.py            # 커스텀 딥러닝 모델 (상세 레이어)
│   └── ensemble_model.py                # 앙상블 모델 (XGBoost, LightGBM 등)
│
├── models/                              # 학습된 모델 저장
│   ├── custom_deep_model.h5
│   ├── custom_best_model.h5
│   └── preprocessor.pkl
│
├── results/                             # 예측 결과 및 시각화
│   ├── predictions.csv
│   └── visualizations/
│       ├── data_distribution.png
│       ├── correlation_matrix.png
│       ├── ward_analysis.html
│       └── predictions.html
│
├── train.py                             # 학습 메인 스크립트
├── predict.py                           # 예측 스크립트
├── visualize.py                         # 시각화 스크립트
├── test_quick_training.py              # 빠른 테스트 스크립트
├── requirements.txt                     # 패키지 의존성
├── README.md                            # 프로젝트 가이드
└── ARCHITECTURE.md                      # 아키텍처 상세 문서
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 생성

```bash
# 도쿄 23구 주택 데이터 생성 (5,000 샘플)
python3 data/generate_tokyo_housing_data.py
```

### 3. 모델 학습

```bash
# 커스텀 딥러닝 모델 학습
python3 train.py --model custom

# AutoML 모델 학습 (시간이 오래 걸림)
python3 train.py --model automl

# 모든 모델 학습
python3 train.py --model all
```

### 4. 예측

```bash
# 학습된 모델로 예측
python3 predict.py --model custom

# 커스텀 데이터로 예측
python3 predict.py --model custom --input your_data.csv
```

### 5. 시각화

```bash
# 데이터 분석 및 예측 결과 시각화
python3 visualize.py
```

## 🧠 딥러닝 아키텍처

### 커스텀 모델 레이어 구조

```
┌─────────────────────────────────────────────┐
│  입력 레이어                                 │
│  ├─ 수치형 특성 (18개)                      │
│  └─ 구(Ward) 임베딩 (23개 → 8차원)         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Dense Block 1 (512 유닛)                   │
│  ├─ Dense Layer                             │
│  ├─ Batch Normalization                     │
│  ├─ ReLU Activation                         │
│  └─ Dropout (30%)                           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Dense Block 2 (256 유닛)                   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Dense Block 3 (128 유닛)                   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Attention 메커니즘                         │
│  ├─ Query, Key, Value 변환                 │
│  ├─ Attention Score 계산                   │
│  └─ Weighted Sum                           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Residual Block 1 (128 유닛)                │
│  ├─ Main Path (Dense → BN → ReLU)          │
│  ├─ Skip Connection                         │
│  └─ Add + ReLU                              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Residual Block 2 (64 유닛)                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  출력 레이어 (가격 예측)                     │
└─────────────────────────────────────────────┘

총 파라미터: ~337,000개
```

**상세 내용은 [ARCHITECTURE.md](ARCHITECTURE.md) 참조**

## 📊 주요 딥러닝 기법

### 1. 임베딩 레이어 (Embedding)
```python
# 카테고리형 변수를 밀집 벡터로 변환
ward_embedding = layers.Embedding(
    input_dim=23,      # 도쿄 23구
    output_dim=8,      # 8차원 벡터
    name='ward_embedding'
)
```
**효과**: 구 간의 유사성을 학습, 차원 감소 (23 → 8)

### 2. Attention 메커니즘
```python
class AttentionLayer(layers.Layer):
    # Query, Key, Value 변환
    # Attention Score 계산
    # 중요 특성에 높은 가중치 부여
```
**효과**: 모델이 중요한 특성에 집중, Transformer의 핵심

### 3. Residual Connection
```python
class ResidualBlock(layers.Layer):
    # Main Path + Skip Connection
    output = main_path(x) + projection(x)
```
**효과**: 깊은 네트워크 학습 가능, 그래디언트 소실 방지

### 4. Batch Normalization
```python
x = layers.BatchNormalization()(x)
```
**효과**: 학습 안정화, 속도 향상

### 5. Dropout
```python
x = layers.Dropout(0.3)(x)
```
**효과**: 과적합 방지, 일반화 성능 향상

## 📈 모델 성능

| 지표 | 값 |
|------|-----|
| **MAE** (평균 절대 오차) | 21.79 백만 엔 (약 2,179만 원) |
| **RMSE** (평균 제곱근 오차) | 27.53 백만 엔 (약 2,753만 원) |
| **R² Score** (결정 계수) | 0.6850 (68.5% 설명력) |

## 🎨 예측 예시

```python
# 예측 #1: 미나토구, 신축 고급 아파트
{
    'ward': 'Minato',
    'area_sqm': 80.0,
    'rooms': 3,
    'floor': 15,
    'building_age': 5.0,
    'distance_to_station_min': 5.0,
    'amenity_score': 85.0
}
→ 예측 가격: 216.21 백만 엔 (약 21.6억 원)

# 예측 #3: 아다치구, 오래된 아파트
{
    'ward': 'Adachi',
    'area_sqm': 70.0,
    'rooms': 3,
    'floor': 3,
    'building_age': 25.0,
    'distance_to_station_min': 15.0,
    'amenity_score': 55.0
}
→ 예측 가격: 73.85 백만 엔 (약 7.4억 원)
```

## 🔬 특성 엔지니어링

### 생성된 파생 특성
1. **price_per_sqm**: 면적당 가격
2. **area_per_room**: 방당 면적
3. **age_category**: 건물 연식 구간 (신축/중고/노후)
4. **station_accessibility**: 역 접근성 (거리 역수)
5. **is_high_floor**: 고층 여부 (10층 이상)
6. **is_spacious**: 넓은 집 여부 (80㎡ 이상)
7. **is_new_building**: 신축 여부 (5년 이내)
8. **total_convenience**: 종합 편의성 점수
9. **area_rooms_interaction**: 면적-방 상호작용
10. **is_premium_ward**: 고급 지역 여부

## 📚 학습 자료

### 코드에서 배울 수 있는 내용
- ✅ 데이터 전처리 및 특성 엔지니어링
- ✅ 임베딩 레이어로 카테고리형 변수 처리
- ✅ Attention 메커니즘 구현
- ✅ Residual Block 구현
- ✅ Batch Normalization & Dropout
- ✅ Early Stopping, LR Scheduling
- ✅ 모델 저장 및 로드
- ✅ 예측 및 평가

### 각 파일의 역할

#### `src/custom_deep_model.py` (500+ 라인)
- **AttentionLayer**: Query, Key, Value를 사용한 셀프 어텐션
- **ResidualBlock**: Skip connection이 있는 잔차 블록
- **CustomDeepHousingModel**: 전체 모델 아키텍처
- 각 레이어의 목적과 동작 원리를 상세히 주석 처리

#### `src/data_preprocessing.py`
- 데이터 로드 및 검증
- 10가지 파생 특성 생성
- 학습/검증/테스트 세트 분할
- 스케일링 및 인코딩

#### `train.py`
- 데이터 파이프라인 구축
- 모델 학습 및 평가
- 콜백 설정 (Early Stopping, LR Scheduling)
- 결과 저장

## 🛠️ 커스터마이징

### 모델 하이퍼파라미터 조정

```python
# src/custom_deep_model.py
model = CustomDeepHousingModel(
    num_features=18,
    num_wards=23,
    embedding_dim=8  # 임베딩 차원 조정
)

model.compile_model(learning_rate=0.001)  # 학습률 조정

model.train(
    epochs=200,      # 에포크 수 조정
    batch_size=32    # 배치 크기 조정
)
```

### 새로운 특성 추가

```python
# src/data_preprocessing.py의 create_features() 함수 수정
def create_features(self, df):
    # 기존 특성...
    
    # 새로운 특성 추가
    df['your_new_feature'] = ...
    
    return df
```

## ⚙️ 요구사항

```
Python >= 3.8
TensorFlow >= 2.13.0
Keras >= 2.13.0
AutoKeras >= 1.1.0
NumPy, Pandas, Scikit-learn
XGBoost, LightGBM
Matplotlib, Seaborn, Plotly
```

## 📖 추가 문서

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: 아키텍처 상세 설명
  - 각 레이어의 수학적 원리
  - Attention 메커니즘 상세
  - Residual Connection 설명
  - 학습 전략 및 최적화

## 🤝 기여

이 프로젝트는 AutoML과 딥러닝 학습을 위한 교육용 프로젝트입니다.

## 📝 라이선스

MIT License

---

## 💡 핵심 포인트

이 프로젝트를 통해 다음을 배울 수 있습니다:

1. **실전 딥러닝 파이프라인**: 데이터 → 전처리 → 학습 → 평가 → 배포
2. **고급 딥러닝 기법**: Attention, Residual, Embedding
3. **AutoML**: 자동 신경망 구조 탐색
4. **모범 사례**: 코드 구조, 주석, 문서화

**코드를 읽고 실행하면서 각 레이어가 어떻게 동작하는지 이해해보세요!** 🚀
