# 도쿄 23구 집값 예측 AI - 상세 가이드

## 📋 프로젝트 개요

이 프로젝트는 AutoML과 커스텀 딥러닝을 활용하여 도쿄 23구의 주택 가격을 예측하는 AI 모델입니다.

## 🏗️ 아키텍처 및 딥러닝 레이어 상세

### 1. 커스텀 딥러닝 모델 구조

```
입력 레이어
├── 수치형 특성 (18개 특성)
└── 카테고리형 특성 (구 - Ward)
    └── 임베딩 레이어 (23개 구 → 8차원 벡터)

특성 결합 레이어
└── Concatenate: 수치형(18) + 임베딩(8) = 26차원
    └── Batch Normalization (입력 정규화)

Dense Block 1 (512 유닛)
├── Dense Layer: 26 → 512
├── Batch Normalization (학습 안정화)
├── ReLU Activation
└── Dropout (30% - 과적합 방지)

Dense Block 2 (256 유닛)
├── Dense Layer: 512 → 256
├── Batch Normalization
├── ReLU Activation
└── Dropout (30%)

Dense Block 3 (128 유닛)
├── Dense Layer: 256 → 128
├── Batch Normalization
├── ReLU Activation
└── Dropout (20%)

Attention 메커니즘
├── Reshape: (None, 128) → (None, 1, 128)
├── Attention Layer
│   ├── Query Transform: W_query (128 x 128)
│   ├── Key Transform: W_key (128 x 128)
│   ├── Value Transform: W_value (128 x 128)
│   ├── Attention Score: Q·K^T / √d_k
│   ├── Softmax (가중치 정규화)
│   └── Weighted Sum: Attention·V
├── Flatten
└── Concatenate with original (128 + 128 = 256)

Residual Block 1 (128 유닛)
├── Main Path
│   ├── Dense: 256 → 128
│   ├── Batch Normalization
│   ├── ReLU
│   ├── Dropout (20%)
│   ├── Dense: 128 → 128
│   └── Batch Normalization
├── Skip Connection (Projection)
│   └── Dense: 256 → 128
└── Add + ReLU + Dropout

Residual Block 2 (64 유닛)
├── Main Path (동일 구조)
├── Skip Connection
└── Add + ReLU + Dropout

최종 Dense Layer
├── Dense: 64 → 32 (ReLU)
└── Dropout (10%)

출력 레이어
└── Dense: 32 → 1 (Linear - 가격 예측)

총 파라미터: ~337,000개
```

### 2. 주요 딥러닝 기법 설명

#### 2.1 임베딩 레이어 (Embedding Layer)
```python
# 도쿄 23구를 저차원 밀집 벡터로 변환
ward_embedding = layers.Embedding(
    input_dim=23,  # 23개 구
    output_dim=8,  # 8차원 벡터로 표현
    name='ward_embedding'
)
```
**목적**: 카테고리형 변수(구 이름)를 연속적인 벡터 공간으로 매핑하여 구 간의 유사성을 학습

#### 2.2 배치 정규화 (Batch Normalization)
```python
x = layers.BatchNormalization()(x)
```
**목적**: 
- 각 미니배치의 활성화 값을 정규화
- 학습 안정화 및 속도 향상
- 내부 공변량 변화(Internal Covariate Shift) 감소

#### 2.3 드롭아웃 (Dropout)
```python
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
```
**목적**: 
- 학습 중 무작위로 30%의 뉴런을 비활성화
- 과적합 방지
- 모델의 일반화 성능 향상

#### 2.4 셀프 어텐션 (Self-Attention)
```python
class AttentionLayer(layers.Layer):
    def call(self, inputs):
        query = tf.matmul(inputs, self.W_query)
        key = tf.matmul(inputs, self.W_key)
        value = tf.matmul(inputs, self.W_value)
        
        score = tf.matmul(query, key, transpose_b=True)
        score = score / tf.math.sqrt(tf.cast(self.units, dtype=tf.float32))
        
        attention_weights = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(attention_weights, value)
        
        return output
```
**목적**: 
- 입력 특성 간의 관계 학습
- 중요한 특성에 더 높은 가중치 부여
- Transformer 아키텍처의 핵심 메커니즘

**동작 원리**:
1. Query, Key, Value 변환 행렬 학습
2. Query와 Key의 유사도 계산 (Attention Score)
3. Softmax로 가중치 정규화
4. Value에 가중치를 적용하여 출력 생성

#### 2.5 잔차 연결 (Residual Connection)
```python
class ResidualBlock(layers.Layer):
    def call(self, inputs):
        # Main path
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.bn2(x)
        
        # Skip connection
        shortcut = self.projection(inputs)
        
        # Add and activate
        x = x + shortcut
        x = tf.nn.relu(x)
        x = self.dropout2(x)
        
        return x
```
**목적**: 
- 그래디언트 소실(Vanishing Gradient) 문제 완화
- 더 깊은 네트워크 학습 가능
- 항등 매핑(Identity Mapping) 학습을 통한 성능 향상

**동작 원리**:
- Skip connection을 통해 입력을 직접 출력에 더함
- 네트워크가 잔차(residual)만 학습하면 됨
- ResNet의 핵심 아이디어

#### 2.6 Adam 옵티마이저
```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,      # 1차 모멘트 지수 감쇠율
    beta_2=0.999,    # 2차 모멘트 지수 감쇠율
    epsilon=1e-07
)
```
**목적**: 
- 적응적 학습률(Adaptive Learning Rate)
- 각 파라미터마다 다른 학습률 적용
- 빠른 수렴과 안정적인 학습

### 3. 특성 엔지니어링

```python
# 1. 파생 특성
price_per_sqm = price / area          # 면적당 가격
area_per_room = area / rooms          # 방당 면적
station_accessibility = 1 / (dist + 1) # 역 접근성

# 2. 범주형 특성
age_category = cut(age, bins=[0,5,10,20,51])  # 연식 구간

# 3. 이진 특성
is_high_floor = (floor >= 10)         # 고층 여부
is_spacious = (area >= 80)            # 넓은 집 여부
is_new_building = (age <= 5)          # 신축 여부
is_premium_ward = ward in [...]       # 고급 지역 여부

# 4. 상호작용 특성
area_rooms_interaction = area * rooms # 면적-방 상호작용

# 5. 종합 점수
total_convenience = (
    amenity_score * 0.4 +
    station_accessibility * 20 * 0.3 +
    has_balcony * 10 * 0.15 +
    has_parking * 10 * 0.15
)
```

### 4. 학습 전략

#### 4.1 Early Stopping
```python
EarlyStopping(
    monitor='val_loss',      # 검증 손실 모니터링
    patience=20,             # 20 에포크 동안 개선 없으면 중단
    restore_best_weights=True # 최적 가중치 복원
)
```

#### 4.2 Learning Rate Scheduling
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,              # 학습률을 절반으로 감소
    patience=10,             # 10 에포크 대기
    min_lr=1e-7              # 최소 학습률
)
```

#### 4.3 Model Checkpointing
```python
ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',
    save_best_only=True      # 최적 모델만 저장
)
```

## 📊 모델 성능

테스트 결과 (10 에포크 학습):
```
MAE (평균 절대 오차):     21.79 백만 엔 (약 2,179만 원)
RMSE (평균 제곱근 오차):   27.53 백만 엔 (약 2,753만 원)
R² Score (결정 계수):      0.6850 (68.5% 설명력)
```

## 🔬 AutoML vs 커스텀 모델 비교

| 특징 | AutoML (AutoKeras) | 커스텀 딥러닝 |
|------|-------------------|--------------|
| 개발 시간 | 빠름 (자동 탐색) | 느림 (수동 설계) |
| 최적화 | 자동 하이퍼파라미터 튜닝 | 수동 조정 필요 |
| 해석성 | 낮음 (블랙박스) | 높음 (명시적 구조) |
| 커스터마이징 | 제한적 | 완전한 제어 |
| 성능 | 일반적으로 우수 | 도메인 지식으로 개선 가능 |

## 📈 예측 예시

```python
# 예측 #1: 미나토구, 신축 고급 아파트
입력:
  - 구: Minato (고급 지역)
  - 면적: 80㎡
  - 방: 3개
  - 층: 15층 (고층)
  - 연식: 5년 (신축)
  - 역 거리: 5분 (매우 가까움)
  - 편의시설: 85점

출력: 216.21 백만 엔 (약 21.6억 원)

# 예측 #3: 아다치구, 오래된 아파트
입력:
  - 구: Adachi (일반 지역)
  - 면적: 70㎡
  - 방: 3개
  - 층: 3층
  - 연식: 25년 (오래됨)
  - 역 거리: 15분
  - 편의시설: 55점

출력: 73.85 백만 엔 (약 7.4억 원)
```

## 🚀 향후 개선 방안

### 1. 모델 아키텍처
- [ ] Transformer 기반 모델 실험
- [ ] GNN (Graph Neural Network)으로 지역 간 관계 모델링
- [ ] 앙상블 (XGBoost + LightGBM + 딥러닝)

### 2. 특성 추가
- [ ] 실제 부동산 데이터 수집
- [ ] 주변 시설 정보 (학교, 병원, 쇼핑몰)
- [ ] 교통 편의성 (지하철 노선 수, 환승역 여부)
- [ ] 시계열 데이터 (가격 변동 추세)

### 3. 학습 전략
- [ ] Cross-validation
- [ ] Hyperparameter tuning (Optuna)
- [ ] Data augmentation
- [ ] Transfer learning

## 📚 참고 자료

### 딥러닝 기법
- **Batch Normalization**: Ioffe & Szegedy (2015)
- **Residual Networks**: He et al. (2016)
- **Attention Mechanism**: Vaswani et al. (2017)
- **Dropout**: Srivastava et al. (2014)

### 프레임워크
- TensorFlow / Keras
- AutoKeras
- XGBoost
- LightGBM

## 💡 핵심 학습 포인트

### 1. 임베딩의 힘
카테고리형 변수를 원-핫 인코딩 대신 임베딩으로 처리하면:
- 차원이 감소 (23 → 8)
- 의미 있는 관계 학습 (고급 지역끼리 가까운 벡터)
- 일반화 성능 향상

### 2. 어텐션 메커니즘
모델이 "어디에 집중할지" 학습:
- 역까지 거리가 중요할 때는 해당 특성에 높은 가중치
- 고급 지역에서는 편의시설 점수에 집중
- 데이터 기반 자동 특성 선택

### 3. 잔차 학습
Skip connection으로:
- 깊은 네트워크도 안정적 학습
- 그래디언트가 직접 흐를 수 있는 경로 제공
- 항등 함수 학습이 기본값

### 4. 정규화의 중요성
Batch Normalization + Dropout:
- 과적합 방지
- 학습 속도 향상
- 더 깊은 네트워크 가능

## 🎯 결론

이 프로젝트는 단순한 회귀 문제를 넘어 현대 딥러닝의 핵심 기법들을 실전에서 적용하는 방법을 보여줍니다:

1. **데이터 전처리**: 특성 엔지니어링, 정규화, 인코딩
2. **모델 설계**: 임베딩, 어텐션, 잔차 연결
3. **학습 전략**: Early stopping, LR scheduling, 체크포인팅
4. **평가 및 해석**: 다양한 지표와 예측 분석

실제 부동산 데이터로 확장하면 상용 수준의 가격 예측 시스템을 구축할 수 있습니다!
