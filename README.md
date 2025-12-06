# 🏠 도쿄 23구 주택 가격 예측 AI (AutoML)

도쿄 23구 부동산 가격을 예측하는 AI 모델입니다. AutoML과 딥러닝을 활용하여 최적의 모델을 자동으로 탐색하고 학습합니다.

## 📋 프로젝트 구조

```
tokyo-housing-prediction/
├── data/
│   ├── __init__.py
│   ├── generate_tokyo_housing_data.py  # 샘플 데이터 생성
│   ├── tokyo_housing_data.csv          # 생성된 데이터
│   ├── train.csv                       # 학습 데이터
│   ├── val.csv                         # 검증 데이터
│   └── test.csv                        # 테스트 데이터
├── models/
│   ├── __init__.py
│   ├── deep_learning_layers.py         # 딥러닝 레이어 구현
│   └── automl_models.py                # AutoML 모델 구현
├── outputs/                            # 학습 결과 저장
├── train_automl.py                     # AutoML 학습 스크립트
├── train_deep_learning.py              # 딥러닝 학습 스크립트
├── requirements.txt                    # 패키지 의존성
└── README.md
```

## 🚀 시작하기

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 생성

```bash
python data/generate_tokyo_housing_data.py
```

### 3. 모델 학습

#### AutoML 학습
```bash
# 앙상블 방식 (권장)
python train_automl.py --method ensemble --n_trials 100

# Optuna 기반 최적화
python train_automl.py --method optuna --n_trials 100
```

#### 딥러닝 학습
```bash
# MLP 모델
python train_deep_learning.py --model mlp --epochs 100

# Transformer 모델
python train_deep_learning.py --model transformer --epochs 100

# ResNet 스타일 모델
python train_deep_learning.py --model resnet --epochs 100

# TabNet 모델
python train_deep_learning.py --model tabnet --epochs 100
```

## 📊 데이터 설명

### 도쿄 23구 정보
| 구 | 일본어 | 기준 가격 (만엔/㎡) |
|---|--------|-----------------|
| Minato | 港区 | 200 |
| Chiyoda | 千代田区 | 180 |
| Chuo | 中央区 | 170 |
| Shibuya | 渋谷区 | 160 |
| Meguro | 目黒区 | 140 |
| ... | ... | ... |

### 피처 목록
| 피처 | 설명 | 타입 |
|------|------|------|
| `ward` | 구 이름 (영문) | 범주형 |
| `ward_jp` | 구 이름 (일본어) | 범주형 |
| `floor_area_sqm` | 면적 (㎡) | 수치형 |
| `num_rooms` | 방 개수 | 수치형 |
| `building_age_years` | 건물 연령 | 수치형 |
| `floor_number` | 층수 | 수치형 |
| `total_floors` | 총 층수 | 수치형 |
| `distance_to_station_min` | 역까지 거리 (분) | 수치형 |
| `building_type` | 건물 유형 | 범주형 |
| `direction` | 방향 | 범주형 |
| `has_parking` | 주차장 유무 | 이진 |
| `has_balcony` | 발코니 유무 | 이진 |
| `has_security` | 보안 시스템 유무 | 이진 |
| `is_corner_unit` | 코너 유닛 여부 | 이진 |
| `price_man_yen` | **가격 (만엔) - 타겟** | 수치형 |

## 🧠 딥러닝 아키텍처

### 1. BasicMLP (Multi-Layer Perceptron)
```python
class BasicMLP(nn.Module):
    """
    기본 MLP 모델
    - 여러 개의 완전 연결 레이어
    - BatchNorm + ReLU + Dropout
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
```

### 2. ResidualNetwork
```python
class ResidualBlock(nn.Module):
    """
    잔차 블록 - Skip Connection으로 깊은 학습 가능
    """
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        return self.relu(out)
```

### 3. TabNetModel
```python
class TabNetBlock(nn.Module):
    """
    TabNet - 순차적 피처 어텐션
    - 피처 선택을 학습
    - 해석 가능한 예측
    """
    def forward(self, x, prior_scales):
        masked_x = x * prior_scales
        # 어텐션 계산
        attention = F.softmax(self.attention(n_a), dim=-1)
        return output, attention, new_prior_scales
```

### 4. TransformerTabular
```python
class TransformerTabular(nn.Module):
    """
    Transformer 기반 테이블 데이터 모델
    - 각 피처를 토큰으로 처리
    - Self-Attention으로 피처 간 관계 학습
    """
    def forward(self, x):
        x = self.feature_embedding(x.unsqueeze(-1))
        x = x + self.position_embedding(positions)
        x = self.transformer(torch.cat([cls_token, x], dim=1))
        return self.output(x[:, 0])  # CLS 토큰 사용
```

## 🔧 AutoML 기능

### 1. Optuna 기반 하이퍼파라미터 최적화
```python
class OptunaAutoML:
    """
    베이지안 최적화로 최적 하이퍼파라미터 탐색
    """
    def fit(self, X, y):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
```

### 2. 자동 앙상블
```python
class AutoEnsemble:
    """
    여러 모델을 자동으로 조합
    - Random Forest, Gradient Boosting, Extra Trees
    - Ridge, Lasso, ElasticNet
    - MLP, KNN
    """
    def fit(self, X, y):
        # 각 모델 학습
        # Optuna로 최적 가중치 탐색
```

### 3. Neural Architecture Search (NAS)
```python
class NeuralArchitectureSearch:
    """
    최적의 딥러닝 아키텍처 자동 탐색
    - 레이어 수, 유닛 수
    - 드롭아웃, 활성화 함수
    - 옵티마이저, 학습률
    """
```

## 📈 성능 메트릭

모델은 다음 메트릭으로 평가됩니다:

- **RMSE** (Root Mean Squared Error): 평균 제곱근 오차
- **MAE** (Mean Absolute Error): 평균 절대 오차
- **R²** (R-squared): 결정 계수
- **MAPE** (Mean Absolute Percentage Error): 평균 절대 백분율 오차

## 📊 예상 성능

| 모델 | RMSE (만엔) | R² |
|------|------------|-----|
| AutoML Ensemble | ~800-1000 | ~0.85 |
| Transformer | ~900-1100 | ~0.82 |
| ResNet | ~950-1150 | ~0.80 |
| MLP | ~1000-1200 | ~0.78 |

## 💡 가격 결정 요인

모델이 학습하는 주요 가격 결정 요인:

1. **위치 (구)**: 미나토구, 치요다구 등 도심 지역이 가장 비쌈
2. **면적**: 면적에 비례하여 가격 상승
3. **역세권**: 역까지 3분 이내면 프리미엄 30%
4. **건물 연령**: 연간 약 1.2% 가격 하락
5. **층수**: 고층일수록 약간의 프리미엄
6. **방향**: 남향 > 남동/남서 > 동/서 > 북향
7. **편의시설**: 주차장, 보안 시스템 등

## 🔮 예측 예시

```python
from models import TokyoHousingAutoML

# 모델 로드
automl = TokyoHousingAutoML()
automl.load('outputs/automl_model.pkl')

# 예측
sample = {
    'ward': 'minato',
    'floor_area_sqm': 70,
    'building_age_years': 5,
    'distance_to_station_min': 3,
    'floor_number': 10,
    ...
}
predicted_price = automl.predict(sample)
print(f"예상 가격: {predicted_price:,} 만엔")
# 예상 가격: 14,500 만엔 (약 1.45억 엔)
```

## 📄 라이선스

MIT License

## 🤝 기여

Pull Request 환영합니다!
