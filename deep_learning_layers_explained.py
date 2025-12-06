"""
딥러닝 레이어 상세 설명 및 구현 예시
각 레이어의 역할과 사용법을 자세히 설명합니다.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def explain_dense_layer():
    """
    Dense Layer (완전 연결층) 설명
    가장 기본적인 신경망 레이어
    """
    print("="*60)
    print("1. Dense Layer (완전 연결층)")
    print("="*60)
    print("""
    역할:
    - 모든 입력 뉴런이 모든 출력 뉴런과 연결됨
    - 선형 변환 후 활성화 함수 적용
    - 수식: output = activation(dot(input, weights) + bias)
    
    주요 파라미터:
    - units: 출력 뉴런 수
    - activation: 활성화 함수 ('relu', 'sigmoid', 'tanh' 등)
    - use_bias: 편향(bias) 사용 여부
    """)
    
    # 예시 코드
    print("\n예시 코드:")
    print("""
    # 128개의 뉴런을 가진 Dense 레이어
    dense_layer = layers.Dense(
        units=128,           # 출력 뉴런 수
        activation='relu',   # ReLU 활성화 함수
        name='dense_1'
    )
    
    # 사용 예시
    inputs = keras.Input(shape=(64,))  # 64차원 입력
    outputs = dense_layer(inputs)      # 128차원 출력
    """)
    
    # 실제 동작 예시
    print("\n실제 동작 예시:")
    example_input = np.random.randn(10, 64)  # 배치 크기 10, 특성 64
    dense_layer = layers.Dense(128, activation='relu')
    example_output = dense_layer(example_input)
    print(f"입력 shape: {example_input.shape}")
    print(f"출력 shape: {example_output.shape}")
    print(f"가중치 shape: {dense_layer.get_weights()[0].shape}")
    print(f"편향 shape: {dense_layer.get_weights()[1].shape}")

def explain_batch_normalization():
    """
    Batch Normalization 설명
    """
    print("\n" + "="*60)
    print("2. Batch Normalization Layer")
    print("="*60)
    print("""
    역할:
    - 각 배치의 입력을 정규화하여 학습 안정화
    - 내부 공변량 이동(Internal Covariate Shift) 감소
    - 학습 속도 향상 및 그래디언트 소실 문제 완화
    
    작동 원리:
    1. 배치의 평균과 분산 계산
    2. 정규화: (x - mean) / sqrt(variance + epsilon)
    3. 스케일 및 시프트: gamma * normalized + beta
    """)
    
    print("\n예시 코드:")
    print("""
    # BatchNormalization 레이어
    bn_layer = layers.BatchNormalization(
        name='bn_1',
        momentum=0.99,      # 이동 평균 업데이트 모멘텀
        epsilon=1e-5        # 분산에 더해지는 작은 값
    )
    
    # 사용 예시
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)  # 정규화
    """)

def explain_dropout():
    """
    Dropout Layer 설명
    """
    print("\n" + "="*60)
    print("3. Dropout Layer")
    print("="*60)
    print("""
    역할:
    - 학습 중 일부 뉴런을 무작위로 비활성화
    - 과적합(Overfitting) 방지
    - 모델의 일반화 성능 향상
    
    작동 원리:
    - 학습 시: 각 뉴런을 rate 확률로 0으로 설정
    - 추론 시: 모든 뉴런 사용 (출력에 1-rate를 곱함)
    """)
    
    print("\n예시 코드:")
    print("""
    # Dropout 레이어 (30%의 뉴런 비활성화)
    dropout_layer = layers.Dropout(
        rate=0.3,    # 비활성화할 뉴런 비율
        name='dropout_1'
    )
    
    # 사용 예시
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
    """)

def explain_residual_connection():
    """
    Residual Connection (잔차 연결) 설명
    """
    print("\n" + "="*60)
    print("4. Residual Connection (잔차 연결)")
    print("="*60)
    print("""
    역할:
    - 입력을 출력에 직접 더하는 연결
    - 그래디언트가 직접 전달되어 깊은 네트워크 학습 가능
    - ResNet에서 처음 도입된 기법
    
    작동 원리:
    - 출력 = F(x) + x (F는 레이어 변환)
    - 입력 x가 직접 출력에 더해짐
    """)
    
    print("\n예시 코드:")
    print("""
    # 잔차 연결 구현
    inputs = keras.Input(shape=(128,))
    
    # 잔차 저장
    residual = inputs
    
    # 레이어 변환
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # 잔차 연결
    outputs = layers.Add()([x, residual])
    """)
    
    # 실제 예시
    print("\n실제 구현 예시:")
    inputs = keras.Input(shape=(128,))
    residual = inputs
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Add()([x, residual])
    model = keras.Model(inputs=inputs, outputs=outputs)
    print("모델 구조:")
    model.summary()

def explain_attention_mechanism():
    """
    Attention Mechanism 설명
    """
    print("\n" + "="*60)
    print("5. Multi-Head Attention")
    print("="*60)
    print("""
    역할:
    - 입력 간의 관계를 학습
    - 중요한 특성에 더 많은 가중치 부여
    - Transformer 아키텍처의 핵심 구성 요소
    
    작동 원리:
    1. Query, Key, Value 벡터 생성
    2. Attention Score 계산: Q·K^T / sqrt(d_k)
    3. Softmax로 가중치 계산
    4. Value에 가중치 적용하여 출력 생성
    """)
    
    print("\n예시 코드:")
    print("""
    # Multi-Head Attention
    attention_layer = layers.MultiHeadAttention(
        num_heads=4,      # 어텐션 헤드 수
        key_dim=32,       # 각 헤드의 키 차원
        name='attention'
    )
    
    # 사용 예시
    # self-attention: 같은 입력을 query, key, value로 사용
    attention_output = attention_layer(query, key, value)
    """)

def explain_optimizer():
    """
    Optimizer 설명
    """
    print("\n" + "="*60)
    print("6. Optimizer (최적화 알고리즘)")
    print("="*60)
    print("""
    역할:
    - 손실 함수를 최소화하도록 가중치 업데이트
    - 학습률(learning rate) 조절
    
    주요 옵티마이저:
    
    1. Adam (Adaptive Moment Estimation)
       - 모멘텀과 RMSprop의 조합
       - 각 파라미터마다 적응적 학습률 사용
       - 가장 널리 사용됨
    
    2. SGD (Stochastic Gradient Descent)
       - 기본 경사 하강법
       - 모멘텀 옵션 가능
    
    3. RMSprop
       - 학습률을 자동으로 조정
    """)
    
    print("\n예시 코드:")
    print("""
    # Adam 옵티마이저
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,    # 학습률
        beta_1=0.9,             # 첫 번째 모멘텀 계수
        beta_2=0.999            # 두 번째 모멘텀 계수
    )
    
    # 모델 컴파일 시 사용
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    """)

def explain_loss_functions():
    """
    Loss Function 설명
    """
    print("\n" + "="*60)
    print("7. Loss Function (손실 함수)")
    print("="*60)
    print("""
    역할:
    - 모델의 예측과 실제 값의 차이를 측정
    - 학습의 목표 함수
    
    회귀 문제에서 주로 사용:
    
    1. MSE (Mean Squared Error)
       - 평균 제곱 오차
       - 큰 오차에 더 큰 페널티
       - 수식: mean((y_true - y_pred)^2)
    
    2. MAE (Mean Absolute Error)
       - 평균 절대 오차
       - 이상치에 덜 민감
       - 수식: mean(|y_true - y_pred|)
    
    3. Huber Loss
       - MSE와 MAE의 조합
       - 이상치에 강건함
    """)
    
    print("\n예시 코드:")
    print("""
    # MSE 손실 함수
    model.compile(
        optimizer='adam',
        loss='mse',           # 또는 'mean_squared_error'
        metrics=['mae', 'mse']
    )
    
    # 커스텀 손실 함수
    def custom_loss(y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        return mse + 0.5 * mae
    """)

def build_complete_example():
    """
    모든 레이어를 사용한 완전한 예시 모델
    """
    print("\n" + "="*60)
    print("8. 완전한 모델 예시 (모든 레이어 사용)")
    print("="*60)
    
    inputs = keras.Input(shape=(7,), name='input')
    
    # Dense Layer
    x = layers.Dense(256, activation='relu', name='dense_1')(inputs)
    print("✓ Dense(256) 레이어 추가")
    
    # Batch Normalization
    x = layers.BatchNormalization(name='bn_1')(x)
    print("✓ BatchNormalization 레이어 추가")
    
    # Dropout
    x = layers.Dropout(0.3, name='dropout_1')(x)
    print("✓ Dropout(0.3) 레이어 추가")
    
    # 두 번째 Dense + BN + Dropout 블록
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    print("✓ 두 번째 블록 추가")
    
    # 잔차 연결을 위한 준비
    residual = x
    
    # 잔차 블록
    x = layers.Dense(64, activation='relu', name='dense_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Dense(64, activation='relu', name='dense_4')(x)
    x = layers.Add(name='residual_connection')([x, residual])
    print("✓ 잔차 연결 블록 추가")
    
    # 최종 출력 레이어
    outputs = layers.Dense(1, name='output')(x)
    print("✓ 출력 레이어 추가")
    
    # 모델 생성
    model = keras.Model(inputs=inputs, outputs=outputs, name='Complete_Example')
    
    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    print("\n완성된 모델 구조:")
    model.summary()
    
    return model

if __name__ == '__main__':
    print("\n" + "="*60)
    print("딥러닝 레이어 상세 설명")
    print("="*60)
    
    # 각 레이어 설명
    explain_dense_layer()
    explain_batch_normalization()
    explain_dropout()
    explain_residual_connection()
    explain_attention_mechanism()
    explain_optimizer()
    explain_loss_functions()
    
    # 완전한 예시 모델
    model = build_complete_example()
    
    print("\n" + "="*60)
    print("모든 설명이 완료되었습니다!")
    print("="*60)
    print("\n이 모델은 도쿄 집값 예측에 사용할 수 있는 완전한 구조입니다.")
    print("각 레이어의 역할을 이해하고 필요에 따라 수정하여 사용하세요.")
