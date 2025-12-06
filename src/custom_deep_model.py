"""
커스텀 딥러닝 모델 (상세한 레이어 구현)

이 모듈은 도쿄 주택 가격 예측을 위한 고급 딥러닝 아키텍처를 구현합니다.
다음 기법들을 포함합니다:
- 임베딩 레이어 (카테고리형 변수)
- Dense 레이어 (여러 은닉층)
- Batch Normalization (학습 안정화)
- Dropout (과적합 방지)
- Attention 메커니즘 (중요 특성 강조)
- Residual Connections (그래디언트 흐름 개선)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class AttentionLayer(layers.Layer):
    """
    셀프 어텐션 레이어
    
    중요한 특성에 더 많은 가중치를 부여하여 모델의 예측 성능을 향상시킵니다.
    """
    
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # 어텐션 가중치를 계산하기 위한 Dense 레이어
        self.W_query = self.add_weight(
            name='W_query',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_key = self.add_weight(
            name='W_key',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_value = self.add_weight(
            name='W_value',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Query, Key, Value 계산
        query = tf.matmul(inputs, self.W_query)
        key = tf.matmul(inputs, self.W_key)
        value = tf.matmul(inputs, self.W_value)
        
        # 어텐션 스코어 계산
        score = tf.matmul(query, key, transpose_b=True)
        score = score / tf.math.sqrt(tf.cast(self.units, dtype=tf.float32))
        
        # Softmax로 정규화
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # 가중 합 계산
        output = tf.matmul(attention_weights, value)
        
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config


class ResidualBlock(layers.Layer):
    """
    Residual Block (잔차 블록)
    
    Skip connection을 통해 그래디언트 소실 문제를 완화하고
    더 깊은 네트워크 학습을 가능하게 합니다.
    """
    
    def __init__(self, units, dropout_rate=0.3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        # 첫 번째 경로 (메인 경로)
        self.dense1 = layers.Dense(units, activation=None)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.dense2 = layers.Dense(units, activation=None)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Skip connection을 위한 projection (차원 맞추기)
        self.projection = layers.Dense(units, activation=None)
        
    def call(self, inputs, training=False):
        # 메인 경로
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection (잔차 연결)
        shortcut = self.projection(inputs)
        
        # 메인 경로와 skip connection 합치기
        x = x + shortcut
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        
        return x
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config


class CustomDeepHousingModel:
    """
    커스텀 딥러닝 모델
    
    아키텍처:
    1. 입력 레이어 (수치형 특성 + 구 임베딩)
    2. 특성 결합 레이어
    3. Dense 레이어 (512 -> 256 -> 128)
    4. Attention 레이어 (중요 특성 강조)
    5. Residual Block (잔차 학습)
    6. 출력 레이어 (가격 예측)
    """
    
    def __init__(self, num_features, num_wards, embedding_dim=8):
        """
        Parameters:
        -----------
        num_features : int
            수치형 특성 개수
        num_wards : int
            구(ward) 개수
        embedding_dim : int
            임베딩 차원
        """
        self.num_features = num_features
        self.num_wards = num_wards
        self.embedding_dim = embedding_dim
        self.model = None
        
    def build_model(self):
        """
        모델 아키텍처 구축
        
        Returns:
        --------
        keras.Model
            구축된 모델
        """
        
        # ====================================================================
        # 1. 입력 레이어
        # ====================================================================
        
        # 수치형 특성 입력
        numerical_input = layers.Input(shape=(self.num_features,), name='numerical_input')
        
        # 구(ward) 카테고리 입력
        ward_input = layers.Input(shape=(1,), name='ward_input')
        
        # ====================================================================
        # 2. 임베딩 레이어 (카테고리형 변수 -> 밀집 벡터)
        # ====================================================================
        
        # 구를 저차원 밀집 벡터로 변환
        ward_embedding = layers.Embedding(
            input_dim=self.num_wards,
            output_dim=self.embedding_dim,
            name='ward_embedding'
        )(ward_input)
        ward_embedding = layers.Flatten(name='ward_embedding_flat')(ward_embedding)
        
        # ====================================================================
        # 3. 특성 결합
        # ====================================================================
        
        # 수치형 특성과 임베딩 결합
        combined = layers.Concatenate(name='feature_combination')([
            numerical_input, 
            ward_embedding
        ])
        
        # 입력 정규화 (배치 정규화)
        x = layers.BatchNormalization(name='input_batch_norm')(combined)
        
        # ====================================================================
        # 4. 첫 번째 Dense 블록 (512 유닛)
        # ====================================================================
        
        x = layers.Dense(512, activation=None, name='dense1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu', name='relu1')(x)
        x = layers.Dropout(0.3, name='dropout1')(x)
        
        # ====================================================================
        # 5. 두 번째 Dense 블록 (256 유닛)
        # ====================================================================
        
        x = layers.Dense(256, activation=None, name='dense2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Activation('relu', name='relu2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        # ====================================================================
        # 6. 세 번째 Dense 블록 (128 유닛)
        # ====================================================================
        
        x = layers.Dense(128, activation=None, name='dense3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.Activation('relu', name='relu3')(x)
        x = layers.Dropout(0.2, name='dropout3')(x)
        
        # ====================================================================
        # 7. Attention 메커니즘 (중요 특성 강조)
        # ====================================================================
        
        # 특성 확장 (어텐션을 위해) - Reshape 사용
        x_expanded = layers.Reshape((1, 128), name='reshape_for_attention')(x)
        attention_output = AttentionLayer(units=128, name='attention')(x_expanded)
        attention_output = layers.Flatten(name='attention_flat')(attention_output)
        
        # 원본과 어텐션 출력 결합
        x = layers.Concatenate(name='attention_concat')([x, attention_output])
        
        # ====================================================================
        # 8. Residual Block (잔차 학습)
        # ====================================================================
        
        x = ResidualBlock(units=128, dropout_rate=0.2, name='residual_block1')(x)
        x = ResidualBlock(units=64, dropout_rate=0.2, name='residual_block2')(x)
        
        # ====================================================================
        # 9. 최종 Dense 레이어
        # ====================================================================
        
        x = layers.Dense(32, activation='relu', name='dense_final')(x)
        x = layers.Dropout(0.1, name='dropout_final')(x)
        
        # ====================================================================
        # 10. 출력 레이어 (가격 예측)
        # ====================================================================
        
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        # ====================================================================
        # 모델 생성
        # ====================================================================
        
        model = Model(
            inputs=[numerical_input, ward_input],
            outputs=output,
            name='TokyoHousingPricePredictor'
        )
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        모델 컴파일
        
        Parameters:
        -----------
        learning_rate : float
            학습률
        """
        
        if self.model is None:
            self.model = self.build_model()
        
        # 옵티마이저 설정 (Adam)
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # 컴파일
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error
            metrics=[
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.MeanSquaredError(name='mse'),
                keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        
        print("✓ 모델 컴파일 완료")
        
    def get_summary(self):
        """모델 구조 출력"""
        if self.model is None:
            self.compile_model()
        
        print("\n" + "=" * 80)
        print("커스텀 딥러닝 모델 구조")
        print("=" * 80)
        self.model.summary()
        print("=" * 80)
        
    def train(self, X_train, ward_train, y_train, 
              X_val, ward_val, y_val,
              epochs=200, batch_size=32):
        """
        모델 학습
        
        Parameters:
        -----------
        X_train : np.ndarray
            학습 특성
        ward_train : np.ndarray
            학습 구 데이터
        y_train : np.ndarray
            학습 타겟
        X_val : np.ndarray
            검증 특성
        ward_val : np.ndarray
            검증 구 데이터
        y_val : np.ndarray
            검증 타겟
        epochs : int
            학습 에포크
        batch_size : int
            배치 크기
            
        Returns:
        --------
        keras.callbacks.History
            학습 히스토리
        """
        
        if self.model is None:
            self.compile_model()
        
        # 콜백 설정
        callbacks = [
            # 조기 종료 (검증 손실이 개선되지 않으면 중단)
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # 학습률 감소 (plateau 도달 시)
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # 최적 모델 저장
            ModelCheckpoint(
                filepath='/workspace/models/custom_best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\n" + "=" * 80)
        print("모델 학습 시작")
        print("=" * 80)
        print(f"에포크: {epochs}")
        print(f"배치 크기: {batch_size}")
        print(f"학습 샘플: {len(X_train)}")
        print(f"검증 샘플: {len(X_val)}")
        print("=" * 80 + "\n")
        
        # 학습
        history = self.model.fit(
            [X_train, ward_train],
            y_train,
            validation_data=([X_val, ward_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "=" * 80)
        print("학습 완료!")
        print("=" * 80)
        
        return history
    
    def evaluate(self, X_test, ward_test, y_test):
        """
        모델 평가
        
        Parameters:
        -----------
        X_test : np.ndarray
            테스트 특성
        ward_test : np.ndarray
            테스트 구 데이터
        y_test : np.ndarray
            테스트 타겟
            
        Returns:
        --------
        dict
            평가 지표
        """
        
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 예측
        y_pred = self.model.predict([X_test, ward_test]).flatten()
        
        # 지표 계산
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print("\n" + "=" * 80)
        print("커스텀 딥러닝 모델 평가 결과")
        print("=" * 80)
        print(f"MAE (평균 절대 오차):     {mae:.2f} 백만 엔")
        print(f"RMSE (평균 제곱근 오차):   {rmse:.2f} 백만 엔")
        print(f"R² Score (결정 계수):      {r2:.4f}")
        print("=" * 80)
        
        return metrics
    
    def predict(self, X, ward):
        """
        예측 수행
        
        Parameters:
        -----------
        X : np.ndarray
            예측할 특성 데이터
        ward : np.ndarray
            구 데이터
            
        Returns:
        --------
        np.ndarray
            예측값
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.model.predict([X, ward]).flatten()
    
    def save(self, filepath):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        self.model.save(filepath)
        print(f"✓ 커스텀 모델 저장: {filepath}")
    
    def load(self, filepath):
        """모델 로드"""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={
                'AttentionLayer': AttentionLayer,
                'ResidualBlock': ResidualBlock,
                'mae': keras.metrics.MeanAbsoluteError(),
                'mse': keras.metrics.MeanSquaredError(),
                'rmse': keras.metrics.RootMeanSquaredError()
            }
        )
        print(f"✓ 커스텀 모델 로드: {filepath}")


if __name__ == "__main__":
    print("커스텀 딥러닝 모델 모듈")
    print("이 모듈은 train.py에서 호출됩니다.")
    
    # 테스트: 모델 구조 확인
    model = CustomDeepHousingModel(num_features=19, num_wards=23)
    model.get_summary()
