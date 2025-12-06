"""
딥러닝 모델 정의 - 도쿄 23구 주택 가격 예측
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers


class TokyoHousingPriceModel:
    """도쿄 주택 가격 예측 딥러닝 모델"""
    
    def __init__(self, input_dim=10, learning_rate=0.001):
        """
        Args:
            input_dim: 입력 특성 차원
            learning_rate: 학습률
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None
    
    def build_model(self, architecture='deep'):
        """
        딥러닝 모델 구성
        Args:
            architecture: 모델 아키텍처 타입 ('deep', 'wide_deep', 'residual')
        Returns:
            컴파일된 모델
        """
        if architecture == 'deep':
            model = self._build_deep_model()
        elif architecture == 'wide_deep':
            model = self._build_wide_deep_model()
        elif architecture == 'residual':
            model = self._build_residual_model()
        else:
            model = self._build_deep_model()
        
        # 모델 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def _build_deep_model(self):
        """깊은 신경망 모델 (Deep Neural Network)"""
        model = models.Sequential([
            # 입력 레이어
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,),
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 히든 레이어 1
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # 히든 레이어 2
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 히든 레이어 3
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 출력 레이어 (회귀 문제이므로 활성화 함수 없음)
            layers.Dense(1, name='price_output')
        ])
        
        return model
    
    def _build_wide_deep_model(self):
        """Wide & Deep 모델 (넓은 부분 + 깊은 부분)"""
        # 입력 레이어
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Wide 부분 (선형 모델)
        wide = layers.Dense(1, use_bias=True)(inputs)
        
        # Deep 부분 (깊은 신경망)
        deep = layers.Dense(128, activation='relu')(inputs)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        
        deep = layers.Dense(256, activation='relu')(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        
        deep = layers.Dense(128, activation='relu')(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.2)(deep)
        
        deep = layers.Dense(64, activation='relu')(deep)
        deep = layers.BatchNormalization()(deep)
        
        # Wide와 Deep 결합
        combined = layers.Concatenate()([wide, deep])
        outputs = layers.Dense(1, name='price_output')(combined)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _build_residual_model(self):
        """Residual Network (잔차 연결) 모델"""
        def residual_block(x, units, dropout_rate=0.2):
            """잔차 블록"""
            # 메인 경로
            main = layers.Dense(units, activation='relu')(x)
            main = layers.BatchNormalization()(main)
            main = layers.Dropout(dropout_rate)(main)
            main = layers.Dense(units, activation='relu')(main)
            main = layers.BatchNormalization()(main)
            
            # 입력과 차원이 다르면 조정
            if x.shape[-1] != units:
                x = layers.Dense(units)(x)
            
            # 잔차 연결
            out = layers.Add()([x, main])
            return layers.Activation('relu')(out)
        
        # 입력 레이어
        inputs = layers.Input(shape=(self.input_dim,))
        
        # 초기 레이어
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # 잔차 블록들
        x = residual_block(x, 256, dropout_rate=0.3)
        x = residual_block(x, 256, dropout_rate=0.3)
        x = residual_block(x, 128, dropout_rate=0.2)
        x = residual_block(x, 128, dropout_rate=0.2)
        x = residual_block(x, 64, dropout_rate=0.2)
        
        # 출력 레이어
        outputs = layers.Dense(1, name='price_output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def get_model_summary(self):
        """모델 구조 요약"""
        if self.model:
            return self.model.summary()
        return "모델이 아직 구성되지 않았습니다."
