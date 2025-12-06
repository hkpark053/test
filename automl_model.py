"""
AutoKeras를 사용한 AutoML 모델 - 도쿄 23구 주택 가격 예측
"""
import autokeras as ak
import numpy as np


class AutoMLTokyoHousing:
    """AutoKeras를 사용한 자동 머신러닝 모델"""
    
    def __init__(self, max_trials=10, epochs=50):
        """
        Args:
            max_trials: AutoKeras가 시도할 최대 모델 수
            epochs: 각 모델의 학습 에포크 수
        """
        self.max_trials = max_trials
        self.epochs = epochs
        self.model = None
        self.best_model = None
    
    def build_automl_model(self, X_train, y_train, X_test, y_test):
        """
        AutoKeras를 사용하여 최적 모델 자동 탐색
        Args:
            X_train: 학습 데이터 특성
            y_train: 학습 데이터 타겟
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 타겟
        Returns:
            학습된 모델
        """
        # AutoKeras StructuredDataRegressor 사용
        # 회귀 문제이므로 StructuredDataRegressor 사용
        self.model = ak.StructuredDataRegressor(
            max_trials=self.max_trials,
            overwrite=True,
            metrics=['mae'],
            loss='mse'
        )
        
        # 모델 학습 (자동으로 최적 아키텍처 탐색)
        print("AutoKeras가 최적 모델을 탐색 중...")
        self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 최적 모델 추출
        self.best_model = self.model.export_model()
        
        return self.best_model
    
    def predict(self, X):
        """예측 수행"""
        if self.model:
            return self.model.predict(X)
        else:
            raise ValueError("모델이 학습되지 않았습니다.")
    
    def evaluate(self, X_test, y_test):
        """모델 평가"""
        if self.model:
            return self.model.evaluate(X_test, y_test)
        else:
            raise ValueError("모델이 학습되지 않았습니다.")
    
    def get_best_model_architecture(self):
        """최적 모델의 아키텍처 반환"""
        if self.best_model:
            return self.best_model.summary()
        return "모델이 아직 학습되지 않았습니다."
