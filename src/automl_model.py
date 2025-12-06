"""
AutoKeras를 사용한 AutoML 모델

이 모듈은 AutoKeras를 사용하여 자동으로 최적의 신경망을 찾습니다.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
from datetime import datetime
import os


class AutoMLHousingModel:
    """AutoKeras 기반 AutoML 모델"""
    
    def __init__(self, max_trials=20, epochs=100, objective='val_mae'):
        """
        Parameters:
        -----------
        max_trials : int
            시도할 모델 아키텍처 수
        epochs : int
            각 모델의 최대 학습 에포크
        objective : str
            최적화할 목표 지표
        """
        self.max_trials = max_trials
        self.epochs = epochs
        self.objective = objective
        self.model = None
        self.search_model = None
        
    def build_search_space(self, input_shape):
        """
        AutoKeras 검색 공간 정의
        
        Parameters:
        -----------
        input_shape : tuple
            입력 데이터 형태
            
        Returns:
        --------
        ak.AutoModel
            AutoKeras 검색 모델
        """
        
        # 입력 노드
        input_node = ak.Input()
        
        # 구조화된 데이터 전처리 블록
        # AutoKeras가 자동으로 최적의 전처리 방법 선택
        output_node = ak.StructuredDataBlock()(input_node)
        
        # Dense 블록 - 자동으로 레이어 수, 유닛 수, 활성화 함수 등을 탐색
        output_node = ak.DenseBlock(
            num_layers=None,  # 자동으로 최적 레이어 수 탐색 (1-5개)
            use_batchnorm=None,  # 배치 정규화 사용 여부 자동 결정
            dropout=None  # 드롭아웃 비율 자동 결정 (0-0.5)
        )(output_node)
        
        # 회귀 헤드
        output_node = ak.RegressionHead(
            loss='mse',
            metrics=['mae', 'mse']
        )(output_node)
        
        # AutoModel 생성
        search_model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            max_trials=self.max_trials,
            objective=self.objective,
            overwrite=True,
            directory='/workspace/models/automl_search',
            project_name=f'tokyo_housing_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        return search_model
    
    def search_and_train(self, X_train, y_train, X_val, y_val):
        """
        모델 구조 탐색 및 학습
        
        Parameters:
        -----------
        X_train : np.ndarray
            학습 데이터
        y_train : np.ndarray
            학습 타겟
        X_val : np.ndarray
            검증 데이터
        y_val : np.ndarray
            검증 타겟
        """
        
        print("=" * 80)
        print("AutoKeras 모델 탐색 시작")
        print("=" * 80)
        print(f"최대 시도 횟수: {self.max_trials}")
        print(f"에포크: {self.epochs}")
        print(f"최적화 목표: {self.objective}")
        print()
        
        # 검색 공간 구축
        self.search_model = self.build_search_space(X_train.shape[1:])
        
        # 콜백 설정
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_mae',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_mae',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # 탐색 및 학습
        self.search_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # 최적 모델 추출
        self.model = self.search_model.export_model()
        
        print("\n" + "=" * 80)
        print("AutoKeras 탐색 완료!")
        print("=" * 80)
        
    def get_best_model_summary(self):
        """최적 모델 구조 출력"""
        if self.model is not None:
            print("\n최적 모델 구조:")
            print("-" * 80)
            self.model.summary()
            print("-" * 80)
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        
        Parameters:
        -----------
        X_test : np.ndarray
            테스트 데이터
        y_test : np.ndarray
            테스트 타겟
            
        Returns:
        --------
        dict
            평가 지표
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 search_and_train()을 실행하세요.")
        
        # 예측
        y_pred = self.model.predict(X_test).flatten()
        
        # 지표 계산
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print("\n" + "=" * 80)
        print("AutoML 모델 평가 결과")
        print("=" * 80)
        print(f"MAE (평균 절대 오차):     {mae:.2f} 백만 엔")
        print(f"RMSE (평균 제곱근 오차):   {rmse:.2f} 백만 엔")
        print(f"R² Score (결정 계수):      {r2:.4f}")
        print("=" * 80)
        
        return metrics
    
    def predict(self, X):
        """
        예측 수행
        
        Parameters:
        -----------
        X : np.ndarray
            예측할 데이터
            
        Returns:
        --------
        np.ndarray
            예측값
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.model.predict(X).flatten()
    
    def save(self, filepath):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        self.model.save(filepath)
        print(f"✓ AutoML 모델 저장: {filepath}")
    
    def load(self, filepath):
        """모델 로드"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ AutoML 모델 로드: {filepath}")


if __name__ == "__main__":
    print("AutoKeras AutoML 모듈")
    print("이 모듈은 train.py에서 호출됩니다.")
