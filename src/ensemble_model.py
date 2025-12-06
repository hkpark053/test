"""
앙상블 모델 - 여러 모델을 결합하여 더 나은 예측 성능 달성
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib


class EnsembleModel:
    """
    앙상블 모델
    
    여러 모델의 예측을 결합하여 더 강건한 예측을 수행합니다:
    - 딥러닝 모델
    - XGBoost
    - LightGBM
    - Random Forest
    """
    
    def __init__(self):
        self.models = {}
        self.weights = None
        
    def add_model(self, name, model, weight=1.0):
        """
        앙상블에 모델 추가
        
        Parameters:
        -----------
        name : str
            모델 이름
        model : object
            모델 객체
        weight : float
            앙상블 가중치
        """
        self.models[name] = {
            'model': model,
            'weight': weight
        }
        
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
        print("\n" + "=" * 80)
        print("XGBoost 학습 시작")
        print("=" * 80)
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=50
        )
        
        print("✓ XGBoost 학습 완료")
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        print("\n" + "=" * 80)
        print("LightGBM 학습 시작")
        print("=" * 80)
        
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        print("✓ LightGBM 학습 완료")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Random Forest 모델 학습"""
        print("\n" + "=" * 80)
        print("Random Forest 학습 시작")
        print("=" * 80)
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        print("✓ Random Forest 학습 완료")
        return model
    
    def predict(self, X, ward=None):
        """
        앙상블 예측
        
        Parameters:
        -----------
        X : np.ndarray
            예측할 데이터
        ward : np.ndarray, optional
            구 데이터 (딥러닝 모델용)
            
        Returns:
        --------
        np.ndarray
            앙상블 예측값
        """
        predictions = []
        weights = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            weight = model_info['weight']
            
            # 모델 타입에 따라 예측 방법 다름
            if hasattr(model, 'predict') and 'custom' in name.lower():
                # 커스텀 딥러닝 모델
                if ward is not None:
                    pred = model.predict([X, ward]).flatten()
                else:
                    continue
            elif hasattr(model, 'predict'):
                # 다른 ML 모델들
                pred = model.predict(X)
            else:
                continue
                
            predictions.append(pred)
            weights.append(weight)
        
        # 가중 평균
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test, ward_test=None):
        """
        앙상블 평가
        
        Parameters:
        -----------
        X_test : np.ndarray
            테스트 데이터
        y_test : np.ndarray
            테스트 타겟
        ward_test : np.ndarray, optional
            테스트 구 데이터
            
        Returns:
        --------
        dict
            평가 지표
        """
        y_pred = self.predict(X_test, ward_test)
        
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
        print("앙상블 모델 평가 결과")
        print("=" * 80)
        print(f"MAE (평균 절대 오차):     {mae:.2f} 백만 엔")
        print(f"RMSE (평균 제곱근 오차):   {rmse:.2f} 백만 엔")
        print(f"R² Score (결정 계수):      {r2:.4f}")
        print("=" * 80)
        
        return metrics
    
    def save(self, filepath):
        """앙상블 모델 저장"""
        save_dict = {}
        
        for name, model_info in self.models.items():
            if 'xgboost' in name.lower() or 'lightgbm' in name.lower() or 'forest' in name.lower():
                save_dict[name] = {
                    'model': model_info['model'],
                    'weight': model_info['weight']
                }
        
        joblib.dump(save_dict, filepath)
        print(f"✓ 앙상블 모델 저장: {filepath}")


if __name__ == "__main__":
    print("앙상블 모델 모듈")
    print("이 모듈은 train.py에서 호출됩니다.")
