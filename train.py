"""
메인 학습 스크립트

도쿄 주택 가격 예측 모델 학습을 실행합니다.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 모듈 임포트
sys.path.append('/workspace/src')
from data_preprocessing import TokyoHousingDataPreprocessor
from automl_model import AutoMLHousingModel
from custom_deep_model import CustomDeepHousingModel
from ensemble_model import EnsembleModel

# 랜덤 시드 설정
np.random.seed(42)
tf.random.set_seed(42)


def train_automl(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    AutoML 모델 학습 및 평가
    
    Parameters:
    -----------
    X_train, y_train : 학습 데이터
    X_val, y_val : 검증 데이터
    X_test, y_test : 테스트 데이터
    
    Returns:
    --------
    AutoMLHousingModel, dict
        학습된 모델과 평가 지표
    """
    print("\n" + "=" * 80)
    print("AutoML 모델 학습")
    print("=" * 80)
    
    # 모델 생성
    automl = AutoMLHousingModel(
        max_trials=10,  # 시간 절약을 위해 10개 시도
        epochs=50,      # 빠른 학습을 위해 50 에포크
        objective='val_mae'
    )
    
    # 학습
    automl.search_and_train(X_train, y_train, X_val, y_val)
    
    # 최적 모델 구조 출력
    automl.get_best_model_summary()
    
    # 평가
    metrics = automl.evaluate(X_test, y_test)
    
    # 모델 저장
    automl.save('/workspace/models/automl_model.h5')
    
    return automl, metrics


def train_custom_deep(X_train, ward_train, y_train, 
                      X_val, ward_val, y_val,
                      X_test, ward_test, y_test):
    """
    커스텀 딥러닝 모델 학습 및 평가
    
    Parameters:
    -----------
    X_train, ward_train, y_train : 학습 데이터
    X_val, ward_val, y_val : 검증 데이터
    X_test, ward_test, y_test : 테스트 데이터
    
    Returns:
    --------
    CustomDeepHousingModel, dict
        학습된 모델과 평가 지표
    """
    print("\n" + "=" * 80)
    print("커스텀 딥러닝 모델 학습")
    print("=" * 80)
    
    # 모델 생성
    num_features = X_train.shape[1]
    num_wards = 23  # 도쿄 23구
    
    model = CustomDeepHousingModel(
        num_features=num_features,
        num_wards=num_wards,
        embedding_dim=8
    )
    
    # 모델 구조 출력
    model.get_summary()
    
    # 컴파일
    model.compile_model(learning_rate=0.001)
    
    # 학습
    history = model.train(
        X_train, ward_train, y_train,
        X_val, ward_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # 평가
    metrics = model.evaluate(X_test, ward_test, y_test)
    
    # 모델 저장
    model.save('/workspace/models/custom_deep_model.h5')
    
    return model, metrics


def train_ensemble(X_train, y_train, X_val, y_val, 
                   X_test, ward_test, y_test, custom_model):
    """
    앙상블 모델 학습 및 평가
    
    Parameters:
    -----------
    X_train, y_train : 학습 데이터
    X_val, y_val : 검증 데이터
    X_test, ward_test, y_test : 테스트 데이터
    custom_model : 학습된 커스텀 딥러닝 모델
    
    Returns:
    --------
    EnsembleModel, dict
        학습된 앙상블 모델과 평가 지표
    """
    print("\n" + "=" * 80)
    print("앙상블 모델 학습")
    print("=" * 80)
    
    ensemble = EnsembleModel()
    
    # 딥러닝 모델 추가
    ensemble.add_model('custom_deep', custom_model.model, weight=2.0)
    
    # XGBoost 학습 및 추가
    xgb_model = ensemble.train_xgboost(X_train, y_train, X_val, y_val)
    ensemble.add_model('xgboost', xgb_model, weight=1.5)
    
    # LightGBM 학습 및 추가
    lgb_model = ensemble.train_lightgbm(X_train, y_train, X_val, y_val)
    ensemble.add_model('lightgbm', lgb_model, weight=1.5)
    
    # Random Forest 학습 및 추가
    rf_model = ensemble.train_random_forest(X_train, y_train)
    ensemble.add_model('random_forest', rf_model, weight=1.0)
    
    # 평가
    metrics = ensemble.evaluate(X_test, y_test, ward_test)
    
    # 모델 저장
    ensemble.save('/workspace/models/ensemble_model.pkl')
    
    return ensemble, metrics


def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description='도쿄 주택 가격 예측 모델 학습')
    parser.add_argument('--model', type=str, default='custom', 
                       choices=['automl', 'custom', 'ensemble', 'all'],
                       help='학습할 모델 종류')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("도쿄 23구 주택 가격 예측 AI - 학습 시작")
    print("=" * 80)
    
    # ========================================================================
    # 1. 데이터 로드 및 전처리
    # ========================================================================
    
    print("\n[1/4] 데이터 로드 및 전처리")
    print("-" * 80)
    
    preprocessor = TokyoHousingDataPreprocessor()
    
    # 데이터 로드
    df = preprocessor.load_data('/workspace/data/tokyo_housing_data.csv')
    
    # 특성 엔지니어링
    df = preprocessor.create_features(df)
    
    # 데이터 분할 및 전처리
    X_train, X_val, X_test, y_train, y_val, y_test, ward_train, ward_val, ward_test = \
        preprocessor.prepare_data(df, test_size=0.2, val_size=0.1)
    
    # 전처리기 저장
    preprocessor.save('/workspace/models/preprocessor.pkl')
    
    # ========================================================================
    # 2. 모델 학습
    # ========================================================================
    
    results = {}
    
    if args.model == 'automl' or args.model == 'all':
        print("\n[2/4] AutoML 모델 학습")
        print("-" * 80)
        automl_model, automl_metrics = train_automl(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        results['automl'] = automl_metrics
    
    if args.model == 'custom' or args.model == 'all':
        print("\n[2/4] 커스텀 딥러닝 모델 학습")
        print("-" * 80)
        custom_model, custom_metrics = train_custom_deep(
            X_train, ward_train, y_train,
            X_val, ward_val, y_val,
            X_test, ward_test, y_test
        )
        results['custom'] = custom_metrics
    
    if args.model == 'ensemble' or args.model == 'all':
        print("\n[3/4] 앙상블 모델 학습")
        print("-" * 80)
        
        # 커스텀 모델이 없으면 먼저 학습
        if 'custom_model' not in locals():
            custom_model, _ = train_custom_deep(
                X_train, ward_train, y_train,
                X_val, ward_val, y_val,
                X_test, ward_test, y_test
            )
        
        ensemble_model, ensemble_metrics = train_ensemble(
            X_train, y_train, X_val, y_val,
            X_test, ward_test, y_test, custom_model
        )
        results['ensemble'] = ensemble_metrics
    
    # ========================================================================
    # 3. 최종 결과 출력
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("학습 완료! 최종 결과 요약")
    print("=" * 80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} 모델:")
        print(f"  MAE:  {metrics['mae']:.2f} 백만 엔")
        print(f"  RMSE: {metrics['rmse']:.2f} 백만 엔")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    print("\n" + "=" * 80)
    print("모든 모델이 /workspace/models/ 디렉토리에 저장되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    main()
