"""
도쿄 23구 주택 가격 예측 AI 학습 스크립트
AutoML과 직접 구성한 딥러닝 모델을 모두 학습합니다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_loader import TokyoHousingDataLoader
from model import TokyoHousingPriceModel
from automl_model import AutoMLTokyoHousing


def train_manual_model(X_train, y_train, X_test, y_test, architecture='deep'):
    """직접 구성한 딥러닝 모델 학습"""
    print("\n" + "="*60)
    print(f"직접 구성한 딥러닝 모델 학습 시작 ({architecture} 아키텍처)")
    print("="*60)
    
    # 모델 생성
    model_builder = TokyoHousingPriceModel(
        input_dim=X_train.shape[1],
        learning_rate=0.001
    )
    
    # 모델 구성
    model = model_builder.build_model(architecture=architecture)
    
    # 모델 구조 출력
    print("\n모델 구조:")
    model.summary()
    
    # 콜백 설정
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_model_{architecture}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    test_loss, test_mae, test_mape = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n테스트 결과:")
    print(f"  Loss (MSE): {test_loss:.2f}")
    print(f"  MAE: {test_mae:.2f} 엔")
    print(f"  MAPE: {test_mape:.2f}%")
    
    # 예측
    y_pred = model.predict(X_test, verbose=0)
    
    return model, history, y_pred


def train_automl_model(X_train, y_train, X_test, y_test):
    """AutoKeras를 사용한 AutoML 모델 학습"""
    print("\n" + "="*60)
    print("AutoKeras AutoML 모델 학습 시작")
    print("="*60)
    
    automl = AutoMLTokyoHousing(max_trials=10, epochs=30)
    best_model = automl.build_automl_model(X_train, y_train, X_test, y_test)
    
    # 평가
    test_results = automl.evaluate(X_test, y_test)
    print(f"\nAutoML 테스트 결과:")
    print(f"  Loss: {test_results[0]:.2f}")
    print(f"  MAE: {test_results[1]:.2f} 엔")
    
    # 예측
    y_pred = automl.predict(X_test)
    
    # 최적 모델 구조 출력
    print("\nAutoKeras가 찾은 최적 모델 구조:")
    automl.get_best_model_architecture()
    
    return automl, y_pred


def plot_results(y_test, y_pred_manual, y_pred_automl, save_dir='results'):
    """결과 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 실제 vs 예측 (수동 모델)
    axes[0, 0].scatter(y_test, y_pred_manual, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('실제 가격 (엔)')
    axes[0, 0].set_ylabel('예측 가격 (엔)')
    axes[0, 0].set_title('직접 구성한 모델: 실제 vs 예측')
    axes[0, 0].grid(True)
    
    # 2. 실제 vs 예측 (AutoML 모델)
    axes[0, 1].scatter(y_test, y_pred_automl, alpha=0.5, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('실제 가격 (엔)')
    axes[0, 1].set_ylabel('예측 가격 (엔)')
    axes[0, 1].set_title('AutoML 모델: 실제 vs 예측')
    axes[0, 1].grid(True)
    
    # 3. 잔차 분포 (수동 모델)
    residuals_manual = y_test.flatten() - y_pred_manual.flatten()
    axes[1, 0].hist(residuals_manual, bins=50, alpha=0.7)
    axes[1, 0].set_xlabel('잔차 (엔)')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].set_title('직접 구성한 모델: 잔차 분포')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].grid(True)
    
    # 4. 잔차 분포 (AutoML 모델)
    residuals_automl = y_test.flatten() - y_pred_automl.flatten()
    axes[1, 1].hist(residuals_automl, bins=50, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('잔차 (엔)')
    axes[1, 1].set_ylabel('빈도')
    axes[1, 1].set_title('AutoML 모델: 잔차 분포')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_results.png', dpi=300, bbox_inches='tight')
    print(f"\n결과 그래프가 {save_dir}/prediction_results.png에 저장되었습니다.")
    plt.close()


def plot_training_history(history, save_dir='results'):
    """학습 히스토리 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss 그래프
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('모델 Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # MAE 그래프
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (엔)')
    axes[1].set_title('모델 MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"학습 히스토리 그래프가 {save_dir}/training_history.png에 저장되었습니다.")
    plt.close()


def main():
    """메인 함수"""
    print("="*60)
    print("도쿄 23구 주택 가격 예측 AI 학습 시작")
    print("="*60)
    
    # 데이터 로드 및 전처리
    print("\n1. 데이터 로드 및 전처리 중...")
    data_loader = TokyoHousingDataLoader(n_samples=5000)
    df = data_loader.load_data()
    
    print(f"   총 데이터 수: {len(df)}")
    print(f"   특성 수: {len(data_loader.get_feature_names())}")
    print(f"\n   데이터 샘플:")
    print(df.head())
    
    X_train, X_test, y_train, y_test, feature_names = data_loader.preprocess_data(df)
    
    print(f"\n   학습 데이터: {X_train.shape}")
    print(f"   테스트 데이터: {X_test.shape}")
    print(f"   가격 범위: {y_train.min():.0f} ~ {y_train.max():.0f} 엔")
    
    # 1. 직접 구성한 딥러닝 모델 학습
    model_deep, history_deep, y_pred_deep = train_manual_model(
        X_train, y_train, X_test, y_test, architecture='deep'
    )
    
    # 학습 히스토리 시각화
    plot_training_history(history_deep)
    
    # 2. AutoML 모델 학습
    automl_model, y_pred_automl = train_automl_model(
        X_train, y_train, X_test, y_test
    )
    
    # 결과 비교 및 시각화
    print("\n" + "="*60)
    print("모델 비교 결과")
    print("="*60)
    
    # 수동 모델 평가
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae_manual = mean_absolute_error(y_test, y_pred_deep)
    mse_manual = mean_squared_error(y_test, y_pred_deep)
    r2_manual = r2_score(y_test, y_pred_deep)
    
    mae_automl = mean_absolute_error(y_test, y_pred_automl)
    mse_automl = mean_squared_error(y_test, y_pred_automl)
    r2_automl = r2_score(y_test, y_pred_automl)
    
    print(f"\n직접 구성한 모델:")
    print(f"  MAE: {mae_manual:.2f} 엔")
    print(f"  MSE: {mse_manual:.2f}")
    print(f"  R² Score: {r2_manual:.4f}")
    
    print(f"\nAutoML 모델:")
    print(f"  MAE: {mae_automl:.2f} 엔")
    print(f"  MSE: {mse_automl:.2f}")
    print(f"  R² Score: {r2_automl:.4f}")
    
    # 결과 시각화
    plot_results(y_test, y_pred_deep, y_pred_automl)
    
    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
