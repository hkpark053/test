#!/usr/bin/env python3
"""
도쿄 23구 주택 가격 예측 - AutoML 학습 스크립트

이 스크립트는 AutoML을 사용하여 최적의 모델을 자동으로 탐색하고 학습합니다.

사용법:
    python train_automl.py --method ensemble --n_trials 100
"""

import argparse
import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.automl_models import (
    TokyoHousingAutoML,
    OptunaAutoML,
    AutoEnsemble,
    NeuralArchitectureSearch,
    AutoMLConfig
)
from data.generate_tokyo_housing_data import generate_housing_data, add_derived_features, split_data


def setup_args():
    """커맨드 라인 인자 설정"""
    parser = argparse.ArgumentParser(description='도쿄 23구 주택 가격 예측 AutoML')
    
    parser.add_argument('--method', type=str, default='ensemble',
                        choices=['optuna', 'ensemble', 'nas'],
                        help='AutoML 방법 (optuna, ensemble, nas)')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='하이퍼파라미터 탐색 횟수')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='생성할 샘플 수')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='결과 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    
    return parser.parse_args()


def load_or_generate_data(n_samples: int, data_dir: str = 'data') -> pd.DataFrame:
    """데이터 로드 또는 생성"""
    data_path = os.path.join(data_dir, 'tokyo_housing_data.csv')
    
    if os.path.exists(data_path):
        print(f"📂 기존 데이터 로드: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"🏠 새 데이터 생성 중 ({n_samples} 샘플)...")
        df = generate_housing_data(n_samples=n_samples)
        df = add_derived_features(df)
        
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"   저장 완료: {data_path}")
    
    return df


def visualize_data(df: pd.DataFrame, output_dir: str):
    """데이터 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 구별 평균 가격
    ax1 = axes[0, 0]
    ward_prices = df.groupby('ward_jp')['price_man_yen'].mean().sort_values(ascending=True)
    ward_prices.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('평균 가격 (만엔)')
    ax1.set_title('도쿄 23구별 평균 주택 가격')
    
    # 2. 면적 vs 가격
    ax2 = axes[0, 1]
    ax2.scatter(df['floor_area_sqm'], df['price_man_yen'], alpha=0.3, s=5)
    ax2.set_xlabel('면적 (㎡)')
    ax2.set_ylabel('가격 (만엔)')
    ax2.set_title('면적 vs 가격')
    
    # 3. 가격 분포
    ax3 = axes[1, 0]
    df['price_man_yen'].hist(bins=50, ax=ax3, color='steelblue', edgecolor='white')
    ax3.set_xlabel('가격 (만엔)')
    ax3.set_ylabel('빈도')
    ax3.set_title('가격 분포')
    
    # 4. 역까지 거리 vs 가격
    ax4 = axes[1, 1]
    ax4.scatter(df['distance_to_station_min'], df['price_man_yen'], alpha=0.3, s=5)
    ax4.set_xlabel('역까지 거리 (분)')
    ax4.set_ylabel('가격 (만엔)')
    ax4.set_title('역까지 거리 vs 가격')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_visualization.png'), dpi=150)
    plt.close()
    
    print(f"📊 시각화 저장: {output_dir}/data_visualization.png")


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    config: AutoMLConfig,
    output_dir: str
):
    """학습 및 평가"""
    
    print("\n" + "="*60)
    print(f"🚀 AutoML 학습 시작 (방법: {method})")
    print("="*60)
    
    # AutoML 모델 생성
    automl = TokyoHousingAutoML(config)
    
    # 학습
    start_time = datetime.now()
    automl.fit(train_df, target_col='price_man_yen', method=method)
    train_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n⏱️ 학습 시간: {train_time:.1f}초")
    
    # 평가
    print("\n📊 모델 평가 결과:")
    
    results = {}
    for name, data in [('train', train_df), ('val', val_df), ('test', test_df)]:
        metrics = automl.evaluate(data)
        results[name] = metrics
        print(f"\n   [{name.upper()}]")
        print(f"   - RMSE: {metrics['rmse']:,.0f} 만엔")
        print(f"   - MAE:  {metrics['mae']:,.0f} 만엔")
        print(f"   - R²:   {metrics['r2']:.4f}")
        print(f"   - MAPE: {metrics['mape']:.2f}%")
    
    # 예측 시각화
    visualize_predictions(automl, test_df, output_dir)
    
    # 모델 저장
    model_path = os.path.join(output_dir, 'automl_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(automl, f)
    print(f"\n💾 모델 저장: {model_path}")
    
    # 결과 저장
    results['train_time_seconds'] = train_time
    results['method'] = method
    results['config'] = {
        'n_trials': config.n_trials,
        'cv_folds': config.cv_folds
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📋 결과 저장: {results_path}")
    
    return automl, results


def visualize_predictions(automl, test_df: pd.DataFrame, output_dir: str):
    """예측 결과 시각화"""
    y_true = test_df['price_man_yen']
    y_pred = automl.predict(test_df)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 실제 vs 예측
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # 대각선
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax1.set_xlabel('실제 가격 (만엔)')
    ax1.set_ylabel('예측 가격 (만엔)')
    ax1.set_title('실제 vs 예측 가격')
    
    # 2. 오차 분포
    ax2 = axes[1]
    errors = y_pred - y_true
    ax2.hist(errors, bins=50, color='steelblue', edgecolor='white')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('예측 오차 (만엔)')
    ax2.set_ylabel('빈도')
    ax2.set_title(f'예측 오차 분포 (평균: {errors.mean():.0f}, 표준편차: {errors.std():.0f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_results.png'), dpi=150)
    plt.close()
    
    print(f"📊 예측 결과 시각화: {output_dir}/prediction_results.png")


def main():
    """메인 함수"""
    args = setup_args()
    
    # 랜덤 시드 설정
    np.random.seed(args.seed)
    
    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.method}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("🏠 도쿄 23구 주택 가격 예측 AutoML")
    print("="*60)
    print(f"   방법: {args.method}")
    print(f"   탐색 횟수: {args.n_trials}")
    print(f"   출력 디렉토리: {output_dir}")
    print("="*60)
    
    # 데이터 로드/생성
    df = load_or_generate_data(args.n_samples)
    print(f"\n📊 데이터 크기: {len(df)} 샘플")
    print(f"   피처 수: {len(df.columns) - 1}")
    print(f"   가격 범위: {df['price_man_yen'].min():,} ~ {df['price_man_yen'].max():,} 만엔")
    
    # 데이터 시각화
    visualize_data(df, output_dir)
    
    # 데이터 분할
    train_df, val_df, test_df = split_data(df)
    print(f"\n📂 데이터 분할:")
    print(f"   학습: {len(train_df)} 샘플")
    print(f"   검증: {len(val_df)} 샘플")
    print(f"   테스트: {len(test_df)} 샘플")
    
    # AutoML 설정
    config = AutoMLConfig(
        n_trials=args.n_trials,
        cv_folds=5,
        random_state=args.seed
    )
    
    # 학습 및 평가
    automl, results = train_and_evaluate(
        train_df, val_df, test_df,
        method=args.method,
        config=config,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("✅ AutoML 학습 완료!")
    print("="*60)
    print(f"\n📁 결과 디렉토리: {output_dir}")
    print("   - automl_model.pkl: 학습된 모델")
    print("   - results.json: 평가 결과")
    print("   - data_visualization.png: 데이터 시각화")
    print("   - prediction_results.png: 예측 결과 시각화")
    

if __name__ == "__main__":
    main()
