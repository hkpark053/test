"""
예측 스크립트

학습된 모델을 사용하여 새로운 데이터에 대한 가격 예측을 수행합니다.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspace/src')
from custom_deep_model import CustomDeepHousingModel


def load_models(model_type='custom'):
    """
    저장된 모델 로드
    
    Parameters:
    -----------
    model_type : str
        로드할 모델 타입
        
    Returns:
    --------
    model, preprocessor
        로드된 모델과 전처리기
    """
    # 전처리기 로드
    preprocessor = joblib.load('/workspace/models/preprocessor.pkl')
    
    # 모델 로드
    if model_type == 'custom':
        model = CustomDeepHousingModel(
            num_features=len(preprocessor['feature_names']),
            num_wards=23
        )
        model.load('/workspace/models/custom_deep_model.h5')
    elif model_type == 'automl':
        from tensorflow import keras
        model = keras.models.load_model('/workspace/models/automl_model.h5')
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    return model, preprocessor


def create_sample_data():
    """
    예측을 위한 샘플 데이터 생성
    
    Returns:
    --------
    pd.DataFrame
        샘플 데이터
    """
    samples = [
        {
            'ward': 'Minato',
            'area_sqm': 80.0,
            'rooms': 3,
            'floor': 15,
            'building_age': 5.0,
            'distance_to_station_min': 5.0,
            'amenity_score': 85.0,
            'has_balcony': 1,
            'has_parking': 1,
            'is_remodeled': 0,
            'description': '미나토구, 신축 고급 아파트, 역 근처'
        },
        {
            'ward': 'Shibuya',
            'area_sqm': 65.0,
            'rooms': 2,
            'floor': 8,
            'building_age': 10.0,
            'distance_to_station_min': 8.0,
            'amenity_score': 75.0,
            'has_balcony': 1,
            'has_parking': 0,
            'is_remodeled': 1,
            'description': '시부야구, 리모델링 완료, 중심가'
        },
        {
            'ward': 'Adachi',
            'area_sqm': 70.0,
            'rooms': 3,
            'floor': 3,
            'building_age': 25.0,
            'distance_to_station_min': 15.0,
            'amenity_score': 55.0,
            'has_balcony': 1,
            'has_parking': 1,
            'is_remodeled': 0,
            'description': '아다치구, 오래된 아파트, 주거 지역'
        },
        {
            'ward': 'Chiyoda',
            'area_sqm': 120.0,
            'rooms': 4,
            'floor': 25,
            'building_age': 2.0,
            'distance_to_station_min': 3.0,
            'amenity_score': 95.0,
            'has_balcony': 1,
            'has_parking': 1,
            'is_remodeled': 0,
            'description': '치요다구, 초고급 펜트하우스, 역 직결'
        },
        {
            'ward': 'Setagaya',
            'area_sqm': 90.0,
            'rooms': 3,
            'floor': 5,
            'building_age': 15.0,
            'distance_to_station_min': 10.0,
            'amenity_score': 70.0,
            'has_balcony': 1,
            'has_parking': 1,
            'is_remodeled': 1,
            'description': '세타가야구, 주거 선호 지역, 넓은 평수'
        }
    ]
    
    return pd.DataFrame(samples)


def preprocess_for_prediction(df, preprocessor):
    """
    예측을 위한 데이터 전처리
    
    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터
    preprocessor : dict
        저장된 전처리기
        
    Returns:
    --------
    X_scaled, ward_encoded
        전처리된 특성과 구 인코딩
    """
    df = df.copy()
    
    # 특성 엔지니어링 (학습 시와 동일하게)
    df['price_per_sqm'] = 0  # 임시값
    df['area_per_room'] = df['area_sqm'] / df['rooms']
    df['age_category'] = pd.cut(df['building_age'], 
                                 bins=[0, 5, 10, 20, 50], 
                                 labels=[0, 1, 2, 3]).astype(int)
    df['station_accessibility'] = 1 / (df['distance_to_station_min'] + 1)
    df['is_high_floor'] = (df['floor'] >= 10).astype(int)
    df['is_spacious'] = (df['area_sqm'] >= 80).astype(int)
    df['is_new_building'] = (df['building_age'] <= 5).astype(int)
    df['total_convenience'] = (
        df['amenity_score'] / 100 * 0.4 +
        df['station_accessibility'] * 20 * 0.3 +
        df['has_balcony'] * 10 * 0.15 +
        df['has_parking'] * 10 * 0.15
    )
    df['area_rooms_interaction'] = df['area_sqm'] * df['rooms']
    
    premium_wards = ['Chiyoda', 'Minato', 'Shibuya', 'Meguro', 'Chuo']
    df['is_premium_ward'] = df['ward'].isin(premium_wards).astype(int)
    
    # 구 인코딩
    ward_encoded = preprocessor['label_encoder'].transform(df['ward'].values)
    
    # 특성 선택
    feature_cols = preprocessor['feature_names']
    X = df[feature_cols].values
    
    # 스케일링
    X_scaled = preprocessor['scaler'].transform(X)
    
    return X_scaled, ward_encoded


def predict_prices(model, X, ward, model_type='custom'):
    """
    가격 예측
    
    Parameters:
    -----------
    model : object
        학습된 모델
    X : np.ndarray
        전처리된 특성
    ward : np.ndarray
        구 인코딩
    model_type : str
        모델 타입
        
    Returns:
    --------
    np.ndarray
        예측된 가격
    """
    if model_type == 'custom':
        predictions = model.predict(X, ward)
    else:
        predictions = model.predict(X).flatten()
    
    return predictions


def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description='도쿄 주택 가격 예측')
    parser.add_argument('--model', type=str, default='custom',
                       choices=['custom', 'automl'],
                       help='사용할 모델')
    parser.add_argument('--input', type=str, default=None,
                       help='입력 CSV 파일 경로 (없으면 샘플 데이터 사용)')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("도쿄 23구 주택 가격 예측")
    print("=" * 80)
    
    # ========================================================================
    # 1. 모델 로드
    # ========================================================================
    
    print("\n[1/3] 모델 로드")
    print("-" * 80)
    
    try:
        model, preprocessor = load_models(args.model)
        print(f"✓ {args.model} 모델 로드 완료")
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        print("먼저 train.py를 실행하여 모델을 학습시켜주세요.")
        return
    
    # ========================================================================
    # 2. 데이터 준비
    # ========================================================================
    
    print("\n[2/3] 데이터 준비")
    print("-" * 80)
    
    if args.input and os.path.exists(args.input):
        df = pd.read_csv(args.input)
        print(f"✓ 입력 파일 로드: {args.input} ({len(df)} 샘플)")
    else:
        df = create_sample_data()
        print(f"✓ 샘플 데이터 생성: {len(df)} 샘플")
    
    # 전처리
    X, ward = preprocess_for_prediction(df, preprocessor)
    
    # ========================================================================
    # 3. 예측
    # ========================================================================
    
    print("\n[3/3] 가격 예측")
    print("-" * 80)
    
    predictions = predict_prices(model, X, ward, args.model)
    
    # ========================================================================
    # 4. 결과 출력
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("예측 결과")
    print("=" * 80)
    
    results = []
    for idx, (_, row) in enumerate(df.iterrows()):
        pred_price = predictions[idx]
        
        result = {
            '번호': idx + 1,
            '구': row['ward'],
            '면적(㎡)': row['area_sqm'],
            '방': row['rooms'],
            '층': row['floor'],
            '연식(년)': row['building_age'],
            '역거리(분)': row['distance_to_station_min'],
            '예측가격(백만엔)': round(pred_price, 2)
        }
        results.append(result)
        
        print(f"\n예측 #{idx + 1}: {row.get('description', '')}")
        print(f"  구: {row['ward']}")
        print(f"  면적: {row['area_sqm']}㎡, 방: {row['rooms']}개, 층: {row['floor']}층")
        print(f"  건물 연식: {row['building_age']}년, 역까지: {row['distance_to_station_min']}분")
        print(f"  편의시설 점수: {row['amenity_score']}")
        print(f"  발코니: {'있음' if row['has_balcony'] else '없음'}, "
              f"주차장: {'있음' if row['has_parking'] else '없음'}")
        print(f"  ► 예측 가격: {pred_price:.2f} 백만 엔 "
              f"(약 {pred_price * 10:.0f} 백만 원)")
    
    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame(results)
    output_path = '/workspace/results/predictions.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print("\n" + "=" * 80)
    print(f"✓ 예측 결과가 저장되었습니다: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
