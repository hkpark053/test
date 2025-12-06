"""
도쿄 23구 주택 가격 샘플 데이터 생성 스크립트

이 스크립트는 도쿄 23구의 실제 특성을 반영한 시뮬레이션 데이터를 생성합니다.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 랜덤 시드 설정 (재현성)
np.random.seed(42)

# 도쿄 23구 리스트
TOKYO_WARDS = [
    'Chiyoda', 'Chuo', 'Minato', 'Shinjuku', 'Bunkyo',
    'Taito', 'Sumida', 'Koto', 'Shinagawa', 'Meguro',
    'Ota', 'Setagaya', 'Shibuya', 'Nakano', 'Suginami',
    'Toshima', 'Kita', 'Arakawa', 'Itabashi', 'Nerima',
    'Adachi', 'Katsushika', 'Edogawa'
]

# 각 구별 평균 집값 계수 (실제 데이터 기반 상대적 가격)
WARD_PRICE_MULTIPLIER = {
    'Chiyoda': 2.5, 'Chuo': 2.2, 'Minato': 2.8, 'Shinjuku': 2.0, 'Bunkyo': 2.1,
    'Taito': 1.5, 'Sumida': 1.3, 'Koto': 1.4, 'Shinagawa': 1.9, 'Meguro': 2.3,
    'Ota': 1.4, 'Setagaya': 1.8, 'Shibuya': 2.6, 'Nakano': 1.5, 'Suginami': 1.6,
    'Toshima': 1.6, 'Kita': 1.3, 'Arakawa': 1.2, 'Itabashi': 1.3, 'Nerima': 1.3,
    'Adachi': 1.1, 'Katsushika': 1.2, 'Edogawa': 1.2
}

def generate_tokyo_housing_data(n_samples=5000):
    """
    도쿄 23구 주택 가격 데이터 생성
    
    Parameters:
    -----------
    n_samples : int
        생성할 샘플 수
        
    Returns:
    --------
    pd.DataFrame
        생성된 주택 데이터
    """
    
    data = []
    
    for _ in range(n_samples):
        # 구 선택
        ward = np.random.choice(TOKYO_WARDS)
        ward_multiplier = WARD_PRICE_MULTIPLIER[ward]
        
        # 면적 (20-200 평방미터)
        area = np.random.normal(70, 30)
        area = np.clip(area, 20, 200)
        
        # 방 개수 (1-5개)
        # 면적에 따라 확률적으로 결정
        if area < 40:
            rooms = np.random.choice([1, 2], p=[0.7, 0.3])
        elif area < 60:
            rooms = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
        elif area < 90:
            rooms = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])
        else:
            rooms = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
        
        # 층수 (1-30층)
        floor = np.random.randint(1, 31)
        
        # 건물 연식 (0-50년)
        building_age = np.random.exponential(15)
        building_age = np.clip(building_age, 0, 50)
        
        # 역까지 거리 (분, 1-30분)
        distance_to_station = np.random.exponential(8)
        distance_to_station = np.clip(distance_to_station, 1, 30)
        
        # 편의시설 점수 (0-100)
        # 역까지 거리가 가까울수록 점수 높음
        amenity_score = 100 - distance_to_station * 2 + np.random.normal(0, 10)
        amenity_score = np.clip(amenity_score, 0, 100)
        
        # 발코니 여부
        has_balcony = np.random.choice([0, 1], p=[0.3, 0.7])
        
        # 주차장 여부
        has_parking = np.random.choice([0, 1], p=[0.4, 0.6])
        
        # 리모델링 여부 (오래된 집일수록 리모델링 확률 높음)
        remodeled_prob = min(0.8, building_age / 50)
        is_remodeled = np.random.choice([0, 1], p=[1-remodeled_prob, remodeled_prob])
        
        # 가격 계산 (백만 엔 단위)
        base_price = 50  # 기본 가격 5천만 엔
        
        # 각 요소별 가격 영향
        price = base_price * ward_multiplier
        price += area * 0.8  # 면적당 80만엔
        price += rooms * 5   # 방당 500만엔
        price += (floor / 10) * 3  # 높은 층 프리미엄
        price -= building_age * 1.5  # 연식에 따른 감가
        price -= distance_to_station * 2  # 역 거리에 따른 감가
        price += (amenity_score / 100) * 20  # 편의시설 점수 영향
        price += has_balcony * 5  # 발코니 프리미엄
        price += has_parking * 10  # 주차장 프리미엄
        price += is_remodeled * 15  # 리모델링 프리미엄
        
        # 노이즈 추가 (현실적인 변동성)
        price *= np.random.normal(1.0, 0.15)
        price = max(price, 10)  # 최소 1천만 엔
        
        data.append({
            'ward': ward,
            'area_sqm': round(area, 2),
            'rooms': rooms,
            'floor': floor,
            'building_age': round(building_age, 1),
            'distance_to_station_min': round(distance_to_station, 1),
            'amenity_score': round(amenity_score, 1),
            'has_balcony': has_balcony,
            'has_parking': has_parking,
            'is_remodeled': is_remodeled,
            'price_million_yen': round(price, 2)
        })
    
    df = pd.DataFrame(data)
    return df


def generate_statistics(df):
    """데이터 통계 출력"""
    print("=" * 80)
    print("도쿄 23구 주택 가격 데이터셋 통계")
    print("=" * 80)
    print(f"\n총 샘플 수: {len(df)}")
    print(f"\n가격 통계 (백만 엔):")
    print(df['price_million_yen'].describe())
    
    print(f"\n구별 평균 가격:")
    ward_avg = df.groupby('ward')['price_million_yen'].mean().sort_values(ascending=False)
    for ward, price in ward_avg.items():
        print(f"  {ward:15s}: {price:8.2f} 백만 엔")
    
    print(f"\n면적 통계 (평방미터):")
    print(df['area_sqm'].describe())
    
    print(f"\n방 개수 분포:")
    print(df['rooms'].value_counts().sort_index())
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("도쿄 23구 주택 가격 데이터 생성 중...")
    
    # 데이터 생성
    df = generate_tokyo_housing_data(n_samples=5000)
    
    # CSV 저장
    output_path = '/workspace/data/tokyo_housing_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ 데이터가 저장되었습니다: {output_path}")
    
    # 통계 출력
    generate_statistics(df)
    
    # 샘플 데이터 미리보기
    print("\n데이터 미리보기 (처음 5개):")
    print(df.head())
