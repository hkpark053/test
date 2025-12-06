"""
도쿄 23구 주택 가격 샘플 데이터 생성기
Tokyo 23-Ward Housing Price Sample Data Generator

실제 도쿄 부동산 시장을 반영한 현실적인 샘플 데이터를 생성합니다.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


# 도쿄 23구 정보 (구 이름, 영문명, 기준 평균 가격 (만엔/㎡), 역세권 프리미엄)
TOKYO_23_WARDS = {
    'chiyoda': {'name_jp': '千代田区', 'base_price': 180, 'station_premium': 1.3, 'crime_rate': 0.02},
    'chuo': {'name_jp': '中央区', 'base_price': 170, 'station_premium': 1.25, 'crime_rate': 0.03},
    'minato': {'name_jp': '港区', 'base_price': 200, 'station_premium': 1.35, 'crime_rate': 0.02},
    'shinjuku': {'name_jp': '新宿区', 'base_price': 130, 'station_premium': 1.2, 'crime_rate': 0.05},
    'bunkyo': {'name_jp': '文京区', 'base_price': 120, 'station_premium': 1.15, 'crime_rate': 0.01},
    'taito': {'name_jp': '台東区', 'base_price': 100, 'station_premium': 1.1, 'crime_rate': 0.04},
    'sumida': {'name_jp': '墨田区', 'base_price': 85, 'station_premium': 1.08, 'crime_rate': 0.03},
    'koto': {'name_jp': '江東区', 'base_price': 90, 'station_premium': 1.1, 'crime_rate': 0.02},
    'shinagawa': {'name_jp': '品川区', 'base_price': 130, 'station_premium': 1.2, 'crime_rate': 0.02},
    'meguro': {'name_jp': '目黒区', 'base_price': 140, 'station_premium': 1.18, 'crime_rate': 0.01},
    'ota': {'name_jp': '大田区', 'base_price': 80, 'station_premium': 1.05, 'crime_rate': 0.03},
    'setagaya': {'name_jp': '世田谷区', 'base_price': 110, 'station_premium': 1.12, 'crime_rate': 0.01},
    'shibuya': {'name_jp': '渋谷区', 'base_price': 160, 'station_premium': 1.3, 'crime_rate': 0.03},
    'nakano': {'name_jp': '中野区', 'base_price': 95, 'station_premium': 1.1, 'crime_rate': 0.02},
    'suginami': {'name_jp': '杉並区', 'base_price': 100, 'station_premium': 1.08, 'crime_rate': 0.01},
    'toshima': {'name_jp': '豊島区', 'base_price': 105, 'station_premium': 1.15, 'crime_rate': 0.04},
    'kita': {'name_jp': '北区', 'base_price': 75, 'station_premium': 1.05, 'crime_rate': 0.03},
    'arakawa': {'name_jp': '荒川区', 'base_price': 70, 'station_premium': 1.05, 'crime_rate': 0.03},
    'itabashi': {'name_jp': '板橋区', 'base_price': 70, 'station_premium': 1.03, 'crime_rate': 0.02},
    'nerima': {'name_jp': '練馬区', 'base_price': 75, 'station_premium': 1.05, 'crime_rate': 0.01},
    'adachi': {'name_jp': '足立区', 'base_price': 55, 'station_premium': 1.02, 'crime_rate': 0.05},
    'katsushika': {'name_jp': '葛飾区', 'base_price': 55, 'station_premium': 1.02, 'crime_rate': 0.03},
    'edogawa': {'name_jp': '江戸川区', 'base_price': 60, 'station_premium': 1.03, 'crime_rate': 0.02},
}


def generate_housing_data(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    도쿄 23구 주택 가격 샘플 데이터 생성
    
    Parameters:
    -----------
    n_samples : int
        생성할 샘플 수
    random_seed : int
        재현성을 위한 랜덤 시드
        
    Returns:
    --------
    pd.DataFrame
        생성된 주택 데이터
    """
    np.random.seed(random_seed)
    
    data = []
    ward_keys = list(TOKYO_23_WARDS.keys())
    
    for _ in range(n_samples):
        # 구 선택 (인기 구역에 더 많은 샘플)
        ward_weights = [TOKYO_23_WARDS[w]['base_price'] for w in ward_keys]
        ward_weights = np.array(ward_weights) / sum(ward_weights)
        ward = np.random.choice(ward_keys, p=ward_weights)
        ward_info = TOKYO_23_WARDS[ward]
        
        # 건물 정보 생성
        building_age = np.random.exponential(15)  # 건물 연령 (지수분포)
        building_age = min(building_age, 60)  # 최대 60년
        
        floor_area = np.random.normal(65, 25)  # 면적 (㎡)
        floor_area = max(20, min(floor_area, 200))  # 20~200㎡
        
        num_rooms = max(1, int(floor_area / 20 + np.random.normal(0, 0.5)))  # 방 개수
        num_rooms = min(num_rooms, 6)
        
        floor_number = np.random.randint(1, 15)  # 층수
        total_floors = max(floor_number, np.random.randint(floor_number, 20))  # 총 층수
        
        # 역까지 거리 (분)
        distance_to_station = np.random.exponential(7)
        distance_to_station = max(1, min(distance_to_station, 30))
        
        # 편의시설
        has_parking = np.random.random() < 0.3  # 주차장
        has_balcony = np.random.random() < 0.7  # 발코니
        has_security = np.random.random() < 0.5  # 보안 시스템
        is_corner_unit = np.random.random() < 0.2  # 코너 유닛
        
        # 건물 타입
        building_type = np.random.choice(['아파트', '맨션', '타워맨션', '빌라'], 
                                          p=[0.4, 0.35, 0.15, 0.1])
        
        # 방향 (남향이 프리미엄)
        direction = np.random.choice(['남', '남동', '남서', '동', '서', '북동', '북서', '북'],
                                      p=[0.25, 0.15, 0.15, 0.12, 0.12, 0.08, 0.08, 0.05])
        
        # 거래 날짜 (최근 5년)
        days_ago = np.random.randint(0, 365 * 5)
        transaction_date = datetime.now() - timedelta(days=days_ago)
        
        # 가격 계산 로직
        base_price = ward_info['base_price']
        
        # 면적 기반 가격
        price = base_price * floor_area
        
        # 역세권 조정 (가까울수록 비쌈)
        if distance_to_station <= 3:
            price *= ward_info['station_premium']
        elif distance_to_station <= 7:
            price *= 1.0 + (ward_info['station_premium'] - 1.0) * 0.5
        elif distance_to_station > 15:
            price *= 0.85
        
        # 건물 연령 조정 (오래될수록 저렴)
        age_factor = max(0.5, 1.0 - building_age * 0.012)
        price *= age_factor
        
        # 층수 조정 (높을수록 비쌈)
        floor_premium = 1.0 + (floor_number - 1) * 0.008
        price *= floor_premium
        
        # 건물 타입 조정
        type_multiplier = {'아파트': 1.0, '맨션': 1.1, '타워맨션': 1.35, '빌라': 0.9}
        price *= type_multiplier[building_type]
        
        # 방향 조정
        direction_multiplier = {'남': 1.05, '남동': 1.03, '남서': 1.02, '동': 1.0, 
                                '서': 0.98, '북동': 0.97, '북서': 0.96, '북': 0.95}
        price *= direction_multiplier[direction]
        
        # 편의시설 조정
        if has_parking:
            price *= 1.08
        if has_balcony:
            price *= 1.03
        if has_security:
            price *= 1.05
        if is_corner_unit:
            price *= 1.04
        
        # 시장 변동 (연도별)
        year = transaction_date.year
        market_trend = {2020: 0.95, 2021: 0.98, 2022: 1.02, 2023: 1.05, 2024: 1.08, 2025: 1.10}
        price *= market_trend.get(year, 1.0)
        
        # 랜덤 노이즈 추가
        price *= np.random.normal(1.0, 0.08)
        
        # 가격을 만엔 단위로 반올림
        price = round(price / 10) * 10
        
        data.append({
            'ward': ward,
            'ward_jp': ward_info['name_jp'],
            'floor_area_sqm': round(floor_area, 1),
            'num_rooms': num_rooms,
            'building_age_years': round(building_age, 1),
            'floor_number': floor_number,
            'total_floors': total_floors,
            'distance_to_station_min': round(distance_to_station, 1),
            'building_type': building_type,
            'direction': direction,
            'has_parking': int(has_parking),
            'has_balcony': int(has_balcony),
            'has_security': int(has_security),
            'is_corner_unit': int(is_corner_unit),
            'transaction_year': year,
            'transaction_month': transaction_date.month,
            'price_man_yen': int(price),  # 만엔 단위
        })
    
    return pd.DataFrame(data)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """파생 피처 추가"""
    df = df.copy()
    
    # 평당 가격
    df['price_per_sqm'] = df['price_man_yen'] / df['floor_area_sqm']
    
    # 방당 면적
    df['area_per_room'] = df['floor_area_sqm'] / df['num_rooms']
    
    # 상대적 층수 (건물 내에서의 위치)
    df['relative_floor'] = df['floor_number'] / df['total_floors']
    
    # 역세권 카테고리
    df['station_category'] = pd.cut(
        df['distance_to_station_min'],
        bins=[0, 3, 7, 15, 100],
        labels=['역세권', '도보권', '버스권', '원거리']
    )
    
    # 건물 연령 카테고리
    df['age_category'] = pd.cut(
        df['building_age_years'],
        bins=[0, 5, 15, 30, 100],
        labels=['신축', '준신축', '중고', '노후']
    )
    
    # 편의시설 점수
    df['amenity_score'] = (
        df['has_parking'] * 2 + 
        df['has_balcony'] * 1 + 
        df['has_security'] * 2 + 
        df['is_corner_unit'] * 1
    )
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    """데이터 분할"""
    from sklearn.model_selection import train_test_split
    
    # 먼저 train+val과 test로 분할
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    
    # train과 val 분할
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
    
    return train, val, test


if __name__ == "__main__":
    # 데이터 생성
    print("🏠 도쿄 23구 주택 가격 데이터 생성 중...")
    df = generate_housing_data(n_samples=10000)
    
    # 파생 피처 추가
    df = add_derived_features(df)
    
    # 데이터 저장
    output_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(output_dir, 'tokyo_housing_data.csv'), index=False, encoding='utf-8')
    
    # 데이터 분할 및 저장
    train, val, test = split_data(df)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8')
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False, encoding='utf-8')
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8')
    
    print(f"✅ 데이터 생성 완료!")
    print(f"   - 전체: {len(df)} 샘플")
    print(f"   - 학습: {len(train)} 샘플")
    print(f"   - 검증: {len(val)} 샘플")
    print(f"   - 테스트: {len(test)} 샘플")
    print(f"\n📊 가격 통계:")
    print(f"   - 평균: {df['price_man_yen'].mean():,.0f} 만엔")
    print(f"   - 중앙값: {df['price_man_yen'].median():,.0f} 만엔")
    print(f"   - 최소: {df['price_man_yen'].min():,.0f} 만엔")
    print(f"   - 최대: {df['price_man_yen'].max():,.0f} 만엔")
