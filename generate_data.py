"""
도쿄 23구 집값 데이터 생성 스크립트
실제 데이터가 없는 경우를 대비해 합성 데이터를 생성합니다.
"""
import numpy as np
import pandas as pd
import random

# 도쿄 23구 목록
TOKYO_WARDS = [
    '千代田区', '中央区', '港区', '新宿区', '文京区',
    '台東区', '墨田区', '江東区', '品川区', '目黒区',
    '大田区', '世田谷区', '渋谷区', '中野区', '杉並区',
    '豊島区', '北区', '荒川区', '板橋区', '練馬区',
    '足立区', '葛飾区', '江戸川区'
]

def generate_tokyo_housing_data(n_samples=5000):
    """
    도쿄 23구 집값 데이터 생성
    
    Parameters:
    -----------
    n_samples : int
        생성할 샘플 수
    
    Returns:
    --------
    pd.DataFrame
        생성된 데이터프레임
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # 구 선택
        ward = random.choice(TOKYO_WARDS)
        
        # 구별 기본 가격 (만엔 단위)
        base_prices = {
            '千代田区': 12000, '中央区': 11000, '港区': 13000,
            '新宿区': 9000, '文京区': 8500, '台東区': 7000,
            '墨田区': 6500, '江東区': 7500, '品川区': 8000,
            '目黒区': 9500, '大田区': 6000, '世田谷区': 8500,
            '渋谷区': 10000, '中野区': 7500, '杉並区': 7000,
            '豊島区': 7000, '北区': 6000, '荒川区': 5500,
            '板橋区': 6000, '練馬区': 5500, '足立区': 5000,
            '葛飾区': 5000, '江戸川区': 5000
        }
        
        base_price = base_prices[ward]
        
        # 면적 (제곱미터)
        area = np.random.normal(60, 20)
        area = max(20, min(150, area))  # 20-150 제곱미터로 제한
        
        # 건축 연도
        year_built = random.randint(1970, 2023)
        age = 2024 - year_built
        
        # 층수
        floor = random.randint(1, 30)
        
        # 역까지 거리 (미터)
        station_distance = np.random.exponential(500)
        station_distance = min(3000, station_distance)
        
        # 방 개수
        rooms = random.choice([1, 2, 3, 4, 5])
        
        # 가격 계산 (만엔)
        price = base_price * (area / 60) * (1 - age * 0.01) * \
                (1 + floor * 0.005) * (1 - station_distance / 10000) * \
                (1 + rooms * 0.1) * np.random.uniform(0.85, 1.15)
        
        price = max(2000, price)  # 최소 2000만엔
        
        data.append({
            'ward': ward,
            'area': round(area, 2),
            'year_built': year_built,
            'age': age,
            'floor': floor,
            'station_distance': round(station_distance, 2),
            'rooms': rooms,
            'price': round(price, 2)
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    # 데이터 생성
    df = generate_tokyo_housing_data(5000)
    
    # CSV로 저장
    df.to_csv('tokyo_housing_data.csv', index=False, encoding='utf-8-sig')
    print(f"데이터 생성 완료: {len(df)}개 샘플")
    print("\n데이터 미리보기:")
    print(df.head())
    print("\n기본 통계:")
    print(df.describe())
