"""
도쿄 23구 주택 가격 데이터 로더 및 전처리
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TokyoHousingDataLoader:
    """도쿄 23구 주택 가격 데이터 로더"""
    
    # 도쿄 23구 목록
    TOKYO_23_WARDS = [
        'Chiyoda', 'Chuo', 'Minato', 'Shinjuku', 'Bunkyo',
        'Taito', 'Sumida', 'Koto', 'Shinagawa', 'Meguro',
        'Ota', 'Setagaya', 'Shibuya', 'Nakano', 'Suginami',
        'Toshima', 'Kita', 'Arakawa', 'Itabashi', 'Nerima',
        'Adachi', 'Katsushika', 'Edogawa'
    ]
    
    def __init__(self, n_samples=5000):
        """
        Args:
            n_samples: 생성할 샘플 데이터 수
        """
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        self.ward_encoder = LabelEncoder()
        
    def generate_sample_data(self):
        """
        도쿄 23구 주택 가격 샘플 데이터 생성
        실제 프로젝트에서는 실제 데이터셋을 사용하세요.
        """
        np.random.seed(42)
        
        data = {
            'ward': np.random.choice(self.TOKYO_23_WARDS, self.n_samples),
            'area_sqm': np.random.normal(60, 20, self.n_samples).clip(20, 150),
            'age_years': np.random.exponential(10, self.n_samples).clip(0, 50),
            'floor': np.random.randint(1, 20, self.n_samples),
            'distance_to_station_m': np.random.normal(500, 300, self.n_samples).clip(50, 2000),
            'nearest_station_lines': np.random.randint(1, 5, self.n_samples),
            'building_type': np.random.choice(['Mansion', 'Apartment', 'House'], self.n_samples),
            'has_parking': np.random.choice([0, 1], self.n_samples, p=[0.3, 0.7]),
            'has_balcony': np.random.choice([0, 1], self.n_samples, p=[0.2, 0.8]),
            'rooms': np.random.choice([1, 2, 3, 4, 5], self.n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
        }
        
        df = pd.DataFrame(data)
        
        # 가격 계산 (실제로는 더 복잡한 관계)
        # 구별 가격 차이, 면적, 연령 등을 고려
        ward_prices = {
            'Chiyoda': 800000, 'Chuo': 750000, 'Minato': 900000,
            'Shinjuku': 700000, 'Bunkyo': 650000, 'Shibuya': 850000,
            'Setagaya': 550000, 'Meguro': 600000, 'Shinagawa': 580000,
            'Nakano': 500000, 'Suginami': 520000, 'Toshima': 480000,
            'Kita': 450000, 'Arakawa': 400000, 'Itabashi': 420000,
            'Nerima': 430000, 'Adachi': 380000, 'Katsushika': 370000,
            'Edogawa': 390000, 'Koto': 500000, 'Sumida': 450000,
            'Taito': 480000, 'Ota': 440000
        }
        
        base_price = df['ward'].map(ward_prices)
        price = (
            base_price * (df['area_sqm'] / 60) * 0.8 +
            base_price * (1 - df['age_years'] / 50) * 0.3 +
            base_price * (df['floor'] / 10) * 0.1 +
            base_price * (1 / (1 + df['distance_to_station_m'] / 1000)) * 0.2 +
            base_price * (df['has_parking'] * 0.1) +
            np.random.normal(0, base_price * 0.1, self.n_samples)
        )
        
        df['price_yen'] = price.clip(200000, 1500000)
        
        return df
    
    def load_data(self, csv_path=None):
        """
        데이터 로드
        Args:
            csv_path: CSV 파일 경로 (None이면 샘플 데이터 생성)
        Returns:
            DataFrame
        """
        if csv_path:
            df = pd.read_csv(csv_path)
        else:
            df = self.generate_sample_data()
        
        return df
    
    def preprocess_data(self, df):
        """
        데이터 전처리
        Args:
            df: 원본 데이터프레임
        Returns:
            X_train, X_test, y_train, y_test: 전처리된 학습/테스트 데이터
        """
        # 카테고리컬 변수 인코딩
        df_processed = df.copy()
        df_processed['ward_encoded'] = self.ward_encoder.fit_transform(df_processed['ward'])
        df_processed['building_type_encoded'] = LabelEncoder().fit_transform(df_processed['building_type'])
        
        # 특성 선택
        feature_columns = [
            'ward_encoded', 'area_sqm', 'age_years', 'floor',
            'distance_to_station_m', 'nearest_station_lines',
            'building_type_encoded', 'has_parking', 'has_balcony', 'rooms'
        ]
        
        X = df_processed[feature_columns].values
        y = df_processed['price_yen'].values
        
        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 스케일링
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def get_feature_names(self):
        """특성 이름 반환"""
        return [
            'ward_encoded', 'area_sqm', 'age_years', 'floor',
            'distance_to_station_m', 'nearest_station_lines',
            'building_type_encoded', 'has_parking', 'has_balcony', 'rooms'
        ]
