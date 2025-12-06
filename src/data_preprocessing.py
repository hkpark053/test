"""
데이터 전처리 및 특성 엔지니어링 모듈

이 모듈은 도쿄 주택 가격 데이터를 로드하고 전처리합니다.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib


class TokyoHousingDataPreprocessor:
    """도쿄 주택 데이터 전처리 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.ward_to_idx = None
        self.idx_to_ward = None
        
    def load_data(self, filepath):
        """
        데이터 로드
        
        Parameters:
        -----------
        filepath : str
            CSV 파일 경로
            
        Returns:
        --------
        pd.DataFrame
            로드된 데이터프레임
        """
        df = pd.read_csv(filepath)
        print(f"✓ 데이터 로드 완료: {len(df)} 샘플")
        return df
    
    def create_features(self, df):
        """
        특성 엔지니어링
        
        Parameters:
        -----------
        df : pd.DataFrame
            원본 데이터프레임
            
        Returns:
        --------
        pd.DataFrame
            새로운 특성이 추가된 데이터프레임
        """
        df = df.copy()
        
        # 1. 면적당 가격
        df['price_per_sqm'] = df['price_million_yen'] / df['area_sqm']
        
        # 2. 방당 면적
        df['area_per_room'] = df['area_sqm'] / df['rooms']
        
        # 3. 건물 연식 구간
        df['age_category'] = pd.cut(df['building_age'], 
                                     bins=[0, 5, 10, 20, 51], 
                                     labels=[0, 1, 2, 3],
                                     include_lowest=True)
        df['age_category'] = df['age_category'].astype(int)
        
        # 4. 역 접근성 점수 (거리 역수)
        df['station_accessibility'] = 1 / (df['distance_to_station_min'] + 1)
        
        # 5. 고층 여부 (10층 이상)
        df['is_high_floor'] = (df['floor'] >= 10).astype(int)
        
        # 6. 넓은 집 여부 (80평방미터 이상)
        df['is_spacious'] = (df['area_sqm'] >= 80).astype(int)
        
        # 7. 신축 여부 (5년 이내)
        df['is_new_building'] = (df['building_age'] <= 5).astype(int)
        
        # 8. 총 편의성 점수 (여러 요소 종합)
        df['total_convenience'] = (
            df['amenity_score'] / 100 * 0.4 +
            df['station_accessibility'] * 20 * 0.3 +
            df['has_balcony'] * 10 * 0.15 +
            df['has_parking'] * 10 * 0.15
        )
        
        # 9. 면적 * 방 개수 상호작용
        df['area_rooms_interaction'] = df['area_sqm'] * df['rooms']
        
        # 10. 구별 고급 지역 여부
        premium_wards = ['Chiyoda', 'Minato', 'Shibuya', 'Meguro', 'Chuo']
        df['is_premium_ward'] = df['ward'].isin(premium_wards).astype(int)
        
        print(f"✓ 특성 엔지니어링 완료: {len(df.columns)} 개 특성")
        
        return df
    
    def prepare_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        학습/검증/테스트 데이터 분할 및 전처리
        
        Parameters:
        -----------
        df : pd.DataFrame
            전처리할 데이터프레임
        test_size : float
            테스트 세트 비율
        val_size : float
            검증 세트 비율
        random_state : int
            랜덤 시드
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test, ward_train, ward_val, ward_test)
        """
        
        # 타겟 변수 분리
        y = df['price_million_yen'].values
        
        # 구(ward) 별도 저장 (임베딩용)
        ward = df['ward'].values
        
        # 구를 숫자로 인코딩
        self.label_encoder.fit(ward)
        ward_encoded = self.label_encoder.transform(ward)
        self.ward_to_idx = dict(zip(self.label_encoder.classes_, 
                                    range(len(self.label_encoder.classes_))))
        self.idx_to_ward = dict(zip(range(len(self.label_encoder.classes_)), 
                                    self.label_encoder.classes_))
        
        # 특성 선택 (ward와 price 제외)
        feature_cols = [col for col in df.columns 
                       if col not in ['ward', 'price_million_yen', 'price_per_sqm']]
        X = df[feature_cols].values
        self.feature_names = feature_cols
        
        # 학습/임시 세트 분할
        X_train, X_temp, y_train, y_temp, ward_train, ward_temp = train_test_split(
            X, y, ward_encoded, test_size=(test_size + val_size), 
            random_state=random_state
        )
        
        # 임시 세트를 검증/테스트로 분할
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test, ward_val, ward_test = train_test_split(
            X_temp, y_temp, ward_temp, test_size=(1 - val_ratio), 
            random_state=random_state
        )
        
        # 스케일링
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print(f"✓ 데이터 분할 완료:")
        print(f"  - 학습 세트: {len(X_train)} 샘플")
        print(f"  - 검증 세트: {len(X_val)} 샘플")
        print(f"  - 테스트 세트: {len(X_test)} 샘플")
        
        return (X_train, X_val, X_test, 
                y_train, y_val, y_test,
                ward_train, ward_val, ward_test)
    
    def save(self, filepath):
        """전처리기 저장"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'ward_to_idx': self.ward_to_idx,
            'idx_to_ward': self.idx_to_ward
        }, filepath)
        print(f"✓ 전처리기 저장: {filepath}")
    
    def load(self, filepath):
        """전처리기 로드"""
        saved_data = joblib.load(filepath)
        self.scaler = saved_data['scaler']
        self.label_encoder = saved_data['label_encoder']
        self.feature_names = saved_data['feature_names']
        self.ward_to_idx = saved_data['ward_to_idx']
        self.idx_to_ward = saved_data['idx_to_ward']
        print(f"✓ 전처리기 로드: {filepath}")


if __name__ == "__main__":
    # 테스트 코드
    preprocessor = TokyoHousingDataPreprocessor()
    
    # 데이터 로드
    df = preprocessor.load_data('/workspace/data/tokyo_housing_data.csv')
    
    # 특성 엔지니어링
    df = preprocessor.create_features(df)
    
    # 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test, ward_train, ward_val, ward_test = \
        preprocessor.prepare_data(df)
    
    print(f"\n특성 목록: {preprocessor.feature_names}")
    print(f"구 매핑: {preprocessor.ward_to_idx}")
