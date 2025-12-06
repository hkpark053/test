"""
AutoKeras를 사용한 도쿄 23구 집값 예측 AutoML 모델
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import autokeras as ak
import tensorflow as tf

def load_and_preprocess_data(csv_path='tokyo_housing_data.csv'):
    """
    데이터 로드 및 전처리
    
    Parameters:
    -----------
    csv_path : str
        CSV 파일 경로
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, label_encoders, scaler)
    """
    # 데이터 로드
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 특성과 타겟 분리
    X = df.drop('price', axis=1)
    y = df['price']
    
    # 범주형 변수 인코딩
    label_encoder = LabelEncoder()
    X['ward_encoded'] = label_encoder.fit_transform(X['ward'])
    X = X.drop('ward', axis=1)
    
    # 수치형 특성 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler

def train_automl_model(X_train, y_train, max_trials=20, epochs=50):
    """
    AutoKeras를 사용한 AutoML 모델 학습
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        학습 데이터
    y_train : pd.Series
        학습 타겟
    max_trials : int
        최대 시도 횟수
    epochs : int
        에포크 수
    
    Returns:
    --------
    autokeras.AutoModel
        학습된 모델
    """
    # AutoKeras 회귀 모델 생성
    # AutoKeras가 자동으로 최적의 아키텍처를 찾습니다
    model = ak.StructuredDataRegressor(
        max_trials=max_trials,
        overwrite=True,
        metrics=['mae', 'mse'],
        loss='mean_squared_error'
    )
    
    # 모델 학습
    print("AutoML 모델 학습 시작...")
    print(f"최대 시도 횟수: {max_trials}, 에포크: {epochs}")
    
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=1
    )
    
    print("학습 완료!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    모델 평가
    
    Parameters:
    -----------
    model : autokeras.AutoModel
        학습된 모델
    X_test : pd.DataFrame
        테스트 데이터
    y_test : pd.Series
        테스트 타겟
    """
    # 예측
    predictions = model.predict(X_test)
    
    # 평가 지표 계산
    mae = np.mean(np.abs(predictions.flatten() - y_test.values))
    mse = np.mean((predictions.flatten() - y_test.values) ** 2)
    rmse = np.sqrt(mse)
    
    # 평균 절대 백분율 오차
    mape = np.mean(np.abs((y_test.values - predictions.flatten()) / y_test.values)) * 100
    
    print("\n=== 모델 평가 결과 ===")
    print(f"MAE (Mean Absolute Error): {mae:.2f} 만엔")
    print(f"MSE (Mean Squared Error): {mse:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} 만엔")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

def show_model_architecture(model):
    """
    AutoKeras가 찾은 최적 모델 아키텍처 출력
    """
    print("\n=== 최적 모델 아키텍처 ===")
    best_model = model.export_model()
    print(best_model.summary())

if __name__ == '__main__':
    # 데이터 로드 및 전처리
    print("데이터 로드 중...")
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess_data()
    
    print(f"학습 데이터: {X_train.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    
    # AutoML 모델 학습
    model = train_automl_model(X_train, y_train, max_trials=10, epochs=30)
    
    # 모델 평가
    metrics = evaluate_model(model, X_test, y_test)
    
    # 모델 아키텍처 출력
    show_model_architecture(model)
    
    # 모델 저장
    model_path = 'automl_tokyo_housing_model'
    model.export_model().save(model_path)
    print(f"\n모델이 '{model_path}'에 저장되었습니다.")
