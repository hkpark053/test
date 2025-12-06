"""
직접 구현한 딥러닝 레이어를 사용한 도쿄 23구 집값 예측 모델
TensorFlow/Keras를 사용하여 다양한 딥러닝 아키텍처를 구현합니다.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_path='tokyo_housing_data.csv'):
    """
    데이터 로드 및 전처리
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
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
    
    # 학습/검증/테스트 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler

def build_mlp_model(input_dim, architecture='deep'):
    """
    다층 퍼셉트론(MLP) 모델 구축
    
    Parameters:
    -----------
    input_dim : int
        입력 특성 차원
    architecture : str
        'deep', 'wide', 'residual' 중 선택
    
    Returns:
    --------
    keras.Model
        컴파일된 모델
    """
    inputs = keras.Input(shape=(input_dim,))
    
    if architecture == 'deep':
        # 깊은 네트워크 (Deep Network)
        x = layers.Dense(256, activation='relu', name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(0.2, name='dropout_3')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_4')(x)
        outputs = layers.Dense(1, name='output')(x)
        
    elif architecture == 'wide':
        # 넓은 네트워크 (Wide Network)
        x = layers.Dense(512, activation='relu', name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.4, name='dropout_1')(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        outputs = layers.Dense(1, name='output')(x)
        
    elif architecture == 'residual':
        # 잔차 연결(Residual Connection)을 사용한 네트워크
        x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        
        # 첫 번째 잔차 블록
        residual = x
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.2, name='dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='dense_3')(x)
        x = layers.Add(name='add_1')([x, residual])  # 잔차 연결
        
        # 두 번째 잔차 블록
        residual = x
        x = layers.Dense(64, activation='relu', name='dense_4')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        x = layers.Dense(64, activation='relu', name='dense_5')(x)
        x = layers.Add(name='add_2')([x, residual])  # 잔차 연결
        
        outputs = layers.Dense(1, name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'MLP_{architecture}')
    
    # 모델 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def build_attention_model(input_dim):
    """
    어텐션 메커니즘을 사용한 모델 구축
    """
    inputs = keras.Input(shape=(input_dim,))
    
    # 특성 임베딩
    x = layers.Dense(128, activation='relu', name='embedding')(inputs)
    x = layers.Reshape((input_dim, 128 // input_dim))(x) if 128 % input_dim == 0 else x
    
    # Multi-Head Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=32, name='attention'
    )(x, x)
    
    # Feed Forward Network
    x = layers.Dense(128, activation='relu', name='ffn_1')(attention_output)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    x = layers.Dense(64, activation='relu', name='ffn_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    
    outputs = layers.Dense(1, name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Attention_Model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def build_ensemble_model(input_dim):
    """
    앙상블 모델 구축 (여러 모델의 예측을 결합)
    """
    inputs = keras.Input(shape=(input_dim,))
    
    # 브랜치 1: 깊은 네트워크
    branch1 = layers.Dense(256, activation='relu')(inputs)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.Dropout(0.3)(branch1)
    branch1 = layers.Dense(128, activation='relu')(branch1)
    branch1 = layers.Dense(64, activation='relu')(branch1)
    branch1 = layers.Dense(1, name='branch1_output')(branch1)
    
    # 브랜치 2: 넓은 네트워크
    branch2 = layers.Dense(512, activation='relu')(inputs)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Dropout(0.4)(branch2)
    branch2 = layers.Dense(256, activation='relu')(branch2)
    branch2 = layers.Dense(1, name='branch2_output')(branch2)
    
    # 브랜치 3: 잔차 네트워크
    branch3 = layers.Dense(128, activation='relu')(inputs)
    branch3 = layers.BatchNormalization()(branch3)
    residual = branch3
    branch3 = layers.Dense(128, activation='relu')(branch3)
    branch3 = layers.Add()([branch3, residual])
    branch3 = layers.Dense(64, activation='relu')(branch3)
    branch3 = layers.Dense(1, name='branch3_output')(branch3)
    
    # 앙상블 (평균)
    ensemble = layers.Average(name='ensemble_output')([branch1, branch2, branch3])
    
    model = models.Model(inputs=inputs, outputs=ensemble, name='Ensemble_Model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    모델 학습
    
    Parameters:
    -----------
    model : keras.Model
        학습할 모델
    X_train : pd.DataFrame
        학습 데이터
    y_train : pd.Series
        학습 타겟
    X_val : pd.DataFrame
        검증 데이터
    y_val : pd.Series
        검증 타겟
    epochs : int
        에포크 수
    batch_size : int
        배치 크기
    
    Returns:
    --------
    keras.callbacks.History
        학습 히스토리
    """
    # 콜백 설정
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            f'best_{model.name}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 모델 학습
    history = model.fit(
        X_train.values,
        y_train.values,
        validation_data=(X_val.values, y_val.values),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    모델 평가
    """
    predictions = model.predict(X_test.values)
    
    mae = np.mean(np.abs(predictions.flatten() - y_test.values))
    mse = np.mean((predictions.flatten() - y_test.values) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test.values - predictions.flatten()) / y_test.values)) * 100
    
    print("\n=== 모델 평가 결과 ===")
    print(f"MAE: {mae:.2f} 만엔")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} 만엔")
    print(f"MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}

def plot_training_history(history, model_name):
    """
    학습 히스토리 시각화
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 그래프
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # MAE 그래프
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title(f'{model_name} - MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=150)
    print(f"학습 히스토리가 '{model_name}_training_history.png'에 저장되었습니다.")

if __name__ == '__main__':
    # 데이터 로드
    print("데이터 로드 중...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler = \
        load_and_preprocess_data()
    
    print(f"학습 데이터: {X_train.shape}")
    print(f"검증 데이터: {X_val.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    
    input_dim = X_train.shape[1]
    
    # 여러 모델 학습 및 비교
    models_to_train = {
        'Deep_MLP': build_mlp_model(input_dim, 'deep'),
        'Wide_MLP': build_mlp_model(input_dim, 'wide'),
        'Residual_MLP': build_mlp_model(input_dim, 'residual'),
        'Ensemble': build_ensemble_model(input_dim)
    }
    
    results = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"{model_name} 모델 학습 시작")
        print(f"{'='*50}")
        
        # 모델 구조 출력
        print("\n모델 구조:")
        model.summary()
        
        # 모델 학습
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32
        )
        
        # 모델 평가
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics
        
        # 학습 히스토리 시각화
        plot_training_history(history, model_name)
    
    # 결과 비교
    print("\n" + "="*50)
    print("모든 모델 비교 결과")
    print("="*50)
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    # 최고 성능 모델 찾기
    best_model_name = results_df['rmse'].idxmin()
    print(f"\n최고 성능 모델: {best_model_name} (RMSE: {results_df.loc[best_model_name, 'rmse']:.2f})")
