"""
빠른 테스트 - 커스텀 딥러닝 모델 학습 (소규모)
"""

import sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspace/src')
from data_preprocessing import TokyoHousingDataPreprocessor
from custom_deep_model import CustomDeepHousingModel

# 랜덤 시드
np.random.seed(42)
tf.random.set_seed(42)

print("\n" + "=" * 80)
print("빠른 테스트: 커스텀 딥러닝 모델")
print("=" * 80)

# 데이터 로드 및 전처리
preprocessor = TokyoHousingDataPreprocessor()
df = preprocessor.load_data('/workspace/data/tokyo_housing_data.csv')
df = preprocessor.create_features(df)
X_train, X_val, X_test, y_train, y_val, y_test, ward_train, ward_val, ward_test = \
    preprocessor.prepare_data(df, test_size=0.2, val_size=0.1)

# 모델 생성
model = CustomDeepHousingModel(
    num_features=X_train.shape[1],
    num_wards=23,
    embedding_dim=8
)

# 모델 구조 출력
model.get_summary()

# 학습 (빠른 테스트를 위해 10 에포크만)
print("\n빠른 학습 테스트 (10 에포크)...")
model.compile_model(learning_rate=0.001)
history = model.train(
    X_train, ward_train, y_train,
    X_val, ward_val, y_val,
    epochs=10,
    batch_size=32
)

# 평가
metrics = model.evaluate(X_test, ward_test, y_test)

# 모델 저장
model.save('/workspace/models/custom_deep_model.h5')
preprocessor.save('/workspace/models/preprocessor.pkl')

print("\n" + "=" * 80)
print("테스트 완료!")
print("=" * 80)
