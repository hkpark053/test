"""
도쿄 23구 주택 가격 예측을 위한 AutoML 모델

이 모듈은 다양한 AutoML 프레임워크를 활용합니다:
1. Optuna 기반 하이퍼파라미터 최적화
2. Custom Neural Architecture Search (NAS)
3. 앙상블 자동 생성
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import json
import os

# ML 라이브러리
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import (
    Ridge, 
    Lasso, 
    ElasticNet,
    BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')


@dataclass
class AutoMLConfig:
    """AutoML 설정"""
    n_trials: int = 100  # 하이퍼파라미터 탐색 횟수
    cv_folds: int = 5  # 교차 검증 폴드 수
    metric: str = 'rmse'  # 최적화 메트릭
    time_budget_seconds: int = 3600  # 시간 제한
    random_state: int = 42


class HyperparameterSpace:
    """
    하이퍼파라미터 탐색 공간 정의
    
    각 모델별로 최적화할 하이퍼파라미터와 그 범위를 정의합니다.
    """
    
    @staticmethod
    def random_forest(trial) -> Dict:
        """Random Forest 하이퍼파라미터 공간"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
    
    @staticmethod
    def gradient_boosting(trial) -> Dict:
        """Gradient Boosting 하이퍼파라미터 공간"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        }
    
    @staticmethod
    def xgboost(trial) -> Dict:
        """XGBoost 하이퍼파라미터 공간"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
    
    @staticmethod
    def lightgbm(trial) -> Dict:
        """LightGBM 하이퍼파라미터 공간"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
    
    @staticmethod
    def mlp(trial) -> Dict:
        """MLP (신경망) 하이퍼파라미터 공간"""
        n_layers = trial.suggest_int('n_layers', 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 512))
        
        return {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        }
    
    @staticmethod
    def neural_architecture(trial) -> Dict:
        """딥러닝 아키텍처 탐색 공간"""
        n_layers = trial.suggest_int('n_layers', 2, 6)
        
        architecture = {
            'layers': [],
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'gelu']),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        }
        
        # 레이어 구성
        prev_units = trial.suggest_int('input_units', 64, 512)
        for i in range(n_layers):
            units = trial.suggest_int(f'units_l{i}', 32, 512)
            use_skip = trial.suggest_categorical(f'skip_l{i}', [True, False]) if i > 0 else False
            
            architecture['layers'].append({
                'units': units,
                'use_skip_connection': use_skip,
            })
        
        return architecture


class OptunaAutoML:
    """
    Optuna 기반 AutoML
    
    베이지안 최적화를 활용하여 최적의 모델과 하이퍼파라미터를 탐색합니다.
    """
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.best_model = None
        self.best_params = None
        self.best_score = float('inf')
        self.study = None
        self.scaler = StandardScaler()
        self.results_history = []
        
    def _create_objective(
        self, 
        model_class,
        param_space_fn: Callable,
        X: np.ndarray,
        y: np.ndarray
    ) -> Callable:
        """Optuna objective 함수 생성"""
        
        def objective(trial):
            params = param_space_fn(trial)
            model = model_class(**params, random_state=self.config.random_state)
            
            # 교차 검증
            scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring='neg_root_mean_squared_error'
            )
            
            rmse = -scores.mean()
            return rmse
        
        return objective
    
    def fit(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'all'):
        """
        AutoML 학습
        
        Parameters:
        -----------
        X : pd.DataFrame
            입력 피처
        y : pd.Series
            타겟 (가격)
        model_type : str
            탐색할 모델 타입 ('rf', 'gb', 'mlp', 'all')
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("⚠️ Optuna가 설치되지 않았습니다. pip install optuna")
            return self
        
        # 데이터 전처리
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델별 탐색
        model_configs = {
            'rf': (RandomForestRegressor, HyperparameterSpace.random_forest),
            'gb': (GradientBoostingRegressor, HyperparameterSpace.gradient_boosting),
            'mlp': (MLPRegressor, HyperparameterSpace.mlp),
        }
        
        if model_type != 'all':
            model_configs = {model_type: model_configs[model_type]}
        
        print(f"🔍 AutoML 탐색 시작 (총 {self.config.n_trials} trials)...")
        
        for name, (model_class, param_space) in model_configs.items():
            print(f"\n📊 {name.upper()} 모델 탐색 중...")
            
            study = optuna.create_study(direction='minimize')
            objective = self._create_objective(model_class, param_space, X_scaled, y)
            
            study.optimize(
                objective,
                n_trials=self.config.n_trials // len(model_configs),
                show_progress_bar=True
            )
            
            if study.best_value < self.best_score:
                self.best_score = study.best_value
                self.best_params = study.best_params
                self.best_model = model_class(**study.best_params, random_state=self.config.random_state)
                self.best_model.fit(X_scaled, y)
                self.study = study
            
            self.results_history.append({
                'model': name,
                'best_rmse': study.best_value,
                'best_params': study.best_params
            })
        
        print(f"\n✅ 최적 모델: RMSE = {self.best_score:.2f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """피처 중요도 반환"""
        if hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(self.best_model.feature_importances_)
        return None


class NeuralArchitectureSearch:
    """
    Neural Architecture Search (NAS)
    
    딥러닝 아키텍처를 자동으로 탐색합니다.
    """
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.best_architecture = None
        self.best_score = float('inf')
        self.search_history = []
        
    def _build_model(self, architecture: Dict, input_dim: int):
        """아키텍처 설정에 따라 PyTorch 모델 생성"""
        import torch
        import torch.nn as nn
        
        layers = []
        prev_dim = input_dim
        
        # 활성화 함수 매핑
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }
        
        for i, layer_config in enumerate(architecture['layers']):
            units = layer_config['units']
            
            # 선형 레이어
            layers.append(nn.Linear(prev_dim, units))
            
            # 배치 정규화
            if architecture.get('use_batch_norm', True):
                layers.append(nn.BatchNorm1d(units))
            
            # 활성화 함수
            layers.append(activations[architecture.get('activation', 'relu')])
            
            # 드롭아웃
            if architecture.get('dropout', 0) > 0:
                layers.append(nn.Dropout(architecture['dropout']))
            
            prev_dim = units
        
        # 출력 레이어
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _evaluate_architecture(
        self,
        architecture: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """아키텍처 평가"""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 생성
        model = self._build_model(architecture, X_train.shape[1])
        model = model.to(device)
        
        # 데이터 준비
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=architecture.get('batch_size', 64),
            shuffle=True
        )
        
        # 옵티마이저 설정
        optimizer_name = architecture.get('optimizer', 'adam')
        lr = architecture.get('learning_rate', 1e-3)
        wd = architecture.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        
        criterion = nn.MSELoss()
        
        # 학습
        model.train()
        for epoch in range(50):  # 빠른 평가를 위해 적은 에폭
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        
        # 평가
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).squeeze()
            rmse = torch.sqrt(criterion(val_pred, y_val_t)).item()
        
        return rmse
    
    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50
    ):
        """아키텍처 탐색 수행"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("⚠️ Optuna가 설치되지 않았습니다.")
            return self
        
        def objective(trial):
            architecture = HyperparameterSpace.neural_architecture(trial)
            rmse = self._evaluate_architecture(architecture, X_train, y_train, X_val, y_val)
            return rmse
        
        print("🧠 Neural Architecture Search 시작...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_architecture = HyperparameterSpace.neural_architecture(study.best_trial)
        self.best_score = study.best_value
        
        print(f"\n✅ 최적 아키텍처 발견: RMSE = {self.best_score:.2f}")
        print(f"   레이어 수: {len(self.best_architecture['layers'])}")
        print(f"   드롭아웃: {self.best_architecture['dropout']:.2f}")
        print(f"   활성화: {self.best_architecture['activation']}")
        
        return self


class AutoEnsemble:
    """
    자동 앙상블 생성기
    
    여러 모델을 조합하여 최적의 앙상블을 구성합니다.
    """
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.base_models = []
        self.ensemble_weights = None
        self.scaler = StandardScaler()
        
    def _create_base_models(self) -> List[Tuple[str, Any]]:
        """기본 모델 후보 생성"""
        return [
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=200, random_state=42)),
            ('ridge', Ridge(alpha=1.0)),
            ('lasso', Lasso(alpha=1.0)),
            ('elastic', ElasticNet(alpha=1.0, l1_ratio=0.5)),
            ('bayesian', BayesianRidge()),
            ('knn', KNeighborsRegressor(n_neighbors=5)),
            ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
        ]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """앙상블 학습"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("⚠️ Optuna가 설치되지 않았습니다.")
            return self
        
        X_scaled = self.scaler.fit_transform(X)
        
        print("🎯 앙상블 모델 학습 중...")
        
        # 기본 모델 학습 및 평가
        model_scores = []
        trained_models = []
        
        for name, model in self._create_base_models():
            try:
                model.fit(X_scaled, y)
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
                rmse = -scores.mean()
                model_scores.append((name, rmse))
                trained_models.append((name, model, rmse))
                print(f"   {name}: RMSE = {rmse:.2f}")
            except Exception as e:
                print(f"   {name}: 학습 실패 - {e}")
        
        # 상위 모델 선택
        trained_models.sort(key=lambda x: x[2])
        self.base_models = [(name, model) for name, model, _ in trained_models[:5]]
        
        # 최적 가중치 탐색
        def objective(trial):
            weights = []
            for i in range(len(self.base_models)):
                w = trial.suggest_float(f'w{i}', 0, 1)
                weights.append(w)
            
            # 정규화
            total = sum(weights)
            weights = [w / total for w in weights]
            
            # 앙상블 예측
            preds = np.zeros(len(y))
            for (name, model), w in zip(self.base_models, weights):
                preds += w * model.predict(X_scaled)
            
            rmse = np.sqrt(mean_squared_error(y, preds))
            return rmse
        
        print("\n🔍 앙상블 가중치 최적화 중...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)
        
        # 최적 가중치 저장
        weights = []
        for i in range(len(self.base_models)):
            weights.append(study.best_params[f'w{i}'])
        total = sum(weights)
        self.ensemble_weights = [w / total for w in weights]
        
        print(f"\n✅ 앙상블 최적화 완료: RMSE = {study.best_value:.2f}")
        print("   모델 가중치:")
        for (name, _), w in zip(self.base_models, self.ensemble_weights):
            print(f"   - {name}: {w:.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """앙상블 예측"""
        X_scaled = self.scaler.transform(X)
        
        preds = np.zeros(len(X))
        for (name, model), w in zip(self.base_models, self.ensemble_weights):
            preds += w * model.predict(X_scaled)
        
        return preds


class TokyoHousingAutoML:
    """
    도쿄 23구 주택 가격 예측 전용 AutoML
    
    데이터 전처리부터 모델 학습, 예측까지 전체 파이프라인을 자동화합니다.
    """
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def _preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """데이터 전처리"""
        df = df.copy()
        
        # 범주형 변수 인코딩
        categorical_cols = ['ward', 'building_type', 'direction', 'station_category', 'age_category']
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # 불필요한 열 제거
        drop_cols = ['ward_jp', 'transaction_year', 'transaction_month']
        drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols)
        
        # 수치형 피처만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        if fit:
            self.feature_names = numeric_df.columns.tolist()
        
        # 스케일링
        if fit:
            return self.scaler.fit_transform(numeric_df)
        else:
            return self.scaler.transform(numeric_df)
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'price_man_yen',
        method: str = 'ensemble'
    ):
        """
        AutoML 학습
        
        Parameters:
        -----------
        df : pd.DataFrame
            학습 데이터
        target_col : str
            타겟 컬럼명
        method : str
            학습 방법 ('optuna', 'ensemble', 'nas')
        """
        # 피처와 타겟 분리
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # 전처리
        X_processed = self._preprocess(X, fit=True)
        
        print(f"📊 학습 데이터: {len(X)} 샘플, {X_processed.shape[1]} 피처")
        
        # 모델 학습
        if method == 'optuna':
            self.model = OptunaAutoML(self.config)
            self.model.fit(pd.DataFrame(X_processed), y)
        elif method == 'ensemble':
            self.model = AutoEnsemble(self.config)
            self.model.fit(pd.DataFrame(X_processed), y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """가격 예측"""
        if 'price_man_yen' in df.columns:
            df = df.drop(columns=['price_man_yen'])
        
        X_processed = self._preprocess(df, fit=False)
        return self.model.predict(pd.DataFrame(X_processed))
    
    def evaluate(self, df: pd.DataFrame, target_col: str = 'price_man_yen') -> Dict:
        """모델 평가"""
        y_true = df[target_col]
        y_pred = self.predict(df)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }


if __name__ == "__main__":
    # 테스트
    print("🏠 도쿄 23구 주택 가격 AutoML 테스트")
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'floor_area_sqm': np.random.normal(65, 20, n_samples),
        'building_age_years': np.random.exponential(15, n_samples),
        'distance_to_station_min': np.random.exponential(7, n_samples),
        'floor_number': np.random.randint(1, 15, n_samples),
        'price_man_yen': np.random.normal(5000, 2000, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['price_man_yen'] = df['price_man_yen'].clip(1000, 20000)
    
    # AutoML 실행
    automl = OptunaAutoML(AutoMLConfig(n_trials=10))
    
    X = df.drop(columns=['price_man_yen'])
    y = df['price_man_yen']
    
    automl.fit(X, y, model_type='rf')
    
    print(f"\n🎯 테스트 완료!")
