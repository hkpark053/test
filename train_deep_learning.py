#!/usr/bin/env python3
"""
도쿄 23구 주택 가격 예측 - 딥러닝 학습 스크립트

이 스크립트는 다양한 딥러닝 아키텍처를 사용하여 주택 가격을 예측합니다.

사용법:
    python train_deep_learning.py --model transformer --epochs 100
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 프로젝트 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.deep_learning_layers import (
    BasicMLP,
    ResidualNetwork,
    TabNetModel,
    TransformerTabular,
    WideAndDeepModel,
    HuberLoss,
    MAPELoss,
    CombinedLoss,
    create_model
)
from data.generate_tokyo_housing_data import generate_housing_data, add_derived_features, split_data


class TokyoHousingDataset(Dataset):
    """도쿄 주택 데이터셋"""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'price_man_yen'):
        self.target = torch.FloatTensor(df[target_col].values)
        
        # 피처 준비
        features_df = df.drop(columns=[target_col])
        
        # 범주형 변수 처리
        categorical_cols = ['ward', 'building_type', 'direction']
        categorical_cols = [c for c in categorical_cols if c in features_df.columns]
        
        # 수치형만 선택
        numeric_df = features_df.select_dtypes(include=[np.number])
        
        self.features = torch.FloatTensor(numeric_df.values)
        self.feature_names = numeric_df.columns.tolist()
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


class Trainer:
    """딥러닝 모델 트레이너"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_type: str = 'mse'
    ):
        self.model = model.to(device)
        self.device = device
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 손실 함수
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'huber':
            self.criterion = HuberLoss()
        elif loss_type == 'combined':
            self.criterion = CombinedLoss()
        else:
            self.criterion = nn.MSELoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 모델에 따라 출력 처리
            output = self.model(batch_x)
            if isinstance(output, tuple):
                pred = output[0]  # TabNet은 (output, attention) 반환
            else:
                pred = output
            
            loss = self.criterion(pred, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * len(batch_x)
            all_preds.extend(pred.detach().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        return avg_loss, rmse
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """평가"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = self.model(batch_x)
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
            
            loss = self.criterion(pred, batch_y)
            
            total_loss += loss.item() * len(batch_x)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        return avg_loss, rmse
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20
    ):
        """학습 실행"""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("\n🚀 학습 시작...")
        print(f"   에폭: {epochs}")
        print(f"   조기 종료 patience: {early_stopping_patience}")
        print("-" * 60)
        
        for epoch in range(epochs):
            # 학습
            train_loss, train_rmse = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_rmse = self.evaluate(val_loader)
            
            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            
            # 스케줄러 업데이트
            self.scheduler.step(val_loss)
            
            # 출력
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, RMSE={train_rmse:.0f} | "
                      f"Val Loss={val_loss:.4f}, RMSE={val_rmse:.0f}")
            
            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️ 조기 종료 (에폭 {epoch+1})")
                break
        
        # 최적 모델 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print("-" * 60)
        print(f"✅ 학습 완료! 최적 검증 손실: {best_val_loss:.4f}")
        
        return self.history
    
    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """예측"""
        self.model.eval()
        all_preds = []
        
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(self.device)
            
            output = self.model(batch_x)
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
            
            all_preds.extend(pred.cpu().numpy())
        
        return np.array(all_preds)


def preprocess_data(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """데이터 전처리"""
    df = df.copy()
    
    # 범주형 변수 인코딩
    categorical_cols = ['ward', 'building_type', 'direction', 'station_category', 'age_category']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # 불필요한 열 제거
    drop_cols = ['ward_jp']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 수치형만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # price_man_yen은 제외하고 스케일링
    feature_cols = [c for c in numeric_cols if c != 'price_man_yen']
    
    if fit:
        if scaler is None:
            scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler


def setup_args():
    """커맨드 라인 인자 설정"""
    parser = argparse.ArgumentParser(description='도쿄 23구 주택 가격 예측 딥러닝')
    
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'resnet', 'tabnet', 'transformer'],
                        help='모델 타입')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='학습률')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='히든 레이어 차원')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='생성할 샘플 수')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='결과 저장 디렉토리')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'huber', 'combined'],
                        help='손실 함수')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    
    return parser.parse_args()


def visualize_training(history: Dict, output_dir: str):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 손실
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('학습 손실')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE
    ax2 = axes[1]
    ax2.plot(history['train_rmse'], label='Train')
    ax2.plot(history['val_rmse'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE (만엔)')
    ax2.set_title('RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    print(f"📊 학습 곡선 저장: {output_dir}/training_curves.png")


def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 실제 vs 예측
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('실제 가격 (만엔)')
    ax1.set_ylabel('예측 가격 (만엔)')
    ax1.set_title('실제 vs 예측')
    ax1.grid(True, alpha=0.3)
    
    # 오차 분포
    ax2 = axes[1]
    errors = y_pred - y_true
    ax2.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('예측 오차 (만엔)')
    ax2.set_ylabel('빈도')
    ax2.set_title(f'오차 분포 (평균: {errors.mean():.0f}, 표준편차: {errors.std():.0f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_results.png'), dpi=150)
    plt.close()
    
    print(f"📊 예측 결과 저장: {output_dir}/prediction_results.png")


def main():
    """메인 함수"""
    args = setup_args()
    
    # 랜덤 시드 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 디바이스: {device}")
    
    # 출력 디렉토리
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'dl_{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("🏠 도쿄 23구 주택 가격 예측 - 딥러닝")
    print("="*60)
    print(f"   모델: {args.model}")
    print(f"   에폭: {args.epochs}")
    print(f"   배치 크기: {args.batch_size}")
    print(f"   학습률: {args.lr}")
    print(f"   손실 함수: {args.loss}")
    print("="*60)
    
    # 데이터 로드/생성
    data_path = 'data/tokyo_housing_data.csv'
    if os.path.exists(data_path):
        print(f"\n📂 데이터 로드: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"\n🏠 데이터 생성 중...")
        df = generate_housing_data(n_samples=args.n_samples)
        df = add_derived_features(df)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
    
    print(f"   샘플 수: {len(df)}")
    
    # 데이터 분할
    train_df, val_df, test_df = split_data(df)
    
    # 전처리
    train_df, scaler = preprocess_data(train_df, fit=True)
    val_df, _ = preprocess_data(val_df, scaler=scaler, fit=False)
    test_df, _ = preprocess_data(test_df, scaler=scaler, fit=False)
    
    # 데이터셋 생성
    train_dataset = TokyoHousingDataset(train_df)
    val_dataset = TokyoHousingDataset(val_df)
    test_dataset = TokyoHousingDataset(test_df)
    
    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 입력 차원
    input_dim = train_dataset.features.shape[1]
    print(f"   입력 차원: {input_dim}")
    
    # 모델 생성
    print(f"\n🧠 모델 생성: {args.model}")
    
    if args.model == 'mlp':
        model = BasicMLP(
            input_dim=input_dim,
            hidden_dims=[args.hidden_dim * 2, args.hidden_dim, args.hidden_dim // 2],
            dropout=0.2
        )
    elif args.model == 'resnet':
        model = ResidualNetwork(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=4,
            dropout=0.1
        )
    elif args.model == 'tabnet':
        model = TabNetModel(
            input_dim=input_dim,
            n_d=64,
            n_a=64,
            n_steps=3
        )
    elif args.model == 'transformer':
        model = TransformerTabular(
            num_features=input_dim,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        )
    
    # 모델 파라미터 수
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   파라미터 수: {n_params:,}")
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=1e-4,
        loss_type=args.loss
    )
    
    # 학습
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=20
    )
    
    # 학습 곡선 시각화
    visualize_training(history, output_dir)
    
    # 테스트 평가
    print("\n📊 테스트 세트 평가:")
    y_true = test_df['price_man_yen'].values
    y_pred = trainer.predict(test_loader)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"   RMSE: {rmse:,.0f} 만엔")
    print(f"   MAE:  {mae:,.0f} 만엔")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # 예측 결과 시각화
    visualize_predictions(y_true, y_pred, output_dir)
    
    # 모델 저장
    model_path = os.path.join(output_dir, f'{args.model}_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': args.model,
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim
        }
    }, model_path)
    print(f"\n💾 모델 저장: {model_path}")
    
    # 결과 저장
    results = {
        'model': args.model,
        'epochs': len(history['train_loss']),
        'test_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        },
        'args': vars(args)
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ 학습 완료!")
    print("="*60)
    print(f"\n📁 결과 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()
