"""
도쿄 23구 주택 가격 예측을 위한 딥러닝 레이어 및 모델

이 모듈은 다양한 딥러닝 아키텍처를 포함합니다:
1. 기본 MLP (Multi-Layer Perceptron)
2. Wide & Deep 네트워크
3. TabNet 스타일 Attention 네트워크
4. Residual 네트워크
5. Transformer 기반 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


# ============================================================================
# 기본 빌딩 블록 레이어들
# ============================================================================

class FeatureEmbedding(nn.Module):
    """
    범주형 피처를 위한 임베딩 레이어
    
    각 범주형 피처를 고정 차원의 벡터로 변환합니다.
    """
    def __init__(self, num_categories: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class NumericalBatchNorm(nn.Module):
    """
    수치형 피처를 위한 Batch Normalization 레이어
    
    수치형 입력을 정규화하여 학습 안정성을 높입니다.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(x)


class GatedLinearUnit(nn.Module):
    """
    GLU (Gated Linear Unit) 레이어
    
    TabNet에서 사용되는 게이트 메커니즘으로,
    피처 선택을 학습할 수 있습니다.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x[:, :self.output_dim] * torch.sigmoid(x[:, self.output_dim:])


class FeatureAttention(nn.Module):
    """
    피처 어텐션 레이어
    
    각 피처의 중요도를 학습하여 가중치를 부여합니다.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 어텐션 가중치 계산
        weights = self.attention(x)
        # 가중치 적용
        attended = x * weights
        return attended, weights


class ResidualBlock(nn.Module):
    """
    잔차 연결 블록 (Residual Block)
    
    깊은 네트워크에서 기울기 소실 문제를 방지합니다.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 블록
    
    채널(피처) 간의 상호 의존성을 모델링합니다.
    """
    def __init__(self, num_features: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_features, num_features // reduction),
            nn.ReLU(),
            nn.Linear(num_features // reduction, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: Global information
        batch_size = x.size(0)
        se_weight = self.excitation(x)
        return x * se_weight


# ============================================================================
# 완전한 딥러닝 모델들
# ============================================================================

class BasicMLP(nn.Module):
    """
    기본 MLP (Multi-Layer Perceptron) 모델
    
    가장 기본적인 피드포워드 신경망으로,
    주택 가격 예측의 베이스라인으로 사용됩니다.
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 출력 레이어 (가격 예측)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class WideAndDeepModel(nn.Module):
    """
    Wide & Deep 모델
    
    Google에서 제안한 추천 시스템 모델을 회귀 문제에 적용.
    - Wide: 선형 모델로 암기(memorization) 담당
    - Deep: DNN으로 일반화(generalization) 담당
    """
    def __init__(
        self,
        num_numerical: int,
        num_categorical: int,
        embedding_dim: int = 8,
        deep_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        
        # Wide 부분: 선형 모델
        wide_dim = num_numerical + num_categorical * embedding_dim
        self.wide = nn.Linear(wide_dim, 1)
        
        # Deep 부분: DNN
        deep_layers = []
        prev_dim = wide_dim
        
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_layers)
        
        # 최종 결합
        self.final = nn.Linear(2, 1)
        
    def forward(
        self, 
        numerical: torch.Tensor, 
        categorical_embedded: torch.Tensor
    ) -> torch.Tensor:
        # 피처 결합
        x = torch.cat([numerical, categorical_embedded], dim=1)
        
        # Wide와 Deep 경로
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        
        # 결합
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.final(combined).squeeze(-1)


class TabNetBlock(nn.Module):
    """
    TabNet 스타일의 어텐션 블록
    
    피처 선택을 순차적으로 수행하며,
    해석 가능한 피처 중요도를 제공합니다.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_d: int = 64,  # Decision step dimension
        n_a: int = 64,  # Attention dimension
        gamma: float = 1.3,  # Coefficient for feature reusage
    ):
        super().__init__()
        
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma
        
        # 공유 레이어
        self.shared_fc = nn.Linear(input_dim, n_d + n_a)
        
        # 어텐션 레이어
        self.attention = nn.Sequential(
            nn.Linear(n_a, input_dim),
            nn.BatchNorm1d(input_dim),
        )
        
        # 출력 레이어
        self.output_fc = nn.Linear(n_d, output_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        prior_scales: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if prior_scales is None:
            prior_scales = torch.ones_like(x)
        
        # 마스킹된 피처
        masked_x = x * prior_scales
        
        # 공유 변환
        shared_out = self.shared_fc(masked_x)
        n_d = shared_out[:, :self.n_d]
        n_a = shared_out[:, self.n_d:]
        
        # 어텐션 계산
        attention = self.attention(n_a)
        attention = attention * prior_scales
        attention = F.softmax(attention, dim=-1)
        
        # 다음 스텝을 위한 prior scales 업데이트
        new_prior_scales = prior_scales * (self.gamma - attention)
        
        # 출력
        output = self.output_fc(torch.relu(n_d))
        
        return output, attention, new_prior_scales


class TabNetModel(nn.Module):
    """
    TabNet 모델 (전체 구현)
    
    순차적인 어텐션 메커니즘을 통해 피처를 선택하고,
    해석 가능한 예측을 제공합니다.
    """
    def __init__(
        self,
        input_dim: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        output_dim: int = 1
    ):
        super().__init__()
        
        self.n_steps = n_steps
        
        # 초기 배치 정규화
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # TabNet 블록들
        self.blocks = nn.ModuleList([
            TabNetBlock(input_dim, n_d, n_d, n_a, gamma)
            for _ in range(n_steps)
        ])
        
        # 최종 출력
        self.final = nn.Linear(n_d, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_bn(x)
        
        prior_scales = torch.ones_like(x)
        aggregated_output = torch.zeros(x.size(0), self.blocks[0].n_d, device=x.device)
        total_attention = torch.zeros_like(x)
        
        for block in self.blocks:
            output, attention, prior_scales = block(x, prior_scales)
            aggregated_output += output
            total_attention += attention
        
        # 어텐션 정규화
        total_attention = total_attention / self.n_steps
        
        final_output = self.final(aggregated_output).squeeze(-1)
        
        return final_output, total_attention


class ResidualNetwork(nn.Module):
    """
    잔차 네트워크 (ResNet for Tabular Data)
    
    깊은 네트워크에서도 효과적인 학습이 가능하도록
    Skip Connection을 활용합니다.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 입력 프로젝션
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 잔차 블록들
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(num_blocks)
        ])
        
        # SE 블록 추가
        self.se_block = SEBlock(hidden_dim)
        
        # 출력 레이어
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.se_block(x)
        
        return self.output(x).squeeze(-1)


class TransformerTabular(nn.Module):
    """
    테이블 데이터를 위한 Transformer 모델
    
    각 피처를 토큰으로 취급하여 Self-Attention을 적용합니다.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        
        # 각 피처를 d_model 차원으로 임베딩
        self.feature_embedding = nn.Linear(1, d_model)
        
        # 위치 임베딩 (피처 위치)
        self.position_embedding = nn.Embedding(num_features, d_model)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # CLS 토큰 (집계용)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 출력 레이어
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 각 피처를 개별 토큰으로 변환
        x = x.unsqueeze(-1)  # (batch, num_features, 1)
        x = self.feature_embedding(x)  # (batch, num_features, d_model)
        
        # 위치 임베딩 추가
        positions = torch.arange(self.num_features, device=x.device)
        x = x + self.position_embedding(positions)
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer 적용
        x = self.transformer(x)
        
        # CLS 토큰의 출력만 사용
        cls_output = x[:, 0]
        
        return self.output(cls_output).squeeze(-1)


class EnsembleModel(nn.Module):
    """
    앙상블 모델
    
    여러 모델의 예측을 결합하여 더 안정적인 예측을 제공합니다.
    """
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        weighted_pred = (predictions * self.weights.view(-1, 1)).sum(dim=0)
        
        return weighted_pred


# ============================================================================
# 손실 함수들
# ============================================================================

class HuberLoss(nn.Module):
    """
    Huber 손실 함수
    
    이상치에 강건한 손실 함수로, MSE와 MAE의 장점을 결합합니다.
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        return loss.mean()


class MAPELoss(nn.Module):
    """
    MAPE (Mean Absolute Percentage Error) 손실
    
    가격 예측에서 퍼센트 기반 오차를 최소화합니다.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (torch.abs(pred - target) / (torch.abs(target) + self.epsilon)).mean() * 100


class CombinedLoss(nn.Module):
    """
    결합 손실 함수
    
    여러 손실 함수를 가중 결합합니다.
    """
    def __init__(
        self, 
        mse_weight: float = 0.5, 
        mae_weight: float = 0.3, 
        mape_weight: float = 0.2
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mape_weight = mape_weight
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.mape = MAPELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_weight * self.mse(pred, target) +
            self.mae_weight * self.mae(pred, target) +
            self.mape_weight * self.mape(pred, target)
        )


# ============================================================================
# 모델 팩토리
# ============================================================================

def create_model(
    model_name: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    모델 생성 팩토리 함수
    
    Parameters:
    -----------
    model_name : str
        모델 이름 ('mlp', 'resnet', 'tabnet', 'transformer')
    input_dim : int
        입력 피처 수
    **kwargs : dict
        모델별 추가 파라미터
        
    Returns:
    --------
    nn.Module
        생성된 모델
    """
    models = {
        'mlp': BasicMLP,
        'resnet': ResidualNetwork,
        'tabnet': TabNetModel,
        'transformer': TransformerTabular,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](input_dim=input_dim, **kwargs)


if __name__ == "__main__":
    # 모델 테스트
    print("🧪 딥러닝 레이어 테스트...")
    
    batch_size = 32
    input_dim = 20
    
    # 랜덤 입력 생성
    x = torch.randn(batch_size, input_dim)
    
    # 각 모델 테스트
    models_to_test = [
        ("BasicMLP", BasicMLP(input_dim)),
        ("ResidualNetwork", ResidualNetwork(input_dim)),
        ("TabNetModel", TabNetModel(input_dim)),
        ("TransformerTabular", TransformerTabular(input_dim)),
    ]
    
    for name, model in models_to_test:
        if name == "TabNetModel":
            output, attention = model(x)
            print(f"✅ {name}: output shape = {output.shape}, attention shape = {attention.shape}")
        else:
            output = model(x)
            print(f"✅ {name}: output shape = {output.shape}")
    
    print("\n✅ 모든 모델 테스트 완료!")
