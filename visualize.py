"""
시각화 스크립트

학습 결과와 예측 결과를 시각화합니다.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일
sns.set_style("whitegrid")
sns.set_palette("husl")


def visualize_data_distribution(df, output_dir):
    """
    데이터 분포 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    output_dir : str
        출력 디렉토리
    """
    print("데이터 분포 시각화 중...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tokyo Housing Data Distribution', fontsize=16, fontweight='bold')
    
    # 1. 가격 분포
    axes[0, 0].hist(df['price_million_yen'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].set_xlabel('Price (Million Yen)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. 면적 분포
    axes[0, 1].hist(df['area_sqm'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('Area Distribution')
    axes[0, 1].set_xlabel('Area (sqm)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. 건물 연식 분포
    axes[0, 2].hist(df['building_age'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 2].set_title('Building Age Distribution')
    axes[0, 2].set_xlabel('Age (years)')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. 방 개수 분포
    room_counts = df['rooms'].value_counts().sort_index()
    axes[1, 0].bar(room_counts.index, room_counts.values, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 0].set_title('Number of Rooms Distribution')
    axes[1, 0].set_xlabel('Rooms')
    axes[1, 0].set_ylabel('Count')
    
    # 5. 구별 평균 가격
    ward_prices = df.groupby('ward')['price_million_yen'].mean().sort_values(ascending=False)
    axes[1, 1].barh(range(len(ward_prices)), ward_prices.values, alpha=0.7, color='teal')
    axes[1, 1].set_yticks(range(len(ward_prices)))
    axes[1, 1].set_yticklabels(ward_prices.index, fontsize=8)
    axes[1, 1].set_title('Average Price by Ward')
    axes[1, 1].set_xlabel('Price (Million Yen)')
    
    # 6. 면적 vs 가격 산점도
    axes[1, 2].scatter(df['area_sqm'], df['price_million_yen'], alpha=0.5, s=10)
    axes[1, 2].set_title('Area vs Price')
    axes[1, 2].set_xlabel('Area (sqm)')
    axes[1, 2].set_ylabel('Price (Million Yen)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 저장: {output_dir}/data_distribution.png")
    plt.close()


def visualize_correlation(df, output_dir):
    """
    상관관계 히트맵
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    output_dir : str
        출력 디렉토리
    """
    print("상관관계 히트맵 생성 중...")
    
    # 수치형 변수만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 저장: {output_dir}/correlation_matrix.png")
    plt.close()


def visualize_ward_analysis(df, output_dir):
    """
    구별 상세 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    output_dir : str
        출력 디렉토리
    """
    print("구별 분석 시각화 중...")
    
    # 구별 통계
    ward_stats = df.groupby('ward').agg({
        'price_million_yen': ['mean', 'median', 'std'],
        'area_sqm': 'mean',
        'building_age': 'mean'
    }).round(2)
    
    ward_stats.columns = ['_'.join(col).strip() for col in ward_stats.columns.values]
    ward_stats = ward_stats.sort_values('price_million_yen_mean', ascending=False)
    
    # Plotly 인터랙티브 차트
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Price by Ward', 'Average Area by Ward',
                       'Average Building Age by Ward', 'Price Std Dev by Ward')
    )
    
    # 1. 평균 가격
    fig.add_trace(
        go.Bar(x=ward_stats.index, y=ward_stats['price_million_yen_mean'],
               name='Avg Price', marker_color='lightblue'),
        row=1, col=1
    )
    
    # 2. 평균 면적
    fig.add_trace(
        go.Bar(x=ward_stats.index, y=ward_stats['area_sqm_mean'],
               name='Avg Area', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # 3. 평균 연식
    fig.add_trace(
        go.Bar(x=ward_stats.index, y=ward_stats['building_age_mean'],
               name='Avg Age', marker_color='lightsalmon'),
        row=2, col=1
    )
    
    # 4. 가격 표준편차
    fig.add_trace(
        go.Bar(x=ward_stats.index, y=ward_stats['price_million_yen_std'],
               name='Price Std', marker_color='plum'),
        row=2, col=2
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=800, showlegend=False,
                     title_text="Ward Analysis Dashboard")
    
    fig.write_html(f'{output_dir}/ward_analysis.html')
    print(f"  ✓ 저장: {output_dir}/ward_analysis.html")


def visualize_predictions(predictions_file, output_dir):
    """
    예측 결과 시각화
    
    Parameters:
    -----------
    predictions_file : str
        예측 결과 CSV 파일
    output_dir : str
        출력 디렉토리
    """
    if not os.path.exists(predictions_file):
        print(f"예측 결과 파일을 찾을 수 없습니다: {predictions_file}")
        return
    
    print("예측 결과 시각화 중...")
    
    df = pd.read_csv(predictions_file)
    
    # 예측 가격 차트
    fig = go.Figure(data=[
        go.Bar(x=df['번호'], y=df['예측가격(백만엔)'],
               text=df['예측가격(백만엔)'].round(2),
               textposition='auto',
               marker_color='indianred')
    ])
    
    fig.update_layout(
        title='Predicted Housing Prices',
        xaxis_title='Sample Number',
        yaxis_title='Price (Million Yen)',
        height=500
    )
    
    fig.write_html(f'{output_dir}/predictions.html')
    print(f"  ✓ 저장: {output_dir}/predictions.html")


def main():
    """메인 실행 함수"""
    
    print("\n" + "=" * 80)
    print("도쿄 주택 가격 데이터 시각화")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    output_dir = '/workspace/results/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # 1. 데이터 로드
    # ========================================================================
    
    data_path = '/workspace/data/tokyo_housing_data.csv'
    
    if not os.path.exists(data_path):
        print(f"\n데이터 파일을 찾을 수 없습니다: {data_path}")
        print("먼저 data/generate_tokyo_housing_data.py를 실행하세요.")
        return
    
    print(f"\n데이터 로드 중: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ 데이터 로드 완료: {len(df)} 샘플")
    
    # ========================================================================
    # 2. 시각화 생성
    # ========================================================================
    
    print("\n시각화 생성 중...")
    print("-" * 80)
    
    # 데이터 분포
    visualize_data_distribution(df, output_dir)
    
    # 상관관계
    visualize_correlation(df, output_dir)
    
    # 구별 분석
    visualize_ward_analysis(df, output_dir)
    
    # 예측 결과
    predictions_file = '/workspace/results/predictions.csv'
    visualize_predictions(predictions_file, output_dir)
    
    # ========================================================================
    # 3. 완료
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("시각화 완료!")
    print("=" * 80)
    print(f"\n모든 시각화 파일이 저장되었습니다: {output_dir}")
    print("\n생성된 파일:")
    print("  - data_distribution.png")
    print("  - correlation_matrix.png")
    print("  - ward_analysis.html (interactive)")
    print("  - predictions.html (interactive)")
    print("=" * 80)


if __name__ == "__main__":
    main()
