import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bland_altman_plot(y_true, y_pred, save_path):
    mean = (y_true + y_pred) / 2
    difference = (y_true - y_pred) / mean * 100  # 백분율 차이 계산
    
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    # 통계 계산 추가
    within_limits = np.sum((difference >= lower_limit) & (difference <= upper_limit))
    percentage_within = (within_limits / len(difference)) * 100
    
    plt.figure(figsize=(10, 6))
    
    # 산점도 스타일 개선
    plt.scatter(mean, difference, alpha=0.5, c='blue', label='Data Points')
    
    # 기준선과 한계선
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
    plt.axhline(upper_limit, color='red', linestyle=':', label='95% Limits of Agreement')
    plt.axhline(lower_limit, color='red', linestyle=':')
    
    # 통계 정보 표시 개선
    plt.text(mean.max()*0.95, mean_diff, f'Mean: {mean_diff:.2f}%', 
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(mean.max()*0.95, upper_limit, f'+1.96 SD: {upper_limit:.2f}%', 
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(mean.max()*0.95, lower_limit, f'-1.96 SD: {lower_limit:.2f}%', 
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(mean.max()*0.95, lower_limit - std_diff/2, 
             f'Within Limits: {percentage_within:.1f}%',
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Mean of Actual and Predicted CO (L/min)')
    plt.ylabel('Percent Difference ((Actual - Predicted) / Mean * 100)')
    plt.title('Bland-Altman Plot for CO Prediction')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # DPI 설정 추가
    plt.savefig(f'{save_path}/bland_altman_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 통계 결과 반환
    stats = {
        'mean_difference': mean_diff,
        'standard_deviation': std_diff,
        'upper_limit': upper_limit,
        'lower_limit': lower_limit,
        'percentage_within_limits': percentage_within
    }
    
    return stats