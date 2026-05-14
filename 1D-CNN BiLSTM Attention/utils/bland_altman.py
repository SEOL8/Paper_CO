import numpy as np
import matplotlib.pyplot as plt


def bland_altman_plot(y_true, y_pred, save_path):
    mean       = (y_true + y_pred) / 2
    difference = (y_true - y_pred) / mean * 100   # percentage difference

    mean_diff  = np.mean(difference)
    std_diff   = np.std(difference)
    upper_loa  = mean_diff + 1.96 * std_diff
    lower_loa  = mean_diff - 1.96 * std_diff

    # PE: half the total LoA width, expressed as a percentage
    pe = (upper_loa - lower_loa) / 2

    within_loa = np.sum((difference >= lower_loa) & (difference <= upper_loa))
    pct_within = within_loa / len(difference) * 100

    plt.figure(figsize=(10, 6))
    plt.scatter(mean, difference, alpha=0.5, c='blue', label='Data Points')
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
    plt.axhline(upper_loa,  color='red', linestyle=':',  label='95% Limits of Agreement')
    plt.axhline(lower_loa,  color='red', linestyle=':')

    x_text = mean.max() * 0.95
    for y_pos, label in [
        (mean_diff, f'Mean: {mean_diff:.2f}%'),
        (upper_loa,  f'+1.96 SD: {upper_loa:.2f}%'),
        (lower_loa,  f'-1.96 SD: {lower_loa:.2f}%'),
        (lower_loa - std_diff / 2, f'Within LoA: {pct_within:.1f}%'),
    ]:
        plt.text(x_text, y_pos, label,
                 verticalalignment='center', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Mean of actual and predicted CO (L/min)')
    plt.ylabel('Percentage difference [(actual − predicted) / mean × 100]')
    plt.title('Bland-Altman plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/bland_altman_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'mean_bias':          mean_diff,
        'std':                std_diff,
        'upper_loa':          upper_loa,
        'lower_loa':          lower_loa,
        'pe':                 pe,
        'pct_within_loa':     pct_within,
    }
