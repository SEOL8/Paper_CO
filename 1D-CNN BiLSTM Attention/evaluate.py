import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm

from utils.metrics import calculate_metrics
from utils.bland_altman import bland_altman_plot


def evaluate_and_visualize(model, test_loader, device, save_path):
    model.eval()
    y_true_list, y_pred_list, pid_list = [], [], []

    with torch.no_grad():
        for ppg, ecg, patient_info, targets, pids in tqdm(test_loader, desc="Evaluating"):
            ppg, ecg, patient_info, targets = (
                ppg.to(device), ecg.to(device),
                patient_info.to(device), targets.to(device)
            )
            outputs = model(ppg, ecg, patient_info)
            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(outputs.cpu().numpy())
            pid_list.extend(pids.numpy())

    y_true    = np.array(y_true_list)
    y_pred    = np.array(y_pred_list)
    pid_array = np.array(pid_list)

    overall = calculate_metrics(y_true, y_pred, loss=mean_squared_error(y_true, y_pred))

    print("\n[Overall Results]")
    print(f"MSE  : {overall['mse']:.4f}")
    print(f"RMSE : {overall['rmse']:.4f}")
    print(f"MAE  : {overall['mae']:.4f}")
    print(f"R2   : {overall['r2']:.4f}")
    print(f"MAPE : {overall['mape']:.4f}%")

    patient_metrics = {}
    for pid in np.unique(pid_array):
        mask = pid_array == pid
        pt, pp = y_true[mask], y_pred[mask]
        patient_metrics[pid] = {
            'mse':  mean_squared_error(pt, pp),
            'rmse': np.sqrt(mean_squared_error(pt, pp)),
            'mae':  mean_absolute_error(pt, pp),
            'r2':   r2_score(pt, pp) if len(np.unique(pt)) > 1 else 0.0,
            'mape': mean_absolute_percentage_error(pt, pp),
        }

    patient_summary = pd.DataFrame.from_dict(patient_metrics, orient='index')
    patient_summary.index.name = 'patient_id'

    print("\n[Patient-wise Summary]")
    for metric in ['mse', 'rmse', 'mae', 'r2', 'mape']:
        print(f"{metric.upper():5s}: {patient_summary[metric].mean():.4f} "
              f"± {patient_summary[metric].std():.4f}")

    results_df = pd.DataFrame({'patient_id': pid_array,
                                'actual_co':    y_true,
                                'predicted_co': y_pred})
    for metric in ['mse', 'rmse', 'mae', 'r2']:
        results_df[f'patient_{metric}'] = results_df['patient_id'].map(
            lambda x: patient_metrics[x][metric])

    results_df.to_csv(f'{save_path}/evaluation_results.csv', index=False)
    patient_summary.to_csv(f'{save_path}/patient_summary.csv')

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual CO (L/min)')
    plt.ylabel('Predicted CO (L/min)')
    plt.title('Actual vs Predicted CO')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f'{save_path}/scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Error distribution
    plt.figure(figsize=(8, 5))
    plt.hist(y_true - y_pred, bins=50, alpha=0.75)
    plt.xlabel('Prediction Error (L/min)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f'{save_path}/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    bland_altman_stats = bland_altman_plot(y_true, y_pred, save_path)

    return {
        **overall,
        'predictions':        y_pred,
        'true_values':        y_true,
        'patient_ids':        pid_array,
        'patient_metrics':    patient_metrics,
        'bland_altman_stats': bland_altman_stats,
    }
