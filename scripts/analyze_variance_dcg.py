import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def _summary_stats(values):
  arr = np.asarray(values, dtype=np.float64)
  return {
      'mean': float(np.mean(arr)),
      'std': float(np.std(arr)),
      'var': float(np.var(arr)),
      'median': float(np.median(arr)),
      'q25': float(np.percentile(arr, 25)),
      'q75': float(np.percentile(arr, 75)),
      'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
  }


def _load_rows(csv_path):
  rows = []
  with open(csv_path, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows.append({
          'estimator': row['estimator'],
          'replicate_id': int(row['replicate_id']),
          'grad_norm': float(row['grad_norm']),
          'runtime_ms': float(row['runtime_ms']),
          'seed': int(row['seed']),
      })
  return rows


def _paired_deltas(rows):
  by_key = {}
  for row in rows:
    key = (row['replicate_id'], row['seed'])
    if key not in by_key:
      by_key[key] = {}
    by_key[key][row['estimator']] = row

  delta_grad = []
  delta_time = []
  for key, pair in by_key.items():
    if 'policygradient' not in pair or 'plrank' not in pair:
      continue
    grad_delta = pair['plrank']['grad_norm'] - pair['policygradient']['grad_norm']
    time_delta = pair['plrank']['runtime_ms'] - pair['policygradient']['runtime_ms']
    delta_grad.append(grad_delta)
    delta_time.append(time_delta)
  return np.asarray(delta_grad, dtype=np.float64), np.asarray(delta_time, dtype=np.float64)


def _print_summary(rows):
  pg_rows = [r for r in rows if r['estimator'] == 'policygradient']
  plr_rows = [r for r in rows if r['estimator'] == 'plrank']
  if not pg_rows or not plr_rows:
    raise ValueError('CSV must contain both policygradient and plrank rows.')

  pg_grad = _summary_stats([r['grad_norm'] for r in pg_rows])
  plr_grad = _summary_stats([r['grad_norm'] for r in plr_rows])
  pg_time = _summary_stats([r['runtime_ms'] for r in pg_rows])
  plr_time = _summary_stats([r['runtime_ms'] for r in plr_rows])

  delta_grad, delta_time = _paired_deltas(rows)
  if delta_grad.size == 0:
    raise ValueError('No paired replicates found for delta analysis.')

  print('Variance Analysis (DCG, paired replicates)')
  print('=========================================')
  print('PolicyGradient grad_norm: mean=%.6f std=%.6f var=%.6f' % (
      pg_grad['mean'], pg_grad['std'], pg_grad['var']))
  print('PL-Rank-1     grad_norm: mean=%.6f std=%.6f var=%.6f' % (
      plr_grad['mean'], plr_grad['std'], plr_grad['var']))
  print('')
  print('PolicyGradient runtime_ms: mean=%.6f std=%.6f var=%.6f median=%.6f IQR=%.6f' % (
      pg_time['mean'], pg_time['std'], pg_time['var'], pg_time['median'], pg_time['iqr']))
  print('PL-Rank-1     runtime_ms: mean=%.6f std=%.6f var=%.6f median=%.6f IQR=%.6f' % (
      plr_time['mean'], plr_time['std'], plr_time['var'], plr_time['median'], plr_time['iqr']))
  print('')
  print('Paired deltas (plrank - policygradient)')
  print('Delta grad_norm: mean=%.6f std=%.6f' % (
      float(np.mean(delta_grad)), float(np.std(delta_grad))))
  print('Delta runtime_ms: mean=%.6f std=%.6f' % (
      float(np.mean(delta_time)), float(np.std(delta_time))))
  print('Frac[PL-Rank grad_norm < PolicyGradient]: %.4f' % float(np.mean(delta_grad < 0.0)))
  print('Frac[PL-Rank runtime_ms < PolicyGradient]: %.4f' % float(np.mean(delta_time < 0.0)))
  print('')
  print('Note: runtime includes Python/TensorFlow synchronization effects. '
        '.numpy() calls inside gradient estimation can introduce device sync '
        'overhead, so runtime comparisons should be interpreted cautiously.')


def _save_plots(rows, grad_plot_path, runtime_plot_path):
  pg_grad = np.asarray([r['grad_norm'] for r in rows if r['estimator'] == 'policygradient'], dtype=np.float64)
  plr_grad = np.asarray([r['grad_norm'] for r in rows if r['estimator'] == 'plrank'], dtype=np.float64)
  pg_time = np.asarray([r['runtime_ms'] for r in rows if r['estimator'] == 'policygradient'], dtype=np.float64)
  plr_time = np.asarray([r['runtime_ms'] for r in rows if r['estimator'] == 'plrank'], dtype=np.float64)

  eps = 1e-12
  log_pg_grad = np.log(np.maximum(pg_grad, eps))
  log_plr_grad = np.log(np.maximum(plr_grad, eps))

  plt.figure(figsize=(10, 4.5))
  plt.subplot(1, 2, 1)
  plt.boxplot([pg_grad, plr_grad], tick_labels=['policygradient', 'plrank'])
  plt.ylabel('grad_norm')
  plt.title('Gradient Norm Distribution')

  plt.subplot(1, 2, 2)
  plt.boxplot([log_pg_grad, log_plr_grad], tick_labels=['policygradient', 'plrank'])
  plt.ylabel('log(grad_norm)')
  plt.title('Log Gradient Norm Distribution')
  plt.tight_layout()
  plt.savefig(grad_plot_path, dpi=160)
  plt.close()

  plt.figure(figsize=(5.5, 4.5))
  plt.boxplot([pg_time, plr_time], tick_labels=['policygradient', 'plrank'])
  plt.ylabel('runtime_ms')
  plt.title('Runtime Distribution')
  plt.tight_layout()
  plt.savefig(runtime_plot_path, dpi=160)
  plt.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv_path', type=str, default='runs/variance_dcg.csv',
                      help='Input variance CSV path.')
  parser.add_argument('--grad_plot_path', type=str, default='runs/variance_dcg_grad.png',
                      help='Output path for grad norm and log-grad plot.')
  parser.add_argument('--runtime_plot_path', type=str, default='runs/variance_dcg_runtime.png',
                      help='Output path for runtime plot.')
  args = parser.parse_args()

  rows = _load_rows(args.csv_path)
  if not rows:
    raise ValueError('No rows found in %s' % args.csv_path)

  grad_dir = os.path.dirname(args.grad_plot_path)
  runtime_dir = os.path.dirname(args.runtime_plot_path)
  if grad_dir:
    os.makedirs(grad_dir, exist_ok=True)
  if runtime_dir:
    os.makedirs(runtime_dir, exist_ok=True)

  _print_summary(rows)
  _save_plots(rows, args.grad_plot_path, args.runtime_plot_path)
  print('Wrote plots to %s and %s' % (args.grad_plot_path, args.runtime_plot_path))


if __name__ == '__main__':
  main()
