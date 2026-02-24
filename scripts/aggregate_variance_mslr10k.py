import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


POLICY_COLOR = '#1f77b4'
PLRANK_COLOR = '#ff7f0e'


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
  delta_runtime = []
  for pair in by_key.values():
    if 'policygradient' not in pair or 'plrank' not in pair:
      continue
    delta_grad.append(pair['plrank']['grad_norm'] - pair['policygradient']['grad_norm'])
    delta_runtime.append(pair['plrank']['runtime_ms'] - pair['policygradient']['runtime_ms'])
  return np.asarray(delta_grad, dtype=np.float64), np.asarray(delta_runtime, dtype=np.float64)


def _compute_fold_metrics(rows):
  pg_grad = np.asarray([r['grad_norm'] for r in rows if r['estimator'] == 'policygradient'], dtype=np.float64)
  pl_grad = np.asarray([r['grad_norm'] for r in rows if r['estimator'] == 'plrank'], dtype=np.float64)
  pg_runtime = np.asarray([r['runtime_ms'] for r in rows if r['estimator'] == 'policygradient'], dtype=np.float64)
  pl_runtime = np.asarray([r['runtime_ms'] for r in rows if r['estimator'] == 'plrank'], dtype=np.float64)

  d_grad, d_runtime = _paired_deltas(rows)
  cohen_d = 0.0
  if d_grad.size > 1:
    std = float(np.std(d_grad, ddof=1))
    if std > 0.0:
      cohen_d = float(np.mean(d_grad) / std)

  metrics = {
      'var_grad_pg': float(np.var(pg_grad)),
      'var_grad_plrank': float(np.var(pl_grad)),
      'variance_ratio': float(np.var(pg_grad) / np.var(pl_grad)),
      'mean_grad_pg': float(np.mean(pg_grad)),
      'mean_grad_plrank': float(np.mean(pl_grad)),
      'median_runtime_pg': float(np.median(pg_runtime)),
      'median_runtime_plrank': float(np.median(pl_runtime)),
      'runtime_ratio': float(np.median(pg_runtime) / np.median(pl_runtime)),
      'frac_plrank_lower_grad': float(np.mean(d_grad < 0.0)),
      'frac_plrank_faster': float(np.mean(d_runtime < 0.0)),
      'delta_grad_mean': float(np.mean(d_grad)),
      'delta_grad_std': float(np.std(d_grad)),
      'delta_grad_frac_lt_zero': float(np.mean(d_grad < 0.0)),
      'delta_grad_cohen_d': float(cohen_d),
  }
  return metrics


def _write_aggregate_csv(path, fold_rows):
  metric_cols = [
      'var_grad_pg',
      'var_grad_plrank',
      'variance_ratio',
      'mean_grad_pg',
      'mean_grad_plrank',
      'median_runtime_pg',
      'median_runtime_plrank',
      'runtime_ratio',
      'frac_plrank_lower_grad',
      'frac_plrank_faster',
  ]
  fieldnames = ['fold'] + metric_cols
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in fold_rows:
      writer.writerow({k: row[k] for k in fieldnames})

    final_row = {'fold': 'mean±std'}
    for col in metric_cols:
      vals = np.asarray([r[col] for r in fold_rows], dtype=np.float64)
      final_row[col] = '%.6f ± %.6f' % (float(np.mean(vals)), float(np.std(vals)))
    writer.writerow(final_row)


def _write_paired_csv(path, fold_rows):
  fields = ['fold', 'delta_grad_mean', 'delta_grad_std', 'delta_grad_frac_lt_zero', 'delta_grad_cohen_d']
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in fold_rows:
      writer.writerow({k: row[k] for k in fields})


def _violin_box(ax, policy_values, plrank_values, ylabel, title):
  parts = ax.violinplot([policy_values, plrank_values], positions=[1, 2], showmeans=False, showextrema=False, widths=0.8)
  for i, body in enumerate(parts['bodies']):
    body.set_facecolor(POLICY_COLOR if i == 0 else PLRANK_COLOR)
    body.set_edgecolor('black')
    body.set_alpha(0.35)
  ax.boxplot([policy_values, plrank_values],
             positions=[1, 2],
             widths=0.25,
             patch_artist=True,
             boxprops=dict(facecolor='white', color='black'),
             medianprops=dict(color='black', linewidth=2),
             whiskerprops=dict(color='black'),
             capprops=dict(color='black'))
  ax.set_xticks([1, 2])
  ax.set_xticklabels(['PolicyGradient', 'PL-Rank-1'])
  ax.set_ylabel(ylabel)
  ax.set_title(title)


def _save_figures(path_grad_pdf, path_runtime_pdf, all_policy_grad, all_plrank_grad, all_policy_runtime, all_plrank_runtime):
  os.makedirs(os.path.dirname(path_grad_pdf), exist_ok=True)
  os.makedirs(os.path.dirname(path_runtime_pdf), exist_ok=True)

  eps = 1e-12
  log_policy_grad = np.log(np.maximum(all_policy_grad, eps))
  log_plrank_grad = np.log(np.maximum(all_plrank_grad, eps))

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
  _violin_box(axes[0], all_policy_grad, all_plrank_grad, 'Gradient norm (L2)', 'Linear scale')
  _violin_box(axes[1], log_policy_grad, log_plrank_grad, 'log(Gradient norm)', 'Log scale')
  fig.suptitle('Gradient Norm Distribution (DCG, MSLR-WEB10k)')
  fig.tight_layout()
  fig.savefig(path_grad_pdf)
  plt.close(fig)

  fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.8))
  _violin_box(ax, all_policy_runtime, all_plrank_runtime, 'Runtime per step (ms)', 'Per-Step Runtime Distribution (DCG, MSLR-WEB10k)')
  fig.tight_layout()
  fig.savefig(path_runtime_pdf)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_csvs', nargs='+', required=True,
                      help='Fold CSV files ordered fold1...fold5.')
  parser.add_argument('--aggregate_csv', type=str, default='runs/variance_mslr10k_aggregate.csv')
  parser.add_argument('--paired_csv', type=str, default='runs/variance_mslr10k_paired_stats.csv')
  parser.add_argument('--grad_pdf', type=str, default='runs/variance_grad_mslr10k_allfolds.pdf')
  parser.add_argument('--runtime_pdf', type=str, default='runs/variance_runtime_mslr10k_allfolds.pdf')
  args = parser.parse_args()

  fold_rows = []
  all_policy_grad = []
  all_plrank_grad = []
  all_policy_runtime = []
  all_plrank_runtime = []

  for i, csv_path in enumerate(args.input_csvs, start=1):
    rows = _load_rows(csv_path)
    metrics = _compute_fold_metrics(rows)
    metrics['fold'] = i
    fold_rows.append(metrics)

    all_policy_grad.extend([r['grad_norm'] for r in rows if r['estimator'] == 'policygradient'])
    all_plrank_grad.extend([r['grad_norm'] for r in rows if r['estimator'] == 'plrank'])
    all_policy_runtime.extend([r['runtime_ms'] for r in rows if r['estimator'] == 'policygradient'])
    all_plrank_runtime.extend([r['runtime_ms'] for r in rows if r['estimator'] == 'plrank'])

  _write_aggregate_csv(args.aggregate_csv, fold_rows)
  _write_paired_csv(args.paired_csv, fold_rows)
  _save_figures(args.grad_pdf,
                args.runtime_pdf,
                np.asarray(all_policy_grad, dtype=np.float64),
                np.asarray(all_plrank_grad, dtype=np.float64),
                np.asarray(all_policy_runtime, dtype=np.float64),
                np.asarray(all_plrank_runtime, dtype=np.float64))

  variance_ratios = np.asarray([r['variance_ratio'] for r in fold_rows], dtype=np.float64)
  runtime_ratios = np.asarray([r['runtime_ratio'] for r in fold_rows], dtype=np.float64)
  print('Wrote %s' % args.aggregate_csv)
  print('Wrote %s' % args.paired_csv)
  print('Wrote %s' % args.grad_pdf)
  print('Wrote %s' % args.runtime_pdf)
  print('Mean variance ratio across folds = %.6f ± %.6f' % (
      float(np.mean(variance_ratios)), float(np.std(variance_ratios))))
  print('Mean runtime ratio across folds = %.6f ± %.6f' % (
      float(np.mean(runtime_ratios)), float(np.std(runtime_ratios))))


if __name__ == '__main__':
  main()
