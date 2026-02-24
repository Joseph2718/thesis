import argparse
import csv
import os

import numpy as np


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


def _compute_summary(rows):
  pg_rows = [r for r in rows if r['estimator'] == 'policygradient']
  pl_rows = [r for r in rows if r['estimator'] == 'plrank']
  if not pg_rows or not pl_rows:
    raise ValueError('Missing policygradient or plrank rows.')

  pg_grad = np.asarray([r['grad_norm'] for r in pg_rows], dtype=np.float64)
  pl_grad = np.asarray([r['grad_norm'] for r in pl_rows], dtype=np.float64)
  pg_runtime = np.asarray([r['runtime_ms'] for r in pg_rows], dtype=np.float64)
  pl_runtime = np.asarray([r['runtime_ms'] for r in pl_rows], dtype=np.float64)

  by_key = {}
  for row in rows:
    key = (row['replicate_id'], row['seed'])
    if key not in by_key:
      by_key[key] = {}
    by_key[key][row['estimator']] = row

  grad_win = []
  runtime_win = []
  for pair in by_key.values():
    if 'policygradient' in pair and 'plrank' in pair:
      grad_win.append(pair['plrank']['grad_norm'] < pair['policygradient']['grad_norm'])
      runtime_win.append(pair['plrank']['runtime_ms'] < pair['policygradient']['runtime_ms'])

  return {
      'var_grad_policygradient': float(np.var(pg_grad)),
      'var_grad_plrank': float(np.var(pl_grad)),
      'mean_runtime_ms_policygradient': float(np.mean(pg_runtime)),
      'mean_runtime_ms_plrank': float(np.mean(pl_runtime)),
      'median_runtime_ms_policygradient': float(np.median(pg_runtime)),
      'median_runtime_ms_plrank': float(np.median(pl_runtime)),
      'frac_plrank_less_grad_norm': float(np.mean(np.asarray(grad_win, dtype=np.float64))),
      'frac_plrank_less_runtime': float(np.mean(np.asarray(runtime_win, dtype=np.float64))),
      'n_pairs': int(len(grad_win)),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', nargs='+', required=True,
                      help='Input variance CSV paths.')
  parser.add_argument('--labels', nargs='+', required=True,
                      help='Dataset labels corresponding to each CSV path.')
  parser.add_argument('--output_csv', type=str, default='runs/variance_summary_table.csv',
                      help='Output summary table CSV path.')
  args = parser.parse_args()

  if len(args.csv) != len(args.labels):
    raise ValueError('--csv and --labels must have same length.')

  out_dir = os.path.dirname(args.output_csv)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  records = []
  for label, csv_path in zip(args.labels, args.csv):
    rows = _load_rows(csv_path)
    summary = _compute_summary(rows)
    summary['dataset'] = label
    records.append(summary)

  fieldnames = [
      'dataset',
      'n_pairs',
      'var_grad_policygradient',
      'var_grad_plrank',
      'mean_runtime_ms_policygradient',
      'mean_runtime_ms_plrank',
      'median_runtime_ms_policygradient',
      'median_runtime_ms_plrank',
      'frac_plrank_less_grad_norm',
      'frac_plrank_less_runtime',
  ]
  with open(args.output_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
      writer.writerow(record)

  print('Wrote summary table to %s' % args.output_csv)
  for record in records:
    print('--- %s ---' % record['dataset'])
    print('Var(grad_norm): pg=%.6f plrank=%.6f' % (
        record['var_grad_policygradient'], record['var_grad_plrank']))
    print('Mean runtime_ms: pg=%.6f plrank=%.6f' % (
        record['mean_runtime_ms_policygradient'], record['mean_runtime_ms_plrank']))
    print('Median runtime_ms: pg=%.6f plrank=%.6f' % (
        record['median_runtime_ms_policygradient'], record['median_runtime_ms_plrank']))
    print('Frac[plrank < policygradient] grad_norm=%.4f runtime=%.4f' % (
        record['frac_plrank_less_grad_norm'], record['frac_plrank_less_runtime']))


if __name__ == '__main__':
  main()
