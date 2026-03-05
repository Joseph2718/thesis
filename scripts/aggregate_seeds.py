import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _read_curve_csv(csv_path):
  points = {'policygradient': [], 'plrank_surrogate': []}
  with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
      if row['heldout_utility'] == '':
        continue
      method = row['method']
      if method not in points:
        continue
      points[method].append({
          'step': int(row['step']),
          'time_ms': float(row['cumulative_time_ms']),
          'utility': float(row['heldout_utility']),
          'total_fw': int(float(row['cumulative_total_generator_forward_passes'])),
          'online_fw': int(float(row['cumulative_online_generator_forward_passes'])),
      })
  return points


def _read_summary_csv(csv_path):
  rows = {}
  with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows[row['method']] = row
  return rows


def main():
  parser = argparse.ArgumentParser(
      description='Aggregate multi-seed PG vs PL-Rank results.')
  parser.add_argument('--run_dirs', nargs='+', required=True,
                      help='Paths to per-seed run directories.')
  parser.add_argument('--output_dir', type=str, required=True)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  all_curves = []
  all_summaries = []
  seed_labels = []

  for run_dir in sorted(args.run_dirs):
    curve_files = glob.glob(os.path.join(run_dir, 'train_set_utility_*.csv'))
    summary_file = os.path.join(run_dir, 'summary_table.csv')
    if not curve_files or not os.path.exists(summary_file):
      print('Skipping %s (missing files)' % run_dir)
      continue

    merged = {'policygradient': [], 'plrank_surrogate': []}
    for cf in curve_files:
      pts = _read_curve_csv(cf)
      for method in merged:
        merged[method].extend(pts.get(method, []))

    all_curves.append(merged)
    all_summaries.append(_read_summary_csv(summary_file))
    seed_labels.append(os.path.basename(run_dir))

  n_seeds = len(all_curves)
  if n_seeds == 0:
    print('No valid run directories found.')
    return
  print('Aggregating %d seeds: %s' % (n_seeds, ', '.join(seed_labels)))

  steps_pg = sorted(set(p['step'] for p in all_curves[0]['policygradient']))
  steps_pl = sorted(set(p['step'] for p in all_curves[0]['plrank_surrogate']))

  def _gather_by_step(method, steps):
    matrix = np.full((n_seeds, len(steps)), np.nan)
    time_matrix = np.full((n_seeds, len(steps)), np.nan)
    for si, curves in enumerate(all_curves):
      step_to_point = {p['step']: p for p in curves[method]}
      for ji, st in enumerate(steps):
        if st in step_to_point:
          matrix[si, ji] = step_to_point[st]['utility']
          time_matrix[si, ji] = step_to_point[st]['time_ms']
    return matrix, time_matrix

  pg_util, pg_time = _gather_by_step('policygradient', steps_pg)
  pl_util, pl_time = _gather_by_step('plrank_surrogate', steps_pl)

  pg_mean = np.nanmean(pg_util, axis=0)
  pg_std = np.nanstd(pg_util, axis=0)
  pl_mean = np.nanmean(pl_util, axis=0)
  pl_std = np.nanstd(pl_util, axis=0)

  pg_time_mean = np.nanmean(pg_time, axis=0)
  pl_time_mean = np.nanmean(pl_time, axis=0)

  baselines = []
  for s in all_summaries:
    if 'topk_retriever_baseline' in s:
      baselines.append(float(s['topk_retriever_baseline']['final_heldout_utility']))
  baseline_mean = float(np.mean(baselines)) if baselines else None

  # --- Utility vs Steps ---
  fig, ax = plt.subplots(figsize=(7.0, 4.3))
  ax.plot(steps_pg, pg_mean, label='PolicyGradient', color='C0')
  ax.fill_between(steps_pg, pg_mean - pg_std, pg_mean + pg_std, alpha=0.2, color='C0')
  ax.plot(steps_pl, pl_mean, label='PL-Rank surrogate', color='C1')
  ax.fill_between(steps_pl, pl_mean - pl_std, pl_mean + pl_std, alpha=0.2, color='C1')
  if baseline_mean is not None:
    ax.axhline(baseline_mean, color='gray', linestyle='--', linewidth=1, label='Top-K baseline')
  ax.set_xlabel('Step')
  ax.set_ylabel('Held-out set utility')
  ax.set_title('Set Utility vs Steps (%d seeds)' % n_seeds)
  ax.legend()
  fig.tight_layout()
  fig.savefig(os.path.join(args.output_dir, 'utility_vs_steps.png'), dpi=160)
  fig.savefig(os.path.join(args.output_dir, 'utility_vs_steps.pdf'))
  plt.close(fig)

  # --- Utility vs Time ---
  fig, ax = plt.subplots(figsize=(7.0, 4.3))
  ax.plot(pg_time_mean, pg_mean, label='PolicyGradient', color='C0')
  ax.fill_between(pg_time_mean, pg_mean - pg_std, pg_mean + pg_std, alpha=0.2, color='C0')
  ax.plot(pl_time_mean, pl_mean, label='PL-Rank surrogate', color='C1')
  ax.fill_between(pl_time_mean, pl_mean - pl_std, pl_mean + pl_std, alpha=0.2, color='C1')
  if baseline_mean is not None:
    ax.axhline(baseline_mean, color='gray', linestyle='--', linewidth=1, label='Top-K baseline')
  ax.set_xlabel('Cumulative wall-clock time (ms)')
  ax.set_ylabel('Held-out set utility')
  ax.set_title('Set Utility vs Time (%d seeds)' % n_seeds)
  ax.legend()
  fig.tight_layout()
  fig.savefig(os.path.join(args.output_dir, 'utility_vs_time.png'), dpi=160)
  fig.savefig(os.path.join(args.output_dir, 'utility_vs_time.pdf'))
  plt.close(fig)

  # --- Summary table ---
  summary_rows = []
  for method in ('policygradient', 'plrank_surrogate', 'topk_retriever_baseline'):
    finals = []
    times_to_thresh = []
    total_fws = []
    online_fws = []
    for s in all_summaries:
      if method not in s:
        continue
      row = s[method]
      finals.append(float(row['final_heldout_utility']))
      t = row.get('time_ms_to_threshold', 'N/A')
      if t != 'N/A' and t != '':
        times_to_thresh.append(float(t))
      total_fws.append(int(float(row['cumulative_total_generator_forward_passes'])))
      online_fws.append(int(float(row['cumulative_online_generator_forward_passes'])))

    if not finals:
      continue
    summary_rows.append({
        'method': method,
        'mean_final_utility': float(np.mean(finals)),
        'std_final_utility': float(np.std(finals)),
        'mean_time_ms_to_threshold': float(np.mean(times_to_thresh)) if times_to_thresh else 'N/A',
        'std_time_ms_to_threshold': float(np.std(times_to_thresh)) if times_to_thresh else 'N/A',
        'mean_total_generator_fw': float(np.mean(total_fws)),
        'mean_online_generator_fw': float(np.mean(online_fws)),
        'n_seeds': len(finals),
    })

  summary_csv_path = os.path.join(args.output_dir, 'aggregate_summary.csv')
  with open(summary_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'method', 'mean_final_utility', 'std_final_utility',
        'mean_time_ms_to_threshold', 'std_time_ms_to_threshold',
        'mean_total_generator_fw', 'mean_online_generator_fw', 'n_seeds'])
    writer.writeheader()
    for row in summary_rows:
      writer.writerow(row)

  print('\n=== Aggregate Summary (%d seeds) ===' % n_seeds)
  for row in summary_rows:
    print('  %s: utility=%.4f±%.4f  time_to_thresh=%s  total_fw=%.0f  online_fw=%.0f' % (
        row['method'],
        row['mean_final_utility'], row['std_final_utility'],
        ('%.0f±%.0f ms' % (row['mean_time_ms_to_threshold'], row['std_time_ms_to_threshold'])
         if row['mean_time_ms_to_threshold'] != 'N/A' else 'N/A'),
        row['mean_total_generator_fw'],
        row['mean_online_generator_fw']))

  print('\nPlots: %s' % args.output_dir)
  print('Summary CSV: %s' % summary_csv_path)


if __name__ == '__main__':
  main()
