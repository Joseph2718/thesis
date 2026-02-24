import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def _fmt_mean_std(values):
  arr = np.asarray(values, dtype=np.float64)
  return '%.6f ± %.6f' % (float(np.mean(arr)), float(np.std(arr)))


def _read_summary(path):
  rows = []
  with open(path, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows.append(row)
  return rows


def _read_train(path):
  rows = []
  with open(path, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows.append(row)
  return rows


def _as_float_or_none(value):
  if value in ('', 'N/A', None):
    return None
  return float(value)


def _as_int_or_none(value):
  if value in ('', 'N/A', None):
    return None
  return int(float(value))


def _get_row(summary_rows, method):
  for row in summary_rows:
    if row['method'] == method:
      return row
  raise ValueError('Method not found in summary: %s' % method)


def _eval_points(train_rows, precompute_fw):
  points = []
  for row in train_rows:
    if row['heldout_utility'] == '':
      continue
    total_fw = int(float(row['cumulative_reward_forward_passes']))
    points.append({
        'step': int(row['step']),
        'time_ms': float(row['cumulative_time_ms']),
        'total_fw': total_fw,
        'online_fw': total_fw - precompute_fw,
        'utility': float(row['heldout_utility']),
    })
  return points


def _final_time_ms(train_rows):
  return float(train_rows[-1]['cumulative_time_ms'])


def _forward_passes_to_step(eval_points, step):
  for row in eval_points:
    if int(row['step']) == step:
      return int(row['total_fw'])
  return None


def _interp_at(points, x_key, x_value):
  x = np.asarray([p[x_key] for p in points], dtype=np.float64)
  y = np.asarray([p['utility'] for p in points], dtype=np.float64)
  if x.size == 0:
    return 0.0
  order = np.argsort(x)
  x = x[order]
  y = y[order]
  return float(np.interp(x_value, x, y, left=y[0], right=y[-1]))


def _interp_curve(points, x_key, x_grid):
  x = np.asarray([p[x_key] for p in points], dtype=np.float64)
  y = np.asarray([p['utility'] for p in points], dtype=np.float64)
  if x.size == 0:
    return np.zeros_like(x_grid, dtype=np.float64)
  order = np.argsort(x)
  x = x[order]
  y = y[order]
  return np.interp(x_grid, x, y, left=y[0], right=y[-1])


def _aggregate_method(method, seed_dirs, common_time_budget, common_total_fw_budget, common_online_fw_budget, grid_size=300):
  final_utility = []
  best_utility = []
  final_em = []
  final_f1 = []
  total_fw = []
  total_time_ms = []
  online_fw = []

  steps_to_threshold = []
  time_to_threshold_ms = []
  fw_to_threshold = []
  online_fw_to_threshold = []

  nauc_time = []
  nauc_total_fw = []
  nauc_online_fw = []
  best_utility_common_time = []
  best_utility_common_total_fw = []
  best_utility_common_online_fw = []
  utility_at_30s = []
  utility_at_60s = []
  utility_at_120s = []
  utility_at_total_budget = []
  utility_at_online_budget = []

  curves_time = []
  curves_total_fw = []
  curves_online_fw = []

  x_time = np.linspace(0.0, common_time_budget, grid_size)
  x_total_fw = np.linspace(0.0, common_total_fw_budget, grid_size)
  x_online_fw = np.linspace(0.0, common_online_fw_budget, grid_size)

  reached = 0

  for seed_dir in seed_dirs:
    summary_rows = _read_summary(os.path.join(seed_dir, 'summary_table.csv'))
    row = _get_row(summary_rows, method)
    train_rows = _read_train(os.path.join(seed_dir, 'train_set_utility_%s.csv' % method))

    precompute_fw = int(float(row['singleton_precompute_forward_passes']))
    points = _eval_points(train_rows, precompute_fw)

    final_utility.append(float(row['final_heldout_utility']))
    best_utility.append(float(np.max([p['utility'] for p in points])))
    final_em.append(float(row['final_heldout_em_approx']))
    final_f1.append(float(row['final_heldout_f1_approx']))

    total_fw_i = int(float(row['cumulative_reward_forward_passes']))
    total_fw.append(total_fw_i)
    total_time_ms.append(_final_time_ms(train_rows))

    online_fw.append(total_fw_i - precompute_fw)

    y_time = _interp_curve(points, 'time_ms', x_time)
    y_total_fw = _interp_curve(points, 'total_fw', x_total_fw)
    y_online_fw = _interp_curve(points, 'online_fw', x_online_fw)

    curves_time.append(y_time)
    curves_total_fw.append(y_total_fw)
    curves_online_fw.append(y_online_fw)
    area_time = float(np.trapz(y_time, x_time))
    area_total = float(np.trapz(y_total_fw, x_total_fw))
    area_online = float(np.trapz(y_online_fw, x_online_fw))
    nauc_time.append(area_time / float(common_time_budget))
    nauc_total_fw.append(area_total / float(common_total_fw_budget))
    nauc_online_fw.append(area_online / float(common_online_fw_budget))
    best_utility_common_time.append(float(np.max(y_time)))
    best_utility_common_total_fw.append(float(np.max(y_total_fw)))
    best_utility_common_online_fw.append(float(np.max(y_online_fw)))
    utility_at_total_budget.append(_interp_at(points, 'total_fw', common_total_fw_budget))
    utility_at_online_budget.append(_interp_at(points, 'online_fw', common_online_fw_budget))
    if common_time_budget >= 30000.0:
      utility_at_30s.append(_interp_at(points, 'time_ms', 30000.0))
    if common_time_budget >= 60000.0:
      utility_at_60s.append(_interp_at(points, 'time_ms', 60000.0))
    if common_time_budget >= 120000.0:
      utility_at_120s.append(_interp_at(points, 'time_ms', 120000.0))

    sthr = _as_int_or_none(row['steps_to_threshold'])
    tthr = _as_float_or_none(row['time_ms_to_threshold'])
    if sthr is not None and tthr is not None:
      reached += 1
      steps_to_threshold.append(sthr)
      time_to_threshold_ms.append(tthr)
      fw_thr = _forward_passes_to_step(points, sthr)
      if fw_thr is not None:
        fw_to_threshold.append(fw_thr)
        online_fw_to_threshold.append(fw_thr - precompute_fw)

  n = len(seed_dirs)
  result = {
      'method': method,
      'n_seeds': n,
      'final_heldout_utility_mean_std': _fmt_mean_std(final_utility),
      'best_heldout_utility_mean_std': _fmt_mean_std(best_utility),
      'approx_em_mean_std': _fmt_mean_std(final_em),
      'approx_f1_mean_std': _fmt_mean_std(final_f1),
      'nauc_utility_vs_time_mean_std': _fmt_mean_std(nauc_time),
      'nauc_utility_vs_total_forwardpasses_mean_std': _fmt_mean_std(nauc_total_fw),
      'nauc_utility_vs_online_forwardpasses_mean_std': _fmt_mean_std(nauc_online_fw),
      'best_utility_within_fixed_time_budget_mean_std': _fmt_mean_std(best_utility_common_time),
      'best_utility_within_fixed_forwardpasses_budget_mean_std': _fmt_mean_std(best_utility_common_total_fw),
      'best_utility_within_fixed_onlinepasses_budget_mean_std': _fmt_mean_std(best_utility_common_online_fw),
      'utility_at_total_passes_budget_mean_std': _fmt_mean_std(utility_at_total_budget),
      'utility_at_online_passes_budget_mean_std': _fmt_mean_std(utility_at_online_budget),
      'total_forward_passes_mean_std': _fmt_mean_std(total_fw),
      'total_wallclock_ms_mean_std': _fmt_mean_std(total_time_ms),
      'online_forward_passes_mean_std': _fmt_mean_std(online_fw),
      'threshold_reached_count': '%d/%d' % (reached, n),
  }
  result['utility_at_30s_mean_std'] = _fmt_mean_std(utility_at_30s) if utility_at_30s else 'N/A'
  result['utility_at_60s_mean_std'] = _fmt_mean_std(utility_at_60s) if utility_at_60s else 'N/A'
  result['utility_at_120s_mean_std'] = _fmt_mean_std(utility_at_120s) if utility_at_120s else 'N/A'
  curve_payload = {
      'x_time': x_time,
      'x_total_fw': x_total_fw,
      'x_online_fw': x_online_fw,
      'curves_time': np.asarray(curves_time, dtype=np.float64),
      'curves_total_fw': np.asarray(curves_total_fw, dtype=np.float64),
      'curves_online_fw': np.asarray(curves_online_fw, dtype=np.float64),
  }

  if reached == n:
    result['steps_to_threshold'] = _fmt_mean_std(steps_to_threshold)
    result['time_ms_to_threshold'] = _fmt_mean_std(time_to_threshold_ms)
    result['forward_passes_to_threshold'] = _fmt_mean_std(fw_to_threshold)
    result['online_forward_passes_to_threshold'] = _fmt_mean_std(online_fw_to_threshold)
  else:
    result['steps_to_threshold'] = 'N/A (reached %d/%d)' % (reached, n)
    result['time_ms_to_threshold'] = 'N/A (reached %d/%d)' % (reached, n)
    if fw_to_threshold:
      result['forward_passes_to_threshold'] = 'N/A (reached %d/%d; reached-only %s)' % (
          reached, n, _fmt_mean_std(fw_to_threshold))
      result['online_forward_passes_to_threshold'] = 'N/A (reached %d/%d; reached-only %s)' % (
          reached, n, _fmt_mean_std(online_fw_to_threshold))
    else:
      result['forward_passes_to_threshold'] = 'N/A (reached %d/%d)' % (reached, n)
      result['online_forward_passes_to_threshold'] = 'N/A (reached %d/%d)' % (reached, n)
  return result, curve_payload


def _plot_mean_std(ax, x, curves, label):
  mean = np.mean(curves, axis=0)
  std = np.std(curves, axis=0)
  ax.plot(x, mean, label=label)
  ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def _save_curve_plot(path,
                     x_label,
                     x_pg,
                     curves_pg,
                     x_pl,
                     curves_pl,
                     cutoff_x,
                     cutoff_label,
                     y_label='Held-out utility'):
  fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))
  _plot_mean_std(ax, x_pg, curves_pg, 'PolicyGradient (true set utility)')
  _plot_mean_std(ax, x_pl, curves_pl, 'PL-Rank surrogate (DCG surrogate)')
  ax.axvline(cutoff_x, color='black', linestyle='--', linewidth=1.0, label=cutoff_label)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.legend()
  ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def _collect_common_budgets(seed_dirs):
  max_times = []
  max_total_fw = []
  max_online_fw = []
  for seed_dir in seed_dirs:
    summary_rows = _read_summary(os.path.join(seed_dir, 'summary_table.csv'))
    for method in ('policygradient', 'plrank_surrogate'):
      row = _get_row(summary_rows, method)
      train_rows = _read_train(os.path.join(seed_dir, 'train_set_utility_%s.csv' % method))
      precompute_fw = int(float(row['singleton_precompute_forward_passes']))
      points = _eval_points(train_rows, precompute_fw)
      max_times.append(max(p['time_ms'] for p in points))
      max_total_fw.append(max(p['total_fw'] for p in points))
      max_online_fw.append(max(p['online_fw'] for p in points))
  return float(np.min(max_times)), float(np.min(max_total_fw)), float(np.min(max_online_fw))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed_dirs', nargs='+', required=True,
                      help='Per-seed output dirs from set_utility_experiment.py')
  parser.add_argument('--output_csv', required=True,
                      help='Path to aggregated summary CSV.')
  parser.add_argument('--utility_vs_time_plot', default='runs/set_utility_experiment/utility_vs_time_allseeds.png')
  parser.add_argument('--utility_vs_forwardpasses_plot', default='runs/set_utility_experiment/utility_vs_forwardpasses_allseeds.png')
  parser.add_argument('--utility_vs_onlinepasses_plot', default='runs/set_utility_experiment/utility_vs_onlinepasses_allseeds.png')
  parser.add_argument('--notes_path', default='runs/set_utility_experiment/aggregate_notes.txt')
  args = parser.parse_args()

  common_time_budget, common_total_fw_budget, common_online_fw_budget = _collect_common_budgets(args.seed_dirs)
  pg_row, pg_curves = _aggregate_method(
      'policygradient', args.seed_dirs, common_time_budget, common_total_fw_budget, common_online_fw_budget)
  pl_row, pl_curves = _aggregate_method(
      'plrank_surrogate', args.seed_dirs, common_time_budget, common_total_fw_budget, common_online_fw_budget)
  rows = [pg_row, pl_row]
  for row in rows:
    row['common_time_budget_ms'] = '%.3f' % common_time_budget
    row['common_total_forward_pass_budget'] = '%.0f' % common_total_fw_budget
    row['common_online_forward_pass_budget'] = '%.0f' % common_online_fw_budget
  fieldnames = [
      'method',
      'n_seeds',
      'common_time_budget_ms',
      'common_total_forward_pass_budget',
      'common_online_forward_pass_budget',
      'final_heldout_utility_mean_std',
      'best_heldout_utility_mean_std',
      'approx_em_mean_std',
      'approx_f1_mean_std',
      'nauc_utility_vs_time_mean_std',
      'nauc_utility_vs_total_forwardpasses_mean_std',
      'nauc_utility_vs_online_forwardpasses_mean_std',
      'utility_at_30s_mean_std',
      'utility_at_60s_mean_std',
      'utility_at_120s_mean_std',
      'utility_at_total_passes_budget_mean_std',
      'utility_at_online_passes_budget_mean_std',
      'best_utility_within_fixed_time_budget_mean_std',
      'best_utility_within_fixed_forwardpasses_budget_mean_std',
      'best_utility_within_fixed_onlinepasses_budget_mean_std',
      'total_forward_passes_mean_std',
      'total_wallclock_ms_mean_std',
      'steps_to_threshold',
      'time_ms_to_threshold',
      'forward_passes_to_threshold',
      'online_forward_passes_mean_std',
      'online_forward_passes_to_threshold',
      'threshold_reached_count',
  ]

  out_dir = os.path.dirname(args.output_csv)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(args.output_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)

  for p in (args.utility_vs_time_plot, args.utility_vs_forwardpasses_plot, args.utility_vs_onlinepasses_plot):
    out_dir = os.path.dirname(p)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
  _save_curve_plot(
      args.utility_vs_time_plot,
      x_label='Cumulative wall-clock time (ms)',
      x_pg=pg_curves['x_time'],
      curves_pg=pg_curves['curves_time'],
      x_pl=pl_curves['x_time'],
      curves_pl=pl_curves['curves_time'],
      cutoff_x=common_time_budget,
      cutoff_label='Common cutoff T')
  _save_curve_plot(
      args.utility_vs_forwardpasses_plot,
      x_label='Cumulative total generator forward passes',
      x_pg=pg_curves['x_total_fw'],
      curves_pg=pg_curves['curves_total_fw'],
      x_pl=pl_curves['x_total_fw'],
      curves_pl=pl_curves['curves_total_fw'],
      cutoff_x=common_total_fw_budget,
      cutoff_label='Common cutoff P')
  _save_curve_plot(
      args.utility_vs_onlinepasses_plot,
      x_label='Cumulative online forward passes (excluding singleton precompute)',
      x_pg=pg_curves['x_online_fw'],
      curves_pg=pg_curves['curves_online_fw'],
      x_pl=pl_curves['x_online_fw'],
      curves_pl=pl_curves['curves_online_fw'],
      cutoff_x=common_online_fw_budget,
      cutoff_label='Common cutoff P_online')

  notes_dir = os.path.dirname(args.notes_path)
  if notes_dir:
    os.makedirs(notes_dir, exist_ok=True)
  with open(args.notes_path, 'w') as handle:
    handle.write('Common budgets used for nAUC and fixed-budget comparisons:\n')
    handle.write('- T (time cutoff, ms): %.3f\n' % common_time_budget)
    handle.write('- P_total (forward-pass cutoff): %.0f\n' % common_total_fw_budget)
    handle.write('- P_online (online forward-pass cutoff): %.0f\n\n' % common_online_fw_budget)
    handle.write('Total forward passes include training reward evaluations, baseline-empty evaluations,\n')
    handle.write('singleton precompute evaluations (surrogate), and held-out validation utility evaluations.\n')
    handle.write('Online forward passes exclude singleton precompute only:\n')
    handle.write('online_forward_passes = total_forward_passes - singleton_precompute_forward_passes\n')
    handle.write('nAUC definitions:\n')
    handle.write('- nAUC_time = (1/T) * integral_0^T utility(t) dt\n')
    handle.write('- nAUC_total = (1/P_total) * integral_0^P_total utility(p_total) dp_total\n')
    handle.write('- nAUC_online = (1/P_online) * integral_0^P_online utility(p_online) dp_online\n')

  print('Wrote aggregate summary to %s' % args.output_csv)
  print('Wrote %s' % args.utility_vs_time_plot)
  print('Wrote %s' % args.utility_vs_forwardpasses_plot)
  print('Wrote %s' % args.utility_vs_onlinepasses_plot)
  print('Wrote %s' % args.notes_path)
  for row in rows:
    print('%s headline compute: forward_passes=%s, wallclock_ms=%s' % (
        row['method'], row['total_forward_passes_mean_std'], row['total_wallclock_ms_mean_std']))


if __name__ == '__main__':
  main()
