import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_csv', type=str, help='CSV created by variance_experiment.py')
  parser.add_argument('output_path', type=str, help='Output image path (e.g., variance_plot.png)')
  args = parser.parse_args()

  grad_norms = {}
  with open(args.input_csv, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      estimator = row['estimator']
      grad_norm = float(row['grad_norm'])
      grad_norms.setdefault(estimator, []).append(grad_norm)

  if not grad_norms:
    raise ValueError('No rows found in %s' % args.input_csv)

  plt.figure(figsize=(8, 4.5))
  bins = 30
  for estimator in sorted(grad_norms.keys()):
    values = np.asarray(grad_norms[estimator], dtype=np.float64)
    plt.hist(values, bins=bins, density=True, alpha=0.45, label=estimator)

  plt.xlabel('Gradient norm (L2)')
  plt.ylabel('Density')
  plt.title('Gradient Norm Distribution by Estimator')
  plt.legend()
  plt.tight_layout()
  plt.savefig(args.output_path, dpi=160)
  print('Wrote plot to %s' % args.output_path)


if __name__ == '__main__':
  main()
