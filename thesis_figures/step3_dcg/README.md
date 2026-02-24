# Step 3 DCG Validation Artifacts

This folder contains final publication-ready artifacts for Step 3 (DCG validation).

## Files

- `variance_grad_mslr10k_allfolds.pdf`
- `variance_runtime_mslr10k_allfolds.pdf`
- `variance_mslr10k_aggregate.csv`

## Command Lines Used

Run all 5 folds (same hyperparameters/seed):

```bash
python variance_experiment.py runs/variance_mslr10k_fold1.csv --dataset MSLR-WEB10k --dataset_info_path local_dataset_info.txt --fold_id 1 --cutoff 5 --minibatch_queries 16 --replicates 200 --num_samples 1 --seed 42
python variance_experiment.py runs/variance_mslr10k_fold2.csv --dataset MSLR-WEB10k --dataset_info_path local_dataset_info.txt --fold_id 2 --cutoff 5 --minibatch_queries 16 --replicates 200 --num_samples 1 --seed 42
python variance_experiment.py runs/variance_mslr10k_fold3.csv --dataset MSLR-WEB10k --dataset_info_path local_dataset_info.txt --fold_id 3 --cutoff 5 --minibatch_queries 16 --replicates 200 --num_samples 1 --seed 42
python variance_experiment.py runs/variance_mslr10k_fold4.csv --dataset MSLR-WEB10k --dataset_info_path local_dataset_info.txt --fold_id 4 --cutoff 5 --minibatch_queries 16 --replicates 200 --num_samples 1 --seed 42
python variance_experiment.py runs/variance_mslr10k_fold5.csv --dataset MSLR-WEB10k --dataset_info_path local_dataset_info.txt --fold_id 5 --cutoff 5 --minibatch_queries 16 --replicates 200 --num_samples 1 --seed 42
```

Aggregate metrics + paired stats + publication figures:

```bash
python scripts/aggregate_variance_mslr10k.py \
  --input_csvs runs/variance_mslr10k_fold1.csv runs/variance_mslr10k_fold2.csv runs/variance_mslr10k_fold3.csv runs/variance_mslr10k_fold4.csv runs/variance_mslr10k_fold5.csv \
  --aggregate_csv runs/variance_mslr10k_aggregate.csv \
  --paired_csv runs/variance_mslr10k_paired_stats.csv \
  --grad_pdf runs/variance_grad_mslr10k_allfolds.pdf \
  --runtime_pdf runs/variance_runtime_mslr10k_allfolds.pdf
```

SGD-aligned convergence check:

```bash
python scripts/train_compare_dcg.py \
  --dataset MSLR-WEB10k \
  --dataset_info_path local_dataset_info.txt \
  --fold_id 1 \
  --cutoff 5 \
  --num_samples 1 \
  --batch_queries 1 \
  --max_steps 200 \
  --eval_every 10 \
  --seed 42 \
  --policy_csv runs/train_dcg_policygradient_sgd.csv \
  --plrank_csv runs/train_dcg_plrank_sgd.csv \
  --plot_steps runs/dcg_vs_steps_sgd.png \
  --plot_time runs/dcg_vs_time_sgd.png \
  --summary_path runs/train_dcg_threshold_summary_sgd.txt
```

## Dataset Version

- Dataset: `MSLR-WEB10k`
- Folds: 1-5 (variance), Fold 1 (SGD convergence check)
- Source layout: `data/MSLR-WEB10K/Fold{1..5}/`
- Metadata file: `local_dataset_info.txt` (`num_unique_feat=131`, `num_nonzero_feat=136`)

## Hardware / Environment

- OS: `macOS-26.3-arm64-arm-64bit`
- CPU: `arm`
- GPU: `none detected by TensorFlow` (`tf_gpus=[]`)
- TensorFlow: `2.12.0`

## Key Numbers

From `variance_mslr10k_aggregate.csv`:

- Mean variance ratio across folds (PG / PL-Rank-1): `2.926487 ± 0.208362`
- Mean median-runtime ratio across folds (PG / PL-Rank-1): `2.112619 ± 0.040034`
- Fraction PL-Rank-1 has lower grad norm: `0.981000 ± 0.011576`
- Fraction PL-Rank-1 is faster: `0.986000 ± 0.009695`

From `runs/train_dcg_threshold_summary_sgd.txt`:

- Threshold (95% best held-out DCG): `3.658454`
- PL-Rank-1: `160 steps / 2888.210 ms`
- PolicyGradient: `N/A / N/A` (did not reach threshold within 200 steps)
