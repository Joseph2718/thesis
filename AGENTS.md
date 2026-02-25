# AGENTS.md

## Cursor Cloud specific instructions

This is a Python research codebase for Learning-to-Rank (LTR) with Plackett-Luce ranking models (SIGIR 2021). It is **not** a web application — it is a CLI experiment runner with two entry points: `run.py` (relevance/DCG optimization) and `fairrun.py` (fairness/disparity optimization).

### Dependencies

- Python 3, NumPy (`<2.0` required — code uses `np.NINF` removed in NumPy 2.0), TensorFlow
- No `requirements.txt` exists; install with: `pip install "numpy<2.0" tensorflow`

### Running experiments

The code requires LETOR-format datasets. See `README.md` for full usage. Before running:

1. Copy `example_datasets_info.txt` to `local_dataset_info.txt` and set dataset paths
2. Create output dir: `mkdir -p local_output`

Example commands are in `README.md`. The dataset name passed via `--dataset` must match one of the hardcoded names in `run.py`/`fairrun.py` (`Webscope_C14_Set1`, `MSLR-WEB10k`, `MSLR-WEB30k`, `istella`) since epoch counts are configured per dataset name.

### Key caveats

- **NumPy version**: Must be `<2.0`. The codebase uses `np.NINF` which was removed in NumPy 2.0.
- **No tests or linting**: The repository has no test suite, no linting configuration, and no build system.
- **No GPU required**: TensorFlow runs on CPU. CUDA warnings in stderr are harmless.
- **Dataset names are hardcoded**: Using an unrecognized dataset name will cause a `NameError` for `n_epochs`. Use one of the four recognized dataset names in your `local_dataset_info.txt`.
- **Synthetic data for testing**: To test without proprietary datasets, generate synthetic LETOR files (`train.txt`, `vali.txt`, `test.txt`) with the format `<label> qid:<id> <feat_id>:<value> ...` and configure `local_dataset_info.txt` accordingly.
