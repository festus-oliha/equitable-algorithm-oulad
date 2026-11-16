# Equitable Algorithmic Recourse for At-Risk Students

Code for the MSc Research work **"Equitable Algorithmic Recourse for At-Risk Students: Fairness and Robustness of Counterfactual Explanations"** by festus O.

The pipeline follows the steps:

| Step | Chapter 3 section | Code |
| --- | --- | --- |
| Load OULAD, build week-4 and week-8 snapshots, split by presentation | 3.4-3.6 | `src/data_prep.py` |
| XGBoost risk model, Optuna tuning, F1 threshold, logistic regression reference, prediction fairness | 3.7 | `src/model.py` |
| Actionability constraints and MAD-scaled distance | 3.8, 3.9.1 | `src/constraints.py` |
| DiCE and Nearest Instance counterfactuals | 3.9.2-3.9.3 | `src/counterfactuals.py` |
| Coverage and quality metrics (RQ1, RQ2) | 3.10 | `src/metrics.py` |
| Burden and recourse fairness (RQ3) | 3.11 | `src/fairness.py`, `src/groups.py` |
| Robustness to retraining (RQ4) | 3.12 | `src/robustness.py` |
| Hypothesis tests H1-H4 | 3.13 | `src/stats_tests.py` |
| Figures | Chapter 4 | `src/figures.py` |
| Runs everything | - | `src/run_all.py` |

All settings are in `config.yaml`.

## Quick start

```bash
# 1. create and activate a virtual environment (Python 3.12 recommended)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. install the libraries
pip install -r requirements.txt

# 3. test the installation on fake data (about 1 minute)
python scripts/make_synthetic_oulad.py
python -m src.run_all --data-dir data/synthetic --quick

# 4. put the 7 OULAD CSV files in data/raw/ and run the real experiment
python -m src.run_all
```

Results appear in `outputs/`:

- `results_summary.md`: all key tables and hypothesis results in one file
- `tables/`: every table as CSV (t01-t10, plus per-student detail)
- `figures/`: burden gap ratio, feature change frequency, robustness
- `models/`: the trained XGBoost models and their thresholds
- `run_log.txt`: a log of the run



## Data

Open University Learning Analytics Dataset (OULAD), CC BY 4.0:
https://analyse.kmi.open.ac.uk/open-dataset

Kuzilek, J., Hlosta, M. and Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.

The data files are not included in this repository (see `.gitignore`).
