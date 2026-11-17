# Equitable Algorithmic Recourse for At-Risk Students

Code for the MSc Research work **"Equitable Algorithmic Recourse for At-Risk Students: Fairness and Robustness of Counterfactual Explanations"** by festus O.

The pipeline follows the steps:

| Step | Code |
| --- | --- | --- |
| Load OULAD, build week-4 and week-8 snapshots, split by presentation |`src/data_prep.py` |
| XGBoost risk model, Optuna tuning, F1 threshold, logistic regression reference, prediction fairness |`src/model.py` |
| Actionability constraints and MAD-scaled distance | `src/constraints.py` |
| DiCE and Nearest Instance counterfactuals | `src/counterfactuals.py` |
| Coverage and quality metrics (RQ1/OBJ1, RQ2/OBJ2) | `src/metrics.py` |
| Burden and recourse fairness (RQ3/OBJ3) |  `src/fairness.py`, `src/groups.py` |
| Robustness to retraining (RQ4) | `src/robustness.py` |
| Hypothesis tests | `src/stats_tests.py` |
| Figures | `src/figures.py` |
| Runs everything | `src/run_all.py` |

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
