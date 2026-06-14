# Banking Cybersecurity EWS — Dummy-Data Smoke Test

This package lets you smoke-test the modelling pipeline (XGBoost + SMOTE +
SHAP) end-to-end on **synthetic data** before HuSSREC ethics clearance and
real data collection are complete.

## Folder structure

```
.
├── requirements.txt
├── generate_dummy_data.py
├── data/
│   └── dummy_gmt_survey_data.csv      <- synthetic 28-item survey data (350 rows)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_modelling_xgboost.ipynb
│   ├── 03_smote_imbalance.ipynb
│   ├── 04_shap_explainability.ipynb
│   └── 05_evaluation.ipynb
├── models/                            <- created when notebooks run
└── outputs/                           <- created when notebooks run (plots, CSVs)
```

## Setup (Windows, VS Code, Python 3.8 venv)

1. Unzip this package into your project folder (e.g.
   `banking-cybersecurity-ews-xgboost-shap/`).

2. Open the folder in VS Code, open a terminal, and create/activate your
   Python 3.8 venv if you haven't already:

   ```bash
   py -3.8 -m venv venv
   .\venv\Scripts\Activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. In VS Code, open `notebooks/01_preprocessing.ipynb` and select the `venv`
   kernel (top-right kernel picker → Python Environments → your venv).

## Running the smoke test

Run the notebooks **in order**, top to bottom (Run All), from inside the
`notebooks/` folder context (or adjust paths if running from elsewhere —
all paths are relative to the project root, e.g. `data/...`, `models/...`,
`outputs/...`):

1. **01_preprocessing.ipynb**
   Loads `data/dummy_gmt_survey_data.csv`, imputes missing values, derives
   SDT composite scores, one-hot encodes categoricals, and writes
   `data/train_processed.csv` / `data/test_processed.csv`.

2. **02_modelling_xgboost.ipynb**
   Trains a baseline XGBoost model, runs a short Optuna hyperparameter
   search (15 trials by default — increase for real data), and saves
   `models/xgb_baseline.json`, `models/xgb_tuned.json`,
   `models/best_params.json`.

3. **03_smote_imbalance.ipynb**
   Applies SMOTE to the training set only, retrains XGBoost on the balanced
   data, compares against the baseline, and saves `models/xgb_smote.json`
   and `data/train_smote.csv`.

4. **04_shap_explainability.ipynb**
   Computes SHAP values for the SMOTE-balanced model and produces global
   (bar, beeswarm) and local (waterfall, dependence) explanation plots in
   `outputs/`.

5. **05_evaluation.ipynb**
   Produces ROC/PR curves, decision-threshold tuning, 5-fold cross-validated
   metrics, and a final model comparison table (`outputs/model_comparison.csv`).

Each notebook ends with a "Smoke test checklist" — use these to confirm each
stage passed before moving to the next notebook.

## Important notes

- **Run notebooks from the project root working directory** (or update the
  relative paths like `data/...`, `models/...`, `outputs/...` at the top of
  each notebook to match wherever you run them from). In VS Code, the
  working directory is usually the folder you opened — if it's the project
  root, no changes are needed.

- **This is synthetic data.** Class balance, feature correlations, and SHAP
  rankings are illustrative only (built with a synthetic "ability/motivation"
  latent factor driving engagement, SDT scores, and risk). They confirm the
  *pipeline runs correctly*, not real findings.

- **Notebook 02's Optuna search** currently evaluates trials on the held-out
  test set for speed during smoke testing. For the real-data run, switch this
  to cross-validated AUC on the training set only (see the markdown note in
  that notebook) to avoid test-set leakage.

- **Regenerating dummy data:** run `python generate_dummy_data.py` from the
  project root to regenerate `data/dummy_gmt_survey_data.csv` (e.g., with a
  different random seed or sample size via the constants at the top of the
  script).

## Once real data arrives (post-HuSSREC clearance)

Replace `data/dummy_gmt_survey_data.csv` with your real survey export, keeping
the **same column names and schema** (28 items: demographics, prior IT
exposure, onboarding engagement, SDT subscales, perceived difficulty,
self-reported performance, and the `at_risk` outcome label). Then re-run
notebooks 01-05 in order. Increase `N_TRIALS` in notebook 02 for a more
thorough hyperparameter search.
