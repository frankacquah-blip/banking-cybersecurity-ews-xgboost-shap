"""06 - Multi-seed sensitivity analysis (addresses M14/M18 flag: no seed-level variance
existed for the single held-out test-set comparison, only 5-fold CV variance).

Retrains baseline and custom_loss+SMOTE with the SAME tuned hyperparameters
(best_params.json / best_wfn.json, found once via the canonical Optuna run, seed=42)
but with the model's own random_state (and SMOTE's random_state) varied across
5 seeds. Train/test split itself is NOT re-drawn (that would conflate split variance
with model variance) -- only downstream randomness (tree construction column/row
subsampling, SMOTE's neighbour interpolation) varies. Evaluated once per seed on the
untouched, fixed test set at the pre-registered threshold=0.5.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

TARGET_COL = 'at_risk'
SDT_SCORE_COLS = ['sdt_autonomy_score', 'sdt_competence_score', 'sdt_relatedness_score']
SEEDS = [42, 0, 1, 7, 123]

train_df = pd.read_csv('data/train_processed.csv')
test_df = pd.read_csv('data/test_processed.csv')
X_train = train_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
y_train = train_df[TARGET_COL].reset_index(drop=True)
X_test = test_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
y_test = test_df[TARGET_COL].reset_index(drop=True)

with open('models/best_params.json') as f:
    best_params = json.load(f)
with open('models/best_wfn.json') as f:
    wfn_config = json.load(f)
w_fn = wfn_config['w_fn']
sdt_weight_strength = wfn_config['sdt_weight_strength']
print('Using fixed tuned hyperparameters (from the canonical seed=42 Optuna run):')
print(best_params, wfn_config)

def get_proba(model, X):
    return expit(model.predict(X, output_margin=True))

def evaluate(model, X, y, threshold=0.5):
    proba = get_proba(model, X)
    preds = (proba >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0),
        'f1': f1_score(y, preds, zero_division=0),
        'roc_auc': roc_auc_score(y, proba),
    }

def make_asymmetric_objective(y_for_weights, sdt_df_for_weights, w_fn, sdt_weight_strength):
    base_weight = np.where(y_for_weights.to_numpy() == 1, w_fn, 1.0)
    sdt_avg = sdt_df_for_weights[SDT_SCORE_COLS].mean(axis=1).to_numpy()
    vulnerability = (6 - sdt_avg) / 5.0
    sample_weight = base_weight * (1 + sdt_weight_strength * vulnerability)
    def objective(y_true, y_pred):
        p = expit(y_pred)
        grad = sample_weight * (p - y_true)
        hess = sample_weight * p * (1 - p)
        return grad, hess
    return objective

rows = []
for seed in SEEDS:
    # baseline: default log-loss objective, tuned architecture, varying random_state only
    baseline_model = xgb.XGBClassifier(objective='binary:logistic', random_state=seed, **best_params)
    baseline_model.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline_model, X_test, y_test)
    rows.append({'model': 'baseline', 'seed': seed, **baseline_metrics})

    # custom_loss + SMOTE: SMOTE re-applied with seed-varying random_state, then model fit with seed-varying random_state
    minority_count = y_train.value_counts().min()
    k_neighbors = max(1, min(5, minority_count - 1))
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
    y_train_smote = pd.Series(y_train_smote)
    smote_objective = make_asymmetric_objective(y_train_smote, X_train_smote, w_fn, sdt_weight_strength)
    custom_model = xgb.XGBClassifier(objective=smote_objective, random_state=seed, **best_params)
    custom_model.fit(X_train_smote, y_train_smote)
    custom_metrics = evaluate(custom_model, X_test, y_test)
    rows.append({'model': 'custom_loss_smote', 'seed': seed, **custom_metrics})

    print(f'seed={seed}  baseline F1={baseline_metrics["f1"]:.3f}  custom_loss_smote F1={custom_metrics["f1"]:.3f}')

results_df = pd.DataFrame(rows)
results_df.to_csv('outputs/multiseed_sensitivity.csv', index=False)

summary = results_df.groupby('model')[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].agg(['mean', 'std'])
print('\nSummary across', len(SEEDS), 'seeds (test set, threshold=0.5):')
print(summary)
summary.to_csv('outputs/multiseed_sensitivity_summary.csv')
print('\nSaved outputs/multiseed_sensitivity.csv and outputs/multiseed_sensitivity_summary.csv')
