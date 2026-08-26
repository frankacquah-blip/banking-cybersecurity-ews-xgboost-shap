"""03 - SMOTE (unchanged logic; now inherits SDT-weighted objective + matched-hyperparameter baseline from 02)"""
import json, os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

RANDOM_SEED = 42
TARGET_COL = 'at_risk'
SDT_SCORE_COLS = ['sdt_autonomy_score', 'sdt_competence_score', 'sdt_relatedness_score']

train_df = pd.read_csv('data/train_processed.csv')
test_df = pd.read_csv('data/test_processed.csv')
X_train = train_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
y_train = train_df[TARGET_COL].reset_index(drop=True)
X_test = test_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
y_test = test_df[TARGET_COL].reset_index(drop=True)

print('Before SMOTE - class distribution:'); print(y_train.value_counts())

minority_count = y_train.value_counts().min()
smote_applicable = minority_count >= 2
if smote_applicable:
    k_neighbors = max(1, min(5, minority_count - 1))
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f'SMOTE applied with k_neighbors={k_neighbors}')
    print('After SMOTE - class distribution:'); print(pd.Series(y_train_smote).value_counts())
else:
    X_train_smote, y_train_smote = X_train.copy(), y_train.copy()

def get_proba(model, X):
    return expit(model.predict(X, output_margin=True))

def evaluate_model(model, X, y, label='model', threshold=0.5):
    proba = get_proba(model, X)
    preds = (proba >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y, preds), 'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0), 'f1': f1_score(y, preds, zero_division=0),
        'roc_auc': roc_auc_score(y, proba),
    }
    print(f'--- {label} ---')
    for k, v in metrics.items():
        print(f'{k:>10}: {v:.4f}')
    return metrics

def make_asymmetric_objective(y_for_weights, sdt_df_for_weights, w_fn, sdt_weight_strength=0.0):
    base_weight = np.where(y_for_weights.to_numpy() == 1, w_fn, 1.0)
    if sdt_weight_strength and len(sdt_df_for_weights) > 0:
        sdt_avg = sdt_df_for_weights[SDT_SCORE_COLS].mean(axis=1).to_numpy()
        vulnerability = (6 - sdt_avg) / 5.0
        sample_weight = base_weight * (1 + sdt_weight_strength * vulnerability)
    else:
        sample_weight = base_weight
    def objective(y_true, y_pred):
        p = expit(y_pred)
        grad = sample_weight * (p - y_true)
        hess = sample_weight * p * (1 - p)
        return grad, hess
    return objective

with open('models/best_params.json') as f:
    best_params = json.load(f)
with open('models/best_wfn.json') as f:
    wfn_config = json.load(f)
print('Loaded from notebook 02:', best_params, wfn_config)

smote_objective = make_asymmetric_objective(y_train_smote, X_train_smote, wfn_config['w_fn'], wfn_config['sdt_weight_strength'])
smote_model = xgb.XGBClassifier(objective=smote_objective, random_state=RANDOM_SEED, **best_params)
smote_model.fit(X_train_smote, y_train_smote)
smote_metrics = evaluate_model(smote_model, X_test, y_test, 'Custom-loss + SMOTE (test set)')

comparison_rows = []
if os.path.exists('models/xgb_baseline.json'):
    baseline_model = xgb.XGBClassifier(); baseline_model.load_model('models/xgb_baseline.json')
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, 'Baseline (reloaded, no SMOTE)')
    comparison_rows.append({'model': 'baseline (default objective)', **baseline_metrics})
if os.path.exists('models/xgb_custom_loss.json'):
    custom_loss_model = xgb.XGBClassifier(); custom_loss_model.load_model('models/xgb_custom_loss.json')
    custom_loss_metrics = evaluate_model(custom_loss_model, X_test, y_test, 'Custom loss (reloaded, no SMOTE)')
    comparison_rows.append({'model': 'custom asymmetric loss (no SMOTE)', **custom_loss_metrics})
comparison_rows.append({'model': 'custom asymmetric loss + SMOTE', **smote_metrics})
comparison_df = pd.DataFrame(comparison_rows).set_index('model')
print(); print(comparison_df)

smote_model.save_model('models/xgb_custom_loss_smote.json')
train_smote_df = pd.DataFrame(X_train_smote, columns=X_train.columns)
train_smote_df[TARGET_COL] = y_train_smote
train_smote_df.to_csv('data/train_smote.csv', index=False)
print('Saved models/xgb_custom_loss_smote.json and data/train_smote.csv')
