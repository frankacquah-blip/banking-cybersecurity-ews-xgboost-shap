"""02 - Modelling (v3: SDT-weighting ON, baseline retrained with matched (tuned) hyperparameters for a clean single-variable ablation)"""
import json, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

RANDOM_SEED = 42
TARGET_COL = 'at_risk'
os.makedirs('models', exist_ok=True)

# FIX 1: SDT-weighting turned ON (was False) -- Optuna now also searches sdt_weight_strength
USE_SDT_WEIGHTING = True
SDT_SCORE_COLS = ['sdt_autonomy_score', 'sdt_competence_score', 'sdt_relatedness_score']

train_df = pd.read_csv('data/train_processed.csv')
test_df = pd.read_csv('data/test_processed.csv')
X_train = train_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
y_train = train_df[TARGET_COL].reset_index(drop=True)
X_test = test_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
y_test = test_df[TARGET_COL].reset_index(drop=True)
print('X_train:', X_train.shape, '| X_test:', X_test.shape)

def get_proba(model, X):
    margin = model.predict(X, output_margin=True)
    return expit(margin)

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

# --- Section 5: CV tuning of hyperparameters + w_fn (+ sdt_weight_strength now that USE_SDT_WEIGHTING=True) ---
N_TRIALS = 15
N_FOLDS = 5

def cv_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
    }
    w_fn = trial.suggest_float('w_fn', 1.0, 5.0)
    sdt_weight_strength = trial.suggest_float('sdt_weight_strength', 0.0, 1.0) if USE_SDT_WEIGHTING else 0.0

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_f1_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
        X_fold_val = X_train.iloc[val_idx].reset_index(drop=True)
        y_fold_val = y_train.iloc[val_idx].reset_index(drop=True)
        fold_objective = make_asymmetric_objective(y_fold_train, X_fold_train, w_fn, sdt_weight_strength)
        fold_model = xgb.XGBClassifier(objective=fold_objective, random_state=RANDOM_SEED, **params)
        fold_model.fit(X_fold_train, y_fold_train)
        fold_proba = get_proba(fold_model, X_fold_val)
        fold_preds = (fold_proba >= 0.5).astype(int)
        fold_f1_scores.append(f1_score(y_fold_val, fold_preds, zero_division=0))
    return float(np.mean(fold_f1_scores))

t0 = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(cv_objective, n_trials=N_TRIALS, show_progress_bar=False)
tuning_seconds = time.time() - t0
print('Best mean CV F1:', study.best_value)
print('Best params:', study.best_params)
print(f'Optuna tuning wall-clock time: {tuning_seconds:.1f}s for {N_TRIALS} trials x {N_FOLDS} folds')

best_params_raw = dict(study.best_params)
best_w_fn = best_params_raw.pop('w_fn')
best_sdt_weight_strength = best_params_raw.pop('sdt_weight_strength', 0.0)

# --- Final custom-loss model on the full training set ---
t0 = time.time()
final_objective = make_asymmetric_objective(y_train, X_train, best_w_fn, best_sdt_weight_strength)
custom_loss_model = xgb.XGBClassifier(objective=final_objective, random_state=RANDOM_SEED, **best_params_raw)
custom_loss_model.fit(X_train, y_train)
custom_loss_train_seconds = time.time() - t0
custom_loss_metrics = evaluate_model(custom_loss_model, X_test, y_test, 'Custom asymmetric-loss XGBoost (test set)')

# FIX 2: baseline retrained with the SAME tuned architecture (best_params_raw) so the ablation isolates
# only the objective function (default log-loss vs custom asymmetric/SDT loss), not architecture+tuning together.
t0 = time.time()
baseline_model = xgb.XGBClassifier(objective='binary:logistic', random_state=RANDOM_SEED, **best_params_raw)
baseline_model.fit(X_train, y_train)
baseline_train_seconds = time.time() - t0
baseline_metrics = evaluate_model(baseline_model, X_test, y_test, 'Baseline XGBoost (same tuned hyperparameters, test set)')

comparison_df = pd.DataFrame([
    {'model': 'baseline (default objective, tuning-matched)', **baseline_metrics},
    {'model': 'custom asymmetric+SDT loss', **custom_loss_metrics},
]).set_index('model')
print(); print(comparison_df)

baseline_model.save_model('models/xgb_baseline.json')
custom_loss_model.save_model('models/xgb_custom_loss.json')
with open('models/best_params.json', 'w') as f:
    json.dump(best_params_raw, f, indent=2)
with open('models/best_wfn.json', 'w') as f:
    json.dump({
        'w_fn': best_w_fn, 'sdt_weight_strength': best_sdt_weight_strength,
        'use_sdt_weighting': USE_SDT_WEIGHTING, 'sdt_score_cols': SDT_SCORE_COLS,
    }, f, indent=2)
with open('models/timing.json', 'w') as f:
    json.dump({
        'optuna_tuning_seconds_total': tuning_seconds, 'optuna_n_trials': N_TRIALS, 'optuna_n_folds': N_FOLDS,
        'custom_loss_final_fit_seconds': custom_loss_train_seconds,
        'baseline_final_fit_seconds': baseline_train_seconds,
    }, f, indent=2)
print('Saved models/xgb_baseline.json, models/xgb_custom_loss.json, models/best_params.json, models/best_wfn.json, models/timing.json')
print('\nFINAL w_fn:', best_w_fn, '| sdt_weight_strength:', best_sdt_weight_strength, '| USE_SDT_WEIGHTING:', USE_SDT_WEIGHTING)
