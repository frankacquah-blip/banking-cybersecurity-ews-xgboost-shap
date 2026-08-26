"""05 - Evaluation (v3: threshold selected via out-of-fold CV predictions, not the test set;
saves fold-level CV results, per-instance predictions for McNemar's test, and timing)."""
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 42
TARGET_COL = 'at_risk'
SDT_SCORE_COLS = ['sdt_autonomy_score', 'sdt_competence_score', 'sdt_relatedness_score']
PRIMARY_MODEL_PATH = 'models/xgb_custom_loss_smote.json'
os.makedirs('outputs', exist_ok=True)

test_df = pd.read_csv('data/test_processed.csv')
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

def get_proba(model, X):
    return expit(model.predict(X, output_margin=True))

MODEL_CANDIDATES = [
    ('baseline', 'models/xgb_baseline.json'),
    ('custom_loss', 'models/xgb_custom_loss.json'),
    ('custom_loss_smote', 'models/xgb_custom_loss_smote.json'),
    ('custom_loss_pruned', 'models/xgb_custom_loss_pruned.json'),
]
models = {}
for name, path in MODEL_CANDIDATES:
    if os.path.exists(path):
        m = xgb.XGBClassifier(); m.load_model(path); models[name] = m
print('Loaded models:', list(models.keys()))

pruned_features = None
if os.path.exists('outputs/pruned_feature_list.json') and 'custom_loss_pruned' in models:
    with open('outputs/pruned_feature_list.json') as f:
        pruned_features = json.load(f)['kept']

def X_for_model(name):
    if name == 'custom_loss_pruned' and pruned_features is not None:
        return X_test[pruned_features]
    return X_test

# --- ROC / PR curves ---
plt.figure(figsize=(6, 5))
for name, m in models.items():
    proba = get_proba(m, X_for_model(name))
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend()
plt.tight_layout(); plt.savefig('outputs/roc_curve.png', dpi=150, bbox_inches='tight'); plt.close()

plt.figure(figsize=(6, 5))
for name, m in models.items():
    proba = get_proba(m, X_for_model(name))
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.plot(rec, prec, label=name)
baseline_rate = y_test.mean()
plt.axhline(baseline_rate, linestyle='--', color='gray', label=f'Prevalence baseline ({baseline_rate:.2f})')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend()
plt.tight_layout(); plt.savefig('outputs/pr_curve.png', dpi=150, bbox_inches='tight'); plt.close()

# --- FIX 3: decision threshold selected via pooled out-of-fold CV predictions (training data only),
# never by looking at the test set. Uses the same 5-fold CV loop as the CV-metrics section below,
# reusing its out-of-fold probabilities.
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

primary_model_name = 'custom_loss_pruned' if 'custom_loss_pruned' in models else (
    'custom_loss_smote' if 'custom_loss_smote' in models else list(models.keys())[0])

cv_fold_results = []
oof_true, oof_proba = [], []
timing = {}

if os.path.exists('data/train_smote.csv') and os.path.exists('models/best_params.json') and os.path.exists('models/best_wfn.json'):
    train_smote_df = pd.read_csv('data/train_smote.csv')
    X_train_smote = train_smote_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
    y_train_smote = train_smote_df[TARGET_COL].reset_index(drop=True)
    with open('models/best_params.json') as f: best_params = json.load(f)
    with open('models/best_wfn.json') as f: wfn_config = json.load(f)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = {m: [] for m in ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']}
    fold_num = 0
    t_cv0 = time.time()
    for train_idx, val_idx in cv.split(X_train_smote, y_train_smote):
        fold_num += 1
        X_fold_train = X_train_smote.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y_train_smote.iloc[train_idx].reset_index(drop=True)
        X_fold_val = X_train_smote.iloc[val_idx].reset_index(drop=True)
        y_fold_val = y_train_smote.iloc[val_idx].reset_index(drop=True)

        fold_objective = make_asymmetric_objective(y_fold_train, X_fold_train, wfn_config['w_fn'], wfn_config['sdt_weight_strength'])
        fold_model = xgb.XGBClassifier(objective=fold_objective, random_state=RANDOM_SEED, **best_params)
        fold_model.fit(X_fold_train, y_fold_train)

        fold_proba = get_proba(fold_model, X_fold_val)
        fold_preds = (fold_proba >= 0.5).astype(int)

        oof_true.extend(y_fold_val.tolist())
        oof_proba.extend(fold_proba.tolist())

        fold_row = {
            'fold': fold_num,
            'roc_auc': roc_auc_score(y_fold_val, fold_proba),
            'f1': f1_score(y_fold_val, fold_preds, zero_division=0),
            'precision': precision_score(y_fold_val, fold_preds, zero_division=0),
            'recall': recall_score(y_fold_val, fold_preds, zero_division=0),
            'accuracy': accuracy_score(y_fold_val, fold_preds),
        }
        cv_fold_results.append(fold_row)
        for k in cv_scores: cv_scores[k].append(fold_row[k])

    timing['cv_5fold_wallclock_seconds'] = time.time() - t_cv0
    for metric, scores in cv_scores.items():
        scores = np.array(scores)
        print(f'{metric:>10}: mean={scores.mean():.4f} | std={scores.std():.4f} | folds={np.round(scores, 3)}')

    cv_fold_df = pd.DataFrame(cv_fold_results)
    cv_fold_df.to_csv('outputs/cv_fold_results.csv', index=False)
    cv_summary_df = cv_fold_df.drop(columns=['fold']).agg(['mean', 'std']).reset_index().rename(columns={'index': 'stat'})
    cv_summary_df.to_csv('outputs/cv_summary.csv', index=False)
    print('Saved outputs/cv_fold_results.csv and outputs/cv_summary.csv')

    # Threshold selection on pooled out-of-fold predictions (training data only, never the test set)
    oof_true = np.array(oof_true); oof_proba = np.array(oof_proba)
    thresholds = np.arange(0.1, 0.91, 0.05)
    oof_rows = []
    for t in thresholds:
        preds = (oof_proba >= t).astype(int)
        oof_rows.append({
            'threshold': round(t, 2),
            'precision': precision_score(oof_true, preds, zero_division=0),
            'recall': recall_score(oof_true, preds, zero_division=0),
            'f1': f1_score(oof_true, preds, zero_division=0),
            'accuracy': accuracy_score(oof_true, preds),
        })
    oof_threshold_df = pd.DataFrame(oof_rows)
    oof_threshold_df.to_csv('outputs/threshold_tuning_oof_train.csv', index=False)
    selected_threshold = float(oof_threshold_df.loc[oof_threshold_df['f1'].idxmax(), 'threshold'])
    print(f"Threshold selected from pooled out-of-fold CV predictions (training data only): {selected_threshold}")
else:
    selected_threshold = 0.5
    print('CV inputs not found - falling back to default threshold 0.5')

# Report the SELECTED threshold's performance on the test set (selection happened on training-only data above)
proba_primary_test = get_proba(models[primary_model_name], X_for_model(primary_model_name))
preds_at_selected = (proba_primary_test >= selected_threshold).astype(int)
print(f"\n'{primary_model_name}' at CV-selected threshold={selected_threshold} on TEST set: "
      f"F1={f1_score(y_test, preds_at_selected, zero_division=0):.3f}, "
      f"precision={precision_score(y_test, preds_at_selected, zero_division=0):.3f}, "
      f"recall={recall_score(y_test, preds_at_selected, zero_division=0):.3f}")

# Also still report the plain 0.5-threshold sweep on the test set, but labelled explicitly as
# descriptive/exploratory (matches the pre-fix threshold_tuning.csv), not as the selection method.
rows = []
proba = proba_primary_test
for t in np.arange(0.1, 0.91, 0.05):
    preds = (proba >= t).astype(int)
    rows.append({
        'threshold': round(t, 2),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'accuracy': accuracy_score(y_test, preds),
    })
pd.DataFrame(rows).to_csv('outputs/threshold_tuning_test_descriptive_only.csv', index=False)

# --- Final model comparison table (threshold=0.5, the pre-registered primary threshold) ---
comparison_rows = []
predictions_export = {'y_true': y_test.tolist()}
for name, m in models.items():
    X_eval = X_for_model(name)
    proba = get_proba(m, X_eval)
    preds = (proba >= 0.5).astype(int)
    comparison_rows.append({
        'model': name, 'accuracy': accuracy_score(y_test, preds), 'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0), 'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, proba),
    })
    predictions_export[f'{name}_proba'] = proba.tolist()
    predictions_export[f'{name}_pred'] = preds.tolist()

comparison_df = pd.DataFrame(comparison_rows).set_index('model')
comparison_df.to_csv('outputs/model_comparison.csv')
print(); print(comparison_df)

# Per-instance predictions -- for McNemar's test (Methods M16), no retraining needed to use this
predictions_df = pd.DataFrame(predictions_export)
predictions_df.to_csv('outputs/test_predictions_per_instance.csv', index=False)
print('Saved outputs/test_predictions_per_instance.csv (per-instance predictions for McNemar\'s test)')

# Confusion matrices at threshold=0.5
confmat_rows = []
for name, m in models.items():
    X_eval = X_for_model(name)
    proba = get_proba(m, X_eval)
    preds = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    confmat_rows.append({'model': name, 'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn})
    print(f'--- {name}: confusion matrix (threshold=0.5) --- TP={tp} FN={fn} FP={fp} TN={tn}')
pd.DataFrame(confmat_rows).to_csv('outputs/confusion_matrices.csv', index=False)

# Inference timing (per-instance, batch of the full test set)
t0 = time.time()
for _ in range(10):
    get_proba(models[primary_model_name], X_for_model(primary_model_name))
inference_seconds_per_batch84 = (time.time() - t0) / 10
timing['inference_seconds_per_84row_batch'] = inference_seconds_per_batch84
timing['inference_seconds_per_instance'] = inference_seconds_per_batch84 / len(y_test)
with open('outputs/timing.json', 'w') as f:
    json.dump(timing, f, indent=2)
print('Saved outputs/timing.json:', timing)
print('\nSelected threshold (from training-only OOF CV):', selected_threshold)
