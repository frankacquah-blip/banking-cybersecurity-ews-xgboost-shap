"""04 - SHAP explainability + pruning (unchanged logic from v2, inherits SDT-weighted config)"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

TARGET_COL = 'at_risk'
MODEL_PATH = 'models/xgb_custom_loss_smote.json'
TRAIN_PATH = 'data/train_smote.csv'
PRUNE_BOTTOM_PCT = 0.15
os.makedirs('outputs', exist_ok=True)

model = xgb.XGBClassifier(); model.load_model(MODEL_PATH)
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv('data/test_processed.csv')
X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

def get_proba(m, X):
    return expit(m.predict(X, output_margin=True))

def evaluate_model(m, X, y, label='model', threshold=0.5):
    proba = get_proba(m, X)
    preds = (proba >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y, preds), 'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0), 'f1': f1_score(y, preds, zero_division=0),
        'roc_auc': roc_auc_score(y, proba),
    }
    print(f'--- {label} ---')
    for k, v in metrics.items(): print(f'{k:>10}: {v:.4f}')
    return metrics

def compute_shap_explanation(m, X):
    # shap.TreeExplainer can't parse xgboost 3.x's JSON base_score format ("[5E-1]").
    # Use xgboost's own native exact TreeSHAP (pred_contribs=True) instead, which is
    # the same algorithm shap.TreeExplainer would call internally, then wrap the
    # result in a shap.Explanation so the existing shap.plots.* calls work unchanged.
    booster = m.get_booster()
    dmat = xgb.DMatrix(X)
    contribs = booster.predict(dmat, pred_contribs=True)  # shape (n, n_features+1), last col = base value
    values = contribs[:, :-1]
    base_values = contribs[:, -1]
    return shap.Explanation(values=values, base_values=base_values, data=X.values, feature_names=list(X.columns))

shap_train = compute_shap_explanation(model, X_train)
print('SHAP values shape:', shap_train.values.shape)

plt.figure(); shap.plots.bar(shap_train, max_display=15, show=False); plt.tight_layout()
plt.savefig('outputs/shap_bar_global_importance.png', dpi=150, bbox_inches='tight'); plt.close()

plt.figure(); shap.plots.beeswarm(shap_train, max_display=15, show=False); plt.tight_layout()
plt.savefig('outputs/shap_beeswarm.png', dpi=150, bbox_inches='tight'); plt.close()

mean_abs_shap = pd.Series(np.abs(shap_train.values).mean(axis=0), index=X_train.columns).sort_values()
mean_abs_shap.to_csv('outputs/shap_feature_importance.csv', header=['mean_abs_shap'])
n_drop = max(1, int(len(mean_abs_shap) * PRUNE_BOTTOM_PCT))
dropped_features = mean_abs_shap.index[:n_drop].tolist()
kept_features = [c for c in X_train.columns if c not in dropped_features]
print(f'Dropping {n_drop} of {len(mean_abs_shap)} features:'); print(dropped_features)
with open('outputs/pruned_feature_list.json', 'w') as f:
    json.dump({'dropped': dropped_features, 'kept': kept_features}, f, indent=2)

SDT_SCORE_COLS = ['sdt_autonomy_score', 'sdt_competence_score', 'sdt_relatedness_score']
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

with open('models/best_params.json') as f: best_params = json.load(f)
with open('models/best_wfn.json') as f: wfn_config = json.load(f)

X_train_pruned = X_train[kept_features].reset_index(drop=True)
X_test_pruned = X_test[kept_features].reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

pruned_objective = make_asymmetric_objective(y_train_reset, X_train_pruned, wfn_config['w_fn'], wfn_config['sdt_weight_strength'])
pruned_model = xgb.XGBClassifier(objective=pruned_objective, random_state=42, **best_params)
pruned_model.fit(X_train_pruned, y_train_reset)

full_metrics = evaluate_model(model, X_test, y_test, 'Full-feature model (test set)')
pruned_metrics = evaluate_model(pruned_model, X_test_pruned, y_test, 'Pruned model (test set)')
pd.DataFrame([
    {'model': f'full ({X_train.shape[1]} features)', **full_metrics},
    {'model': f'pruned ({len(kept_features)} features)', **pruned_metrics},
]).set_index('model')

top_feature_name = mean_abs_shap.index[-1]
print('Top feature by mean |SHAP|:', top_feature_name)
plt.figure(); shap.plots.scatter(shap_train[:, top_feature_name], show=False); plt.tight_layout()
plt.savefig(f'outputs/shap_dependence_{top_feature_name}.png', dpi=150, bbox_inches='tight'); plt.close()

shap_test = compute_shap_explanation(model, X_test)
sample_idx = int(np.argmax((y_test.values == 1) & (get_proba(model, X_test) >= 0.5)))  # a correctly-identified at-risk case
print('Sample idx:', sample_idx, '| Actual label:', y_test.iloc[sample_idx], '| Predicted proba:', get_proba(model, X_test.iloc[[sample_idx]])[0])
plt.figure(); shap.plots.waterfall(shap_test[sample_idx], show=False); plt.tight_layout()
plt.savefig(f'outputs/shap_waterfall_sample{sample_idx}.png', dpi=150, bbox_inches='tight'); plt.close()

pruned_model.save_model('models/xgb_custom_loss_pruned.json')
print('Saved models/xgb_custom_loss_pruned.json')
