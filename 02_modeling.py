"""
DSC 148 – LOL Project
Notebook 2: Feature Engineering + Modeling
Models: Logistic Regression, Naive Bayes, Random Forest (baseline), LightGBM (proposed)
Task 1: Win Prediction (binary)
Task 2: Comeback Classification (losing at 10min → win at end)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('outputs', exist_ok=True)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report,
                             RocCurveDisplay)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3d4257', 'axes.labelcolor': '#c8ccd4',
    'xtick.color': '#c8ccd4', 'ytick.color': '#c8ccd4',
    'text.color': '#c8ccd4', 'grid.color': '#2d3145', 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12, 'axes.labelsize': 10,
})
GOLD = '#C89B3C'; BLUE = '#5B8AF5'; RED = '#E84057'
GREEN = '#0BC4AA'; PURPLE = '#9B59B6'; ORANGE = '#E67E22'

# ── Load Data ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('lol_2024.csv', low_memory=False)
df_raw['date'] = pd.to_datetime(df_raw['date'])
teams = df_raw[(df_raw['position'] == 'team') &
               (df_raw['datacompleteness'] == 'complete')].copy()
blue = teams[teams['side'] == 'Blue'].copy().reset_index(drop=True)

print("=" * 65)
print("FEATURE ENGINEERING + MODELING PIPELINE")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """All feature engineering in one place for reproducibility."""
    f = df.copy()

    # ── Group 1: Early Game (10min) ──────────────────────────────────────────
    f['gd10']  = f['golddiffat10']
    f['xpd10'] = f['xpdiffat10']
    f['csd10'] = f['csdiffat10']
    f['kd10']  = f['killsat10'] - f['deathsat10']

    # ── Group 2: Mid Game (15min) ────────────────────────────────────────────
    f['gd15']  = f['golddiffat15']
    f['xpd15'] = f['xpdiffat15']
    f['csd15'] = f['csdiffat15']
    f['kd15']  = f['killsat15'] - f['deathsat15']

    # ── Group 3: Trajectory Features (novel angle) ───────────────────────────
    # How fast is the lead growing or shrinking?
    f['gold_momentum'] = f['gd15'] - f['gd10']   # positive = snowballing
    f['xp_momentum']   = f['xpd15'] - f['xpd10']
    f['kill_momentum'] = f['kd15'] - f['kd10']

    # ── Group 4: Objective Control ───────────────────────────────────────────
    f['dragon_diff']  = f['dragons'] - f['opp_dragons']
    f['baron_diff']   = f['barons']  - f['opp_barons']
    f['herald_diff']  = f['heralds'] - f['opp_heralds']
    f['tower_diff']   = f['towers']  - f['opp_towers']
    f['plate_diff']   = f['turretplates'] - f['opp_turretplates']

    # ── Group 5: First Objectives (binary) ──────────────────────────────────
    f['first_obj_sum'] = (f['firstblood'].fillna(0) +
                          f['firstdragon'].fillna(0) +
                          f['firstherald'].fillna(0) +
                          f['firsttower'].fillna(0))

    # ── Group 6: Interaction Features ────────────────────────────────────────
    f['gold_x_baron'] = f['gd15'] * f['baron_diff']    # baron+lead is deadly
    f['gold_x_dragon'] = f['gd15'] * f['dragon_diff']

    # ── Group 7: Patch version (encode as float) ──────────────────────────────
    def patch_to_float(p):
        try:
            parts = str(p).split('.')
            return float(parts[0]) + float(parts[1]) / 100
        except Exception:
            return 14.0
    f['patch_num'] = f['patch'].apply(patch_to_float)

    # ── Group 8: League encoding (label encode by avg win rate) ──────────────
    league_wr = df.groupby('league')['result'].mean()
    f['league_wr'] = f['league'].map(league_wr).fillna(0.5)

    return f

blue_fe = engineer_features(blue)

# Feature groups for ablation study
FEATURE_GROUPS = {
    'Early (10m)':     ['gd10', 'xpd10', 'csd10', 'kd10'],
    'Mid (15m)':       ['gd15', 'xpd15', 'csd15', 'kd15'],
    'Trajectory':      ['gold_momentum', 'xp_momentum', 'kill_momentum'],
    'Objectives':      ['dragon_diff', 'baron_diff', 'herald_diff',
                        'tower_diff', 'plate_diff', 'first_obj_sum'],
    'Interactions':    ['gold_x_baron', 'gold_x_dragon'],
    'Context':         ['patch_num', 'league_wr'],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Drop rows with NaN in feature columns
blue_clean = blue_fe.dropna(subset=ALL_FEATURES + ['result']).copy()
X = blue_clean[ALL_FEATURES].values
y = blue_clean['result'].values.astype(int)

print(f"\nFeature set: {len(ALL_FEATURES)} features")
print(f"Samples:     {len(X):,}")
print(f"Class balance: {y.mean():.3f} (win rate)")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: WIN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("TASK 1: WIN PREDICTION (5-fold stratified CV)")
print("─" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    'Naive Bayes': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                       random_state=42, n_jobs=-1))
    ]),
    'LightGBM (Proposed)': lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1
    ),
}

results_task1 = {}
for name, model in models.items():
    accs, f1s, aucs = [], [], []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if isinstance(model, lgb.LGBMClassifier):
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        else:
            model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_prob))

    results_task1[name] = {
        'Accuracy': np.mean(accs),
        'F1':       np.mean(f1s),
        'AUC-ROC':  np.mean(aucs),
        'Acc_std':  np.std(accs),
        'F1_std':   np.std(f1s),
        'AUC_std':  np.std(aucs),
    }
    print(f"  {name:<25} Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
          f"F1={np.mean(f1s):.4f}  AUC={np.mean(aucs):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: COMEBACK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("TASK 2: COMEBACK PREDICTION (losing at 10m, can you win?)")
print("─" * 65)

# Filter to only games where blue team is losing at 10m
losing_mask = blue_clean['gd10'] < -500
blue_losing = blue_clean[losing_mask].copy()
X_cb = blue_losing[ALL_FEATURES].values
y_cb = blue_losing['result'].values.astype(int)

print(f"  Comeback dataset: {len(X_cb):,} games where blue trails at 10m")
print(f"  Comeback rate:    {y_cb.mean():.3f} ({y_cb.sum()} comebacks)")

results_task2 = {}
cv_cb = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Re-instantiate models fresh for task 2
models_t2 = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                   class_weight='balanced'))
    ]),
    'Naive Bayes': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                       class_weight='balanced',
                                       random_state=42, n_jobs=-1))
    ]),
    'LightGBM (Proposed)': lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1 - y_cb.mean()) / y_cb.mean(),
        random_state=42, verbose=-1
    ),
}

for name, model in models_t2.items():
    f1s, aucs, precs, recs = [], [], [], []
    for train_idx, val_idx in cv_cb.split(X_cb, y_cb):
        X_tr, X_val = X_cb[train_idx], X_cb[val_idx]
        y_tr, y_val = y_cb[train_idx], y_cb[val_idx]

        if isinstance(model, lgb.LGBMClassifier):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        else:
            model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_val, y_prob))
        precs.append(precision_score(y_val, y_pred, zero_division=0))
        recs.append(recall_score(y_val, y_pred, zero_division=0))

    results_task2[name] = {
        'F1':        np.mean(f1s),
        'AUC-ROC':   np.mean(aucs),
        'Precision': np.mean(precs),
        'Recall':    np.mean(recs),
        'F1_std':    np.std(f1s),
    }
    print(f"  {name:<25} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
          f"AUC={np.mean(aucs):.4f}  Prec={np.mean(precs):.4f}  Rec={np.mean(recs):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("ABLATION STUDY (LightGBM, removing one feature group at a time)")
print("─" * 65)

ablation_results = {}
baseline_features = ALL_FEATURES

# Full model baseline
lgb_full = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=31, random_state=42, verbose=-1)
full_aucs = cross_val_score(lgb_full, X, y, cv=cv, scoring='roc_auc')
ablation_results['Full Model'] = full_aucs.mean()
print(f"  {'Full Model':<30} AUC = {full_aucs.mean():.4f}")

for group_name, group_feats in FEATURE_GROUPS.items():
    remaining = [f for f in ALL_FEATURES if f not in group_feats]
    X_abl = blue_clean[remaining].values
    lgb_abl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, random_state=42, verbose=-1)
    abl_aucs = cross_val_score(lgb_abl, X_abl, y, cv=cv, scoring='roc_auc')
    drop = full_aucs.mean() - abl_aucs.mean()
    ablation_results[f'w/o {group_name}'] = abl_aucs.mean()
    print(f"  {'w/o ' + group_name:<30} AUC = {abl_aucs.mean():.4f}  "
          f"(drop = {drop:+.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (train full LightGBM on all data)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("FEATURE IMPORTANCE (full LightGBM)")
print("─" * 65)

lgb_final = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                num_leaves=31, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbose=-1)
lgb_final.fit(X, y)

feat_importance = pd.Series(
    lgb_final.feature_importances_,
    index=ALL_FEATURES
).sort_values(ascending=False)
print(feat_importance.head(10).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("PATCH SENSITIVITY ANALYSIS")
print("─" * 65)

# Train on early patches, test on late patches
blue_clean_sorted = blue_clean.sort_values('patch_num')
split_idx = int(len(blue_clean_sorted) * 0.7)
train_data = blue_clean_sorted.iloc[:split_idx]
test_data  = blue_clean_sorted.iloc[split_idx:]

X_tr_p = train_data[ALL_FEATURES].values
y_tr_p = train_data['result'].values.astype(int)
X_te_p = test_data[ALL_FEATURES].values
y_te_p = test_data['result'].values.astype(int)

patch_models = {
    'Logistic Regression': Pipeline([('s', StandardScaler()),
                                      ('c', LogisticRegression(max_iter=1000))]),
    'LightGBM': lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1),
}
for nm, pm in patch_models.items():
    pm.fit(X_tr_p, y_tr_p)
    auc = roc_auc_score(y_te_p, pm.predict_proba(X_te_p)[:, 1])
    acc = accuracy_score(y_te_p, pm.predict(X_te_p))
    print(f"  {nm:<25} AUC={auc:.4f}  Acc={acc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 6: Model Comparison ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Model Performance Comparison – Task 1: Win Prediction', fontsize=14, color='white')

model_names = list(results_task1.keys())
colors_m = [BLUE, GREEN, ORANGE, GOLD]
metrics = ['Accuracy', 'F1', 'AUC-ROC']

for ax, metric in zip(axes, metrics):
    vals = [results_task1[m][metric] for m in model_names]
    stds = [results_task1[m].get(f'{metric.split("-")[0]}_std', 0) for m in model_names]
    bars = ax.bar(range(len(model_names)), vals, color=colors_m, alpha=0.85,
                  yerr=stds, capsize=5, error_kw={'color': 'white', 'linewidth': 1.5})
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n.replace(' (Proposed)', '\n(Proposed)') for n in model_names],
                       fontsize=8, rotation=15)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    baseline = min(vals)
    ax.set_ylim(baseline - 0.03, 1.0)
    ax.axhline(0.5, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f'{val:.4f}', ha='center', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig6_model_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n✓ Figure 6 saved: Model Comparison")

# ── Figure 7: Comeback Task Results ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Model Performance – Task 2: Comeback Prediction', fontsize=14, color='white')

for ax, metric in zip(axes, ['F1', 'AUC-ROC']):
    vals = [results_task2[m][metric] for m in model_names]
    stds = [results_task2[m].get(f'{metric.split("-")[0]}_std', 0) for m in model_names]
    bars = ax.bar(range(len(model_names)), vals, color=colors_m, alpha=0.85,
                  yerr=stds, capsize=5, error_kw={'color': 'white', 'linewidth': 1.5})
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n.replace(' (Proposed)', '\n(Proposed)') for n in model_names],
                       fontsize=8, rotation=15)
    ax.set_title(f'Comeback: {metric}')
    ax.set_ylabel(metric)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f'{val:.4f}', ha='center', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig7_comeback_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 7 saved: Comeback Comparison")

# ── Figure 8: Ablation Study ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0f1117')

abl_names = list(ablation_results.keys())
abl_vals  = list(ablation_results.values())
full_val  = ablation_results['Full Model']
bar_colors = [GOLD if n == 'Full Model' else
              (RED if v < full_val - 0.002 else GREEN)
              for n, v in zip(abl_names, abl_vals)]

bars = ax.bar(range(len(abl_names)), abl_vals, color=bar_colors, alpha=0.85)
ax.axhline(full_val, color='white', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xticks(range(len(abl_names)))
ax.set_xticklabels(abl_names, rotation=25, fontsize=9)
ax.set_title('Ablation Study – Feature Group Contribution (AUC-ROC)', fontsize=13, color='white')
ax.set_ylabel('AUC-ROC')
baseline_abl = min(abl_vals)
ax.set_ylim(baseline_abl - 0.02, full_val + 0.01)
for bar, val in zip(bars, abl_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0005,
            f'{val:.4f}', ha='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('outputs/fig8_ablation.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 8 saved: Ablation Study")

# ── Figure 9: Feature Importance ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0f1117')

top_feats = feat_importance.head(15)
bars = ax.barh(top_feats.index[::-1], top_feats.values[::-1],
               color=PURPLE, alpha=0.85)
ax.set_title('LightGBM Feature Importance (Top 15)', fontsize=13, color='white')
ax.set_xlabel('Importance Score')
for bar, val in zip(bars, top_feats.values[::-1]):
    ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('outputs/fig9_feature_importance.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 9 saved: Feature Importance")

# ── Figure 10: Confusion Matrices ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Confusion Matrices (5-Fold CV, Win Prediction)', fontsize=13, color='white')

# Refit each model on full data for confusion matrix visualization
cm_models = {
    'Logistic Regression': Pipeline([('s', StandardScaler()),
                                      ('c', LogisticRegression(max_iter=1000))]),
    'Naive Bayes':          Pipeline([('s', StandardScaler()), ('c', GaussianNB())]),
    'Random Forest':        Pipeline([('s', StandardScaler()),
                                      ('c', RandomForestClassifier(n_estimators=200,
                                                                    random_state=42))]),
    'LightGBM (Proposed)':  lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1),
}

from sklearn.model_selection import cross_val_predict
for ax, (nm, cm_model) in zip(axes, cm_models.items()):
    y_pred_cv = cross_val_predict(cm_model, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred_cv)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False, linewidths=0.5)
    ax.set_title(nm.replace(' (Proposed)', '\n(Proposed)'), fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(['Loss', 'Win'])
    ax.set_yticklabels(['Loss', 'Win'])

plt.tight_layout()
plt.savefig('outputs/fig10_confusion_matrices.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 10 saved: Confusion Matrices")

# ── Save model results to CSV for report ──────────────────────────────────────
task1_df = pd.DataFrame(results_task1).T[['Accuracy', 'F1', 'AUC-ROC']]
task2_df = pd.DataFrame(results_task2).T[['F1', 'AUC-ROC', 'Precision', 'Recall']]
task1_df.to_csv('outputs/results_task1.csv')
task2_df.to_csv('outputs/results_task2.csv')

print("\n" + "=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)
print("\nTask 1 – Win Prediction:")
print(task1_df.round(4).to_string())
print("\nTask 2 – Comeback Prediction:")
print(task2_df.round(4).to_string())
print("\nAll outputs saved to outputs/")

# Save the trained LightGBM model and feature list for dashboard
import pickle
with open('lgb_model.pkl', 'wb') as f:
    pickle.dump({'model': lgb_final, 'features': ALL_FEATURES}, f)
print("✓ LightGBM model saved for dashboard")
