"""
DSC 148 – LOL Project
Notebook 1: Exploratory Data Analysis
Oracle Elixir 2024 Professional Match Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3d4257',
    'axes.labelcolor': '#c8ccd4',
    'xtick.color': '#c8ccd4',
    'ytick.color': '#c8ccd4',
    'text.color': '#c8ccd4',
    'grid.color': '#2d3145',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

GOLD   = '#C89B3C'
BLUE   = '#5B8AF5'
RED    = '#E84057'
GREEN  = '#0BC4AA'
PURPLE = '#9B59B6'

# ── Load & Filter ──────────────────────────────────────────────────────────────
df_raw = pd.read_csv('lol_2024.csv', low_memory=False)
df_raw['date'] = pd.to_datetime(df_raw['date'])

# Team-level rows only for most analysis
teams = df_raw[df_raw['position'] == 'team'].copy()

# Drop incomplete games
teams = teams[teams['datacompleteness'] == 'complete'].reset_index(drop=True)

# Compute game-level stats (blue team perspective)
blue = teams[teams['side'] == 'Blue'].copy()
red  = teams[teams['side'] == 'Red'].copy()

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total rows (all):        {len(df_raw):>10,}")
print(f"Team rows (complete):    {len(teams):>10,}")
print(f"Unique games:            {teams['gameid'].nunique():>10,}")
print(f"Leagues covered:         {teams['league'].nunique():>10,}")
print(f"Patches covered:         {teams['patch'].nunique():>10,}")
print(f"Date range:              {teams['date'].min().date()} → {teams['date'].max().date()}")
print(f"Blue win rate:           {blue['result'].mean():>10.3f}")

# ── Figure 1: Dataset Overview ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Oracle Elixir 2024 – Dataset Overview', fontsize=16, color='white', y=1.01)

# 1a: Games per league
ax = axes[0, 0]
league_counts = teams.groupby('league')['gameid'].nunique().sort_values(ascending=True)
bars = ax.barh(league_counts.index, league_counts.values, color=BLUE, alpha=0.85)
ax.set_title('Games per League')
ax.set_xlabel('Number of Games')
for bar, val in zip(bars, league_counts.values):
    ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontsize=9, color='white')

# 1b: Win rate by side
ax = axes[0, 1]
win_by_side = teams.groupby('side')['result'].mean()
colors = [BLUE, RED]
bars = ax.bar(win_by_side.index, win_by_side.values, color=colors, alpha=0.85, width=0.5)
ax.axhline(0.5, color='white', linestyle='--', alpha=0.4, linewidth=1)
ax.set_title('Win Rate by Side')
ax.set_ylabel('Win Rate')
ax.set_ylim(0, 0.7)
for bar, val in zip(bars, win_by_side.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f'{val:.3f}', ha='center', fontsize=11, color='white', fontweight='bold')

# 1c: Games over time (monthly)
ax = axes[0, 2]
monthly = teams.set_index('date').resample('ME')['gameid'].nunique()
ax.fill_between(monthly.index, monthly.values, alpha=0.3, color=GOLD)
ax.plot(monthly.index, monthly.values, color=GOLD, linewidth=2)
ax.set_title('Games per Month')
ax.set_xlabel('Month')
ax.set_ylabel('Games')
ax.tick_params(axis='x', rotation=30)

# 1d: Game length distribution
ax = axes[1, 0]
gl_min = teams['gamelength'] / 60
ax.hist(gl_min, bins=40, color=GREEN, alpha=0.8, edgecolor='none')
ax.axvline(gl_min.mean(), color='white', linestyle='--', linewidth=1.5,
           label=f'Mean: {gl_min.mean():.1f}m')
ax.set_title('Game Length Distribution')
ax.set_xlabel('Minutes')
ax.set_ylabel('Count')
ax.legend(fontsize=9)

# 1e: Gold diff at 15 distribution
ax = axes[1, 1]
ax.hist(blue['golddiffat15'], bins=50, color=PURPLE, alpha=0.8, edgecolor='none')
ax.axvline(0, color='white', linestyle='--', linewidth=1.5)
ax.set_title('Gold Diff at 15min (Blue Perspective)')
ax.set_xlabel('Gold Difference')
ax.set_ylabel('Count')

# 1f: Win rate by patch (top 8 patches)
ax = axes[1, 2]
patch_wr = (blue.groupby('patch')['result'].agg(['mean', 'count'])
              .query('count >= 30')
              .sort_values('patch'))
ax.bar(range(len(patch_wr)), patch_wr['mean'], color=BLUE, alpha=0.85)
ax.axhline(0.5, color='white', linestyle='--', alpha=0.4)
ax.set_xticks(range(len(patch_wr)))
ax.set_xticklabels(patch_wr.index, rotation=45, fontsize=8)
ax.set_title('Blue Win Rate by Patch')
ax.set_ylabel('Win Rate')
ax.set_ylim(0.4, 0.6)

plt.tight_layout()
plt.savefig('outputs/fig1_dataset_overview.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n✓ Figure 1 saved: Dataset Overview")

# ── Figure 2: Feature Correlations with Win ────────────────────────────────────
feature_cols = [
    'golddiffat10', 'xpdiffat10', 'csdiffat10',
    'golddiffat15', 'xpdiffat15', 'csdiffat15',
    'killsat10', 'deathsat10', 'killsat15', 'deathsat15',
    'firstblood', 'firstdragon', 'firstherald', 'firstbaron', 'firsttower',
    'dragons', 'barons', 'heralds', 'towers',
    'turretplates', 'gamelength',
]

correlations = blue[feature_cols + ['result']].corr()['result'].drop('result').sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0f1117')
colors_corr = [RED if v < 0 else GREEN for v in correlations.values]
bars = ax.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.85)
ax.axvline(0, color='white', linewidth=0.8)
ax.set_title('Feature Correlation with Win (Blue Team)', fontsize=14, color='white')
ax.set_xlabel('Pearson Correlation')
for bar, val in zip(bars, correlations.values):
    xpos = val + 0.005 if val >= 0 else val - 0.005
    ha = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha=ha, fontsize=8, color='white')

plt.tight_layout()
plt.savefig('outputs/fig2_correlations.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 2 saved: Feature Correlations")

# ── Figure 3: Comeback Analysis ────────────────────────────────────────────────
# Define "comeback" as: losing at 10min (golddiffat10 < -500) but winning
blue_valid = blue.dropna(subset=['golddiffat10', 'golddiffat15', 'result']).copy()
blue_valid['losing_at_10'] = blue_valid['golddiffat10'] < -500
blue_valid['losing_at_15'] = blue_valid['golddiffat15'] < -500
blue_valid['comeback_from_10'] = blue_valid['losing_at_10'] & (blue_valid['result'] == 1)
blue_valid['comeback_from_15'] = blue_valid['losing_at_15'] & (blue_valid['result'] == 1)

losing_at_10 = blue_valid[blue_valid['losing_at_10']]
losing_at_15 = blue_valid[blue_valid['losing_at_15']]

comeback_rate_10 = losing_at_10['result'].mean()
comeback_rate_15 = losing_at_15['result'].mean()
normal_wr = blue_valid[~blue_valid['losing_at_10']]['result'].mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Comeback Analysis – Can You Win From Behind?', fontsize=14, color='white')

# 3a: Win rate conditional on gold state at 10
ax = axes[0]
categories = ['Winning\nat 10m', 'Even\nat 10m', 'Losing\nat 10m']
win_rates = [
    blue_valid[blue_valid['golddiffat10'] > 500]['result'].mean(),
    blue_valid[blue_valid['golddiffat10'].between(-500, 500)]['result'].mean(),
    blue_valid[blue_valid['golddiffat10'] < -500]['result'].mean(),
]
colors_cb = [GREEN, GOLD, RED]
bars = ax.bar(categories, win_rates, color=colors_cb, alpha=0.85, width=0.5)
ax.axhline(0.5, color='white', linestyle='--', alpha=0.4)
ax.set_title('Win Rate by Gold State at 10m')
ax.set_ylabel('Win Rate')
ax.set_ylim(0, 1)
for bar, val in zip(bars, win_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
            f'{val:.3f}', ha='center', fontsize=12, color='white', fontweight='bold')

# 3b: Gold diff at 10 vs outcome (violin)
ax = axes[1]
winners_gd10 = blue_valid[blue_valid['result'] == 1]['golddiffat10']
losers_gd10  = blue_valid[blue_valid['result'] == 0]['golddiffat10']
vp = ax.violinplot([losers_gd10, winners_gd10], positions=[0, 1], showmedians=True)
for i, (body, color) in enumerate(zip(vp['bodies'], [RED, GREEN])):
    body.set_facecolor(color)
    body.set_alpha(0.7)
vp['cmedians'].set_color('white')
vp['cmins'].set_color('white')
vp['cmaxes'].set_color('white')
vp['cbars'].set_color('white')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Loss', 'Win'])
ax.set_title('Gold Diff at 10m by Outcome')
ax.set_ylabel('Gold Differential')
ax.axhline(0, color='white', linestyle='--', alpha=0.3)

# 3c: Comeback rate by deficit size
ax = axes[2]
bins = [-5000, -2000, -1500, -1000, -500]
labels = ['>2000', '1500-2000', '1000-1500', '500-1000']
comeback_rates = []
counts = []
for i in range(len(bins) - 1):
    mask = blue_valid['golddiffat10'].between(bins[i], bins[i+1])
    subset = blue_valid[mask]
    comeback_rates.append(subset['result'].mean() if len(subset) > 0 else 0)
    counts.append(len(subset))

bars = ax.bar(labels, comeback_rates, color=PURPLE, alpha=0.85)
ax.set_title('Comeback Rate by Deficit at 10m')
ax.set_ylabel('Win Rate (from deficit)')
ax.set_xlabel('Gold Deficit (Blue behind by)')
ax.set_ylim(0, 0.6)
ax.tick_params(axis='x', rotation=20)
for bar, val, cnt in zip(bars, comeback_rates, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f'{val:.3f}\n(n={cnt})', ha='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('outputs/fig3_comeback_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 3 saved: Comeback Analysis")

# ── Figure 4: Objective Control Analysis ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Objective Control & Win Correlation', fontsize=14, color='white')

# 4a: Dragons
ax = axes[0, 0]
for n_drag in range(5):
    mask = blue_valid['dragons'] == n_drag
    wr = blue_valid[mask]['result'].mean() if mask.sum() > 10 else np.nan
    cnt = mask.sum()
    if not np.isnan(wr):
        ax.bar(n_drag, wr, color=plt.cm.RdYlGn(wr), alpha=0.85)
        ax.text(n_drag, wr + 0.01, f'{wr:.2f}\n(n={cnt})',
                ha='center', fontsize=9, color='white')
ax.axhline(0.5, color='white', linestyle='--', alpha=0.4)
ax.set_title('Win Rate by Dragons Secured')
ax.set_xlabel('Dragons')
ax.set_ylabel('Win Rate')
ax.set_ylim(0, 1)

# 4b: Barons
ax = axes[0, 1]
for n_bar in range(4):
    mask = blue_valid['barons'] == n_bar
    wr = blue_valid[mask]['result'].mean() if mask.sum() > 10 else np.nan
    cnt = mask.sum()
    if not np.isnan(wr):
        ax.bar(n_bar, wr, color=plt.cm.RdYlGn(wr), alpha=0.85)
        ax.text(n_bar, wr + 0.01, f'{wr:.2f}\n(n={cnt})',
                ha='center', fontsize=9, color='white')
ax.axhline(0.5, color='white', linestyle='--', alpha=0.4)
ax.set_title('Win Rate by Barons Secured')
ax.set_xlabel('Barons')
ax.set_ylabel('Win Rate')
ax.set_ylim(0, 1)

# 4c: First objectives heatmap of win rates
ax = axes[1, 0]
obj_cols = ['firstblood', 'firstdragon', 'firstherald', 'firstbaron', 'firsttower']
obj_labels = ['First Blood', 'First Dragon', 'First Herald', 'First Baron', 'First Tower']
wr_with    = [blue_valid[blue_valid[c] == 1]['result'].mean() for c in obj_cols]
wr_without = [blue_valid[blue_valid[c] == 0]['result'].mean() for c in obj_cols]
x = np.arange(len(obj_cols))
ax.bar(x - 0.2, wr_with,    0.35, label='Secured', color=GREEN, alpha=0.85)
ax.bar(x + 0.2, wr_without, 0.35, label='Not Secured', color=RED, alpha=0.85)
ax.axhline(0.5, color='white', linestyle='--', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(obj_labels, rotation=20, fontsize=9)
ax.set_title('Win Rate: Secured vs. Not Secured First Objectives')
ax.set_ylabel('Win Rate')
ax.legend(fontsize=9)
ax.set_ylim(0, 0.85)

# 4d: Turret plates vs win
ax = axes[1, 1]
plate_bins = [0, 2, 4, 6, 8, 100]
plate_labels = ['0-2', '3-4', '5-6', '7-8', '9+']
plate_wr = []
for i in range(len(plate_bins) - 1):
    mask = blue_valid['turretplates'].between(plate_bins[i], plate_bins[i+1] - 1)
    plate_wr.append(blue_valid[mask]['result'].mean())
ax.bar(plate_labels, plate_wr, color=GOLD, alpha=0.85)
ax.axhline(0.5, color='white', linestyle='--', alpha=0.4)
ax.set_title('Win Rate by Turret Plates Taken')
ax.set_xlabel('Turret Plates')
ax.set_ylabel('Win Rate')
ax.set_ylim(0, 0.85)

plt.tight_layout()
plt.savefig('outputs/fig4_objectives.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 4 saved: Objective Analysis")

# ── Figure 5: League Comparison ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('League-Level Patterns', fontsize=14, color='white')

league_stats = (blue_valid.groupby('league')
                .agg(
                    avg_gamelength=('gamelength', 'mean'),
                    avg_gd15=('golddiffat15', 'mean'),
                    avg_gd15_std=('golddiffat15', 'std'),
                    blue_wr=('result', 'mean'),
                    n_games=('gameid', 'count')
                )
                .query('n_games >= 50')
                .sort_values('avg_gamelength'))

# 5a: Average game length by league
ax = axes[0]
bars = ax.barh(league_stats.index, league_stats['avg_gamelength'] / 60,
               color=BLUE, alpha=0.85)
ax.set_title('Average Game Length by League')
ax.set_xlabel('Minutes')
for bar, val in zip(bars, league_stats['avg_gamelength'] / 60):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}m', va='center', fontsize=9, color='white')

# 5b: Gold diff variance (game volatility)
ax = axes[1]
bars = ax.barh(league_stats.index, league_stats['avg_gd15_std'],
               color=PURPLE, alpha=0.85)
ax.set_title('Gold Diff Volatility at 15m (Std Dev)')
ax.set_xlabel('Standard Deviation of GD@15')
for bar, val in zip(bars, league_stats['avg_gd15_std']):
    ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}', va='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('outputs/fig5_leagues.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Figure 5 saved: League Comparison")

# ── Summary Statistics Table ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY EDA FINDINGS")
print("=" * 60)
print(f"Blue side win rate:       {blue['result'].mean():.3f}")
print(f"Comeback rate (down 500+ gold at 10m): {comeback_rate_10:.3f}")
print(f"Comeback rate (down 500+ gold at 15m): {comeback_rate_15:.3f}")
print(f"Avg game length:          {teams['gamelength'].mean()/60:.1f} min")
print(f"Strongest predictor:      {correlations.abs().idxmax()} (r={correlations.abs().max():.3f})")
print(f"\nTop 5 features by |correlation| with win:")
for feat, corr in correlations.abs().sort_values(ascending=False).head(5).items():
    print(f"  {feat:<25} r = {correlations[feat]:+.3f}")

print("\nAll EDA figures saved to outputs/")
