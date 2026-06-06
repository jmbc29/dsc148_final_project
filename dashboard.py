"""
DSC 148 - LoL Project
Streamlit Dashboard: Live Win & Comeback Predictor
Run with: streamlit run dashboard.py

Notes:
- Self-contained: trains a Logistic Regression (our best model) on lol_2024.csv at
  startup, so there is no LightGBM dependency and nothing to pickle beforehand.
- Place lol_2024.csv in the same folder as this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -- Page Config ----------------------------------------------------------------
st.set_page_config(
    page_title="LoL Win Predictor",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Custom CSS -----------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1, h2, h3 { color: #C89B3C !important; }
    .stSlider > div { color: #c8ccd4; }
</style>
""", unsafe_allow_html=True)

# -- Feature order (must match build_feature_vector below) ----------------------
FEATURE_ORDER = ['gd10', 'xpd10', 'csd10', 'kd10',
                 'gd15', 'xpd15', 'csd15', 'kd15',
                 'gold_momentum', 'xp_momentum', 'kill_momentum',
                 'dragon_diff', 'baron_diff', 'herald_diff',
                 'tower_diff', 'plate_diff', 'first_obj_sum',
                 'gold_x_baron', 'gold_x_dragon',
                 'patch_num', 'league_wr']


def _patch_to_float(p):
    try:
        parts = str(p).split('.')
        return float(parts[0]) + float(parts[1]) / 100
    except Exception:
        return 14.0


# -- Train the model (Logistic Regression) on the dataset -----------------------
@st.cache_resource
def load_model():
    """Train Logistic Regression on lol_2024.csv using the same 21 features the
    sliders produce. Cached so it only runs once per session."""
    df = pd.read_csv('lol_2024.csv', low_memory=False)
    teams = df[(df['position'] == 'team') & (df['datacompleteness'] == 'complete')]
    blue = teams[teams['side'] == 'Blue'].copy()

    f = pd.DataFrame(index=blue.index)
    f['gd10'] = blue['golddiffat10']
    f['xpd10'] = blue['xpdiffat10']
    f['csd10'] = blue['csdiffat10']
    f['kd10'] = blue['killsat10'] - blue['deathsat10']
    f['gd15'] = blue['golddiffat15']
    f['xpd15'] = blue['xpdiffat15']
    f['csd15'] = blue['csdiffat15']
    f['kd15'] = blue['killsat15'] - blue['deathsat15']
    f['gold_momentum'] = f['gd15'] - f['gd10']
    f['xp_momentum'] = f['xpd15'] - f['xpd10']
    f['kill_momentum'] = f['kd15'] - f['kd10']
    # Objective features use the same single-team approximations as the sliders,
    # so training and live inputs are transformed identically.
    f['dragon_diff'] = blue['dragons'] - (3 - blue['dragons'])
    f['baron_diff'] = blue['barons']
    f['herald_diff'] = blue['heralds']
    f['tower_diff'] = blue['towers'] - 6
    f['plate_diff'] = blue['turretplates'] - 5
    f['first_obj_sum'] = (blue['firstblood'].fillna(0) + blue['firstdragon'].fillna(0) +
                          blue['firstherald'].fillna(0) + blue['firsttower'].fillna(0))
    f['gold_x_baron'] = f['gd15'] * f['baron_diff']
    f['gold_x_dragon'] = f['gd15'] * f['dragon_diff']
    f['patch_num'] = blue['patch'].apply(_patch_to_float)
    league_wr = blue.groupby('league')['result'].mean()
    f['league_wr'] = blue['league'].map(league_wr).fillna(0.5)
    f['result'] = blue['result'].astype(int)

    f = f.dropna(subset=FEATURE_ORDER + ['result'])
    X = f[FEATURE_ORDER].values
    y = f['result'].values.astype(int)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    model.fit(X, y)
    return model


try:
    model = load_model()
except FileNotFoundError:
    st.error("Could not find **lol_2024.csv**. Download it from "
             "https://oracleselixir.com/tools/downloads and place it in the same "
             "folder as this app, then refresh.")
    st.stop()


# -- Feature builder (must match training order) --------------------------------
def build_feature_vector(gd10, xpd10, csd10, kd10,
                         gd15, xpd15, csd15, kd15,
                         dragons, barons, heralds,
                         towers, plates,
                         firstblood, firstdragon, firstherald,
                         firstbaron, firsttower):
    gold_momentum = gd15 - gd10
    xp_momentum = xpd15 - xpd10
    kill_momentum = kd15 - kd10
    dragon_diff = dragons - (3 - dragons)
    baron_diff = barons
    herald_diff = heralds
    tower_diff = towers - 6
    plate_diff = plates - 5
    first_obj_sum = firstblood + firstdragon + firstherald + firsttower
    gold_x_baron = gd15 * baron_diff
    gold_x_dragon = gd15 * dragon_diff
    return np.array([[gd10, xpd10, csd10, kd10,
                      gd15, xpd15, csd15, kd15,
                      gold_momentum, xp_momentum, kill_momentum,
                      dragon_diff, baron_diff, herald_diff,
                      tower_diff, plate_diff, first_obj_sum,
                      gold_x_baron, gold_x_dragon,
                      patch_num, league_wr]])


# -- Sidebar Inputs -------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Game State Input")
st.sidebar.markdown("### 🕐 At 10 Minutes")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    gd10 = st.slider("Gold Diff", -4000, 4000, 0, 100, key='gd10',
                     help="Your team's gold advantage")
    xpd10 = st.slider("XP Diff", -3000, 3000, 0, 100, key='xpd10')
with col_s2:
    csd10 = st.slider("CS Diff", -30, 30, 0, 1, key='csd10')
    kd10 = st.slider("Kill Diff", -6, 6, 0, 1, key='kd10')

st.sidebar.markdown("### 🕐 At 15 Minutes")
col_s3, col_s4 = st.sidebar.columns(2)
with col_s3:
    gd15 = st.slider("Gold Diff", -5000, 5000, 0, 100, key='gd15')
    xpd15 = st.slider("XP Diff", -4000, 4000, 0, 100, key='xpd15')
with col_s4:
    csd15 = st.slider("CS Diff", -40, 40, 0, 1, key='csd15')
    kd15 = st.slider("Kill Diff", -10, 10, 0, 1, key='kd15')

st.sidebar.markdown("### 🏆 Objectives")
col_o1, col_o2 = st.sidebar.columns(2)
with col_o1:
    dragons = st.number_input("Your Dragons", 0, 5, 0)
    barons = st.number_input("Your Barons", 0, 4, 0)
    heralds = st.number_input("Your Heralds", 0, 2, 0)
with col_o2:
    towers = st.number_input("Your Towers", 0, 11, 3)
    plates = st.number_input("Your Plates", 0, 15, 5)

st.sidebar.markdown("### 🎯 First Objectives")
firstblood = int(st.sidebar.checkbox("First Blood", value=False))
firstdragon = int(st.sidebar.checkbox("First Dragon", value=False))
firstherald = int(st.sidebar.checkbox("First Herald", value=False))
firstbaron = int(st.sidebar.checkbox("First Baron", value=False))
firsttower = int(st.sidebar.checkbox("First Tower", value=False))

st.sidebar.markdown("### 🌐 Context")
league_wr = st.sidebar.slider("League Avg Win Rate", 0.45, 0.55, 0.50, 0.01)
patch_num = st.sidebar.slider("Patch", 14.01, 14.20, 14.10, 0.01)

# -- Prediction -----------------------------------------------------------------
X_input = build_feature_vector(
    gd10, xpd10, csd10, kd10,
    gd15, xpd15, csd15, kd15,
    dragons, barons, heralds,
    towers, plates,
    firstblood, firstdragon, firstherald, firstbaron, firsttower
)
try:
    win_prob = model.predict_proba(X_input)[0, 1]
except Exception:
    win_prob = 0.5

# -- Main Dashboard -------------------------------------------------------------
st.title("⚔️  League of Legends Win Predictor")
st.markdown("*DSC 148 - Data Mining | UCSD*")
st.markdown("---")

col1, col2, col3 = st.columns([1, 1.5, 1])
with col2:
    st.markdown("### 🎯 Win Probability")
    prob_pct = win_prob * 100
    color = "#0BC4AA" if win_prob > 0.6 else ("#E84057" if win_prob < 0.4 else "#C89B3C")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Win Probability (%)", 'font': {'color': 'white', 'size': 16}},
        number={'font': {'color': color, 'size': 48}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': '#1a1d27',
            'bordercolor': '#3d4257',
            'steps': [
                {'range': [0, 40], 'color': '#2d1117'},
                {'range': [40, 60], 'color': '#1a1d27'},
                {'range': [60, 100], 'color': '#0d2017'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 2},
                          'thickness': 0.75, 'value': 50}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='#0f1117', font_color='white',
                            height=300, margin=dict(t=30, b=10, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    verdict = "🟢 Likely WIN" if win_prob > 0.6 else ("🔴 Likely LOSS" if win_prob < 0.4 else "🟡 Coin Flip")
    st.markdown(f"<h3 style='text-align:center; color:{color}'>{verdict}</h3>",
                unsafe_allow_html=True)

# -- Key Metrics Row ------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Game State Summary")
m1, m2, m3, m4, m5 = st.columns(5)
momentum = gd15 - gd10
m1.metric("Gold Diff @15m", f"{gd15:+,}", f"{momentum:+,} since 10m")
m2.metric("Kill Diff @15m", f"{kd15:+d}", f"{kd15-kd10:+d} since 10m")
m3.metric("Objectives", f"🐉{dragons} 🏰{barons}", f"{towers} towers")
m4.metric("Trajectory", "Snowballing" if momentum > 500 else ("Stable" if abs(momentum) < 500 else "Falling"),
          f"{momentum:+,} gold momentum")
m5.metric("First Objs", f"{firstblood+firstdragon+firstherald+firstbaron+firsttower}/5",
          "objectives secured")

# -- Trajectory + Radar ---------------------------------------------------------
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### 💰 Gold Trajectory")
    fig_gold = go.Figure()
    fig_gold.add_trace(go.Scatter(
        x=[0, 10, 15], y=[0, gd10, gd15], mode='lines+markers',
        line=dict(color='#C89B3C', width=3), marker=dict(size=10, color='#C89B3C'),
        fill='tozeroy', fillcolor='rgba(200,155,60,0.15)', name='Gold Lead'))
    fig_gold.add_hline(y=0, line_color='white', line_dash='dash', opacity=0.3)
    fig_gold.update_layout(paper_bgcolor='#0f1117', plot_bgcolor='#1a1d27',
                           font_color='white', height=280,
                           xaxis=dict(title='Game Time (min)', tickvals=[0, 10, 15]),
                           yaxis=dict(title='Gold Differential'),
                           margin=dict(t=20, b=40, l=50, r=20))
    st.plotly_chart(fig_gold, use_container_width=True)
with col_b:
    st.markdown("### 🎯 Objective Radar")
    categories = ['Gold Lead', 'XP Lead', 'Kill Lead', 'Objectives', 'Turret Control']
    values = [
        min(max((gd15 + 5000) / 10000, 0), 1),
        min(max((xpd15 + 4000) / 8000, 0), 1),
        min(max((kd15 + 10) / 20, 0), 1),
        (dragons + barons * 2 + heralds) / 15,
        min(towers / 11, 1),
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(91,138,245,0.3)',
        line=dict(color='#5B8AF5', width=2)))
    fig_radar.update_layout(
        polar=dict(bgcolor='#1a1d27',
                   radialaxis=dict(visible=True, range=[0, 1], color='white'),
                   angularaxis=dict(color='white')),
        paper_bgcolor='#0f1117', font_color='white',
        height=280, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    st.plotly_chart(fig_radar, use_container_width=True)

# -- Comeback Context -----------------------------------------------------------
if gd10 < -500:
    st.markdown("---")
    st.warning(f"⚠️ **Comeback Scenario**: You trail by {abs(gd10):,} gold at 10 minutes. "
               f"Historical win rate from this deficit is about 31%. "
               f"The model gives you **{win_prob*100:.1f}%** based on the current trajectory.")

# -- Model Info -----------------------------------------------------------------
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model:** Logistic Regression, trained live on the Oracle Elixir 2024 dataset
    (about 5,000 Blue-side professional matches).

    We use Logistic Regression here because it was the best model in our experiments,
    beating Naive Bayes, Random Forest, and LightGBM on both win prediction and
    comeback classification once the trajectory and interaction features were
    engineered by hand.

    **Features (21 total):** early game state at 10 minutes, mid game at 15 minutes,
    trajectory/momentum terms, objective control, interaction terms, and context
    (patch, league).

    **Task 1 win prediction (5-fold CV):** Logistic Regression AUC 0.785, ahead of
    LightGBM (0.771), Random Forest (0.770), and Naive Bayes (0.754).
    """)

st.markdown("---")
st.markdown("<center><small>DSC 148 Data Mining Project | UCSD | Oracle Elixir Data</small></center>",
            unsafe_allow_html=True)
