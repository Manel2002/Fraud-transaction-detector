import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import time

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════
# GLOBAL CSS — dark security theme
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}

.stApp {
    background: #080c14;
}

/* animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,170,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,170,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    animation: gridPulse 8s ease-in-out infinite;
}

@keyframes gridPulse {
    0%,100% { opacity: 0.6; }
    50%      { opacity: 1;   }
}

/* ── Header ── */
.hero-wrapper {
    position: relative;
    padding: 2.5rem 0 1.5rem;
    text-align: center;
    overflow: hidden;
}

.hero-glow {
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(0,255,170,0.12) 0%, transparent 70%);
    pointer-events: none;
    animation: glowPulse 4s ease-in-out infinite;
}

@keyframes glowPulse {
    0%,100% { opacity: 0.7; transform: translateX(-50%) scale(1);   }
    50%      { opacity: 1;   transform: translateX(-50%) scale(1.1); }
}

.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #00ffaa;
    border: 1px solid rgba(0,255,170,0.3);
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
    background: rgba(0,255,170,0.05);
    animation: fadeSlideDown 0.6s ease both;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 0.5rem;
    animation: fadeSlideDown 0.7s ease both;
}

.hero-title span {
    background: linear-gradient(135deg, #00ffaa 0%, #00b4d8 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    letter-spacing: 0.1em;
    animation: fadeSlideDown 0.8s ease both;
}

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0);     }
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin: 1.5rem 0;
    animation: fadeIn 1s ease 0.3s both;
}

.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.3s;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00ffaa, #00b4d8);
    opacity: 0;
    transition: opacity 0.3s;
}

.metric-card:hover {
    border-color: rgba(0,255,170,0.3);
    transform: translateY(-2px);
}

.metric-card:hover::before { opacity: 1; }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #00ffaa;
}

.metric-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 2px;
    font-family: 'Space Mono', monospace;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #64748b;
    border-radius: 8px;
    padding: 8px 20px;
    border: none;
    background: transparent;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: rgba(0,255,170,0.1) !important;
    color: #00ffaa !important;
    border: 1px solid rgba(0,255,170,0.25) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Scenario cards ── */
.scenario-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem;
    transition: all 0.3s;
    height: 100%;
    animation: fadeIn 0.6s ease both;
}

.scenario-card:hover {
    border-color: rgba(0,255,170,0.25);
    background: rgba(0,255,170,0.04);
}

.scenario-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 4px;
    color: #e2e8f0;
}

.scenario-note {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #64748b;
    line-height: 1.5;
    margin-bottom: 12px;
}

/* ── Risk badges ── */
.risk-high {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #f87171;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    font-family: 'Space Mono', monospace;
    animation: riskPulse 1.5s ease-in-out infinite;
}

.risk-medium {
    display: inline-block;
    background: rgba(251,146,60,0.15);
    border: 1px solid rgba(251,146,60,0.4);
    color: #fb923c;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    font-family: 'Space Mono', monospace;
}

.risk-low {
    display: inline-block;
    background: rgba(0,255,170,0.10);
    border: 1px solid rgba(0,255,170,0.3);
    color: #00ffaa;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    font-family: 'Space Mono', monospace;
}

@keyframes riskPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3);  }
    50%      { box-shadow: 0 0 12px 4px rgba(239,68,68,0.15); }
}

/* ── Gauge ── */
.gauge-wrapper {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    animation: fadeIn 0.5s ease;
}

.gauge-pct {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}

.gauge-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #64748b;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

/* ── Progress bar ── */
.prob-bar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin: 10px 0 4px;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}

.section-header h2 {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0;
    color: #e2e8f0;
}

.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00ffaa;
    animation: dotBlink 2s ease-in-out infinite;
}

@keyframes dotBlink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00ffaa20, #00b4d820);
    border: 1px solid rgba(0,255,170,0.35);
    color: #00ffaa;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.2s;
    letter-spacing: 0.04em;
}

.stButton > button:hover {
    background: rgba(0,255,170,0.15);
    border-color: rgba(0,255,170,0.6);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,255,170,0.15);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00ffaa, #00b4d8);
    color: #080c14;
    border: none;
    font-weight: 700;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 24px rgba(0,255,170,0.35);
    transform: translateY(-2px);
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    overflow: hidden;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Scanning animation ── */
.scan-line {
    position: relative;
    overflow: hidden;
}
.scan-line::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,170,0.06), transparent);
    animation: scan 2.5s ease-in-out infinite;
}
@keyframes scan {
    0%   { left: -100%; }
    100% { left: 200%;  }
}

/* ── Upload area ── */
.stFileUploader {
    border: 2px dashed rgba(0,255,170,0.2) !important;
    border-radius: 12px !important;
    background: rgba(0,255,170,0.02) !important;
    transition: border-color 0.3s !important;
}

.stFileUploader:hover {
    border-color: rgba(0,255,170,0.45) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #00ffaa !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(0,180,216,0.1);
    border: 1px solid rgba(0,180,216,0.3);
    color: #00b4d8;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.2s;
}

.stDownloadButton > button:hover {
    background: rgba(0,180,216,0.2);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════
@st.cache_resource
def load_model():
    path = 'fraud_detection_pipeline.joblib'
    if not os.path.exists(path):
        st.error("Model file not found. Run Section 12 in your notebook first.")
        st.stop()
    return joblib.load(path)

bundle    = load_model()
pipeline  = bundle['pipeline']
meta      = bundle['metadata']
FEATURES  = meta['all_features']
THRESHOLD = meta['best_threshold']



#net= network 
#pos=point of sale 
#misc=miscellaneous
CATEGORIES = [
    'shopping_net', 'mis_net', 'grocery_pos', 'shopping_pos',
    'gas_transport', 'misc_pos', 'grocery_net', 'travel',
    'entertainment', 'personal_care', 'kids_pets',
    'food_dining', 'home', 'health_fitness'
]

HIGH_RISK_CATS = {'shopping_net', 'misc_net', 'grocery_pos'}

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#0d1117',
    'axes.edgecolor':   '#1e293b',
    'axes.labelcolor':  '#94a3b8',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'text.color':       '#e2e8f0',
    'grid.color':       '#1e293b',
    'grid.alpha':       0.5,
})


# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════
def engineer_features(df):
    df = df.copy()
    if 'hour' not in df.columns:
        df['hour'] = 12
    df['is_night']   = df['hour'].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)
    df['is_weekend'] = 0
    df['log_amount'] = np.log1p(df['Transaction_Amount'])

    # ── these 4 were missing — causing the KeyError ──
    df['amount_to_city_ratio'] = df['Transaction_Amount'] / (df['City_Population'] + 1)
    df['day_of_week']          = df.get('hour', 12) % 7   # proxy since we have no date
    df['is_high_risk_cat']     = df['Merchant_Category'].isin(
                                     HIGH_RISK_CATS).astype(int)
    df['is_senior']            = (df['Age'] >= 60).astype(int)

  

    return df
def predict_single(category, amount, city_pop, age, gender, hour):
    row = pd.DataFrame([{
        'Merchant_Category':   category,
        'Transaction_Amount':  amount,
        'City_Population':     city_pop,
        'Age':                 age,
        'Gender':              gender,
        'hour':                hour,
    }])
    row = engineer_features(row)
    return pipeline.predict_proba(row[FEATURES])[0][1]

def risk_level(prob):
    if prob >= 0.80:   return "HIGH",   "#ef4444", "🔴"
    elif prob >= 0.30: return "MEDIUM", "#fb923c", "🟠"
    else:              return "LOW",    "#00ffaa", "🟢"

def render_risk_result(prob):
    level, color, icon = risk_level(prob)
    pct = prob * 100

    bar_color = color
    st.markdown(f"""
    <div class="gauge-wrapper scan-line">
        <div class="gauge-pct" style="color:{color}">{pct:.1f}%</div>
        <div class="gauge-label">fraud probability</div>
        <div class="prob-bar-track" style="margin-top:14px">
            <div class="prob-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:0.6rem;color:#475569;margin-top:4px">
            <span>0%</span><span>30%</span><span>80%</span><span>100%</span>
        </div>
        <div style="margin-top:16px">
            <span class="risk-{'high' if level=='HIGH' else 'medium' if level=='MEDIUM' else 'low'}">
                {icon} {level} RISK
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-glow"></div>
    <div class="hero-badge">◈ FRAUD INTELLIGENCE SYSTEM · ACTIVE</div>
    <div class="hero-title">Bank<span>Guard</span> AI</div>
    <div class="hero-sub">SEMI-SUPERVISED · XGBOOST · SPARKOV DATASET </div>
</div>
""", unsafe_allow_html=True)

# ── Metric bar ──
pr_auc     = meta.get('best_pr_auc', 0)
train_rows = meta.get('train_rows', 0)
fraud_rate = meta.get('fraud_rate', 0)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">Model</div>
        <div class="metric-value" style="font-size:1.1rem">XGBoost</div>
        <div class="metric-sub">retrained · semi-supervised</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">PR-AUC Score</div>
        <div class="metric-value">{pr_auc:.4f}</div>
        <div class="metric-sub">primary metric · max 1.0</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Trained On</div>
        <div class="metric-value">{train_rows:,}</div>
        <div class="metric-sub">transactions · Sparkov 2019</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Decision Threshold</div>
        <div class="metric-value">{THRESHOLD:.3f}</div>
        <div class="metric-sub">Best threshold</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Features Used</div>
        <div class="metric-value">{len(FEATURES)}</div>
        <div class="metric-sub">universal · cross-dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "⚡  Live Scenarios",
    "🔬  Manual Test",
    "📂  Batch Analysis"
])


# ──────────────────────────────────────────────────
# TAB 1 — SCENARIOS
# ──────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <h2>Live Demo Scenarios</h2>
    </div>
    <p style="color:#64748b;font-size:0.85rem;margin-bottom:1.5rem;font-family:'Space Mono',monospace">
        Pre-built transaction patterns. Each one tests a different fraud signal the model learned.
    </p>
    """, unsafe_allow_html=True)

    scenarios = {
        "Normal Grocery": dict(
            category='grocery_pos', amount=45.0, city_pop=500000,
            age=34, gender='F', hour=14,
            note="Afternoon grocery run in a large city. Classic low-risk pattern.",
            icon="🛒"
        ),
        "3AM Online Shop": dict(
            category='shopping_net', amount=890.0, city_pop=2500,
            age=22, gender='M', hour=2,
            note="3AM purchase online from a tiny town. High-risk category + night + large amount.",
            icon="🌙"
        ),
        "Travel Booking": dict(
            category='travel', amount=350.0, city_pop=200000,
            age=45, gender='M', hour=10,
            note="Morning travel purchase in a medium city. Typical legitimate pattern.",
            icon="✈️"
        ),
        "Midnight Misc": dict(
            category='misc_net', amount=1200.0, city_pop=800,
            age=67, gender='F', hour=1,
            note="1AM misc purchase, elderly customer, tiny village. Multiple risk flags.",
            icon="⚠️"
        ),
    }

    cols = st.columns(4)
    for idx, (name, s) in enumerate(scenarios.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="scenario-card">
                <div style="font-size:1.8rem;margin-bottom:8px">{s['icon']}</div>
                <div class="scenario-title">{name}</div>
                <div class="scenario-note">{s['note']}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"▶ Analyze", key=f"sc_{idx}", use_container_width=True):
                with st.spinner("Scanning..."):
                    time.sleep(0.4)
                    prob = predict_single(
                        s['category'], s['amount'], s['city_pop'],
                        s['age'], s['gender'], s['hour']
                    )
                render_risk_result(prob)

                level, color, icon = risk_level(prob)
                why = []
                if s['hour'] < 6 or s['hour'] >= 22:
                    why.append("Night transaction (10pm–6am)")
                if s['category'] in HIGH_RISK_CATS:
                    why.append(f"High-risk category: {s['category']}")
                if s['amount'] > 500:
                    why.append(f"Large amount: ${s['amount']:,.0f}")
                if s['city_pop'] < 5000:
                    why.append(f"Very small city ({s['city_pop']:,} people)")
                if s['age'] >= 60:
                    why.append("Senior customer (higher targeting rate)")

                if why:
                    st.markdown(
                        "<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:0.68rem;color:#64748b'>"
                        + "<br>".join(f"→ {w}" for w in why)
                        + "</div>",
                        unsafe_allow_html=True
                    )


# ──────────────────────────────────────────────────
# TAB 2 — MANUAL TEST
# ──────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <h2>Manual Transaction Test</h2>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
    
        category = st.selectbox("Merchant Category", CATEGORIES,
                                help="Type of store where the transaction happened")
        amount   = st.number_input("Transaction Amount ($)", 0.0, 10000.0, 250.0, step=10.0)
        city_pop = st.number_input("City Population", 100, 5_000_00, 10_000, step=1000)
        age      = st.slider("Customer Age", 18, 95, 35)
        gender   = st.radio("Gender", ['M', 'F'], horizontal=True)
        hour     = st.slider("Transaction Hour (0 = midnight, 12 = noon, 23 = 11pm)", 0, 23, 14)

        is_night = hour < 6 or hour >= 22
        ratio    = amount / (city_pop + 1)

        st.markdown(f"""
        <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap">
            <span style="font-family:'Space Mono',monospace;font-size:0.68rem;
                         padding:4px 10px;border-radius:6px;
                         background:{'rgba(239,68,68,0.1)' if is_night else 'rgba(0,255,170,0.07)'};
                         color:{'#f87171' if is_night else '#00ffaa'};
                         border:1px solid {'rgba(239,68,68,0.3)' if is_night else 'rgba(0,255,170,0.2)'}">
                {'⚠ NIGHT TRANSACTION' if is_night else '✓ DAYTIME'}
            </span>
            <span style="font-family:'Space Mono',monospace;font-size:0.68rem;
                         padding:4px 10px;border-radius:6px;
                         background:rgba(255,255,255,0.04);
                         color:#64748b;border:1px solid rgba(255,255,255,0.07)">
                ratio: {ratio:.6f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        run = st.button("🔍 Run Fraud Analysis", use_container_width=True, type="primary")

    with right:
        if run:
            with st.spinner("Analyzing transaction..."):
                time.sleep(0.5)
                prob = predict_single(category, amount, city_pop, age, gender, hour)

            render_risk_result(prob)

            level, color, icon = risk_level(prob)

            st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                        color:#64748b;letter-spacing:0.1em;margin-bottom:8px">
            SIGNAL BREAKDOWN
            </div>
            """, unsafe_allow_html=True)

            signals = [
                ("Merchant Category", category,
                 "HIGH RISK" if category in HIGH_RISK_CATS else "Normal",
                 category in HIGH_RISK_CATS),
                ("Transaction Time", f"{hour:02d}:00",
                 "Night — fraud peaks here" if is_night else "Daytime — normal",
                 is_night),
                ("Amount", f"${amount:,.2f}",
                 "Large transaction" if amount > 500 else "Normal range",
                 amount > 500),
                ("City Size", f"{city_pop:,} people",
                 "Small town — higher risk" if city_pop < 10_000 else "Normal city",
                 city_pop < 10_000),
                ("Customer Age", f"{age} years",
                 "Senior — higher targeting" if age >= 60 else "Normal age group",
                 age >= 60),
            ]

            for sig_name, sig_val, sig_note, is_risky in signals:
                dot_color = "#ef4444" if is_risky else "#00ffaa"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:8px 0;
                            border-bottom:1px solid rgba(255,255,255,0.04)">
                    <div style="width:7px;height:7px;border-radius:50%;
                                background:{dot_color};flex-shrink:0"></div>
                    <div style="flex:1">
                        <div style="font-size:0.78rem;color:#e2e8f0;font-weight:600">{sig_name}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#64748b">{sig_val} — {sig_note}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


        else:
            st.markdown("""
            <div style="height:320px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        border:1px dashed rgba(255,255,255,0.08);border-radius:12px;
                        color:#334155;font-family:'Space Mono',monospace;font-size:0.75rem;
                        text-align:center;gap:12px">
                <div style="font-size:2rem;opacity:0.3">🛡️</div>
                <div>Fill in the transaction details<br>and click Analyze</div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────
# TAB 3 — BATCH
# ──────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <h2>Batch CSV Analysis</h2>
    </div>
    <p style="color:#64748b;font-size:0.85rem;margin-bottom:1.5rem;font-family:'Space Mono',monospace">
        Upload a CSV file to scan multiple transactions at once and get a downloadable risk report.
    </p>
    """, unsafe_allow_html=True)

    with st.expander("📋 Expected CSV Format — click to see"):
        sample = pd.DataFrame([{
            'Merchant_Category': 'shopping_net',
            'Transaction_Amount': 150.0,
            'City_Population': 50000,
            'Age': 35,
            'Gender': 'F',
            'hour': 14
        }])
        st.dataframe(sample, use_container_width=True)
        st.download_button(
            "⬇️ Download Template CSV",
            sample.to_csv(index=False).encode(),
            "template.csv", "text/csv"
        )

    with st.expander("⚙️ Column Mapping — if your CSV uses different column names"):
        col_a, col_b = st.columns(2)
        map_cat    = col_a.text_input("Category column",   "Merchant_Category")
        map_amount = col_a.text_input("Amount column",     "Transaction_Amount")
        map_pop    = col_b.text_input("City pop column",   "City_Population")
        map_age    = col_b.text_input("Age column",        "Age")

    uploaded = st.file_uploader(
        "Drop your CSV here or click to browse",
        type="csv",
        help="Must contain at minimum: Merchant_Category, Transaction_Amount, Age"
    )

    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(data):,} transactions from {uploaded.name}")

            rename_map = {
                map_cat: 'Merchant_Category', map_amount: 'Transaction_Amount',
                map_pop: 'City_Population',   map_age: 'Age',
            }
            rename_map = {k: v for k, v in rename_map.items() if k != v}
            if rename_map:
                data = data.rename(columns=rename_map)

            for col, default in [('Gender','M'), ('hour', 12), ('City_Population', 50000)]:
                if col not in data.columns:
                    data[col] = default

            data = engineer_features(data)

            with st.spinner("Running fraud detection across all transactions..."):
                time.sleep(0.3)
                probs          = pipeline.predict_proba(data[FEATURES])[:, 1]
                data['Risk_Score'] = probs
                data['Risk_%']     = (probs * 100).round(2)
                data['Verdict']    = np.where(
                    probs >= 0.80, "HIGH",
                    np.where(probs >= 0.30, "MEDIUM", "LOW")
                )

            # ── Metric row ──
            n_high   = (probs >= 0.80).sum()
            n_medium = ((probs >= 0.30) & (probs < 0.80)).sum()
            n_low    = (probs < 0.30).sum()

            st.markdown(f"""
            <div class="metric-row" style="margin-top:1.5rem">
                <div class="metric-card">
                    <div class="metric-label">Total Scanned</div>
                    <div class="metric-value">{len(data):,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">High Risk</div>
                    <div class="metric-value" style="color:#ef4444">{n_high:,}</div>
                    <div class="metric-sub">{n_high/len(data)*100:.2f}% of total</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Medium Risk</div>
                    <div class="metric-value" style="color:#fb923c">{n_medium:,}</div>
                    <div class="metric-sub">{n_medium/len(data)*100:.2f}% of total</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Low Risk</div>
                    <div class="metric-value" style="color:#00ffaa">{n_low:,}</div>
                    <div class="metric-sub">{n_low/len(data)*100:.2f}% of total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Charts ──
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor('#0d1117')

            # Distribution
            axes[0].hist(probs, bins=50, color='#00ffaa', edgecolor='#080c14', alpha=0.85)
            axes[0].axvline(0.30, color='#fb923c', ls='--', lw=1.5, label='Medium 0.30')
            axes[0].axvline(0.80, color='#ef4444', ls='--', lw=1.5, label='High 0.80')
            axes[0].set_title('Fraud Probability Distribution', fontsize=12, fontweight='bold', color='#e2e8f0')
            axes[0].set_xlabel('Risk Score')
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.2)

            # Verdict pie
            sizes  = [n_low, n_medium, n_high]
            labels = ['Low', 'Medium', 'High']
            colors = ['#00ffaa', '#fb923c', '#ef4444']
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if nonzero:
                s2, l2, c2 = zip(*nonzero)
                axes[1].pie(s2, labels=l2, colors=c2, autopct='%1.1f%%',
                            textprops={'color': '#e2e8f0', 'fontsize': 9},
                            pctdistance=0.82,
                            wedgeprops={'edgecolor': '#080c14', 'linewidth': 2})
                axes[1].set_title('Risk Distribution', fontsize=12, fontweight='bold', color='#e2e8f0')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Table ──
            st.markdown("""
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                        color:#64748b;letter-spacing:0.1em;margin:1.2rem 0 0.5rem">
            TOP FLAGGED TRANSACTIONS
            </div>
            """, unsafe_allow_html=True)

            show_cols = [c for c in
                ['Merchant_Category', 'Transaction_Amount', 'Age',
                 'hour', 'Risk_%', 'Verdict'] if c in data.columns]

            top_flagged = data.sort_values('Risk_Score', ascending=False).head(100)
            st.dataframe(top_flagged[show_cols], use_container_width=True)

            # ── Download ──
            st.download_button(
                "⬇️ Download Full Risk Report (CSV)",
                data.to_csv(index=False).encode(),
                "bankguard_results.csv", "text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Required columns: Merchant_Category, Transaction_Amount, City_Population, Age")

# ── Footer ──
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;
            font-size:0.65rem;color:#1e293b;padding:1rem 0 0.5rem;letter-spacing:0.1em">
    BANKGUARD AI · M1 AI & DATA SCIENCE · SPARKOV DATASET · XGBOOST
</div>
""", unsafe_allow_html=True)
