import os
import pickle
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Risk Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom dark-themed CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base palette ────────────────────────────────────────────────────── */
:root {
    --bg:       #0e1117;
    --card:     #161b27;
    --card2:    #1a2035;
    --accent:   #3b82f6;
    --accent2:  #6366f1;
    --success:  #22c55e;
    --warn:     #f59e0b;
    --danger:   #ef4444;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
    --border:   #2d3748;
}

html, body, [class*="css"] { background-color: var(--bg); color: var(--text); }

/* ── hero banner ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
}
.hero h1 { font-size: 2rem; font-weight: 800; color: #fff; margin: 0 0 .4rem; }
.hero p  { color: #bfdbfe; margin: 0; font-size: 1rem; }

/* ── stat cards ──────────────────────────────────────────────────────── */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
              color: var(--muted); margin-bottom: .3rem; }
.stat-value { font-size: 2rem; font-weight: 700; color: var(--text); }

/* ── section heading ─────────────────────────────────────────────────── */
.section-head {
    color: var(--accent);
    font-size: 1.1rem;
    font-weight: 700;
    border-left: 4px solid var(--accent);
    padding-left: .7rem;
    margin: 1.5rem 0 .8rem;
}

/* ── account card ────────────────────────────────────────────────────── */
.acct-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 2rem;
}
.risk-score-num { font-size: 3rem; font-weight: 900; line-height: 1; }
.rank-badge {
    display: inline-block;
    background: #14532d;
    color: #4ade80;
    border-radius: 20px;
    padding: .15rem .75rem;
    font-size: .78rem;
    font-weight: 600;
    margin-top: .4rem;
}
.signal-label { font-size: .72rem; color: var(--muted); text-transform: uppercase;
                letter-spacing: .06em; margin-bottom: .2rem; }
.signal-value { font-size: 1.6rem; font-weight: 700; }

/* ── explanation box ─────────────────────────────────────────────────── */
.explain-box {
    background: var(--card2);
    border-left: 4px solid var(--accent2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: var(--text);
    line-height: 1.65;
    font-size: .95rem;
    margin-top: 1rem;
}

/* ── suspicious pill ─────────────────────────────────────────────────── */
.pill-suspicious {
    display: inline-flex; align-items: center; gap: .45rem;
    color: #ef4444; font-size: 1.4rem; font-weight: 800;
}
.pill-suspicious::before {
    content: '';
    display: inline-block;
    width: 14px; height: 14px;
    background: #ef4444;
    border-radius: 50%;
}
.pill-safe {
    display: inline-flex; align-items: center; gap: .45rem;
    color: #22c55e; font-size: 1.4rem; font-weight: 800;
}
.pill-safe::before {
    content: '';
    display: inline-block;
    width: 14px; height: 14px;
    background: #22c55e;
    border-radius: 50%;
}

/* ── precision table ─────────────────────────────────────────────────── */
.prec-row { display:flex; justify-content:space-between;
            border-bottom:1px solid var(--border); padding: .4rem 0; }
.prec-row:last-child { border-bottom: none; }

/* ── sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (cached)
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_DIR = "pipeline_outputs"


@st.cache_data
def load_predictions():
    path = os.path.join(PIPELINE_DIR, "oof_predictions.csv")
    return pd.read_csv(path)


@st.cache_data
def load_aggregated():
    path = os.path.join(PIPELINE_DIR, "X_aggregated.csv")
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_resource
def load_model():
    path = os.path.join(PIPELINE_DIR, "lightgbm_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_shap_data():
    path = os.path.join(PIPELINE_DIR, "shap_data.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    path = os.path.join(PIPELINE_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def check_pipeline_exists():
    return os.path.isdir(PIPELINE_DIR) and os.path.exists(
        os.path.join(PIPELINE_DIR, "oof_predictions.csv")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ AML Risk System")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠 Home", "📊 Global SHAP Analysis", "🔍 Account Risk Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Hackathon Track — Risk Scoring & Prioritization<br>"
        "Model: LightGBM + SHAP</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: risk colour
# ─────────────────────────────────────────────────────────────────────────────
def score_to_color(score_0_100):
    if score_0_100 >= 80:
        return "#ef4444"
    if score_0_100 >= 50:
        return "#f59e0b"
    return "#22c55e"


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ AML Account Risk Scoring &amp; Prioritization</h1>
        <p>Analyze, rank, and explain money-laundering risks at the account level.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── summary stats (live if pipeline exists, else illustrative) ──────────
    if check_pipeline_exists():
        pred_df = load_predictions()
        total_accounts = len(pred_df)
        threshold_raw = 0.50
        flagged = int((pred_df["risk_score"] >= threshold_raw).sum())
        metrics = load_metrics()
        roc_auc_val = metrics["oof_roc_auc"] if metrics else 0.9687
        pr_auc_val  = metrics["oof_pr_auc_ap"] if metrics else 0.5066
    else:
        total_accounts, flagged, roc_auc_val, pr_auc_val = 22310, 237, 0.9687, 0.5066

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "TOTAL ACCOUNTS ANALYZED", f"{total_accounts:,}"),
        (c2, "FLAGGED SUSPICIOUS ACCOUNTS", f"{flagged:,}"),
        (c3, "MODEL ROC-AUC", f"{roc_auc_val:.4f}"),
        (c4, "MODEL PR-AUC (AP)", f"{pr_auc_val:.4f}"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── project description ─────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="section-head">About This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        Financial crime costs the global economy **trillions of dollars** every year.
        Anti-Money Laundering (AML) analysts are overwhelmed with alerts, most of which turn
        out to be false positives. This system is designed to **cut through the noise**.

        Rather than looking at individual transactions, we aggregate every account's full
        transaction history into a rich behavioural profile — then score each account on a
        scale of **0 (safe) to 100 (highly suspicious)**. Accounts are ranked so investigators
        can work down the list from highest to lowest risk, maximising the number of genuine
        threats caught per hour of investigator time.

        The model uses **LightGBM** (a fast gradient-boosting algorithm) trained with
        5-fold cross-validation to handle the extreme class imbalance (~1% of accounts
        are actually suspicious). Predictions are fully explainable using **SHAP values**,
        so every risk score comes with a plain-language reason.
        """)

    with col_r:
        st.markdown('<div class="section-head">How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("📥", "Ingest raw transactions"),
            ("🔧", "Engineer behavioural features"),
            ("🏦", "Aggregate to account level"),
            ("🤖", "Score with LightGBM (5-fold CV)"),
            ("💡", "Explain with SHAP values"),
            ("📋", "Rank & prioritize accounts"),
        ]
        for icon, text in steps:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.8rem;
                        background:#161b27;border-radius:10px;padding:.6rem 1rem;
                        margin-bottom:.5rem;border:1px solid #2d3748;'>
                <span style='font-size:1.3rem'>{icon}</span>
                <span style='font-size:.95rem'>{text}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── key signals explained ───────────────────────────────────────────────
    st.markdown('<div class="section-head">Key Risk Signals the Model Monitors</div>',
                unsafe_allow_html=True)

    signals = [
        ("🆕 Account Age", "Brand-new accounts with large transactions are a classic red flag."),
        ("🌍 Cross-Border Activity", "Transactions crossing high-risk country corridors elevate risk significantly."),
        ("⚡ Velocity Spikes", "Sudden surges in transaction frequency or amount vs. the account's own history."),
        ("🚨 PEP / Sanctions", "Any connection to politically exposed persons or sanctioned entities."),
        ("💰 Large Single Transfers", "One-off transfers that dwarf the account's normal behaviour (layering signal)."),
        ("🔄 Currency Mismatches", "Sending in one currency, receiving in another — a structuring indicator."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(signals):
        cols[i % 3].markdown(f"""
        <div style='background:#161b27;border:1px solid #2d3748;border-radius:12px;
                    padding:1rem;margin-bottom:.8rem;'>
            <div style='font-weight:700;margin-bottom:.3rem'>{title}</div>
            <div style='color:#94a3b8;font-size:.88rem'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── pipeline status ─────────────────────────────────────────────────────
    st.markdown("---")
    if check_pipeline_exists():
        st.success("✅ Pipeline outputs detected — all analysis pages are live.")
    else:
        st.warning(
            "⚠️ **`pipeline_outputs/` folder not found.**  "
            "Run the notebook first to generate `oof_predictions.csv`, "
            "`X_aggregated.csv`, `lightgbm_model.pkl`, and `shap_data.pkl`. "
            "The analysis pages will then become fully interactive."
        )


elif page == "📊 Global SHAP Analysis":

    st.markdown("""
    <div class="hero">
        <h1>📊 Global SHAP Analysis</h1>
        <p>Understand what the model cares about — across all 22,000+ accounts.</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_pipeline_exists():
        st.error("Pipeline outputs not found. Please run the notebook first.")
        st.stop()

    shap_data = load_shap_data()
    X_agg = load_aggregated()
    pred_df = load_predictions()
    metrics = load_metrics()

    shap_values_raw = shap_data["shap_values"]
    base_value      = shap_data["base_value"]
    feature_names   = shap_data["feature_names"]

    X_for_shap = X_agg[feature_names] if feature_names else X_agg

    # Rebuild Explanation object
    explanation = shap.Explanation(
        values=shap_values_raw,
        base_values=np.full(shap_values_raw.shape[0], base_value),
        data=X_for_shap.values,
        feature_names=feature_names,
    )

    # ── Performance summary ─────────────────────────────────────────────────
    st.markdown('<div class="section-head">📈 Performance & Ranking Quality</div>',
                unsafe_allow_html=True)

    # Precision@K
    if metrics:
        p_at_k = metrics.get("p_at_k_standard", {})
        ks     = [10, 25, 50, 100]
        cols   = st.columns(len(ks))
        for col, k in zip(cols, ks):
            key = f"Precision@{k}"
            val = p_at_k.get(key, None)
            display = f"{val*100:.1f}%" if val is not None else "—"
            col.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Precision @ {k}</div>
                <div class="stat-value">{display}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Precision / Recall @ K curve ────────────────────────────────────────
    col_chart, col_info = st.columns([2, 1], gap="large")

    with col_chart:
        st.markdown('<div class="section-head">Investigator Effort Curve (Precision/Recall @ K)</div>',
                    unsafe_allow_html=True)

        if metrics and "precision_at_k" in metrics:
            prec_k   = metrics["precision_at_k"]
            recall_k = metrics["recall_at_k"]
        else:
            # Recompute from predictions
            sorted_df = pred_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
            total_pos = int(sorted_df["true_label"].sum())
            prec_k, recall_k = [], []
            for k in range(1, min(501, len(sorted_df) + 1)):
                tp = sorted_df.head(k)["true_label"].sum()
                prec_k.append(tp / k)
                recall_k.append(tp / max(total_pos, 1))

        fig, ax1 = plt.subplots(figsize=(9, 4), facecolor="#161b27")
        ax1.set_facecolor("#161b27")
        ax2 = ax1.twinx()

        ax1.plot(range(1, len(prec_k) + 1), prec_k,   color="#ef4444", lw=2, label="Precision@K")
        ax2.plot(range(1, len(recall_k) + 1), recall_k, color="#3b82f6", lw=2, ls="--", label="Recall@K")

        for ax in [ax1, ax2]:
            ax.set_facecolor("#161b27")
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("#2d3748")

        ax1.set_xlabel("K (Accounts Investigated)", color="#94a3b8")
        ax1.set_ylabel("Precision@K", color="#ef4444")
        ax2.set_ylabel("Recall@K",    color="#3b82f6")
        ax1.tick_params(axis="y", labelcolor="#ef4444")
        ax2.tick_params(axis="y", labelcolor="#3b82f6")
        ax1.set_ylim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   facecolor="#161b27", edgecolor="#2d3748", labelcolor="#e2e8f0")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_info:
        st.markdown('<div class="section-head">What This Means</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explain-box">
        This chart answers the question every AML team asks: <strong>"If I can only
        review the top N accounts this week, how many real criminals will I catch?"</strong>
        <br><br>
        The <span style="color:#ef4444"><b>red line</b></span> (Precision@K) shows
        what fraction of the top K flagged accounts are genuinely suspicious.
        At K = 10, Precision = 100% — every single one of the top 10 alerts is real.
        <br><br>
        The <span style="color:#3b82f6"><b>blue dashed line</b></span> (Recall@K)
        shows how many of all known criminals we've captured by reviewing K accounts.
        <br><br>
        A high Precision at low K means investigators waste <em>zero</em> time on false
        positives when working the most urgent cases.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Beeswarm ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">🐝 Global Feature Importance (SHAP Beeswarm)</div>',
                unsafe_allow_html=True)

    fig_bee, ax_bee = plt.subplots(figsize=(10, 7), facecolor="#161b27")
    ax_bee.set_facecolor("#161b27")
    shap.plots.beeswarm(explanation, max_display=15, show=False, color_bar=True)
    plt.gcf().set_facecolor("#161b27")
    for spine in plt.gca().spines.values():
        spine.set_color("#2d3748")
    plt.gca().tick_params(colors="#94a3b8")
    plt.gca().set_facecolor("#161b27")
    plt.title("Global Feature Importance — SHAP Beeswarm", color="#e2e8f0", fontsize=13)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close("all")

    # ── global explanation paragraph ────────────────────────────────────────
    st.markdown("""
    <div class="explain-box">
    <b>What is this chart saying?</b><br><br>
    Each dot represents one account. Dots to the <b>right</b> pushed the risk score
    <em>higher</em>; dots to the <b>left</b> pulled it <em>lower</em>. The colour tells you
    whether that feature's value was high (pink/red) or low (blue) for that account.<br><br>
    By far, the most critical red flag is a <b>brand-new account profile</b> — it immediately
    shifts the model toward a high-risk prediction. Conversely, established, mature accounts are
    consistently treated as safe. The model also monitors over 40 secondary signals — specific
    transaction modes, currency mismatches, sanction threats — most of which have near-zero impact
    on the average user. They act as a safety net, grouping together to trigger an alert only when
    a rare, highly specific combination of anomalies occurs simultaneously.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Bar summary ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📊 Mean Absolute SHAP — Top 20 Features</div>',
                unsafe_allow_html=True)

    mean_abs = np.abs(shap_values_raw).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = mean_abs[top_idx]

    fig_bar, ax_bar = plt.subplots(figsize=(9, 6), facecolor="#161b27")
    ax_bar.set_facecolor("#161b27")
    bars = ax_bar.barh(top_names[::-1], top_vals[::-1], color="#3b82f6", edgecolor="none")
    ax_bar.tick_params(colors="#94a3b8")
    ax_bar.set_xlabel("Mean |SHAP value|", color="#94a3b8")
    ax_bar.spines[:].set_color("#2d3748")
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)
    plt.title("Feature Importance (Mean Absolute SHAP)", color="#e2e8f0", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

elif page == "🔍 Account Risk Explorer":
    st.dataframe(risk_df)
    st.markdown("""
    <div class="hero">
        <h1>🔍 Account Risk Explorer</h1>
        <p>Search any account ID to see its risk score, key signals, and a plain-language explanation.</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_pipeline_exists():
        st.error("Pipeline outputs not found. Please run the notebook first.")
        st.stop()

    pred_df  = load_predictions()
    X_agg    = load_aggregated()
    shap_data = load_shap_data()

    shap_values_raw = shap_data["shap_values"]
    base_value      = shap_data["base_value"]
    feature_names   = shap_data["feature_names"]

    X_for_shap = X_agg[feature_names] if feature_names else X_agg

    # Build a ranked lookup
    ranked_df = pred_df.copy()
    ranked_df["risk_score_100"] = (ranked_df["risk_score"] * 100).round(1)
    ranked_df = ranked_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1

    # Map account → row index in X_agg for SHAP
    acct_to_xidx = {acct: i for i, acct in enumerate(X_agg.index)}

    # ── Account selector ────────────────────────────────────────────────────
    col_sel, col_search = st.columns([3, 1], gap="small")
    with col_sel:
        # Dropdown ranked highest to lowest
        sorted_accounts = ranked_df["Sender_account"].astype(str).tolist()
        selected_account = st.selectbox(
            "Select Account ID to investigate (ranked highest to lowest risk):",
            sorted_accounts,
        )
    with col_search:
        st.markdown("<br>", unsafe_allow_html=True)
        manual_id = st.text_input("Or type an Account ID:", placeholder="e.g. 900000000123")

    # Resolve final account
    if manual_id.strip():
        try:
            manual_id_cast = type(ranked_df["Sender_account"].iloc[0])(manual_id.strip())
            if manual_id_cast in ranked_df["Sender_account"].values:
                selected_account = str(manual_id_cast)
            else:
                st.warning(f"Account ID **{manual_id}** not found in the dataset.")
        except Exception:
            st.warning("Invalid Account ID format.")

    # Fetch row
    sel_row = ranked_df[ranked_df["Sender_account"].astype(str) == str(selected_account)].iloc[0]
    risk_100   = sel_row["risk_score_100"]
    risk_color = score_to_color(risk_100)
    rank_val   = int(sel_row["rank"])
    is_susp    = int(sel_row.get("true_label", 0)) if "true_label" in sel_row else None

    # Verification label
    if is_susp == 1:
        label_html = '<span class="pill-suspicious">Suspicious</span>'
    elif is_susp == 0:
        label_html = '<span class="pill-safe">Safe</span>'
    else:
        label_html = '<span style="color:#94a3b8;font-size:1rem">Unknown</span>'

    # ── Account card ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="acct-card">', unsafe_allow_html=True)

    c_score, c_label, c_signals = st.columns([2, 2, 4], gap="large")

    with c_score:
        st.markdown(f"""
        <div>
            <div style='color:#94a3b8;font-size:.85rem;margin-bottom:.3rem'>Risk Score (0–100)</div>
            <div class='risk-score-num' style='color:{risk_color}'>{risk_100}</div>
            <div class='rank-badge'>↑ Rank #{rank_val}</div>
        </div>
        """, unsafe_allow_html=True)

    with c_label:
        st.markdown(f"""
        <div>
            <div style='color:#94a3b8;font-size:.85rem;margin-bottom:.5rem'>Verification Label</div>
            {label_html}
        </div>
        """, unsafe_allow_html=True)

    # Pull account-level features for signals
    acct_key = sel_row["Sender_account"]
    if acct_key in X_agg.index:
        row_feat = X_agg.loc[acct_key]
        total_sent = row_feat.get("amount_local_npr_sum", None)
        tx_count   = row_feat.get("amount_local_npr_count", None)
        cross_pct  = row_feat.get("cross_border_flag_mean", None)
    else:
        total_sent = tx_count = cross_pct = None

    with c_signals:
        st.markdown("<div style='color:#94a3b8;font-size:.85rem;margin-bottom:.5rem'>Account Key Signals</div>",
                    unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.markdown(f"""
        <div><div class='signal-label'>Total Sent (NPR)</div>
        <div class='signal-value'>{f"{total_sent:,.0f}" if total_sent is not None else "—"}</div></div>
        """, unsafe_allow_html=True)
        s2.markdown(f"""
        <div><div class='signal-label'>Tx Count</div>
        <div class='signal-value'>{f"{int(tx_count):,}" if tx_count is not None else "—"}</div></div>
        """, unsafe_allow_html=True)
        s3.markdown(f"""
        <div><div class='signal-label'>Cross-Border Tx</div>
        <div class='signal-value'>{f"{cross_pct*100:.0f}%" if cross_pct is not None else "—"}</div></div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Risk drivers heading ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-head">Why did this account get this score? (Risk Drivers)</div>',
        unsafe_allow_html=True,
    )

    # ── SHAP waterfall ───────────────────────────────────────────────────────
    xidx = acct_to_xidx.get(acct_key, None)
    if xidx is not None:
        single_exp = shap.Explanation(
            values=shap_values_raw[xidx],
            base_values=base_value,
            data=X_for_shap.iloc[xidx].values,
            feature_names=feature_names,
        )

        col_wf, col_nl = st.columns([3, 2], gap="large")

        with col_wf:
            fig_wf = plt.figure(figsize=(9, 5), facecolor="#161b27")
            ax_wf  = fig_wf.add_subplot(111)
            ax_wf.set_facecolor("#161b27")
            shap.plots.waterfall(single_exp, max_display=10, show=False)
            plt.gcf().set_facecolor("#161b27")
            plt.gca().set_facecolor("#161b27")
            for sp in plt.gca().spines.values():
                sp.set_color("#2d3748")
            plt.gca().tick_params(colors="#94a3b8")
            plt.title(f"Risk Drivers — Account {acct_key}", color="#e2e8f0", fontsize=12)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close("all")

        with col_nl:
            # ── Plain-language explanation ───────────────────────────────────
            st.markdown(
                '<div class="section-head" style="margin-top:.5rem">Plain-Language Explanation</div>',
                unsafe_allow_html=True,
            )

            # Identify top positive/negative drivers
            vals = shap_values_raw[xidx]
            pos_mask = vals > 0
            neg_mask = vals < 0
            top_pos = sorted(zip(vals[pos_mask], np.array(feature_names)[pos_mask]),
                             reverse=True)[:3]
            top_neg = sorted(zip(np.abs(vals[neg_mask]), np.array(feature_names)[neg_mask]),
                             reverse=True)[:2]

            # Risk level text
            if risk_100 >= 80:
                level_text = "🔴 **High Risk** — This account should be prioritized for investigation."
                core_msg = (
                    "Our system has identified this account as highly suspicious. "
                    "While we treat the average account as low-risk, this specific account "
                    "showed a dramatic deviation from normal behavior, spiking into the danger zone."
                )
            elif risk_100 >= 50:
                level_text = "🟡 **Medium Risk** — This account displays some unusual behaviour."
                core_msg = (
                    "This account has shown some patterns that are less common among normal accounts. "
                    "It doesn't necessarily mean fraud, but a secondary review is recommended."
                )
            else:
                level_text = "🟢 **Low Risk** — This account appears consistent with normal behaviour."
                core_msg = (
                    "This account's transaction activity looks normal. No major red flags were found. "
                    "It can be deprioritized in the current investigator queue."
                )

            # Feature name prettifier
            def pretty(name):
                return name.replace("_", " ").replace("  ", " ").title()

            drivers_text = ""
            if top_pos:
                driver_list = ", ".join([f"**{pretty(n)}**" for _, n in top_pos])
                drivers_text += (
                    f"The biggest factors **pushing the risk score up** were: {driver_list}. "
                )
            if top_neg:
                mitig_list = ", ".join([f"**{pretty(n)}**" for _, n in top_neg])
                drivers_text += (
                    f"On the other hand, {mitig_list} helped to slightly pull the score back down."
                )

            full_explanation = f"{core_msg}\n\n{drivers_text}"

            st.markdown(
                f'<div class="explain-box">{level_text}<br><br>{full_explanation}</div>',
                unsafe_allow_html=True,
            )

        # ── Additional feature breakdown table ───────────────────────────────
        st.markdown(
            '<div class="section-head">Top SHAP Contributors for This Account</div>',
            unsafe_allow_html=True,
        )

        shap_series = pd.Series(vals, index=feature_names)
        top_features = shap_series.abs().nlargest(10)
        display_df = pd.DataFrame({
            "Feature":    top_features.index,
            "SHAP Value": shap_series[top_features.index].round(4),
            "Direction":  shap_series[top_features.index].apply(
                lambda v: "⬆ Increases Risk" if v > 0 else "⬇ Decreases Risk"
            ),
            "Feature Value": [
                round(float(X_for_shap.iloc[xidx][f]), 4)
                for f in top_features.index
            ],
        }).reset_index(drop=True)

        st.dataframe(
            display_df.style.applymap(
                lambda v: "color: #ef4444" if "Increases" in str(v) else "color: #22c55e",
                subset=["Direction"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("SHAP data for this account is not available in the saved pipeline output.")

    # ── Risk gauge ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Risk Score Gauge</div>', unsafe_allow_html=True)

    fig_g, ax_g = plt.subplots(figsize=(6, 1.5), facecolor="#161b27")
    ax_g.set_facecolor("#161b27")

    # gradient bar
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax_g.imshow(gradient, aspect="auto", extent=[0, 100, 0, 1],
                cmap="RdYlGn_r", alpha=0.85)
    ax_g.axvline(risk_100, color="white", lw=3, ymin=0, ymax=1)
    ax_g.text(risk_100, 1.15, f"{risk_100}", ha="center", va="bottom",
              color="white", fontsize=14, fontweight="bold",
              transform=ax_g.get_xaxis_transform())

    ax_g.set_xlim(0, 100)
    ax_g.set_yticks([])
    ax_g.set_xlabel("Risk Score (0 = Safe, 100 = Highly Suspicious)", color="#94a3b8", fontsize=10)
    ax_g.tick_params(colors="#94a3b8")
    for sp in ax_g.spines.values():
        sp.set_color("#2d3748")

    plt.tight_layout()
    st.pyplot(fig_g)
    plt.close(fig_g)
