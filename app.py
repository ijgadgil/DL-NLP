"""
SecurePay - Real-Time Fraud Detection using Deep Neural Networks
Streamlit front-end.   Run:  streamlit run app.py
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from securepay.baseline import RuleBasedDetector
from securepay.data import CITIES, FOREIGN_CITIES, GeneratorConfig, generate_transactions
from securepay.datasets import SPARKOV, SPARKOV_TEST_START, SPECS
from securepay.features import feature_names, n_features
from securepay.metrics import evaluate, recall_by_type
from securepay.pipeline import IMBALANCE_STRATEGIES, TrainConfig, TrainedModel, prepare, train
from securepay.streaming import StreamingDetector, summarise_latency

ROOT = Path(__file__).parent
ART_DIRS = {"synthetic": ROOT / "artifacts", "sparkov": ROOT / "artifacts" / "sparkov"}
DATASET_LABELS = {"synthetic": "SecurePay synthetic", "sparkov": "Kaggle Sparkov"}

# ----------------------------------------------------------------------------- design tokens
INK, MUTED, LINE = "#1B2733", "#5B6B78", "#DDE3E7"
TEAL, OCHRE, RISK, LEGIT, RULES_C = "#0F766E", "#B7791F", "#C2364B", "#6F8AA6", "#8A94A0"
FONT = "IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"

st.set_page_config(page_title="SecurePay", page_icon=str(ROOT / "assets" / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
st.logo(str(ROOT / "assets" / "logo.svg"), size="large", icon_image=str(ROOT / "assets" / "icon.svg"))

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp {{ font-family: {FONT}; }}
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th, label, input, textarea,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stSidebarNavLink"] span:not([data-testid="stIconMaterial"]),
[data-testid="stNavSectionHeader"], button p {{ font-family: {FONT}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; }}
.block-container {{ padding: 1.6rem 2.2rem 1rem 2.2rem; max-width: 1500px; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
[data-testid="stSidebarNavLink"] span {{ font-size: .93rem; }}

.sp-head {{ display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: .9rem; }}
.sp-head h1 {{ font-size: 1.45rem; font-weight: 600; color: {INK}; margin: 0; padding: 0; letter-spacing: -.01em; }}
.sp-head p {{ color: {MUTED}; margin: .2rem 0 0 0; font-size: .92rem; }}
.sp-chip {{ border: 1px solid {LINE}; background: #FFFFFF; border-radius: 6px; padding: 3px 10px;
           font-size: .8rem; color: {MUTED}; white-space: nowrap; }}

[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 8px; padding: 10px 14px; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .8rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; color: {INK}; }}
[class*="st-key-card"] {{ background: #FFFFFF; border-radius: 8px; }}
.stMarkdown p.sp-note, p.sp-note {{ color: {MUTED}; font-size: .82rem; line-height: 1.45; margin: .1rem 0 0 0; }}
.sp-verdict {{ border-radius: 8px; padding: 12px 14px; font-weight: 600; margin-bottom: .6rem; }}
.sp-verdict.block {{ background: rgba(194,54,75,.09); color: {RISK}; border: 1px solid rgba(194,54,75,.35); }}
.sp-verdict.ok {{ background: rgba(15,118,110,.08); color: {TEAL}; border: 1px solid rgba(15,118,110,.3); }}
.sp-verdict span {{ display: block; font-weight: 400; color: {MUTED}; font-size: .85rem; margin-top: 2px; }}
.sp-steps {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: .2rem; }}
.sp-steps div {{ flex: 1 1 140px; border: 1px solid {LINE}; border-radius: 8px; background: #FFFFFF;
                 padding: 8px 10px; font-size: .82rem; color: {MUTED}; }}
.sp-steps b {{ color: {INK}; font-weight: 600; display: block; font-size: .86rem; }}
.sp-sig {{ width: 100%; border-collapse: collapse; font-size: .84rem; margin-top: .3rem; }}
.sp-sig td {{ padding: 5px 2px; border-bottom: 1px solid {LINE}; color: {INK}; border-left: none; border-right: none; }}
.sp-sig tr td:first-child {{ color: {MUTED}; }}
.sp-sig tr td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 500; }}
.sp-empty {{ border: 1px dashed {LINE}; border-radius: 8px; padding: 28px; text-align: center; color: {MUTED}; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def header(title: str, subtitle: str = "", chip: bool = True):
    tag = f'<div class="sp-chip">{DATASET_LABELS[st.session_state.dataset]}</div>' if chip else ""
    st.markdown(f'<div class="sp-head"><div><h1>{title}</h1><p>{subtitle}</p></div>{tag}</div>',
                unsafe_allow_html=True)


_card_ids = itertools.count()


def card(height=None):
    """A white bordered panel."""
    kw = {"height": height} if height else {}
    return st.container(border=True, key=f"card_{next(_card_ids)}", **kw)


def style(fig, height=320, title=None, legend=True):
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top", pad=dict(t=6),
                   font=dict(size=14, color=INK)) if title else None,
        font=dict(family=FONT, size=12, color=INK), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=(62 if legend else 36) if title else (34 if legend else 10), b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None, font=dict(size=11)),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(gridcolor=LINE, linecolor=LINE, zeroline=False, title_font=dict(size=11, color=MUTED))
    fig.update_yaxes(gridcolor=LINE, linecolor=LINE, zeroline=False, title_font=dict(size=11, color=MUTED))
    return fig


def chart(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def empty(msg: str, page: str | None = None):
    st.markdown(f'<div class="sp-empty">{msg}</div>', unsafe_allow_html=True)
    if page:
        st.page_link(PAGES[page], label=f"Go to {page}")


def model_color(name: str) -> str:
    if name.startswith("Rule"):
        return RULES_C
    return OCHRE if name.startswith("GRU") else TEAL


def money(x) -> str:
    return st.session_state.spec.money(x)


CLS_COLORS = {"legit": LEGIT, "fraud": RISK}


# ----------------------------------------------------------------------------- state
@st.cache_data(show_spinner="Generating synthetic transactions")
def cached_generate(n_users, n_days, victims, seed):
    return generate_transactions(GeneratorConfig(n_users, n_days, victims, seed))


@st.cache_resource(show_spinner="Building behavioural features (about 30 s the first time)")
def cached_prepare(dataset_key, seq_len, _df, _users, _spec, _cuts):
    return prepare(_df, _users, seq_len, _spec, cuts=_cuts)


@st.cache_data(show_spinner="Loading dataset")
def cached_parquet(path: str, mtime: float):
    return pd.read_parquet(path)


def available_datasets() -> list[str]:
    out = ["synthetic"]
    if (ART_DIRS["sparkov"] / "transactions.parquet").exists():
        out.append("sparkov")
    return out


def load_dataset(name: str):
    ss = st.session_state
    art = ART_DIRS[name]
    ss.dataset, ss.spec, ss.art, ss.models = name, SPECS[name], art, {}
    ss.cuts = (SPARKOV_TEST_START - pd.Timedelta(days=91), SPARKOV_TEST_START) if name == "sparkov" else None
    ss.eval_cache, ss.stream_out, ss.score_out = None, None, None
    tx = art / "transactions.parquet"
    if tx.exists():
        ss.df = cached_parquet(str(tx), tx.stat().st_mtime)
        ss.users = cached_parquet(str(art / "users.parquet"), (art / "users.parquet").stat().st_mtime)
        ss.dataset_key = f"{name}-artifacts-{tx.stat().st_mtime}"
        for p in sorted(art.glob("model_*.pt"), key=lambda p: p.name != "model_lstm.pt"):
            tm = TrainedModel.load(p)
            if tm.spec.name == name:
                ss.models[tm.config.cell.upper()] = tm
    else:
        ss.df, ss.users = cached_generate(800, 75, 0.12, 42)
        ss.dataset_key = (800, 75, 0.12, 42)


def prepared(seq_len: int):
    ss = st.session_state
    return cached_prepare(str(ss.dataset_key), seq_len, ss.df, ss.users, ss.spec, ss.cuts)


def rules_detector():
    return RuleBasedDetector(st.session_state.spec)


def saved_results() -> dict | None:
    p = st.session_state.art / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_scores():
    """Scores of every detector on the test window, cached per dataset + model set."""
    ss = st.session_state
    key = (str(ss.dataset_key), tuple((k, id(m)) for k, m in ss.models.items()))
    if ss.eval_cache and ss.eval_cache["key"] == key:
        return ss.eval_cache
    rules = rules_detector()
    seq_len = next(iter(ss.models.values())).config.seq_len
    data = prepared(seq_len)
    rows = data.test_rows
    with st.spinner("Scoring the test window"):
        scores = {"Rule-based": rules.score(data.raw.iloc[rows])}
        thr = {"Rule-based": rules.threshold}
        for name, tm in ss.models.items():
            d = data if tm.config.seq_len == seq_len else prepared(tm.config.seq_len)
            scores[name] = tm.score_rows(d, d.test_rows)
            thr[name] = tm.threshold
    y = data.y[rows]
    ss.eval_cache = dict(key=key, scores=scores, thr=thr, y=y, rows=rows,
                         ftype=ss.df.fraud_type.iloc[rows],
                         metrics={k: evaluate(y, s, thr[k]) for k, s in scores.items()},
                         period=(ss.df.timestamp.iloc[rows[0]], ss.df.timestamp.iloc[rows[-1]]))
    return ss.eval_cache


if "df" not in st.session_state:
    load_dataset("synthetic")
ss = st.session_state


# ============================================================================= pages
def page_dashboard():
    if not ss.models:
        header("Overview", "Deep sequence models versus static rules for card-fraud detection")
        empty("No trained model for this dataset yet. Train an LSTM or GRU to see results here.", "Train models")
        return
    ev = test_scores()
    m = ev["metrics"]
    best = max((k for k in m if k != "Rule-based"), key=lambda k: m[k]["f1"])
    p0, p1 = ev["period"]
    header("Overview", f"Hold-out period {p0:%d %b %Y} to {p1:%d %b %Y}, scored transaction by transaction")
    lat = (saved_results() or {}).get("latency", {})

    c = st.columns(5)
    c[0].metric("Transactions scored", f"{len(ev['y']):,}")
    c[1].metric("Actual fraud", f"{int(ev['y'].sum()):,}", f"{ev['y'].mean():.2%} of traffic", delta_color="off",
                delta_arrow="off")
    c[2].metric(f"{best} F1 score", f"{m[best]['f1']:.3f}", f"{m[best]['f1'] - m['Rule-based']['f1']:+.3f} vs rules")
    c[3].metric("Rule-based F1 score", f"{m['Rule-based']['f1']:.3f}",
                f"{m['Rule-based']['fp']:,} false alarms", delta_color="off", delta_arrow="off")
    c[4].metric("Latency per transaction (p95)", f"{lat['p95']:.2f} ms" if lat else "n/a",
                f"{lat['throughput']:,.0f} per second" if lat else "run train.py", delta_color="off",
                delta_arrow="off")

    left, right = st.columns([3, 2], gap="medium")
    with left, card():
        names = list(m)
        fig = go.Figure()
        fig.add_bar(y=names, x=[m[k]["tp"] for k in names], name="Fraud caught", orientation="h",
                    marker_color=TEAL)
        fig.add_bar(y=names, x=[m[k]["fp"] for k in names], name="False alarms", orientation="h",
                    marker_color=RISK, opacity=.85,
                    text=[f"{m[k]['tp']:,} caught, {m[k]['fp']:,} false alarms" for k in names],
                    textposition="outside", cliponaxis=False, textfont=dict(size=12, color=INK))
        xmax = max(m[k]["tp"] + m[k]["fp"] for k in names)
        fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"), xaxis_title="alerts raised",
                          xaxis_range=[0, xmax * 1.62])
        chart(style(fig, 330, "Alerts an analyst would have to review"))
        st.markdown(f'<p class="sp-note">Rules raise {m["Rule-based"]["fp"] + m["Rule-based"]["tp"]:,} alerts to '
                    f'catch {m["Rule-based"]["tp"]:,} frauds; {best} raises {m[best]["fp"] + m[best]["tp"]:,} to '
                    f'catch {m[best]["tp"]:,}.</p>', unsafe_allow_html=True)
    with right, card():
        rows = [dict(detector=k, metric=x, value=m[k][x]) for k in m for x in ("precision", "recall", "f1")]
        fig = px.bar(pd.DataFrame(rows), x="metric", y="value", color="detector", barmode="group",
                     color_discrete_map={k: model_color(k) for k in m})
        fig.update_yaxes(range=[0, 1], title=None)
        fig.update_xaxes(title=None)
        chart(style(fig, 330, "Precision, recall and F1"))
        st.markdown('<p class="sp-note">Measured on the chronological test set the models never trained on.</p>',
                    unsafe_allow_html=True)

    st.markdown(f"""<div class="sp-steps">
<div><b>1. Transaction arrives</b>amount, time, merchant, location, device</div>
<div><b>2. Customer state</b>running averages, devices, recent activity</div>
<div><b>3. {n_features(ss.spec)} behavioural features</b>each compared with the customer's own past</div>
<div><b>4. Last {next(iter(ss.models.values())).config.seq_len} transactions</b>fed as one sequence</div>
<div><b>5. LSTM / GRU + attention</b>outputs a fraud probability</div>
<div><b>6. Decision</b>alert when above the tuned threshold</div>
</div>""", unsafe_allow_html=True)


def page_data():
    df = ss.df
    sub = (SPARKOV.notes if ss.dataset == "sparkov"
           else "Synthetic card transactions with per-customer habits and three injected fraud patterns")
    top = st.columns([6, 1.2], vertical_alignment="center")
    with top[0]:
        header("Data explorer", sub, chip=False)
    if ss.dataset == "synthetic":
        with top[1].popover("New dataset", width="stretch"):
            with st.form("gen", border=False):
                n_users = st.number_input("Customers", 100, 5000, 800, 100)
                n_days = st.number_input("Days", 20, 180, 75, 5)
                victims = st.slider("Share of customers defrauded", 0.02, 0.3, 0.12, 0.01)
                seed = st.number_input("Random seed", 0, 9999, 42)
                if st.form_submit_button("Generate dataset", type="primary", width="stretch"):
                    ss.df, ss.users = cached_generate(int(n_users), int(n_days), float(victims), int(seed))
                    ss.dataset_key = (int(n_users), int(n_days), float(victims), int(seed))
                    ss.eval_cache = ss.stream_out = ss.score_out = None
                    st.rerun()

    c = st.columns(5)
    c[0].metric("Transactions", f"{len(df):,}")
    c[1].metric("Customers", f"{df.user_id.nunique():,}")
    c[2].metric("Fraudulent", f"{int(df.is_fraud.sum()):,}")
    c[3].metric("Fraud rate", f"{df.is_fraud.mean():.2%}")
    c[4].metric("Period", f"{(df.timestamp.max() - df.timestamp.min()).days} days")

    view = st.segmented_control("View", ["Distributions", "Fraud patterns", "Customer timeline", "Records"],
                                default="Distributions", label_visibility="collapsed")
    lab = df.is_fraud.map({0: "legit", 1: "fraud"})
    if view in (None, "Distributions"):
        cols = st.columns(3, gap="medium")
        with cols[0], card():
            fig = px.histogram(df.assign(cls=lab), x=np.log10(df.amount.clip(lower=1)), color="cls", nbins=50,
                               histnorm="probability density", barmode="overlay", color_discrete_map=CLS_COLORS,
                               opacity=.75, labels={"x": f"log10 amount ({ss.spec.currency})"})
            chart(style(fig, 340, "Transaction amount"))
        with cols[1], card():
            fig = px.histogram(df.assign(cls=lab, hour=df.timestamp.dt.hour), x="hour", color="cls", nbins=24,
                               histnorm="probability", barmode="group", color_discrete_map=CLS_COLORS)
            fig.update_yaxes(title="share within class")
            chart(style(fig, 340, "Hour of day"))
        with cols[2], card():
            cat = (df.assign(cls=lab).groupby(["category", "cls"]).size()
                   .groupby(level=1, group_keys=False).apply(lambda s: s / s.sum()).rename("share").reset_index())
            fig = px.bar(cat, y="category", x="share", color="cls", barmode="group", orientation="h",
                         color_discrete_map=CLS_COLORS)
            fig.update_yaxes(title=None)
            chart(style(fig, 340, "Merchant category"))
    elif view == "Fraud patterns":
        fr = df[df.is_fraud == 1]
        a, b = st.columns([2, 3], gap="medium")
        with a, card():
            if ss.dataset == "sparkov":
                byh = fr.timestamp.dt.hour.value_counts().reindex(range(24), fill_value=0).reset_index()
                byh.columns = ["hour", "count"]
                fig = px.bar(byh, x="hour", y="count", color_discrete_sequence=[RISK])
                fig.update_yaxes(title="fraudulent transactions")
                chart(style(fig, 330, "Fraud by hour of day", legend=False))
            else:
                counts = fr.fraud_type.value_counts().reset_index()
                fig = px.bar(counts, x="count", y="fraud_type", orientation="h", color="fraud_type",
                             color_discrete_sequence=[RISK, OCHRE, "#6B5CA5"])
                fig.update_yaxes(title=None, autorange="reversed")
                chart(style(fig, 330, "Fraudulent transactions by pattern", legend=False))
        with b, card(height=376):
            if ss.dataset == "sparkov":
                legit = df[df.is_fraud == 0]
                nights = [22, 23, 0, 1, 2, 3]
                span = fr.groupby("user_id").timestamp.agg(lambda t: (t.max() - t.min()).total_seconds() / 86400)
                st.markdown(f"""
**What fraud looks like in Sparkov** (labelled yes/no only)

| Signal | Fraud | Legit |
|---|---|---|
| Median amount | {money(fr.amount.median())} | {money(legit.amount.median())} |
| Between 22:00 and 04:00 | {fr.timestamp.dt.hour.isin(nights).mean():.0%} | {legit.timestamp.dt.hour.isin(nights).mean():.0%} |
| Under {ss.spec.currency}50 | {(fr.amount < 50).mean():.0%} | {(legit.amount < 50).mean():.0%} |
| Fraud transactions per affected card | {fr.groupby('user_id').size().median():.0f} | |
| Days from first to last fraud | {span.median():.1f} | |

It behaves like account takeover: a short night-time burst of larger purchases. No device or
foreign-transaction data exists, so those signals stay constant.
""")
            else:
                st.markdown("""
| Pattern | What happens | Why static rules struggle |
|---|---|---|
| **Account takeover** | 2 to 5 high-value buys (electronics, gift cards, crypto) from a new device, often in another city or country | Caught, but the same rules also fire on legitimate travel and big purchases |
| **Card testing** | 2 to 6 tiny probe charges within minutes, then a large one | Each small charge looks harmless on its own |
| **Behavioural drift** | 2 to 4 inflated purchases at the customer's quiet hours, in categories they rarely use, from their own device | Nothing crosses a fixed threshold; only this customer's history reveals it |
""")
    elif view == "Customer timeline":
        victims = list(df[df.is_fraud == 1].user_id.unique())
        a, b = st.columns([1, 3], gap="medium")
        with a, card(height=420):
            uid = st.selectbox("Customer (fraud victims first)",
                               victims + sorted(set(df.user_id) - set(victims))[:200])
            u = df[df.user_id == uid]
            prof = ss.users.set_index("user_id").loc[uid]
            st.metric("Transactions", f"{len(u):,}")
            st.metric("Median spend", money(u.amount.median()))
            st.metric("Fraudulent", int(u.is_fraud.sum()))
            st.markdown(f'<p class="sp-note">Home: {prof.home_city}</p>', unsafe_allow_html=True)
        with b, card():
            fig = px.scatter(u.assign(cls=u.is_fraud.map({0: "legit", 1: "fraud"})), x="timestamp", y="amount",
                             color="cls", log_y=True, color_discrete_map=CLS_COLORS,
                             hover_data=["category", "city", "device_id", "fraud_type"])
            fig.update_traces(marker=dict(size=7, line=dict(width=.5, color="white")))
            fig.update_yaxes(title=f"amount ({ss.spec.currency}, log scale)")
            fig.update_xaxes(title=None)
            chart(style(fig, 374, "Every transaction for this customer"))
    else:
        with card():
            f1, f2 = st.columns([1, 4])
            only_fraud = f1.toggle("Fraud only")
            show = df[df.is_fraud == 1] if only_fraud else df
            f2.markdown(f'<p class="sp-note">Showing the first 2,000 of {len(show):,} rows</p>',
                        unsafe_allow_html=True)
            st.dataframe(show.head(2000), width="stretch", height=360, hide_index=True)


def page_train():
    if ss.dataset == "sparkov":
        split = ("Kaggle's split: fraudTest.csv is the test set; the last 3 months of fraudTrain.csv validate")
    else:
        split = "Chronological split: first 60% of days train, next 15% validate, last 25% test"
    header("Train models", split)
    left, right = st.columns([2, 3], gap="medium")
    with left, card():
        with st.form("train", border=False):
            cells = st.multiselect("Architecture", ["LSTM", "GRU"], default=["LSTM", "GRU"])
            a, b = st.columns(2)
            imbalance = a.selectbox("Class-imbalance handling", IMBALANCE_STRATEGIES, index=0)
            epochs = b.slider("Epochs", 2, 30, 5 if ss.dataset == "sparkov" else 8)
            seq_len = st.slider("Sequence length (transactions per customer)", 3, 30, 10)
            with st.expander("Advanced settings"):
                a, b = st.columns(2)
                hidden = a.select_slider("Hidden units", [16, 32, 64, 128, 256], 64)
                layers = b.slider("Recurrent layers", 1, 3, 2)
                dropout = a.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
                lr = b.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], 2e-3)
                smote_ratio = a.slider("SMOTE fraud:legit ratio", 0.02, 0.5, 0.15, 0.01)
                batch = b.select_slider("Batch size", [64, 128, 256, 512, 1024], 256)
                alpha = a.slider("Focal loss alpha", 0.1, 0.95, 0.75, 0.05)
                gamma = b.slider("Focal loss gamma", 0.0, 5.0, 2.0, 0.5)
            go_btn = st.form_submit_button("Train models", type="primary", width="stretch")
        st.markdown('<p class="sp-note">Sparkov takes several minutes per model on a CPU. '
                    '<code>python train.py --dataset sparkov</code> is faster.</p>' if ss.dataset == "sparkov"
                    else '<p class="sp-note">About a minute per model on a laptop CPU.</p>', unsafe_allow_html=True)

    with right:
        if go_btn and cells:
            with card():
                data = prepared(seq_len)
                st.markdown(f'<p class="sp-note">{len(data.train_rows):,} training, {len(data.val_rows):,} validation '
                            f'and {len(data.test_rows):,} test sequences; '
                            f'{int(data.y[data.train_rows].sum()):,} frauds in training</p>', unsafe_allow_html=True)
                ss.art.mkdir(parents=True, exist_ok=True)
                for cell in cells:
                    cfg = TrainConfig(cell=cell.lower(), seq_len=seq_len, hidden=hidden, layers=layers,
                                      dropout=dropout, epochs=epochs, batch_size=batch, lr=lr, imbalance=imbalance,
                                      smote_ratio=smote_ratio, focal_alpha=alpha, focal_gamma=gamma)
                    bar = st.progress(0.0, text=f"{cell}: preparing")

                    def cb(epoch, logs, bar=bar, cfg=cfg, cell=cell):
                        bar.progress(epoch / cfg.epochs, text=f"{cell}: epoch {epoch} of {cfg.epochs}, "
                                                              f"validation PR-AUC {logs['val_pr_auc']:.3f}")

                    tm = train(data, cfg, on_epoch=cb)
                    ss.models[cell] = tm
                    tm.save(ss.art / f"model_{cell.lower()}.pt")
                    bar.progress(1.0, text=f"{cell}: done in {tm.train_seconds:.0f} s, threshold {tm.threshold:.3f}")
                if not str(ss.dataset_key).startswith(f"{ss.dataset}-artifacts"):
                    ss.df.to_parquet(ss.art / "transactions.parquet")
                    ss.users.to_parquet(ss.art / "users.parquet")
                ss.eval_cache = ss.stream_out = ss.score_out = None
                st.toast("Models trained and saved")
        if not ss.models:
            empty("No models yet. Choose settings and select Train models.")
            return
        with card():
            st.dataframe(pd.DataFrame([dict(Model=k, Imbalance=m.config.imbalance, Sequence=m.config.seq_len,
                                            Epochs=m.config.epochs, Threshold=round(m.threshold, 3),
                                            Parameters=f"{sum(p.numel() for p in m.model.parameters()):,}",
                                            Seconds=round(m.train_seconds)) for k, m in ss.models.items()]),
                         hide_index=True, width="stretch")
            fig = go.Figure()
            for k, m in ss.models.items():
                h = pd.DataFrame(m.history)
                fig.add_scatter(x=h.epoch, y=h.val_pr_auc, name=k, mode="lines+markers",
                                line=dict(color=model_color(k), width=2))
            fig.update_yaxes(title="validation PR-AUC")
            fig.update_xaxes(title="epoch", dtick=1)
            chart(style(fig, 260, "Learning curves (best epoch is kept)"))


def page_evaluate():
    header("Evaluation", "Deep models against the rule-based baseline on the test period")
    if not ss.models:
        empty("Train a model to evaluate it.", "Train models")
        return
    ev = test_scores()
    m, scores, thr, y = ev["metrics"], ev["scores"], ev["thr"], ev["y"]
    tabs = st.tabs(["Summary", "Curves", "Errors", "Threshold", "Ablation"])

    with tabs[0]:
        tbl = pd.DataFrame(m).T
        show = tbl[["precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "tp", "fp", "fn"]]
        show.columns = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "False-positive rate",
                        "Fraud caught", "False alarms", "Fraud missed"]
        with card():
            st.dataframe(show.style.format({c: "{:.3f}" for c in ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]}
                                           | {"False-positive rate": "{:.4f}", "Fraud caught": "{:,.0f}",
                                              "False alarms": "{:,.0f}", "Fraud missed": "{:,.0f}"}),
                         width="stretch")
        a, b = st.columns(2, gap="medium")
        with a, card():
            rows = [dict(detector=k, metric=x, value=m[k][x]) for k in m for x in ("precision", "recall", "f1")]
            fig = px.bar(pd.DataFrame(rows), x="metric", y="value", color="detector", barmode="group",
                         color_discrete_map={k: model_color(k) for k in m})
            fig.update_yaxes(range=[0, 1], title=None)
            fig.update_xaxes(title=None)
            chart(style(fig, 290, "Precision, recall and F1"))
        with b, card():
            rt = pd.DataFrame({k: recall_by_type(ev["ftype"], (s >= thr[k]).astype(int)) for k, s in scores.items()})
            fig = px.bar(rt.reset_index().melt(id_vars="type"), x="type", y="value", color="variable",
                         barmode="group", color_discrete_map={k: model_color(k) for k in scores})
            fig.update_yaxes(range=[0, 1], title="recall")
            fig.update_xaxes(title=None)
            chart(style(fig, 290, "Fraud caught, by pattern"))

    with tabs[1]:
        a, b = st.columns(2, gap="medium")
        with a, card():
            fig = go.Figure()
            for k, s in scores.items():
                p, r, _ = precision_recall_curve(y, s)
                fig.add_scatter(x=r, y=p, name=f"{k} (AP {m[k]['pr_auc']:.2f})", line=dict(color=model_color(k)))
            fig.update_xaxes(title="recall", range=[0, 1])
            fig.update_yaxes(title="precision", range=[0, 1.02])
            chart(style(fig, 400, "Precision-recall curves"))
        with b, card():
            k = next(n for n in scores if n != "Rule-based")
            dist = pd.DataFrame({"risk": scores[k], "cls": np.where(y == 1, "fraud", "legit")})
            fig = px.histogram(dist, x="risk", color="cls", nbins=40, histnorm="percent", barmode="overlay",
                               color_discrete_map=CLS_COLORS, opacity=.75, log_y=True)
            fig.add_vline(x=thr[k], line_dash="dash", line_color=INK, annotation_text="threshold")
            fig.update_xaxes(title=f"{k} fraud probability")
            fig.update_yaxes(title="% of class (log)")
            chart(style(fig, 400, f"How {k} separates fraud from legitimate"))

    with tabs[2]:
        cols = st.columns(len(m), gap="medium")
        for c, (k, v) in zip(cols, m.items()):
            with c, card():
                z = [[v["tn"], v["fp"]], [v["fn"], v["tp"]]]
                fig = px.imshow(z, text_auto=",d", x=["predicted legit", "predicted fraud"],
                                y=["actual legit", "actual fraud"],
                                color_continuous_scale=[[0, "#F4F6F7"], [1, model_color(k)]], aspect="auto")
                fig.update_traces(textfont_size=14)
                fig.update_layout(coloraxis_showscale=False)
                chart(style(fig, 300, k, legend=False))
                st.markdown(f'<p class="sp-note">{v["fp"]:,} false alarms, {v["fn"]:,} frauds missed</p>',
                            unsafe_allow_html=True)

    with tabs[3]:
        a, b = st.columns([1, 2], gap="medium")
        with a, card(height=420):
            k = st.selectbox("Model", [n for n in scores if n != "Rule-based"])
            t = st.slider("Alert threshold", 0.01, 0.99, float(np.clip(thr[k], 0.01, 0.99)), 0.01)
            mm = evaluate(y, scores[k], t)
            x1, x2 = st.columns(2)
            x1.metric("Precision", f"{mm['precision']:.3f}")
            x2.metric("Recall", f"{mm['recall']:.3f}")
            x1.metric("False alarms", f"{mm['fp']:,}")
            x2.metric("Fraud missed", f"{mm['fn']:,}")
            st.markdown(f'<p class="sp-note">Tuned threshold {thr[k]:.3f} maximises F1 on validation data.</p>',
                        unsafe_allow_html=True)
        with b, card():
            grid = np.linspace(0.01, 0.99, 50)
            curve = pd.DataFrame([dict(t=g, **{x: evaluate(y, scores[k], g)[x] for x in ("precision", "recall", "f1")})
                                  for g in grid])
            fig = go.Figure()
            for x, col in (("precision", TEAL), ("recall", OCHRE), ("f1", INK)):
                fig.add_scatter(x=curve.t, y=curve[x], name=x, line=dict(color=col, width=2))
            fig.add_vline(x=t, line_dash="dash", line_color=RISK)
            fig.update_xaxes(title="alert threshold")
            fig.update_yaxes(range=[0, 1.02])
            chart(style(fig, 374, "Trade-off as the threshold moves"))

    with tabs[4]:
        res = saved_results()
        saved = (res or {}).get("metrics", {})
        if not any(k.startswith("LSTM (") and not k.endswith("smote+focal)") for k in saved):
            empty("Run <code>python train.py --ablation</code> to compare SMOTE, focal loss, both and neither "
                  "with the same LSTM.")
        else:
            a, b = st.columns([3, 2], gap="medium")
            ab = pd.DataFrame(saved).T[["precision", "recall", "f1", "pr_auc", "fp", "fn"]].astype(float)
            ab.columns = ["Precision", "Recall", "F1", "PR-AUC", "False alarms", "Fraud missed"]
            with a, card():
                st.dataframe(ab.style.format({c: "{:.3f}" for c in ["Precision", "Recall", "F1", "PR-AUC"]}
                                             | {"False alarms": "{:,.0f}", "Fraud missed": "{:,.0f}"}),
                             width="stretch")
            with b, card():
                lst = ab[ab.index.str.startswith("LSTM")].copy()
                lst.index = [i.replace("LSTM (", "").rstrip(")").replace("smote", "SMOTE").replace("focal", "focal loss")
                             .replace("+", " + ").replace("none", "neither") for i in lst.index]
                fig = px.bar(lst.reset_index(), y="index", x="F1", orientation="h", color_discrete_sequence=[TEAL],
                             text_auto=".3f")
                fig.update_yaxes(title=None, autorange="reversed")
                fig.update_xaxes(title="F1", range=[0, 1])
                chart(style(fig, 250, "LSTM F1 by imbalance strategy", legend=False))
            st.markdown('<p class="sp-note">Same LSTM and data, only the imbalance handling changes.</p>',
                        unsafe_allow_html=True)


def page_stream():
    header("Live stream", "Test-period transactions replayed one at a time, as a bank would receive them")
    if not ss.models:
        empty("Train a model to run the stream.", "Train models")
        return
    data = prepared(next(iter(ss.models.values())).config.seq_len)
    first = int(data.test_rows.min())
    test_df = ss.df.iloc[first:]
    with card():
        c = st.columns([1.2, 1.4, 1.6, 1.2, 1.1], vertical_alignment="bottom")
        name = c[0].selectbox("Model", list(ss.models))
        n_events = c[1].select_slider("Transactions", [500, 1000, 2000, 5000, 10000], 2000)
        offset = c[2].slider("Start position in test period", 0, max(len(test_df) - n_events, 0), 0, 500)
        speed = c[3].selectbox("Pace", ["Fastest", "Readable", "Slow"])
        start = c[4].button("Start stream", type="primary", width="stretch")
    tm = ss.models[name]
    delay = {"Fastest": 0, "Readable": 0.004, "Slow": 0.02}[speed]
    refresh = {"Fastest": 250, "Readable": 50, "Slow": 20}[speed]

    k = st.columns(6)
    slots = [x.empty() for x in k]
    a, b = st.columns([3, 2], gap="medium")
    with a, card(height=440):
        chart_slot = st.empty()
    with b, card(height=440):
        feed_note = st.empty()
        feed_slot = st.empty()

    def render(out: pd.DataFrame, final: bool):
        lat = summarise_latency(out.latency_ms)
        vals = [("Processed", f"{len(out):,}"), (f"{name} alerts", f"{int(out.dl_alert.sum()):,}"),
                ("Rule alerts", f"{int(out.rule_alert.sum()):,}"), ("Actual fraud", f"{int(out.is_fraud.sum()):,}"),
                ("p95 latency", f"{lat['p95']:.2f} ms"), ("Throughput", f"{lat['throughput']:,.0f} / s")]
        for s, (lbl, v) in zip(slots, vals):
            s.metric(lbl, v)
        tail = out.tail(600)
        fig = go.Figure()
        legit, fraud = tail[tail.is_fraud == 0], tail[tail.is_fraud == 1]
        fig.add_scatter(x=legit.timestamp, y=legit.risk, mode="markers", name="legit",
                        marker=dict(size=5, color=LEGIT, opacity=.45))
        fig.add_scatter(x=fraud.timestamp, y=fraud.risk, mode="markers", name="fraud",
                        marker=dict(size=8, color=RISK, line=dict(width=1, color="white")))
        fig.add_hline(y=tm.threshold, line_dash="dash", line_color=INK, annotation_text="alert threshold",
                      annotation_font_size=11)
        fig.update_yaxes(title="fraud probability", range=[-0.03, 1.03])
        fig.update_xaxes(title=None)
        with chart_slot:
            chart(style(fig, 400, "Risk score of each transaction"), key=f"stream_{len(out)}_{final}")
        alerts = out[out.dl_alert | out.rule_alert].iloc[::-1].head(60)
        alerts = alerts.assign(Outcome=np.select(
            [alerts.dl_alert & (alerts.is_fraud == 1), alerts.dl_alert, alerts.is_fraud == 1],
            ["Fraud caught", "False alarm", "Rules: fraud"], "Rules: false alarm"),
            rules_fired=alerts.rules_fired.str.replace(r"(R\d)_[a-z_]+", r"\1", regex=True))
        feed_note.markdown('<p class="sp-note">Newest first. R1 large amount, R2 spend spike, R3 velocity, '
                           'R4 foreign, R5 night spike, R6 new device, R7 far and high-risk.</p>',
                           unsafe_allow_html=True)
        feed_slot.dataframe(
            alerts[["timestamp", "amount", "risk", "Outcome", "rules_fired"]]
            .rename(columns={"timestamp": "Time", "amount": "Amount", "risk": "Risk", "rules_fired": "Rules fired"})
            .style.format({"Risk": "{:.2f}", "Amount": ss.spec.currency + "{:,.0f}", "Time": "{:%d %b %H:%M}"}),
            width="stretch", hide_index=True, height=362)

    if start:
        with st.spinner("Building every customer's profile from their history"):
            det = StreamingDetector(tm, ss.users)
            det.warm_up(ss.df.iloc[:first + offset])
        results = []
        events = test_df.iloc[offset:offset + n_events]
        for i, r in enumerate(det.stream(events), 1):
            results.append(r)
            if delay:
                time.sleep(delay)
            if i % refresh == 0:
                render(pd.DataFrame(results), False)
        ss.stream_out = (name, pd.DataFrame(results))

    if ss.stream_out and ss.stream_out[0] == name:
        out = ss.stream_out[1]
        render(out, True)
        y = out.is_fraud.values
        summ = pd.DataFrame({name: evaluate(y, out.risk.values, tm.threshold),
                             "Rule-based": evaluate(y, out.rule_alert.astype(float).values, 0.5)}).T
        with st.popover("Stream summary"):
            st.dataframe(summ[["precision", "recall", "f1", "tp", "fp", "fn"]].round(3), width="stretch")
            lat = summarise_latency(out.latency_ms)
            st.markdown(f'<p class="sp-note">Latency per transaction, including state update and model call: '
                        f'median {lat["p50"]:.2f} ms, p95 {lat["p95"]:.2f} ms, p99 {lat["p99"]:.2f} ms.</p>',
                        unsafe_allow_html=True)
    elif not start:
        for s, lbl in zip(slots, ["Processed", "Model alerts", "Rule alerts", "Actual fraud", "p95 latency",
                                  "Throughput"]):
            s.metric(lbl, "–")
        with chart_slot:
            empty("Choose how many transactions to replay and select Start stream.")
        feed_slot.markdown('<p class="sp-note">Alerts from the model and the rules appear here, newest first.</p>',
                           unsafe_allow_html=True)


def page_score():
    header("Score a transaction", "Try a payment for a real customer in the dataset and see how each detector reacts")
    if not ss.models:
        empty("Train a model to score transactions.", "Train models")
        return
    df = ss.df
    a, b, c = st.columns([1.15, 1.25, 1.3], gap="medium")
    with a, card(height=604):
        name = st.selectbox("Model", list(ss.models))
        tm = ss.models[name]
        uid = st.selectbox("Customer", sorted(df.user_id.unique()))
        hist = df[df.user_id == uid]
        prof = ss.users.set_index("user_id").loc[uid]
        h = hist.timestamp.dt.hour
        st.markdown(f'<p class="sp-note">Home {prof.home_city}. {len(hist):,} past transactions, median '
                    f'{money(hist.amount.median())}, usually active {int(h.quantile(.1))}:00 to '
                    f'{int(h.quantile(.9))}:00.</p>', unsafe_allow_html=True)
        st.dataframe(hist.tail(7)[["timestamp", "amount", "category"]]
                     .rename(columns={"timestamp": "Recent", "amount": "Amount", "category": "Category"})
                     .style.format({"Amount": ss.spec.currency + "{:,.0f}", "Recent": "{:%d %b %H:%M}"}),
                     hide_index=True, width="stretch", height=282)
    with b, card(height=604):
        with st.form("score", border=False):
            x1, x2 = st.columns(2)
            amount = x1.number_input(f"Amount ({ss.spec.currency})", 1.0, 1_000_000.0,
                                     float(round(hist.amount.median(), 0)))
            hour = x2.slider("Hour of day", 0, 23, 14)
            category = st.selectbox("Merchant category", list(ss.spec.categories))
            all_cities = {**CITIES, **FOREIGN_CITIES}
            if ss.dataset == "synthetic":
                x1, x2 = st.columns(2)
                city = x1.selectbox("City", list(all_cities), index=list(all_cities).index(prof.home_city))
                device = x2.selectbox("Device", list(hist.device_id.unique()) + ["NEW-DEVICE"])
            else:
                km = st.slider("Distance from home (km)", 0, 3000, 50, 10)
                device = hist.device_id.iloc[0]
            minutes_after = st.number_input("Minutes since the customer's last transaction", 1, 100000, 600)
            burst = st.slider(f"Tiny {ss.spec.currency}10 charges just before (card testing)", 0, 6, 0)
            sub = st.form_submit_button("Score transaction", type="primary", width="stretch")
    if sub:
        det = StreamingDetector(tm, ss.users)
        det.warm_up(hist)
        last = hist.timestamp.max()
        ts = (last + pd.Timedelta(minutes=minutes_after)).normalize() + pd.Timedelta(hours=hour)
        if ts <= last:
            ts = last + pd.Timedelta(minutes=minutes_after)
        if ss.dataset == "synthetic":
            lat0, lon0 = all_cities[city]
            country = "IN" if city in CITIES else "FOREIGN"
        else:
            lat0, lon0 = prof.home_lat + km / 111.0, prof.home_lon
            city, country = f"{km} km from home", ss.spec.default_country
        base = dict(user_id=uid, category=category, lat=lat0, lon=lon0, city=city, country=country,
                    device_id=device, is_fraud=None, fraud_type=None)
        burst_cat = "ecommerce" if "ecommerce" in ss.spec.categories else "shopping_net"
        for j in range(burst):
            det.process({**base, "txn_id": f"B{j}", "timestamp": ts - pd.Timedelta(minutes=2 * (burst - j)),
                         "amount": 10.0, "category": burst_cat})
        r = det.process({**base, "txn_id": "NEW", "timestamp": ts, "amount": amount})
        f = dict(zip(feature_names(ss.spec), det.engine.states[uid].history[-1]))
        ss.score_out = dict(uid=uid, model=name, r=r, f=f, desc=det.rules.descriptions)

    with c, card(height=604):
        out = ss.score_out
        if not out or out["uid"] != uid or out["model"] != name:
            empty("Set up a transaction and select Score transaction.")
            return
        r, f = out["r"], out["f"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=r["risk"] * 100, number=dict(suffix="%", font=dict(size=34, color=INK)),
            gauge=dict(axis=dict(range=[0, 100], tickcolor=LINE), bar=dict(color=RISK if r["dl_alert"] else TEAL),
                       bgcolor="#F4F6F7", borderwidth=0,
                       threshold=dict(line=dict(color=INK, width=3), value=tm.threshold * 100))))
        style(fig, 190, legend=False).update_layout(margin=dict(l=28, r=28, t=26, b=4))
        chart(fig)
        if r["dl_alert"]:
            st.markdown(f'<div class="sp-verdict block">Hold and verify with the customer<span>{name} risk is above '
                        f'the {tm.threshold:.0%} alert threshold</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sp-verdict ok">Approve<span>{name} finds this consistent with the '
                        f'customer\'s behaviour</span></div>', unsafe_allow_html=True)
        fired = [x for x in r["rules_fired"].split(", ") if x]
        rules_txt = "; ".join(out["desc"][x] for x in fired) if fired else "none"
        st.markdown(f'<p class="sp-note"><b>Rules fired:</b> {rules_txt}</p>', unsafe_allow_html=True)
        sig = [("Amount vs usual (z-score)", f"{f['amount_zscore']:+.2f}"),
               ("Hours since previous", f"{np.expm1(f['log_hours_since_last']):.1f}"),
               ("Transactions in last 10 min", f"{int(f['txns_last_10min'])}"),
               ("Km from home", f"{np.expm1(f['log_km_from_home']):,.0f}"),
               ("New device", "yes" if f["new_device"] else "no"),
               ("New category for customer", "yes" if f["new_category"] else "no"),
               ("Hour vs usual (0 to 1)", f"{f['hour_deviation']:.2f}")]
        st.markdown('<table class="sp-sig">' + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sig)
                    + "</table>", unsafe_allow_html=True)


# ============================================================================= navigation
PAGES = {
    "Overview": st.Page(page_dashboard, title="Overview", icon=":material/space_dashboard:", default=True),
    "Live stream": st.Page(page_stream, title="Live stream", icon=":material/stream:", url_path="stream"),
    "Score a transaction": st.Page(page_score, title="Score a transaction", icon=":material/credit_score:",
                                   url_path="score"),
    "Evaluation": st.Page(page_evaluate, title="Evaluation", icon=":material/monitoring:", url_path="evaluation"),
    "Train models": st.Page(page_train, title="Train models", icon=":material/model_training:", url_path="train"),
    "Data explorer": st.Page(page_data, title="Data explorer", icon=":material/table_chart:", url_path="data"),
}
nav = st.navigation({"Monitor": [PAGES["Overview"], PAGES["Live stream"], PAGES["Score a transaction"]],
                     "Model": [PAGES["Evaluation"], PAGES["Train models"], PAGES["Data explorer"]]})

with st.sidebar:
    opts = available_datasets()
    choice = st.selectbox("Dataset", opts, index=opts.index(ss.dataset), format_func=DATASET_LABELS.get,
                          help="Kaggle Sparkov appears after running python train.py --dataset sparkov")
    if choice != ss.dataset:
        load_dataset(choice)
        st.rerun()
    st.markdown(f'<p class="sp-note">{len(ss.df):,} transactions, {ss.df.is_fraud.mean():.2%} fraud</p>',
                unsafe_allow_html=True)
    if ss.models:
        st.markdown('<p class="sp-note">Models: ' + ", ".join(ss.models) + "</p>", unsafe_allow_html=True)
    else:
        st.markdown('<p class="sp-note">No trained models for this dataset</p>', unsafe_allow_html=True)

nav.run()
