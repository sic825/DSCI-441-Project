import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pages._style import inject, _CSS

st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(_CSS, unsafe_allow_html=True)

# ── Title block ───────────────────────────────────────────────────────────────
st.title("Hybrid Music Recommendation System")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; '
    'font-size:1.0em; color:#555; margin-top:-0.4em;">'
    'Simon Chen &nbsp;·&nbsp; Thoi Quach &nbsp;·&nbsp; '
    'Lehigh University &nbsp;·&nbsp; DSCI 441 — Statistical Machine Learning &nbsp;·&nbsp; '
    'Spring 2026 &nbsp;·&nbsp; Instructor: Dr. Yari'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Headline metrics ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Warm NDCG@10", "0.1099", delta="≈ CF baseline (p = 0.72)", delta_color="off")
    st.caption("n = 1,000 warm users · 80/20 hold-out")
with col2:
    st.metric("Cold-Start LT NDCG@10", "0.0168", delta="+6.7× vs Content-only", delta_color="normal")
    st.caption("n = 869 cold-start users · top-100 excluded")
with col3:
    st.metric("Catalog Coverage Δ (Hybrid − CF)", "+0.025", delta="p = 0.004, d = 0.099", delta_color="off")
    st.caption("Paired bootstrap · 10 K resamples")

st.divider()
st.info(
    "Navigate via the sidebar: **Overview** (paper) · **How It Works** (methodology) · "
    "**Live Demo** (interactive models) · **About** (team & references)."
)
