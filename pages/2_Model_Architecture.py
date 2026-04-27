import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from styling import inject, figcap, callout

st.set_page_config(page_title="Model Architecture", page_icon="⚙️", layout="wide")
inject()

ROOT = Path(__file__).parent.parent
R    = ROOT / "results"

@st.cache_data
def load_user_info():
    return pd.read_parquet(R / "user_info.parquet")

user_info = load_user_info()

st.title("Model Architecture")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; color:#555;">'
    'System design, scoring formulas, and algorithmic detail'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── 1. System Architecture ────────────────────────────────────────────────────
st.markdown("## 1. System Architecture")
st.markdown("""
The recommender consists of two independently trained retrieval models whose
candidate lists are merged at query time by the `HybridRecommender` class.
The following data-flow diagram summarizes the full pipeline.
""")

_col_l, _col_c, _col_r = st.columns([1, 2, 1])
with _col_c:
    st.graphviz_chart("""
digraph G {
    rankdir=LR; bgcolor="#FFFFFF";
    node [shape=box, style="filled,rounded", fillcolor="#EEF3F9",
          color="#1F4E79", fontcolor="#1A1A1A", fontname=Helvetica, fontsize=11]
    edge [color="#1F4E79", fontcolor="#555", fontname=Helvetica, fontsize=9]

    MSD  [label="MSD Taste Profile\\n661K users × 98K songs"]
    ALS  [label="ALS (implicit)\\nfactors=64, iter=20\\nα·log1p(counts)"]
    CF   [label="CF Candidates\\ntop-50 per user"]

    Spot [label="Spotify Audio Features\\n114K tracks, 9 dimensions"]
    Mat  [label="Cross-dataset match\\nexact + fuzzy → 7,611 songs"]
    kNN  [label="k-NN (cosine, brute)\\nStandardScaler"]
    CB   [label="Content Candidates\\ntop-50 by similarity"]

    Hyb  [label="HybridRecommender\\nwarm: adaptive α blend\\ncold: genre RRF"]
    Out  [label="Top-k Recommendations"]

    MSD -> ALS -> CF  -> Hyb
    Spot -> Mat -> kNN -> CB -> Hyb
    Hyb -> Out
}
""", use_container_width=True)
figcap(
    "Figure 1: End-to-end data flow. The two retrieval components are trained independently; "
    "the HybridRecommender merges their outputs at serving time."
)

st.markdown("""
**Training data sources:**

- **MSD Taste Profile** (Echo Nest / Columbia): 48M play events filtered
  to users with ≥ 20 songs and songs with ≥ 50 listeners → 661,089 users,
  98,485 songs, 40.3M interactions (sparsity 99.94%).

- **Spotify Tracks Dataset** (HuggingFace): 114,000 tracks with 9 Spotify
  audio features. Cross-dataset matching via cleaned title+artist (exact) and
  `token_set_ratio` ≥ 90 (fuzzy) yields 7,611 deduplicated song pairs.

**Serving time cost (per query):**

| Step | Complexity |
|---|---|
| ALS recommend | *O*(1) dot product |
| k-NN query | *O*(\\|catalog\\|·*d*) ≈ *O*(68K) |
| RRF / blend | *O*(*k*) |
""")
st.divider()

# ── 2. Warm-User Blending — Adaptive Alpha ────────────────────────────────────
st.markdown("## 2. Warm-User Blending — Adaptive Alpha")
st.markdown("""
For a user with *n* training interactions, the blending weight assigned to the
CF score is given by:
""")
st.latex(r"\alpha(n) = \sigma\!\left(\frac{\ln(1 + n) - 2}{1.5}\right), \qquad \sigma(x) = \frac{1}{1+e^{-x}}")
st.markdown("""
The inflection point of the sigmoid occurs at ln(1 + *n*) = 2, i.e., *n* ≈ 6.4
interactions, where CF and content contribute equally. Before blending, scores
from each model are min-max normalized within their own top-50 candidate pool to
place CF ALS dot-products and content cosine similarities on the same [0, 1] scale.
The blended score for song *s* is:
""")
st.latex(r"\text{score}(s) = \alpha \cdot \mathrm{CF}_\text{norm}(s) + (1-\alpha) \cdot \mathrm{Content}_\text{norm}(s)")
st.markdown("""
For songs appearing in only one candidate pool, the missing contribution is zero,
penalizing single-source songs proportionally.
""")

n_range = np.linspace(0, 600, 600)
alpha_curve = 1.0 / (1.0 + np.exp(-((np.log1p(n_range) - 2.0) / 1.5)))
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=n_range, y=alpha_curve, mode="lines", name="α(n)",
    line=dict(color="#1F4E79", width=2.5),
))
fig.add_trace(go.Histogram(
    x=user_info["n_interactions"], nbinsx=40, name="Test users",
    opacity=0.35, marker_color="#C0392B", yaxis="y2",
))
fig.add_hline(y=0.5, line_dash="dot", line_color="#888",
              annotation_text="α = 0.5  (balanced)", annotation_position="right")
fig.update_layout(
    xaxis_title="n_history (training interactions)",
    yaxis=dict(title="Alpha (CF weight)", range=[0, 1], gridcolor="#E8E8E8"),
    yaxis2=dict(title="# Test users", overlaying="y", side="right", showgrid=False),
    legend=dict(x=0.55, y=0.25, bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#D0D8E4", borderwidth=1),
    height=380, paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Helvetica Neue, Arial, sans-serif", color="#1A1A1A", size=12),
    margin=dict(l=50, r=50, t=20, b=50),
)
st.plotly_chart(fig, use_container_width=True)
figcap(
    "Figure 2: Adaptive alpha as a function of interaction count (blue curve) "
    "and the distribution of training interaction counts for the 1,000 test users (red "
    f"histogram). Observed alpha range: "
    f"{user_info['adaptive_alpha'].min():.3f}–{user_info['adaptive_alpha'].max():.3f} "
    f"(median {user_info['adaptive_alpha'].median():.3f}). The alpha floor of ≈ 0.59 "
    "reflects the MIN_USER_COUNT = 20 filter applied during model training."
)

alpha_examples = [
    (1,   "1 (cold)"),
    (7,   "7 (balanced)"),
    (20,  "20 (filter floor)"),
    (50,  "50"),
    (200, "200"),
    (500, "500+"),
]
alpha_html = "".join(
    f"<li><em>n</em>&nbsp;=&nbsp;{label}: "
    f"α&nbsp;=&nbsp;{1/(1+np.exp(-((np.log1p(n)-2)/1.5))):.3f}</li>"
    for n, label in alpha_examples
)
callout("Representative α Values", f"<ul>{alpha_html}</ul>")

callout(
    "Normalization Rationale",
    "<p>ALS recommendation scores are inner products of 64-dimensional factor vectors; "
    "their magnitude depends on training scale. Cosine similarities live in [−1, 1] "
    "by construction. Without normalization, the model with larger raw scores silently "
    "dominates the blend regardless of the alpha setting. Min-max within each candidate "
    "pool is the minimum-assumption normalization that preserves intra-pool ranking while "
    "making the scales commensurate.</p>",
)
st.divider()

# ── 3. Cold-Start Blending — Genre-Conditioned RRF ───────────────────────────
st.markdown("## 3. Cold-Start Blending — Genre-Conditioned RRF")
st.markdown("""
When no user history is available, the system requires an alternative signal to
augment the content k-NN list. We use the seed song's Spotify genre tag to
construct a genre-local popularity prior, then fuse the two ranked lists with
Reciprocal Rank Fusion (Cormack et al., 2009):
""")
st.latex(r"\text{RRF}(s) = \sum_{i \,\in\, \{\text{content},\, \text{popularity}\}} \frac{1}{60 + \mathrm{rank}_i(s)}")
st.markdown(r"""
The damping constant *k* = 60 follows the original paper; it reduces sensitivity
to top-rank placement errors. Each ranked list contributes up to top-50 candidates.
Songs appearing in both lists accumulate two RRF terms, creating a natural
amplification for items that are both acoustically similar to the seed and
contextually popular within the same genre.
""")

_col_l2, _col_c2, _col_r2 = st.columns([1, 2, 1])
with _col_c2:
    st.graphviz_chart("""
digraph CS {
    rankdir=TB; bgcolor="#FFFFFF";
    node [shape=box, style="filled,rounded", fillcolor="#EEF3F9",
          color="#1F4E79", fontcolor="#1A1A1A", fontname=Helvetica, fontsize=10]
    edge [color="#1F4E79", fontname=Helvetica, fontsize=9]

    Seed  [label="Seed Song (no user history)"]
    Genre [label="Lookup genre\\nfrom metadata catalog"]
    GLP   [label="Genre-local popularity\\ntop-50 by MSD play count"]
    FB    [label="Global top-50\\n(fallback)"]
    kNN   [label="Content k-NN\\ntop-50 similar songs"]
    RRF   [label="RRF fusion\\nRRF(s) = Σ 1/(60 + rankᵢ(s))"]
    Out   [label="Top-k cold-start recommendations"]

    Seed -> Genre -> GLP [label="genre known"]
    Genre -> FB [label="Unknown"]
    GLP -> RRF; FB -> RRF
    Seed -> kNN -> RRF
    RRF -> Out
}
""", use_container_width=True)
figcap(
    "Figure 3: Cold-start recommendation pipeline. Genre-conditioned popularity "
    "is used when the seed song's genre can be identified from the metadata catalog; "
    "otherwise global popularity is the fallback."
)

callout(
    "Design Decisions",
    "<p><strong>Why genre conditioning?</strong> Global popularity rewards songs popular "
    "across all genres, which is weakly related to any specific seed. Genre-conditioning "
    "restricts the popularity signal to songs that share the seed's musical context, "
    "increasing the probability of double-contribution items in the RRF fusion.</p>"
    "<p><strong>Why RRF over min-max blending?</strong> The cold-start content and popularity "
    "pools have near-zero overlap with global top songs; min-max normalization is unstable "
    "when pool overlap is sparse. RRF is order-statistic-based and degrades gracefully "
    "when only one pool contributes.</p>"
    "<p><strong>Fallback behavior:</strong> If the seed genre is Unknown or absent from the "
    "metadata catalog, the system falls back to global top-50 popularity, ensuring cold-start "
    "always returns exactly <em>k</em> recommendations.</p>",
)
