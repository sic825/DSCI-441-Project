import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from styling import inject, figcap

st.set_page_config(page_title="Overview", page_icon="📄", layout="wide")
inject()

# ── Title block ───────────────────────────────────────────────────────────────
st.title("A Hybrid Music Recommendation System")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; font-size:1.0em; color:#555; margin-bottom:0.2em;">'
    'Integrating Collaborative Filtering and Content-Based Filtering with '
    'Adaptive Blending and Genre-Conditioned Reciprocal Rank Fusion'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown("""
<div style="font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.4; margin-bottom: 0.6rem;">
<p style="font-size:1.05em; font-style:italic; margin:0; color:#1A1A1A;">Simon Chen, Thoi Quach</p>
<p style="font-size:0.95em; margin:0.2em 0 0 0; color:#666;">Lehigh University &nbsp;·&nbsp; DSCI 441: Statistical Machine Learning</p>
<p style="font-size:0.85em; margin:0.2em 0 0 0; color:#888;">Spring 2026</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Abstract ──────────────────────────────────────────────────────────────────
st.markdown("## Abstract")
st.markdown("""
We present a hybrid music recommendation system that combines implicit-feedback
Alternating Least Squares (ALS) collaborative filtering with audio-feature-based
k-nearest-neighbor content filtering. The system addresses two persistent challenges
in offline recommender evaluation: the cold-start problem for new users without
interaction history, and popularity bias that inflates offline metrics for
non-personalized baselines.

For warm users (those with sufficient interaction history), the system employs
an adaptive alpha blending strategy in which the CF weight increases
monotonically with the user's interaction count via a sigmoid function. For
cold-start users, recommendations are produced by Reciprocal Rank Fusion
(Cormack et al., 2009) over a content k-NN list and a genre-conditioned
popularity prior.

We evaluate across three protocols: warm-standard, warm-long-tail (Steck, 2011),
and cold-start. Bootstrap 95% confidence intervals and paired bootstrap hypothesis
tests (10,000 resamples) with Cohen's *d* effect sizes accompany all reported
metrics. Three headline findings emerge: (1) Hybrid achieves NDCG@10 = 0.1099,
statistically indistinguishable from the CF ceiling (p = 0.72, *d* = −0.006
[negligible]); (2) on cold-start long-tail evaluation, Hybrid achieves a 6.7×
relative NDCG improvement over Content-only (p < 0.001, *d* = 0.29 [small]);
(3) adaptive alpha is the only blending strategy that does not significantly degrade
warm-user accuracy.
""")
st.divider()

# ── Project Arc ───────────────────────────────────────────────────────────────
st.markdown("## Project Arc")
st.markdown("""
In Milestone 1, we built two independent recommenders: an implicit-feedback ALS
model on the Million Song Dataset Taste Profile (661,089 users × 98,485 songs) and
a content-based k-NN on Spotify audio features (114,000 tracks). Initial evaluation
highlighted three issues: variant-track contamination in content k-NN, NaN metadata
in CF outputs due to a 12.5% catalog overlap, and the absence of any actual hybrid
model.

In Milestone 2, we (1) refactored the pipeline into reusable `src/` modules and
fixed the M1 bugs; (2) implemented two distinct hybrid blending strategies (adaptive
min-max for warm users, RRF + genre-conditioned popularity for cold-start);
(3) built a rigorous three-protocol evaluation including Steck-style
popularity-bias correction; (4) ran an ablation study validating adaptive alpha as
the only blending strategy that does not degrade warm accuracy; (5) deployed the
system as the interactive Streamlit app on this site.
""")
st.divider()

# ── Headline Findings ─────────────────────────────────────────────────────────
st.markdown("## Headline Findings")
st.markdown("""
**Warm-user equivalence.** On both the standard and long-tail warm protocols, the
Hybrid (adaptive) system is statistically indistinguishable from the CF ceiling
(NDCG@10 = 0.1099 vs 0.1100, p = 0.72, *d* = −0.006 [negligible]). The adaptive
alpha distribution for the 1,000-user test population ranges from 0.593 to 0.946
(median 0.767), so the CF component dominates for every test user. The content
contribution adds marginal catalog diversity at zero accuracy cost. Ablation
confirms this is not accidental: fixed equal weighting (α = 0.5) and Reciprocal
Rank Fusion both significantly degrade warm accuracy relative to CF (p < 0.0001
and p = 0.0002 respectively), establishing that adaptive weighting is a necessary
design choice.

**Cold-start advantage under bias correction.** Under standard cold-start
evaluation, the non-personalized Popularity baseline achieves NDCG@10 = 0.0998,
outperforming Hybrid's 0.0278. This is a textbook instance of popularity bias
(Steck, 2011): popular songs appear disproportionately in held-out sets regardless
of personalization quality. Long-tail correction (top-100 globally most-played songs
excluded from ground truth) reverses the picture entirely: Popularity collapses to
NDCG@10 = 0.000 while Hybrid achieves 0.0168, a 6.7× relative improvement over
Content-only (p < 0.001, *d* = 0.29 [small]). Under fair evaluation the
genre-conditioned RRF cold-start strategy provides a meaningful personalization
signal from audio features alone.

**Diversity and catalog coverage.** Hybrid and CF produce recommendations with
statistically indistinguishable intra-list similarity (ILS ≈ 0.27) and genre
coverage (≈ 0.27 unique genres per 10-item list). Content-only recommendations
cluster tightly (ILS = 0.904), a consequence of near-duplicate audio feature
vectors in the Spotify catalog. Hybrid recommendations appear in the Spotify
metadata catalog 0.025 songs per list more than CF recommendations (p = 0.004,
*d* = 0.099 [negligible]), the only statistically detectable diversity difference.
The structural explanation is alpha-bounding: with α ≥ 0.59 for every test user,
the content component's catalog contribution is small relative to the CF pool.
""")
st.divider()

# ── Headline metrics (quick-reference summary) ────────────────────────────────
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

# ── Navigation ────────────────────────────────────────────────────────────────
st.markdown(
    "*Navigate via the sidebar: **Model Architecture** (system design and scoring formulas) · "
    "**Statistical Analysis** (evaluation protocols and hypothesis testing) · "
    "**Results** (figures, tables, and conclusions) · "
    "**Live Demo** (interactive recommendation explorer) · "
    "**About** (team, acknowledgments, and references).*"
)
