import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from styling import inject, figcap, callout

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")
inject()

ROOT = Path(__file__).parent.parent
R    = ROOT / "results"


def _metric_row(cards):
    """Render a horizontal row of styled metric cards. cards = [(label, value, subtitle)]."""
    cols = st.columns(len(cards))
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div style="background:#F8FAFC;border:1px solid #D0E0F0;border-radius:8px;'
                f'padding:14px 16px;text-align:center;min-height:100px;">'
                f'<div style="font-family:Helvetica Neue,Arial,sans-serif;font-size:0.72em;'
                f'color:#555;text-transform:uppercase;letter-spacing:0.08em;'
                f'margin-bottom:6px;">{label}</div>'
                f'<div style="font-family:Helvetica Neue,Arial,sans-serif;font-size:1.6em;'
                f'font-weight:700;color:#1F4E79;line-height:1.1;">{value}</div>'
                f'<div style="font-family:Helvetica Neue,Arial,sans-serif;font-size:0.78em;'
                f'color:#888;font-style:italic;margin-top:5px;">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


st.title("Results")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; color:#555;">'
    'Empirical evaluation across warm, cold-start, ablation, and diversity protocols'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── §1 Warm-User Performance ──────────────────────────────────────────────────
st.markdown("## 1. Warm-User Performance")

st.markdown("""
Hybrid achieves statistical equivalence with the CF ceiling on warm-user evaluation,
and the result is robust to the long-tail popularity-bias correction. The adaptive
alpha mechanism assigns CF-dominant weights to all test users (α ∈ [0.593, 0.946]),
so the two models produce nearly identical ranked lists. The content-only baseline
sits near floor: the 7,611-song content catalog covers only 12.5% of the CF song
space, making content-indexed songs rare in test users' held-out sets.
""")

_metric_row([
    ("NDCG@10 Hybrid",  "0.1099",   "CI [0.1008, 0.1196]"),
    ("NDCG@10 CF",      "0.1100",   "CI [0.1006, 0.1195]"),
    ("Δ (Hybrid − CF)", "−0.00004", "p = 0.72"),
    ("Cohen's d",       "−0.006",   "negligible"),
])

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
st.image(str(R / "poster_three_protocol_comparison.png"), use_container_width=True)
figcap(
    "Figure 4: NDCG@10 with 95% bootstrap confidence intervals across the three "
    "evaluation protocols. Hybrid (adaptive) and CF are statistically indistinguishable "
    "on both warm protocols (p = 0.72 and p = 0.70). Content-only is at floor on warm "
    "evaluation because the test users' held-out songs are rarely in the 7,611-song "
    "content catalog."
)

st.markdown("**Table 1.** Warm-user NDCG@10 (standard protocol, *n* = 1,000).")
st.markdown("""
| Model | NDCG@10 | 95% CI |
|---|---|---|
| CF (ALS) | 0.1100 | [0.1006, 0.1195] |
| **Hybrid** | **0.1099** | [0.1008, 0.1196] |
| Popularity | 0.0196 | [0.0159, 0.0236] |
| Content | 0.0005 | [0.0000, 0.0012] |
""")

callout(
    "Structural Explanation",
    "<p>The adaptive alpha distribution of the test population explains Hybrid ≈ CF. "
    "Because the CF training pipeline filters to users with at least 20 interactions, "
    "every test user has <em>n</em> ≥ 20, which corresponds to α ≥ 0.59. The empirically "
    "observed range is α ∈ [0.593, 0.946] with a median of 0.767. At these weights, the "
    "content contribution (1 − α) ≤ 0.41 is small relative to CF, and the content "
    "catalog's 12.5% overlap with the CF song space further attenuates the blending effect. "
    "The system performs exactly as designed: for high-interaction users, CF dominates and "
    "the content component adds marginal diversity at no accuracy cost.</p>",
)
st.divider()

# ── §2 Cold-Start Performance ──────────────────────────────────────────────────
st.markdown("## 2. Cold-Start Performance")

st.markdown("""
Standard cold-start evaluation places the non-personalized Popularity baseline first.
This is a popularity bias artifact: heavy-listener test sets contain globally popular
songs simply because those users have listened to them, making popularity an
accidentally effective baseline under this protocol. Long-tail correction removes
the top-100 globally most-played songs from ground truth, eliminating the artifact.
Under this bias-corrected evaluation, Popularity collapses to zero, Content-only barely
registers, and Hybrid is the only model achieving meaningful performance.
""")

_metric_row([
    ("NDCG@10 Hybrid (LT)",  "0.0168", "p < 0.001 vs Popularity"),
    ("NDCG@10 Pop. (LT)",    "0.000",  "collapses under correction"),
    ("NDCG@10 Content (LT)", "0.0025", "CI [0.0013, 0.0042]"),
    ("Relative improvement", "6.7×",   "Hybrid vs Content, d = 0.29"),
])

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
st.image(str(R / "poster_coldstart_standard_vs_longtail.png"), use_container_width=True)
figcap(
    "Figure 5: Cold-start NDCG@10 under standard evaluation (left) and long-tail "
    "correction (right, Steck 2011). Popularity-based models collapse to zero under "
    "long-tail correction; Hybrid is the only model that achieves non-trivial "
    "performance under both protocols."
)

st.markdown("**Table 2.** Cold-start NDCG@10 (*n* = 871).")
st.markdown("""
| Model | Standard | Long-tail |
|---|---|---|
| Popularity | 0.0998 | 0.000 |
| CF Cold-Start | 0.0998 | 0.000 |
| **Hybrid** | **0.0278** | **0.0168** |
| Content | 0.0029 | 0.0025 |
""")

callout(
    "Methodological Caveat",
    "<p>Popularity collapsing to zero under long-tail correction is partly tautological: "
    "the exclusion criterion targets precisely the songs it recommends. The Hybrid result "
    "is more robust, as its content k-NN component recommends songs outside the top-100 "
    "long tail. We report both protocols to give an honest picture — the standard result "
    "shows the practical user-facing landscape, while long-tail shows performance under "
    "a popularity-bias-corrected evaluation.</p>",
)
st.divider()

# ── §3 Ablation Study ─────────────────────────────────────────────────────────
st.markdown("## 3. Ablation Study")

st.markdown("""
Three warm-path blending strategies were evaluated against the CF ceiling on the same
1,000-user hold-out. Only adaptive alpha achieves statistical equivalence with CF.
Fixed equal weighting (α = 0.5) and Reciprocal Rank Fusion both significantly degrade
warm-user accuracy. The mechanism is direct: the adaptive formula correctly assigns
CF-dominant weights to all test users, while fixed equal weighting dilutes the stronger
CF signal regardless of interaction count. The same ordering holds under long-tail
correction.
""")

_metric_row([
    ("Adaptive α vs CF",    "p = 0.69",   "Δ = −0.00004, d = −0.006"),
    ("Fixed α = 0.5 vs CF", "p < 0.0001", "d = −0.192 (negligible)"),
    ("RRF vs CF",           "p = 0.0002", "d = −0.157 (negligible)"),
])

callout(
    "Note on p-value",
    "<p>The Adaptive α vs CF p-value here (0.69) differs slightly from the p = 0.72 reported "
    "in §1 for the same comparison. Both come from independent 10,000-resample bootstrap runs "
    "on the same data split; the ±0.03 gap reflects bootstrap variance rather than a "
    "methodological discrepancy. The conclusion is identical in both: non-significant, "
    "negligible effect size.</p>",
)

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
st.image(str(R / "poster_warm_hybrid_ablation.png"), use_container_width=True)
figcap(
    "Figure 6: NDCG@10 across three hybrid blending variants versus the CF baseline under "
    "standard (left) and long-tail (right) warm evaluation. Adaptive alpha (Δ = −0.00004, "
    "p = 0.69) is the only variant statistically equivalent to CF; Fixed α = 0.5 and RRF "
    "are significantly worse under both protocols (p < 0.0001 and p = 0.0002 respectively)."
)
st.divider()

# ── §4 Recommendation Diversity ───────────────────────────────────────────────
st.markdown("## 4. Recommendation Diversity")

st.markdown("""
Hybrid and CF produce recommendations with statistically indistinguishable diversity
across all three metrics: intra-list similarity, genre coverage, and catalog coverage.
The one detectable difference — Hybrid recommendations appear in the Spotify metadata
catalog 0.025 songs per list more than CF — is statistically significant but negligible
in magnitude. Content-only recommendations cluster tightly because the Spotify audio
feature space contains many near-duplicate tracks with similarity ≈ 1.0.
""")

_metric_row([
    ("ILS Hybrid",               "≈ 0.27",  "≈ CF, no sig. difference"),
    ("ILS Content",               "0.904",   "near-duplicate clusters"),
    ("Coverage Δ (Hybrid − CF)", "+0.025",  "p = 0.004, d = 0.099"),
    ("Genre Coverage",            "≈ 0.27",  "no sig. difference across models"),
])

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
st.image(str(R / "poster_diversity_comparison.png"), use_container_width=True)
figcap(
    "Figure 7: Diversity metrics (ILS, genre coverage, catalog coverage) for all four "
    "models at k = 10 with 95% bootstrap confidence intervals. Content ILS ≈ 0.90 "
    "reflects near-duplicate audio feature clusters. CF and Hybrid are statistically "
    "indistinguishable on all three metrics."
)
st.divider()

# ── §5 Conclusions and Limitations ────────────────────────────────────────────
st.markdown("## 5. Conclusions and Limitations")
st.markdown("""
We draw three conclusions. First, for users with sufficient interaction history,
the hybrid system matches CF accuracy without degradation — the adaptive alpha
strategy correctly learns to up-weight CF for experienced users, so no ranking
quality is sacrificed relative to the CF ceiling. Second, the hybrid's cold-start
mode meaningfully outperforms both content-only and popularity baselines under
fair long-tail evaluation, with the genre-conditioned RRF providing a non-trivial
personalization signal from audio features alone. Third, the choice of blending
strategy matters: fixed-weight and rank-fusion variants both degrade warm accuracy
relative to CF, confirming that strategy selection should be an explicit design
decision rather than an implementation detail.
""")

callout(
    "Limitations",
    "<p><strong>Catalog overlap.</strong> The 12.5% catalog overlap (7,611 content-indexed "
    "songs out of 98,485 CF songs) means that warm hybrid recommendations are predominantly "
    "CF-sourced; the content component can only contribute when the CF candidate pool "
    "intersects the content catalog.</p>"
    "<p><strong>Alpha floor.</strong> The adaptive alpha floor of α ≥ 0.59 (set by the "
    "minimum-20-interaction user filter) means the system was never evaluated on genuinely "
    "cold-warm boundary users where content should provide the most benefit. A supplementary "
    "study drawing from users with 1–10 interactions would be more informative on this axis.</p>"
    "<p><strong>Tautological exclusion.</strong> The long-tail evaluation result for the "
    "Popularity model is partly tautological: the exclusion criterion removes exactly the "
    "songs that Popularity recommends by design, so the zero NDCG is not purely a reflection "
    "of inability to personalize.</p>"
    "<p><strong>Single hold-out split.</strong> The evaluation relies on a single "
    "leave-fraction-out split per user; leave-<em>k</em>-out replication across multiple "
    "random seeds would provide more reliable variance estimates.</p>",
)
