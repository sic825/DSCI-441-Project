import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from styling import inject, figcap, callout

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")
inject()

ROOT = Path(__file__).parent.parent
R    = ROOT / "results"

st.title("Results")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; color:#555;">'
    'Empirical evaluation across warm, cold-start, ablation, and diversity protocols'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── 4.1 Warm-User Performance ─────────────────────────────────────────────────
st.markdown("## 1. Warm-User Performance")

st.image(str(R / "poster_three_protocol_comparison.png"), use_container_width=True)
figcap(
    "Figure 4: NDCG@10 with 95% bootstrap confidence intervals across the three "
    "evaluation protocols. Hybrid (adaptive) and CF are statistically indistinguishable "
    "on both warm protocols (p = 0.72 and p = 0.70). Content-only is at floor on warm "
    "evaluation because the test users' held-out songs are rarely in the 7,611-song "
    "content catalog."
)

st.markdown("""
On both warm protocols, Hybrid (adaptive) is statistically indistinguishable from
the CF baseline. Standard warm evaluation yields NDCG@10 = 0.1099 for Hybrid
versus 0.1100 for CF (Δ = −0.00004, p = 0.72, *d* = −0.006 [negligible]).
The same result holds under long-tail correction (Δ = −0.00004, p = 0.70,
*d* = −0.006). Both models significantly outperform the Popularity baseline
(NDCG@10 = 0.0196, p < 0.001, *d* = 0.61 [medium]) and the Content-only
baseline (NDCG@10 ≈ 0.0005, essentially at floor).
""")

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

# ── 4.2 Cold-Start Performance ────────────────────────────────────────────────
st.markdown("## 2. Cold-Start Performance")

st.image(str(R / "poster_coldstart_standard_vs_longtail.png"), use_container_width=True)
figcap(
    "Figure 5: Cold-start NDCG@10 under standard evaluation (left) and long-tail "
    "correction (right, Steck 2011). Popularity-based models collapse to zero under "
    "long-tail correction; Hybrid is the only model that achieves non-trivial "
    "performance under both protocols."
)

st.markdown("""
Under standard cold-start evaluation (all held-out items scored), Popularity
achieves NDCG@10 = 0.0998, outperforming Hybrid's 0.0278 (p < 0.001, *d* = −0.38
[small]). This apparent dominance of a non-personalized baseline is a textbook
instance of the popularity bias described by Steck (2011): popular songs appear
frequently in hold-out sets because users have genuinely listened to them, and a
model recommending popular songs exploits this correlation without any personalization.

Long-tail correction reveals the true picture. After removing the top-100 globally
most-played songs from held-out ground truth, Popularity collapses to NDCG@10 = 0.000,
while Hybrid achieves NDCG@10 = 0.0168 (p < 0.001, *d* = 0.33 [small]). Content-only
achieves 0.0025, so Hybrid represents a 6.7× relative improvement (p < 0.001,
*d* = 0.29 [small]).
""")

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

# ── 4.3 Blending Strategy Ablation ───────────────────────────────────────────
st.markdown("## 3. Ablation Study")

st.image(str(R / "poster_warm_hybrid_ablation.png"), use_container_width=True)
figcap(
    "Figure 6: NDCG@10 across three hybrid blending variants versus the CF baseline under "
    "standard (left) and long-tail (right) warm evaluation. Adaptive alpha (Δ = −0.00004, "
    "p = 0.69) is the only variant statistically equivalent to CF; Fixed α = 0.5 and RRF "
    "are significantly worse under both protocols (p < 0.0001 and p = 0.0002 respectively)."
)

st.markdown("""
To validate the choice of adaptive alpha, we compare three warm-path variants —
Adaptive (current default), Fixed *α* = 0.5, and Reciprocal Rank Fusion — against
the CF ceiling on the same 1,000-user hold-out. Adaptive achieves Δ = −0.00004
(p = 0.69, *d* = −0.006 [negligible]), confirming equivalence to CF. Fixed *α* = 0.5
and RRF are both significantly worse than CF (p < 0.0001 for Fixed, p = 0.0002 for
RRF) with *d* = −0.192 and *d* = −0.157 respectively (approaching small effect).
The same ordering holds under long-tail correction.

The empirical result validates the design rationale: equal content weighting dilutes
CF signal for the warm population, where CF is the stronger model. Adaptive alpha
adapts correctly by assigning CF-dominant weights to all users in the test set.
""")
st.divider()

# ── 4.4 Recommendation Diversity ─────────────────────────────────────────────
st.markdown("## 4. Recommendation Diversity")

st.image(str(R / "poster_diversity_comparison.png"), use_container_width=True)
figcap(
    "Figure 7: Diversity metrics (ILS, genre coverage, catalog coverage) for all four "
    "models at k = 10 with 95% bootstrap confidence intervals. Content ILS ≈ 0.90 "
    "reflects near-duplicate audio feature clusters. CF and Hybrid are statistically "
    "indistinguishable on all three metrics."
)

st.markdown("""
We report three diversity metrics at *k* = 10: Intra-List Similarity (ILS),
genre coverage, and catalog coverage. Content recommendations are highly similar
(ILS = 0.904), consistent with the known near-duplicate contamination in the
Spotify audio feature space — the k-NN index finds acoustically near-identical
tracks. CF and Hybrid produce substantially more diverse lists (ILS ≈ 0.27),
with confidence intervals fully overlapping. Genre coverage is similar across all
models (≈ 0.25–0.30 unique genres per 10 recommendations) with no significant
differences. Hybrid recommendations appear in the metadata catalog 0.025 songs/list
more often than CF recommendations (p = 0.004), a statistically detectable but
negligible effect (*d* = 0.099). Hybrid offers no meaningful diversity advantage
over CF alone, consistent with alpha-bounding limiting the content contribution.
""")
st.divider()

# ── 5. Conclusions and Limitations ───────────────────────────────────────────
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
