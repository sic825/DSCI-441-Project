import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pages._style import inject, figcap

st.set_page_config(page_title="Overview", page_icon="📄", layout="wide")
inject()

ROOT = Path(__file__).parent.parent
R    = ROOT / "results"

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("A Hybrid Music Recommendation System")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; font-size:1.0em; color:#555;">'
    'Integrating Collaborative Filtering and Content-Based Filtering with '
    'Adaptive Blending and Genre-Conditioned Reciprocal Rank Fusion'
    '</p>',
    unsafe_allow_html=True,
)
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
monotonically with the user's interaction count via a sigmoid function. For cold-start
users, recommendations are produced by Reciprocal Rank Fusion (Cormack et al., 2009)
over a content k-NN list and a genre-conditioned popularity prior.

We evaluate across three protocols: warm-standard, warm-long-tail (Steck, 2011
popularity bias correction), and cold-start. Bootstrap 95% confidence intervals
and paired bootstrap hypothesis tests (10,000 resamples) with Cohen's *d* effect
sizes accompany all reported metrics. Three headline findings emerge:
(1) Hybrid achieves NDCG@10 = 0.1099, statistically indistinguishable from the
CF ceiling (p = 0.72, *d* = −0.006 [negligible]);
(2) on cold-start long-tail evaluation, Hybrid achieves a 6.7× relative NDCG
improvement over Content-only (p < 0.001, *d* = 0.29 [small]);
(3) adaptive alpha is the only blending strategy that does not significantly
degrade warm-user accuracy. Structural limitations include a 12.5% content
catalog overlap with the CF song space and an empirical alpha floor of 0.60
imposed by the user-activity filter applied during model training.
""")
st.divider()

# ── 1. Introduction ───────────────────────────────────────────────────────────
st.markdown("## 1. Introduction")
st.markdown("""
Collaborative filtering systems trained on implicit feedback exploit the co-occurrence
structure of user–item interactions to surface personalized recommendations. In music
streaming, this approach scales to hundreds of millions of users and produces highly
accurate rankings for experienced users. However, it offers nothing to new users with
no interaction history — the canonical *cold-start problem*. Content-based methods,
which represent items by their intrinsic audio or textual features, can recommend
without any user history but have no mechanism to learn individual preferences.
A second, less-discussed failure mode afflicts both approaches in offline evaluation:
*popularity bias*. Standard leave-one-out or hold-out evaluation protocols reward
models that recommend globally popular songs, because popular items are
disproportionately likely to appear in any user's held-out ground truth. This
inflates the measured quality of non-personalized popularity baselines relative to
their practical utility (Steck, 2011).

This paper presents a hybrid system that combines an ALS collaborative filter
operating on the Million Song Dataset Taste Profile (661,089 users, 98,485 songs)
with a k-NN content model trained on Spotify audio features for 7,611 cross-dataset
matched tracks. We ask three research questions: (RQ1) Can hybrid blending improve
over CF-alone for warm users? (RQ2) Does the hybrid's cold-start mode outperform
content and popularity baselines under fair evaluation? (RQ3) Which blending
strategy best preserves warm-user accuracy?
""")
st.divider()

# ── 2. Methodology Summary ────────────────────────────────────────────────────
st.markdown("## 2. Methodology Summary")
st.markdown("""
The collaborative filter is an Alternating Least Squares model (Hu et al., 2008)
trained with 64 latent factors, regularization 0.1, and 20 iterations on
log₁₊-transformed, alpha-scaled play counts. The content model is a scikit-learn
brute-force cosine-similarity k-nearest-neighbor index fitted on nine z-score
normalized audio features (danceability, energy, loudness, speechiness,
acousticness, instrumentalness, liveness, valence, tempo) for 7,611 deduplicated
tracks matched between the MSD metadata and the Spotify Tracks Dataset.

For warm users, the hybrid blends min-max normalized CF and content candidate
scores with an adaptive weight *α* = σ((log(1 + *n*) − 2) / 1.5), where *n*
is the user's training interaction count and σ is the sigmoid function. For
cold-start users, Reciprocal Rank Fusion (RRF, *k* = 60) merges the content
k-NN ranking with a genre-conditioned popularity ranking derived from the
seed song's Spotify genre tag. Full algorithmic detail, interactive plots,
and annotated formulas are available on the *How It Works* page.
""")
st.divider()

# ── 3. Evaluation Protocols ───────────────────────────────────────────────────
st.markdown("## 3. Evaluation Protocols")
st.markdown("""
All warm-user evaluation uses an 80/20 interaction hold-out on a stratified
sample of 1,000 active users drawn from the training corpus. Cold-start evaluation
draws a separate sample of 871 users whose most-played in-catalog song serves as
the sole seed. NDCG@10, HitRate@10, and Recall@10 are computed per user and
aggregated with 1,000-resample bootstrap 95% confidence intervals (percentile method).
Paired bootstrap hypothesis tests (10,000 resamples, two-sided, centered under H₀)
and Cohen's *d* = mean(Δ)/std(Δ, ddof=1) accompany every comparison.

Two long-tail correction protocols follow Steck (2011): the top-100 globally
most-played songs are excluded from held-out ground truth before NDCG scoring.
This removes the evaluation advantage of models that recommend popular hits, and
serves as a stress-test for the warm ablation and the cold-start comparisons.
Effect sizes are classified as negligible (*d* < 0.2), small (0.2–0.5),
medium (0.5–0.8), or large (≥ 0.8) following Cohen (1988).
""")
st.divider()

# ── 4. Results ────────────────────────────────────────────────────────────────
st.markdown("## 4. Results")

# 4.1 Warm
st.markdown("### 4.1  Warm-User Performance")
st.markdown("""
On both warm protocols, Hybrid (adaptive) is statistically indistinguishable from
the CF baseline. Standard warm evaluation yields NDCG@10 = 0.1099 for Hybrid
versus 0.1100 for CF (Δ = −0.00004, p = 0.72, *d* = −0.006 [negligible]).
The same result holds under long-tail correction (Δ = −0.00004, p = 0.70,
*d* = −0.006). Both models significantly outperform the Popularity baseline
(NDCG@10 = 0.0196, p < 0.001, *d* = 0.61 [medium]) and the Content-only
baseline (NDCG@10 ≈ 0.0005, essentially at floor).

The structural explanation for Hybrid ≈ CF is the adaptive alpha distribution
of the test population. Because the CF training pipeline filters to users with
at least 20 interactions, every test user has *n* ≥ 20, which corresponds to
α ≥ 0.59. The empirically observed range is α ∈ [0.593, 0.946] with a median of
0.767. At these weights, the content contribution (1 − α) ≤ 0.41 is small
relative to CF, and the content catalog's 12.5% overlap with the CF song space
further attenuates the blending effect. The system performs exactly as designed:
for high-interaction users, CF dominates and the content component adds marginal
diversity at no accuracy cost.
""")

col_img, col_txt = st.columns([5, 3])
with col_img:
    st.image(str(R / "poster_three_protocol_comparison.png"), use_container_width=True)
    figcap(
        "Figure 1: NDCG@10 with 95% bootstrap confidence intervals across the three "
        "evaluation protocols. Hybrid (adaptive) and CF are statistically indistinguishable "
        "on both warm protocols (p = 0.72 and p = 0.70). Content-only is at floor on warm "
        "evaluation because the test users' held-out songs are rarely in the 7,611-song "
        "content catalog."
    )
with col_txt:
    st.markdown("**Table 1.** Warm-user NDCG@10 (standard protocol, *n* = 1,000).")
    st.markdown("""
| Model | NDCG@10 | 95% CI |
|---|---|---|
| CF (ALS) | 0.1100 | [0.1006, 0.1195] |
| **Hybrid** | **0.1099** | [0.1008, 0.1196] |
| Popularity | 0.0196 | [0.0159, 0.0236] |
| Content | 0.0005 | [0.0000, 0.0012] |
""")

# 4.2 Cold-start
st.markdown("### 4.2  Cold-Start Performance")
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
*d* = 0.29 [small]). We note an important caveat: Popularity collapsing to zero is
partly tautological — the exclusion criterion targets precisely the songs it
recommends. The Hybrid result is more robust, as its content k-NN component
recommends songs outside the top-100 long tail.
""")

col_img2, col_txt2 = st.columns([5, 3])
with col_img2:
    st.image(str(R / "poster_coldstart_standard_vs_longtail.png"), use_container_width=True)
    figcap(
        "Figure 2: Cold-start NDCG@10 under standard evaluation (left) and long-tail "
        "correction (right, Steck 2011). Popularity-based models collapse to zero under "
        "long-tail correction; Hybrid is the only model that achieves non-trivial "
        "performance under both protocols."
    )
with col_txt2:
    st.markdown("**Table 2.** Cold-start NDCG@10 (*n* = 871).")
    st.markdown("""
| Model | Standard | Long-tail |
|---|---|---|
| Popularity | 0.0998 | 0.000 |
| CF Cold-Start | 0.0998 | 0.000 |
| **Hybrid** | **0.0278** | **0.0168** |
| Content | 0.0029 | 0.0025 |
""")

# 4.3 Ablation
st.markdown("### 4.3  Blending Strategy Ablation")
st.markdown("""
To validate the choice of adaptive alpha, we compare three warm-path variants —
Adaptive (current default), Fixed *α* = 0.5, and Reciprocal Rank Fusion — against
the CF ceiling on the same 1,000-user hold-out. Adaptive achieves Δ = −0.00004
(p = 0.69, *d* = −0.006 [negligible]), confirming equivalence to CF. Fixed *α* = 0.5
and RRF are both significantly worse than CF (p < 0.0001 for Fixed, p = 0.0002 for
RRF) with *d* = −0.192 and *d* = −0.157 respectively (approaching small effect).
The same ordering holds under long-tail correction. The empirical result validates
the design rationale: equal content weighting dilutes CF signal for the warm
population, where CF is the stronger model. Adaptive alpha adapts correctly by
assigning CF-dominant weights to all users in the test set.
""")

st.image(str(R / "poster_warm_hybrid_ablation.png"), use_container_width=True)
figcap(
    "Figure 3: NDCG@10 across three hybrid blending variants versus the CF baseline under "
    "standard (left) and long-tail (right) warm evaluation. Adaptive alpha (Δ = −0.00004, "
    "p = 0.69) is the only variant statistically equivalent to CF; Fixed *α* = 0.5 and RRF "
    "are significantly worse under both protocols (p < 0.0001 and p = 0.0002 respectively)."
)

# 4.4 Diversity
st.markdown("### 4.4  Recommendation Diversity")
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

st.image(str(R / "poster_diversity_comparison.png"), use_container_width=True)
figcap(
    "Figure 4: Diversity metrics (ILS, genre coverage, catalog coverage) for all four "
    "models at k = 10 with 95% bootstrap confidence intervals. Content ILS ≈ 0.90 "
    "reflects near-duplicate audio feature clusters. CF and Hybrid are statistically "
    "indistinguishable on all three metrics."
)

st.divider()

# ── 5. Conclusions ────────────────────────────────────────────────────────────
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

Several structural limitations constrain these findings. The 12.5% catalog overlap
(7,611 content-indexed songs out of 98,485 CF songs) means that warm hybrid
recommendations are predominantly CF-sourced; the content component can only
contribute when the CF candidate pool intersects the content catalog. The adaptive
alpha floor of *α* ≥ 0.59 (set by the minimum-20-interaction user filter) means
the system was never evaluated on genuinely cold-warm boundary users where content
should provide the most benefit; a supplementary study drawing from users with
1–10 interactions would be more informative on this axis. The long-tail evaluation
result for the Popularity model is partly tautological: the exclusion criterion
removes exactly the songs that Popularity recommends by design, so the zero NDCG
is not purely a reflection of the model's inability to personalize. Finally, the
evaluation relies on a single leave-fraction-out split per user; leave-*k*-out
replication across multiple random seeds would provide more reliable variance
estimates.
""")
st.divider()

# ── References ────────────────────────────────────────────────────────────────
st.markdown("## References")
st.markdown("""
1. Bertin-Mahieux, T., Ellis, D. P. W., Whitman, B., & Lamere, P. (2011).
   The Million Song Dataset. *Proceedings of the 12th International Society
   for Music Information Retrieval Conference (ISMIR 2011)*, Miami, FL.

2. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
   (2nd ed.). Lawrence Erlbaum Associates.

3. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank
   Fusion Outperforms Condorcet and Individual Rank Learning Methods.
   *Proceedings of the 32nd International ACM SIGIR Conference on Research and
   Development in Information Retrieval*, 758–759.

4. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit
   Feedback Datasets. *Proceedings of the 8th IEEE International Conference on
   Data Mining (ICDM 2008)*, 263–272.

5. Maharshipandya (2023). Spotify Tracks Dataset.
   HuggingFace Datasets. 114,000 tracks across 114 genres.

6. Steck, H. (2011). Item Popularity and Recommendation Accuracy.
   *Proceedings of the 5th ACM Conference on Recommender Systems (RecSys 2011)*,
   125–132.
""")
