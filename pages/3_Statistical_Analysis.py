import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from styling import inject, callout

st.set_page_config(page_title="Statistical Analysis", page_icon="📊", layout="wide")
inject()

st.title("Statistical Analysis")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; color:#555;">'
    'Evaluation protocols, confidence intervals, and hypothesis testing methodology'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── 1. Evaluation Protocols ───────────────────────────────────────────────────
st.markdown("## 1. Evaluation Protocols")
st.markdown("""
All warm-user evaluation uses an 80/20 interaction hold-out on a stratified
sample of 1,000 active users drawn from the training corpus. Cold-start evaluation
draws a separate sample of 871 users whose most-played in-catalog song serves as
the sole seed. NDCG@10, HitRate@10, and Recall@10 are computed per user and
aggregated with bootstrap 95% confidence intervals.

Four distinct protocols are used across the evaluation suite:
""")
st.markdown(
    '<div style="margin-left:1.5em;">'
    '<p><strong>1. Warm-standard.</strong> 80/20 interaction hold-out; all held-out items eligible for scoring. Tests whether the hybrid blending improves over CF-alone when a rich interaction history is available. n = 1,000 users.</p>'
    '<p><strong>2. Warm-long-tail.</strong> Same 80/20 split; top-100 globally most-played songs removed from held-out ground truth before NDCG scoring. This protocol follows Steck (2011) and removes the evaluation advantage of models that recommend popular hits, providing a popularity-bias-corrected view of ranking quality. n ≈ 999 users (users whose entire held-out set consists of top-100 songs are excluded).</p>'
    '<p><strong>3. Cold-start standard.</strong> A separate cohort; the user\'s most-played song in the content catalog (7,611 songs) serves as the sole input seed. Held-out ground truth is unrestricted. Tests whether the genre-conditioned RRF cold-start strategy produces better personalized rankings than content or popularity baselines. n = 871.</p>'
    '<p><strong>4. Cold-start long-tail.</strong> Same seed selection and cohort; top-100 globally most-played songs excluded from ground truth. This is the most stringent protocol: it removes the popularity shortcut and forces evaluation on songs that require genuine personalization signal to surface. n ≈ 869.</p>'
    '</div>',
    unsafe_allow_html=True,
)

callout(
    "Why Long-Tail Correction Matters",
    "<p>Under standard evaluation, a model that always recommends globally popular songs "
    "scores well because popular items appear disproportionately in any user's held-out "
    "set — not because they are genuinely personalized. Steck (2011) showed this inflates "
    "measured quality for non-personalized baselines. Long-tail correction exposes this "
    "artifact: the Popularity baseline achieves NDCG@10 = 0.0998 under standard cold-start "
    "evaluation but collapses to 0.000 after exclusion, since the excluded songs are "
    "precisely what it recommends. The Hybrid result (0.0168 under long-tail) is not "
    "subject to this tautology.</p>",
)
st.divider()

# ── 2. Bootstrap Confidence Intervals ────────────────────────────────────────
st.markdown("## 2. Bootstrap Confidence Intervals")
st.markdown("""
All aggregate metric values are reported with 95% bootstrap confidence intervals.
For a sample of per-user metric values {x₁, x₂, …, xₙ}, we draw B = 1,000
bootstrap resamples with replacement from the user index and compute the sample
mean on each resample. The percentile method gives the confidence interval:
""")
st.latex(r"[\hat{\theta}_{0.025},\ \hat{\theta}_{0.975}]")
st.markdown("""
where the percentiles are taken over the B bootstrap means. This quantifies
uncertainty in the *mean across users*, not in individual-user point estimates.
Users are the resample unit throughout — each resample draws a new set of n users
(with replacement) from the evaluation pool, preserving the user-level structure
of the data.

The bootstrap CI is preferred over normal-theory intervals for three reasons:
(1) per-user metric distributions are skewed — most users score near zero, with
a long right tail of users who match many recommendations; (2) NDCG@K is
bounded and discrete for small K, violating normality assumptions; (3) the
percentile method makes no assumptions about the shape of the sampling distribution
and is consistent under mild regularity conditions.
""")
st.divider()

# ── 3. Paired Bootstrap Hypothesis Tests ─────────────────────────────────────
st.markdown("## 3. Paired Bootstrap Hypothesis Tests")
st.markdown("""
For every model comparison, we report a two-sided paired bootstrap hypothesis
test with B = 10,000 resamples. Let aᵢ and bᵢ be the NDCG@10 scores for user i
under models A and B respectively. The observed test statistic is the mean
per-user difference:
""")
st.latex(r"\bar{\delta}_\text{obs} = \frac{1}{n}\sum_{i=1}^{n}(a_i - b_i)")
st.markdown("""
Under H₀: E[aᵢ − bᵢ] = 0, the null distribution is estimated by centering each
bootstrap resample's mean difference at zero:
""")
st.latex(
    r"p = \frac{\#\left\{|\bar{\delta}^*| \geq |\bar{\delta}_\text{obs}|\right\}}{B},"
    r"\quad \bar{\delta}^* = \overline{(a_i^* - b_i^*)} - \bar{\delta}_\text{obs}"
)
st.markdown("""
The centering step — subtracting the observed mean from each bootstrap mean —
ensures the null distribution is centered at zero, giving a correctly calibrated
two-sided test under H₀. Without centering, the bootstrap p-value is a test of
whether the observed mean is extreme relative to a distribution centered at the
observed mean, which is vacuous.

Pairing is on user_id: both models must be evaluated on the same user for the
difference to be defined. In practice, all warm-path models share the identical
1,000-user hold-out split, and all cold-start models share the same 871-user
cohort, so the paired condition is always satisfied.
""")

callout(
    "Interpretation Convention",
    "<p>We adopt α = 0.05 as the significance threshold. However, statistical "
    "significance is never reported alone — every p-value is accompanied by Cohen's "
    "<em>d</em> effect size. A small p with a negligible effect (|<em>d</em>| < 0.2) "
    "is flagged as a statistically detectable but practically negligible difference. "
    "This is the case for all Hybrid vs CF warm comparisons.</p>",
)
st.divider()

# ── 4. Effect Sizes: Cohen's d ────────────────────────────────────────────────
st.markdown("## 4. Effect Sizes: Cohen's *d*")
st.markdown("""
Cohen's *d* for paired differences measures the practical magnitude of the
difference between two models, independent of sample size. For per-user
differences δᵢ = aᵢ − bᵢ:
""")
st.latex(r"d = \frac{\bar{\delta}}{\mathrm{std}(\delta,\ \mathrm{ddof}=1)}")
st.markdown("""
The denominator uses ddof = 1 (sample standard deviation) because the per-user
differences are treated as a sample from a population of potential users, not as
a census. The sign convention is A − B: positive *d* means model A is better,
negative means model B is better.
""")

callout(
    "Effect Size Thresholds (Cohen 1988)",
    "<ul>"
    "<li>|<em>d</em>| &lt; 0.2 — <strong>negligible</strong>: practically indistinguishable</li>"
    "<li>0.2 ≤ |<em>d</em>| &lt; 0.5 — <strong>small</strong>: detectable in large samples</li>"
    "<li>0.5 ≤ |<em>d</em>| &lt; 0.8 — <strong>medium</strong>: visible to careful observation</li>"
    "<li>|<em>d</em>| ≥ 0.8 — <strong>large</strong>: obvious practical difference</li>"
    "</ul>"
    "<p style='margin-top:0.5em;'>Key result summary: Hybrid vs CF warm comparisons "
    "yield <em>d</em> ≈ −0.006 (negligible). Ablation variants Fixed-0.5 and RRF "
    "yield <em>d</em> ≈ −0.19 and −0.16 (negligible). Cold-start Hybrid vs "
    "Content-only under long-tail yields <em>d</em> = 0.29 (small).</p>",
)
