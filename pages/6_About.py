import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from styling import inject

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
inject()

st.title("About This Work")
st.divider()

# ── Authors ───────────────────────────────────────────────────────────────────
st.markdown("## Authors and Affiliations")
st.markdown("""
<table>
<tr><th>Name</th><th>Affiliation</th><th>Role</th></tr>
<tr><td><strong>Simon Chen</strong></td>
    <td>Lehigh University</td>
    <td>Collaborative filtering, hybrid architecture, evaluation pipeline</td></tr>
<tr><td><strong>Thoi Quach</strong></td>
    <td>Lehigh University</td>
    <td>Content-based filtering, cold-start design, Streamlit application</td></tr>
</table>

<p style="margin:0.3em 0 0 0;"><strong>Course:</strong> DSCI 441 — Statistical Machine Learning</p>
<p style="margin:0.2em 0 0 0;"><strong>Instructor:</strong> Masoud Yari</p>
<p style="margin:0.2em 0 0 0;"><strong>Semester:</strong> Spring 2026</p>
""", unsafe_allow_html=True)

st.divider()

# ── Acknowledgments ───────────────────────────────────────────────────────────
st.markdown("## Acknowledgments")
st.markdown("""
We thank Masoud Yari for guidance on the statistical methodology, particularly
the emphasis on bootstrap confidence intervals and effect-size reporting.
The Million Song Dataset was created by Thierry Bertin-Mahieux, Daniel P. W. Ellis,
Brian Whitman, and Paul Lamere at Columbia University. The Spotify Tracks Dataset
was compiled by Maharshi Pandya and made available via HuggingFace Datasets.
The `implicit` library (Ben Frederickson) provides the ALS implementation used
for collaborative filtering; the damping-constant choice for Reciprocal Rank Fusion
follows Cormack, Clarke & Buettcher (2009).
""")

st.divider()

# ── Code availability ─────────────────────────────────────────────────────────
st.markdown("## Code Availability")
st.markdown("""
The full source code, trained model artifacts (excluding large binary files committed
via Git LFS), and results CSVs are available at:

- **GitHub repository:** [github.com/sic825/DSCI-441-Project](https://github.com/sic825/DSCI-441-Project)

Reproduction requires the MSD Taste Profile Subset and the Spotify Tracks Dataset
(see `README.md` for download instructions). A `conda` environment specification
is provided in `environment.yml`.
""")

st.divider()

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown("## Technical Stack")
st.markdown("""
The system is implemented in Python 3.11 using `implicit` 0.7.2 (ALS),
`scikit-learn` (k-NN), `pandas` / `numpy` / `scipy` (data processing and statistics),
`plotly` (interactive visualizations), and `streamlit` 1.56.0 (web application).
Model artifacts are serialized with `joblib`; precomputed recommendation tables
use Apache Parquet via `pyarrow`.

| Component | Library | Version |
|---|---|---|
| Collaborative filtering | `implicit` (ALS) | 0.7.2 |
| Content-based filtering | `scikit-learn` NearestNeighbors | 1.3.x |
| Statistical tests | hand-rolled bootstrap (NumPy) | — |
| Web application | `streamlit` | 1.56.0 |
| Interactive plots | `plotly` | 5.x |
""")

