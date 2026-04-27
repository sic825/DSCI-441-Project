# A Hybrid Music Recommendation System

Integrating Collaborative Filtering and Content-Based Filtering with Adaptive Blending and Genre-Conditioned Reciprocal Rank Fusion.

**Authors:** Simon Chen, Thoi Quach
**Course:** DSCI 441 — Statistical Machine Learning, Lehigh University, Spring 2026
**Instructor:** Masoud Yari

---

## Project Description

This project implements and rigorously evaluates a hybrid music recommendation system that combines two independent retrieval models:

- **Collaborative Filtering** via implicit-feedback Alternating Least Squares (Hu, Koren & Volinsky, 2008) trained on the Million Song Dataset Taste Profile (661,089 users × 98,485 songs, 40.3M interactions).
- **Content-Based Filtering** via cosine k-NN over standardized 9-dimensional Spotify audio features (114,000 tracks, deduplicated to 7,611 in-catalog songs).

The hybrid uses two distinct blending strategies depending on user state:

- **Warm users** (rich interaction history): adaptive-alpha min-max blending where the CF weight increases monotonically with interaction count via a sigmoid function.
- **Cold-start users** (no history, single seed song): Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009) over a content k-NN list and a genre-conditioned popularity prior.

Evaluation spans three protocols (warm-standard, warm-long-tail, cold-start standard and long-tail) with bootstrap 95% confidence intervals, paired bootstrap hypothesis tests (10,000 resamples), and Cohen's d effect sizes on every comparison. A long-tail correction (Steck, 2011) is applied to control for popularity bias in offline evaluation.

The full pipeline is wrapped in a six-page Streamlit application that includes an interactive demo using ALS fold-in inference for synthetic users not seen during training.

### Headline Findings

- Hybrid is statistically equivalent to CF on warm-user evaluation (p = 0.72, d = -0.006), robust under long-tail correction. Adaptive alpha is the only blending strategy that does not significantly degrade warm-user accuracy (validated via ablation against fixed alpha = 0.5 and RRF).
- Under bias-corrected long-tail evaluation, Hybrid achieves a 6.7× relative NDCG improvement over Content-only recommendations (p < 0.001, d = 0.29) and is the only model that performs reasonably under both standard and long-tail protocols.
- The 12.5% catalog overlap between MSD and Spotify is a structural ceiling on warm-user hybrid blending influence; this finding is documented openly.

---

## Data Sources

This project requires two external datasets that are not included in the repository due to size.

### Million Song Dataset (Taste Profile Subset)

- **Source:** http://millionsongdataset.com/tasteprofile/
- **Files needed:**
  - `train_triplets.txt` — user-song-playcount triplets (~3 GB unzipped)
  - `track_metadata.db` — SQLite database with title and artist metadata (~700 MB)
- **Citation:** Bertin-Mahieux, T., Ellis, D. P. W., Whitman, B., & Lamere, P. (2011). The Million Song Dataset. *Proc. ISMIR 2011*.

### Spotify Tracks Dataset

- **Source:** HuggingFace `maharshipandya/spotify-tracks-dataset`
- **Direct URL:** https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv
- **File needed:** `dataset.csv` (~21 MB)
- 114,000 tracks with audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo) plus genre and popularity metadata.

### Where to place the data

```
DSCI-441-Project/
└── dataset/
    ├── taste-profile/
    │   ├── train_triplets.txt
    │   └── track_metadata.db
    └── spotify-tracks/
        └── dataset.csv
```

The `src/data.py` module resolves these paths automatically relative to the repo root.

---

## Setup

### Required packages

This project uses Python 3.11 with both conda and pip dependencies. All required packages are pinned in `environment.yml` and `requirements.txt`.

Core dependencies:

- `implicit==0.7.2` — ALS collaborative filtering
- `scikit-learn==1.8.0` — k-NN, scaler, train/test utilities
- `pandas==3.0.2`, `numpy==2.4.3`, `scipy==1.17.1` — data manipulation
- `streamlit==1.56.0` — interactive web application
- `plotly==6.7.0`, `matplotlib==3.10.9`, `seaborn==0.13.2` — visualization
- `rapidfuzz` — fuzzy string matching for cross-dataset entity resolution
- `datasets==4.8.4` — HuggingFace dataset access
- `pyarrow==24.0.0` — parquet I/O for precomputed app artifacts

### Environment setup with conda

```bash
conda env create -f environment.yml
conda activate dsci441
python -m ipykernel install --user --name dsci441 --display-name "Python (dsci441)"
```

### Or pip-only fallback

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Reproduce the evaluation results

After downloading the data into `dataset/` and activating the environment, run the notebooks in the following order:

```bash
cd notebooks
jupyter lab
```

Then execute, in order:

1. `content_demo.ipynb` — builds the matched content catalog and trains the k-NN model
2. `cf_demo.ipynb` — trains the ALS CF model (~4-6 minutes on a typical laptop)
3. `final_evaluation.ipynb` — runs warm-user evaluation across 1,000 test users
4. `cold_start_evaluation.ipynb` — runs cold-start evaluation across 871 users
5. `warm_longtail.ipynb` — applies popularity-bias correction to warm evaluation
6. `diversity_metrics.ipynb` — computes intra-list similarity, genre coverage, catalog coverage
7. `warm_hybrid_ablation.ipynb` — compares three blending strategies (adaptive, fixed α=0.5, RRF)

All results are saved to `results/` as CSV (per-user metrics, hypothesis tests, aggregated statistics) and PNG (figures used in the paper and Streamlit app).

### 2. Run the Streamlit web application

```bash
streamlit run app.py
```

The application launches at http://localhost:8501 and includes six pages:

- **Overview** — abstract, project arc, headline findings
- **Model Architecture** — CF and content models, hybrid blending strategies, system diagrams
- **Statistical Analysis** — evaluation protocols, bootstrap methodology, hypothesis testing
- **Results** — per-protocol findings with figures and tables
- **Live Demo** — three interactive demonstrations: Build Your Taste (synthetic-user fold-in), Cold-Start, Warm-User
- **About** — authors, acknowledgments, code availability

The Live Demo's "Build Your Taste" tab uses ALS fold-in to compute recommendations for users not in the training data, demonstrating the full hybrid pipeline against a user constructed at request time from the user's selected songs.

---

## Repository Structure

```
DSCI-441-Project/
├── app.py                          # Streamlit entry point
├── styling.py                      # Shared CSS for the web application
├── pages/                          # Streamlit multi-page components
│   ├── 1_Overview.py
│   ├── 2_Model_Architecture.py
│   ├── 3_Statistical_Analysis.py
│   ├── 4_Results.py
│   ├── 5_Live_Demo.py
│   └── 6_About.py
├── src/                            # Reusable Python modules
│   ├── data.py                     # MSD + Spotify loading, cross-dataset matching, dedup
│   ├── models.py                   # CFModel, ContentModel, PopularityBaseline, CFColdStart
│   ├── hybrid.py                   # HybridRecommender (adaptive + RRF strategies)
│   └── evaluation.py               # Precision/Recall/NDCG/HitRate, bootstrap CIs, hypothesis tests
├── notebooks/                      # Demonstration and evaluation notebooks
├── scripts/
│   └── build_demo_library.py       # Curates the 78-song Build Your Taste library
├── results/                        # Computed metrics, hypothesis tests, figures, parquet caches
├── dataset/                        # External data (gitignored, see Data Sources above)
├── models/                         # Trained model artifacts (gitignored)
├── environment.yml                 # Conda environment spec
├── requirements.txt                # Pip fallback environment spec
├── CLAUDE.md                       # Project context for AI-assisted development
└── README.md                       # This file
```

---

## Methodology Highlights

This project applies the following statistical methods to ensure rigorous evaluation:

- **Bootstrap confidence intervals** (1,000 resamples) on every aggregate metric
- **Paired bootstrap hypothesis tests** (10,000 resamples) for model-vs-model comparisons
- **Cohen's d effect sizes** alongside p-values to distinguish statistical significance from practical magnitude
- **Popularity bias correction** (Steck, 2011) excluding the top-100 globally most-played songs from held-out ground truth in long-tail protocols
- **Ablation study** comparing three independent blending strategies (adaptive alpha, fixed alpha, Reciprocal Rank Fusion) to validate the design choice

Code for all evaluation logic is in `src/evaluation.py`. The hybrid blending logic is in `src/hybrid.py`, with separate code paths for warm-user min-max blending and cold-start RRF fusion.

---

## References

1. Bertin-Mahieux, T., Ellis, D. P. W., Whitman, B., & Lamere, P. (2011). The Million Song Dataset. *Proc. International Society for Music Information Retrieval Conference (ISMIR)*.
2. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. *Proc. ACM SIGIR*.
3. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. *Proc. IEEE International Conference on Data Mining (ICDM)*.
4. Steck, H. (2011). Item Popularity and Recommendation Accuracy. *Proc. ACM Conference on Recommender Systems (RecSys)*.

---

## Acknowledgments

We thank Masoud Yari for guidance on the statistical methodology, particularly the emphasis on bootstrap confidence intervals and effect-size reporting. The Million Song Dataset was created by Thierry Bertin-Mahieux, Daniel P. W. Ellis, Brian Whitman, and Paul Lamere at Columbia University. The Spotify Tracks Dataset was compiled by Maharshi Pandya and made available via HuggingFace Datasets. The `implicit` library (Ben Frederickson) provides the ALS implementation used for collaborative filtering.

---

## Code Availability

GitHub repository: [github.com/sic825/DSCI-441-Project](https://github.com/sic825/DSCI-441-Project)