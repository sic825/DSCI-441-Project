# DSCI 441 Final Project — Hybrid Music Recommendation System

This file is read automatically by Claude Code on startup. Keep it accurate.

## Project Snapshot

- **Title:** A Hybrid Music Recommendation System: Integrating Collaborative Filtering and Content-Based Filtering
- **Course:** DSCI 441 (Statistical Machine Learning), Lehigh University, Spring 2026
- **Instructor:** Dr. Yari
- **Team:** Simon Chen, Thoi Quach
- **Milestone:** M2 final report — due April 27, 11:59 PM
- **Goal:** A+ submission. Hold to a high standard.

## Current Repo State (as of M1 submission)

```
DSCI-441-Project/
├── notebooks/
│   ├── matrix_fact_model.ipynb    # ALS CF, working
│   ├── content_based_model.ipynb  # k-NN content, working but with bugs
│   └── evaluation.ipynb           # HitRate/Recall on CF, GenreConsistency on content
├── models/                         # joblib pickles (gitignored or not committed)
├── dataset/                        # MSD + Spotify CSVs (gitignored, see README)
├── README.md
├── requirements.txt
├── environment.yml                 # pinned versions
└── CLAUDE.md                       # this file
```

## Architecture (as of M1 — needs M2 work)

### Collaborative Filtering (CF) — DONE in M1
- `notebooks/matrix_fact_model.ipynb`
- Library: `implicit.als.AlternatingLeastSquares`
- Hyperparameters: factors=64, regularization=0.1, iterations=20, alpha=40, random_state=42
- Data: MSD Taste Profile, filtered to MIN_USER_COUNT=20 + MIN_SONG_COUNT=50
- After filter: 661,089 users × 98,485 songs, 40,266,961 interactions
- Preprocessing: log1p on play counts, then alpha scaling
- Saved artifacts: `als_model.pkl`, `user_to_idx.pkl`, `song_to_idx.pkl`, `idx_to_song.pkl`, `user_item_matrix.pkl`

### Content-Based (CB) — DONE in M1, has bugs
- `notebooks/content_based_model.ipynb`
- Library: `sklearn.neighbors.NearestNeighbors(metric='cosine', algorithm='brute')`
- Data: HuggingFace `maharshipandya/spotify-tracks-dataset` (114K tracks)
- 9 audio features: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- Preprocessing: StandardScaler (z-score)
- Cross-dataset matching: cleaned title + artist exact match yields 11,293 unique songs after dedup; fuzzy match with rapidfuzz token_set_ratio > 90 yields 12,307 (currently NOT used)
- Saved artifacts: `content_model.pkl`, `scaler.pkl`, `song_features.csv`

### Hybrid — NOT YET IMPLEMENTED
This is the M2 deliverable. Spec below.

## Known Bugs and Gaps from M1 (must fix in M2)

### Critical
1. **Variant track contamination in CB.** Top recs for "I Ran (So Far Away)" are 5 other versions of "I Ran" by the same artist with similarity = 1.0. Need dedup by (title_clean, artist_clean) keeping highest popularity BEFORE building the k-NN index.
2. **NaN metadata in CF output.** `evaluation.ipynb` Cell 15 shows 17/20 CF top recs have NaN title/artist/genre because joining CF song_ids back to `song_features` only succeeds for the 11.3K matched subset. CF recommends from the full 98K catalog. Fix: gracefully label as "Unknown" instead of dropping, OR expand metadata coverage by using fuzzy matches.
3. **CF only 23/100 overlap with content catalog** for a sample user (Cell 18). This limits hybrid scope.
4. **No PopularityBaseline model.** Required as a floor for the hypothesis test and for credibility.

### Methodology
5. **No Precision@K or NDCG@K.** Currently only HitRate@K and Recall@K on CF, and GenreConsistency@K + AvgSimilarity on CB. Spec promised all four ranking metrics across all models.
6. **No bootstrap confidence intervals.** Every metric is a point estimate. Statistical methods rubric requires distributions / CLT / bootstrapping.
7. **No hypothesis test.** Need paired bootstrap or Wilcoxon signed-rank: Hybrid vs Popularity, Hybrid vs CF-only, Hybrid vs Content-only.
8. **HitRate@K and Recall@K resample users independently.** Same seed=42 saves it now but it's fragile. Pull sampling out, share user set across metrics.
9. **Genre consistency uses n_samples=100.** Too small for stable estimates. Bump to 500+.
10. **No held-out test split with timestamps.** Currently leave-one-out on training data. Acceptable but worth documenting.

### Code organization
11. **Hardcoded `~/Desktop/projects/...` path** in `matrix_fact_model.ipynb` Cell 1. Grader can't run this.
12. **Dead/empty cells.** mf Cell 3 (broken `pd.read_table` stub), mf Cells 20-22, content Cell 33, evaluation Cells 19-21 + 25-26.
13. **Everything in notebooks, nothing reusable.** Streamlit app needs importable `src/` modules.
14. **`OPENBLAS_NUM_THREADS=1` warning.** Set the env var before importing implicit. Cuts ALS training time significantly.

## Cross-Dataset Matching Details

Two datasets, no shared IDs (MSD uses MSD song IDs, Spotify uses Spotify track IDs).

Current strategy in `content_based_model.ipynb`:
1. `clean_text()` — lowercase, strip parens, remove `feat.` clauses, strip non-alphanumeric, collapse whitespace
2. Exact merge on (title_clean, artist_clean) → 21,754 row merge → 11,293 unique songs after dedup
3. Fuzzy match with rapidfuzz `token_set_ratio` threshold 90 on artist (after exact title match) → 12,307 songs (computed but NOT used downstream)

Hybrid model can only run for songs in the matched subset. Outside, CF-only or content-only.

## EDA Findings (from M1, keep for poster/video)

- **Audio feature distributions:** approximately normal — valence, danceability, tempo. Right-skewed — speechiness, instrumentalness, acousticness, liveness. Left-skewed — loudness, energy.
- **Strongest correlation:** energy ↔ loudness (0.73). Most pairs weakly correlated, low multicollinearity, good for k-NN.
- **Spotify popularity:** bimodal (spike near 0, second mass at 50-70). Motivates non-parametric methods.
- **Sparsity:** user-item matrix is ~99.94% sparse after filtering.

## M1 Baseline Numbers (target table for M2 final eval)

| Metric | M1 Result |
|---|---|
| HitRate@10 (CF) | 0.121 |
| Recall@10 (CF) | 0.111 |
| HitRate@100 (CF) | 0.348 |
| GenreConsistency@5 (CB) | 0.372 |
| AvgSimilarity@5 (CB) | 0.967 |

ALS training time on M1: 6:14 (with OpenBLAS thread thrashing — fix in M2).

## M2 Hybrid Model Spec

```python
class HybridRecommender:
    """
    Combines CF (ALS) and CB (k-NN cosine) with cold-start handling.

    alpha_strategy:
        'adaptive': alpha = sigmoid((log(n_history) - 2) / 1.5)
            -> ~7 interactions: alpha ~ 0.5 (balanced)
            -> 1 interaction: alpha ~ 0.25 (lean content)
            -> 100+ interactions: alpha ~ 0.85 (lean CF)
        'fixed': use the alpha argument
        'switching': alpha=1.0 if n_history > 20 else alpha=0.0
    """
    def __init__(self, cf_model, content_model, alpha_strategy='adaptive', alpha=0.5):
        ...

    def recommend(self, user_id=None, seed_song=None, k=10, return_components=False):
        # 1. Get top-50 candidates from each available model
        # 2. MIN-MAX normalize scores within each candidate pool
        #    (CRITICAL: CF scores and cosine sims live on different scales,
        #     unnormalized blending silently lets one dominate)
        # 3. Blend: final = alpha * cf_norm + (1-alpha) * content_norm
        # 4. Return top-k by blended score
        # 5. Cold-start (no user_id): pure content from seed_song
        # 6. Warm user, no seed_song: blend CF top-50 with content recs of user's most-played song
        ...
```

Score normalization is the make-or-break detail. Min-max within candidate pool prevents one model from silently dominating. Document the choice in the docstring.

## M2 Deliverables (due April 27, 11:59 PM)

1. **Recorded video** (~15 min, all team members on camera)
2. **One-page poster** (PDF)
3. **Code** (this repo, clean and runnable)
4. **Web app** (Streamlit, deployed to Streamlit Community Cloud)

Late penalty: -10 per 12 hours past deadline.

## Tech Stack and Constraints

- Python 3.11, conda env `dsci441` (see `environment.yml`)
- Use `implicit==0.7.2` for ALS — implicit feedback, NOT explicit. RMSE is NOT applicable here.
- Use sklearn `NearestNeighbors` for CB k-NN
- Streamlit==1.56.0 for web app
- All metrics must report bootstrap 95% CIs (use `scipy.stats.bootstrap` or hand-roll with `np.random.choice`)
- Set `os.environ['OPENBLAS_NUM_THREADS'] = '1'` before `import implicit` to avoid thread thrashing

## Target src/ Structure (M2)

```
src/
├── __init__.py
├── data.py         # MSD load + filter, Spotify load, cross-dataset matching, train/test split
├── models.py       # CFModel, ContentModel, PopularityBaseline (all expose .recommend(...))
├── hybrid.py       # HybridRecommender (see spec above)
└── evaluation.py   # Precision/Recall/NDCG/HitRate@K, bootstrap CIs, paired hypothesis tests
```

Notebooks become thin demos that import from `src/`. Original M1 notebooks renamed to `legacy_*.ipynb` as fallback.

## User Preferences

- Direct, critical feedback. No padding, no over-praising. Push back when something is wrong.
- Concise code comments in Simon's voice — short and practical, not tutorial-style.
- No em-dashes in prose.
- Match Dr. Yari's conventions where applicable (eigenvalue scree plots, 1/n covariance form).
- This is a graded grad-school project aiming for A+. Hold to a high standard. Flag anything that would lose points.
- Use bootstrap CIs everywhere statistically meaningful. Don't accept "looks better" without a test.
- When unsure about a methodology choice, stop and ask rather than guessing.

## M2 Plan (Run in Order)

**Block 1 — Refactor + bug fixes**
Pull notebook logic into `src/`, fix the variant-track dedup, fix the NaN metadata issue, expand match coverage with fuzzy matches, set OpenBLAS env var, remove dead cells, replace hardcoded paths.

**Block 2 — Hybrid + final evaluation**
Implement `src/hybrid.py` per spec above. Build `notebooks/final_evaluation.ipynb` covering Precision/Recall/NDCG/HitRate@K=5/10/20 across PopularityBaseline, CF-only, Content-only, Hybrid, with bootstrap CIs and paired hypothesis tests.

**Block 3 — Streamlit app**
Three modes: cold-start (seed song), warm user (user_id), comparison view (CF vs Content vs Hybrid side-by-side). Imports from `src/`.

**Block 4 — Final polish**
Update README with reproduction steps, add results CSVs to `results/`, generate poster, record video.
