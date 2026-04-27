import os
# Must be set before importing implicit to prevent OpenBLAS from spawning its
# own threadpool, which fights with implicit's threading and causes ~6x slowdown.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from implicit.als import AlternatingLeastSquares

from .data import AUDIO_FEATURES, build_user_item_matrix, dedup_tracks


class CFModel:
    """
    Wraps implicit ALS for the MSD Taste Profile (implicit feedback).
    Stores the user-item matrix alongside the ALS model so .recommend() is
    self-contained without requiring external matrix lookups.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 20,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self._model = None
        self._user_item = None
        self._user_to_idx = None
        self._idx_to_song = None

    def fit(self, train_df: pd.DataFrame) -> 'CFModel':
        """Train ALS on a triplets DataFrame with columns (uid, sid, count)."""
        mat, mappings = build_user_item_matrix(train_df)
        self._user_to_idx = mappings['user_to_idx']
        self._idx_to_song = mappings['idx_to_song']
        self._user_item = mat

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self._model.fit(mat * self.alpha)
        return self

    def recommend(
        self, user_id: str, k: int = 10, return_scores: bool = True
    ) -> pd.DataFrame:
        """Return top-k CF recommendations for user_id."""
        if user_id not in self._user_to_idx:
            raise ValueError(f"User '{user_id}' not in training data.")
        uidx = self._user_to_idx[user_id]
        idxs, scores = self._model.recommend(
            uidx,
            self._user_item[uidx],
            N=k,
            filter_already_liked_items=True,
        )
        song_ids = [self._idx_to_song[int(i)] for i in idxs]
        df = pd.DataFrame({'song_id': song_ids})
        if return_scores:
            df['cf_score'] = scores
        return df

    def recommend_with_metadata(
        self, user_id: str, k: int, metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Recommend and left-join display metadata (title, artist_name, track_genre).
        Songs outside the matched ~12K subset are labeled 'Unknown' rather than
        left as NaN. Fixes the NaN metadata bug from M1.

        metadata_df must have columns: song_id, title, artist_name, track_genre.
        """
        recs = self.recommend(user_id, k, return_scores=True)
        merged = recs.merge(
            metadata_df[['song_id', 'title', 'artist_name', 'track_genre']],
            on='song_id',
            how='left',
        )
        merged[['title', 'artist_name', 'track_genre']] = (
            merged[['title', 'artist_name', 'track_genre']].fillna('Unknown')
        )
        return merged

    def save(self, model_dir: str):
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model,        d / 'als_model.pkl')
        joblib.dump(self._user_to_idx,  d / 'user_to_idx.pkl')
        joblib.dump(self._idx_to_song,  d / 'idx_to_song.pkl')
        joblib.dump(self._user_item,    d / 'user_item_matrix.pkl')

    @classmethod
    def load(cls, model_dir: str) -> 'CFModel':
        """Load pre-trained model artifacts from pkl files."""
        d = Path(model_dir)
        obj = cls()
        obj._model        = joblib.load(d / 'als_model.pkl')
        obj._user_to_idx  = joblib.load(d / 'user_to_idx.pkl')
        obj._idx_to_song  = joblib.load(d / 'idx_to_song.pkl')
        obj._user_item    = joblib.load(d / 'user_item_matrix.pkl')
        return obj


class ContentModel:
    """
    Wraps sklearn NearestNeighbors (cosine) for content-based recommendations.
    Calls dedup_tracks() before fitting to eliminate variant-track contamination
    (e.g. 'I Ran', 'I Ran (L Remix)', 'I Ran (So Far Away)' -> similarity=1.0 bug).
    """

    def __init__(self, n_neighbors: int = 11):
        self.n_neighbors = n_neighbors
        self._nn = None
        self._scaler = None
        self._song_features = None   # deduped DataFrame stored for lookup
        self._X_scaled = None
        self._songid_to_idx = None

    def fit(self, matched_df: pd.DataFrame) -> 'ContentModel':
        """
        Fit on the output of match_datasets().
        Dedup runs first so each (title_clean, artist_clean) pair appears only once
        in the k-NN index.
        """
        df = dedup_tracks(matched_df).reset_index(drop=True)
        self._song_features = df

        X = df[AUDIO_FEATURES].values.astype(np.float32)
        self._scaler = StandardScaler()
        self._X_scaled = self._scaler.fit_transform(X)

        self._nn = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=self.n_neighbors,
        )
        self._nn.fit(self._X_scaled)

        self._songid_to_idx = {
            sid: i for i, sid in enumerate(df['song_id'])
        }
        return self

    def recommend(self, seed_song_id: str, k: int = 10) -> pd.DataFrame:
        """Return k most similar songs (excludes the seed itself)."""
        if seed_song_id not in self._songid_to_idx:
            raise ValueError(f"Song '{seed_song_id}' not in content index.")
        idx = self._songid_to_idx[seed_song_id]
        dists, idxs = self._nn.kneighbors(
            self._X_scaled[idx].reshape(1, -1),
            n_neighbors=k + 1,
        )
        rec_idxs = idxs[0][1:]
        rec_dists = dists[0][1:]

        rec_df = self._song_features.iloc[rec_idxs][
            ['song_id', 'title', 'artist_name', 'track_genre']
        ].copy()
        rec_df['similarity'] = 1 - rec_dists
        return rec_df.reset_index(drop=True)

    def save(self, model_dir: str):
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._nn,     d / 'content_model.pkl')
        joblib.dump(self._scaler, d / 'scaler.pkl')
        self._song_features.to_csv(d / 'song_features.csv', index=False)

    @classmethod
    def load(cls, model_dir: str, n_neighbors: int = 11) -> 'ContentModel':
        """Load pre-saved model artifacts."""
        d = Path(model_dir)
        obj = cls(n_neighbors=n_neighbors)
        obj._nn           = joblib.load(d / 'content_model.pkl')
        obj._scaler       = joblib.load(d / 'scaler.pkl')
        obj._song_features = pd.read_csv(d / 'song_features.csv')
        X = obj._song_features[AUDIO_FEATURES].values.astype(np.float32)
        obj._X_scaled     = obj._scaler.transform(X)
        obj._songid_to_idx = {
            sid: i for i, sid in enumerate(obj._song_features['song_id'])
        }
        return obj


class CFColdStart:
    """
    CF cannot recommend for a user with no fitted vector.  In cold-start
    evaluation, the honest behavior is to fall back to popularity -- CF
    degenerates to its weakest baseline.  This wrapper makes that explicit
    so the degradation is benchmarkable rather than hidden.
    """

    def __init__(self, cf_model, popularity_model):
        self.cf  = cf_model
        self.pop = popularity_model

    def recommend(self, user_id=None, seed_song=None, k: int = 10) -> pd.DataFrame:
        """Always returns popularity recs regardless of seed or user_id."""
        return self.pop.recommend(user_id=None, k=k)


class PopularityBaseline:
    """
    Non-personalized baseline: recommends globally most-played songs.
    Required as the floor for hypothesis tests against the hybrid model.
    """

    def __init__(self):
        self._top_songs = None  # Series: song_id -> total play count, sorted desc

    def fit(self, train_df: pd.DataFrame) -> 'PopularityBaseline':
        """Compute top songs by total play count from triplets (uid, sid, count)."""
        self._top_songs = (
            train_df.groupby('sid')['count']
                    .sum()
                    .sort_values(ascending=False)
        )
        return self

    def recommend(self, user_id=None, k: int = 10) -> pd.DataFrame:
        """Return top-k globally popular songs. user_id is ignored (non-personalized)."""
        top_k = self._top_songs.head(k)
        return pd.DataFrame({
            'song_id':          top_k.index,
            'popularity_score': top_k.values,
        }).reset_index(drop=True)
