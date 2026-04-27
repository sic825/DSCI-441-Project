import numpy as np
import pandas as pd


class HybridRecommender:
    """
    Combines CF (ALS) and CB (k-NN cosine) with cold-start handling.

    alpha_strategy:
        'adaptive'  : alpha = sigmoid((log1p(n_history) - 2) / 1.5)
                          n=1    -> alpha ≈ 0.25  (lean content)
                          n=7    -> alpha ≈ 0.50  (balanced)
                          n=100+ -> alpha ≈ 0.85  (lean CF)
        'fixed'     : always use self.alpha
        'switching' : alpha=1.0 if n_history > 20 else 0.0
        'rrf'       : Reciprocal Rank Fusion (warm path); no alpha, no normalization.
                      RRF score = 1/(60+rank_cf) + 1/(60+rank_ct).
                      alpha_used=NaN in output; hybrid_score holds the RRF score.

    Score normalization: min-max within each model's top-50 candidate pool
    before blending.  CF scores (ALS dot-product) and cosine similarities
    live on different ranges; raw blending silently lets one model dominate.

    Cross-catalog overlap:
        CF operates on ~98K songs; the content k-NN index covers only ~7.6K
        deduped tracks.  The intersection at init time is ~2.7K songs.
        Songs outside the overlap receive a score from only one model;
        their hybrid_score is penalized proportionally (alpha * cf_norm for
        CF-only, (1-alpha) * content_norm for content-only).
    """

    def __init__(
        self,
        cf_model,
        content_model,
        metadata_catalog: pd.DataFrame,
        popularity_model=None,
        alpha_strategy: str = 'adaptive',
        alpha: float = 0.5,
        cold_start_content_weight: float = 0.7,
    ):
        self.cf_model                  = cf_model
        self.content_model             = content_model
        self.metadata_catalog          = metadata_catalog   # song_id, title, artist_name, track_genre
        self.popularity_model          = popularity_model
        self.alpha_strategy            = alpha_strategy
        self.alpha                     = alpha
        self.cold_start_content_weight = cold_start_content_weight

        # Compute overlap once; used to label source in recommend() output.
        cf_ids      = set(cf_model._idx_to_song.values())
        content_ids = set(content_model._songid_to_idx.keys())
        self._overlap_ids = cf_ids & content_ids

        # Lazy-built inverse mapping (song_id -> matrix column index) for CF eval.
        self.__song_to_idx = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _song_to_idx(self):
        """Inverse of _idx_to_song, built once on first use."""
        if self.__song_to_idx is None:
            self.__song_to_idx = {
                sid: idx for idx, sid in self.cf_model._idx_to_song.items()
            }
        return self.__song_to_idx

    def _get_n_history(self, user_id: str) -> int:
        uidx = self.cf_model._user_to_idx.get(user_id)
        if uidx is None:
            return 0
        return self.cf_model._user_item[uidx].nnz

    def _compute_alpha(self, user_id: str) -> float:
        if self.alpha_strategy == 'fixed':
            return self.alpha
        n = self._get_n_history(user_id)
        if self.alpha_strategy == 'switching':
            return 1.0 if n > 20 else 0.0
        # adaptive
        return float(1.0 / (1.0 + np.exp(-((np.log1p(n) - 2.0) / 1.5))))

    def _get_most_played_song(self, user_id: str):
        """Return the user's most-played song by log1p-weighted play count."""
        uidx = self.cf_model._user_to_idx.get(user_id)
        if uidx is None:
            return None
        row = self.cf_model._user_item[uidx]
        if row.nnz == 0:
            return None
        best_col = int(row.toarray().flatten().argmax())
        return self.cf_model._idx_to_song[best_col]

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize; returns 0.5 everywhere if all values are equal."""
        mn, mx = scores.min(), scores.max()
        if mx == mn:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - mn) / (mx - mn)

    def _cf_candidates(self, user_id: str, k: int = 50,
                        known_song_ids=None) -> dict:
        """
        Get CF top-k as {song_id: raw_score}.

        known_song_ids: optional set of song_ids to use as the filter mask
        instead of the full user row.  Pass training-only song_ids during
        evaluation so held-out items are not filtered from CF's pool.
        """
        if user_id not in self.cf_model._user_to_idx:
            return {}
        uidx = self.cf_model._user_to_idx[user_id]

        if known_song_ids is not None:
            # Build a user row containing only the known (training) interactions.
            full = self.cf_model._user_item[uidx].toarray().flatten()
            s2i  = self._song_to_idx()
            mask = np.zeros(full.shape, dtype=bool)
            for sid in known_song_ids:
                if sid in s2i:
                    mask[s2i[sid]] = True
            from scipy.sparse import csr_matrix
            user_row = csr_matrix(full * mask)
        else:
            user_row = self.cf_model._user_item[uidx]

        idxs, scores = self.cf_model._model.recommend(
            uidx, user_row, N=k, filter_already_liked_items=True
        )
        return {self.cf_model._idx_to_song[int(i)]: float(s)
                for i, s in zip(idxs, scores)}

    def _content_candidates(self, seed_song_id: str, k: int = 50) -> dict:
        """Get content top-k as {song_id: similarity}."""
        if seed_song_id not in self.content_model._songid_to_idx:
            return {}
        recs = self.content_model.recommend(seed_song_id, k=k)
        return dict(zip(recs['song_id'], recs['similarity']))

    def _attach_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(
            self.metadata_catalog[['song_id', 'title', 'artist_name']],
            on='song_id', how='left',
        )
        df[['title', 'artist_name']] = df[['title', 'artist_name']].fillna('Unknown')
        return df

    # ── Public interface ──────────────────────────────────────────────────────

    def recommend(
        self,
        user_id=None,
        seed_song=None,
        k: int = 10,
        known_song_ids=None,
    ) -> pd.DataFrame:
        """
        Warm path returns:
            song_id, title, artist_name,
            hybrid_score, cf_score, content_score, alpha_used, source
            source: 'both' | 'cf_only' | 'content_only'

        Cold-start path (user_id=None) returns RRF columns instead:
            song_id, title, artist_name,
            rrf_score, content_rank, popularity_rank, source
            source: 'both' | 'content_only' | 'popularity_only'
            See _cold_start() for RRF citation.

        Parameters
        ----------
        user_id : str, optional
            MSD user_id.  Required for any CF contribution.
        seed_song : str, optional
            MSD song_id used to seed the content model.
            If omitted for a warm user, defaults to their most-played
            training song.
        k : int
            Number of recommendations to return.
        known_song_ids : set, optional
            For evaluation only: song_ids to filter from CF's candidate pool
            (i.e. the user's 80% training songs).  Allows held-out items to
            surface from CF.  When None, CF filters all known interactions.
        """
        if user_id is None and seed_song is None:
            raise ValueError("Provide at least one of user_id or seed_song.")

        # ── Cold start ──
        if user_id is None:
            return self._cold_start(seed_song, k)

        # ── Warm user ──
        if seed_song is None:
            seed_song = self._get_most_played_song(user_id)

        if self.alpha_strategy == 'rrf':
            return self._warm_rrf(user_id, seed_song, k, known_song_ids)

        alpha = self._compute_alpha(user_id)

        cf_raw  = self._cf_candidates(user_id, k=50, known_song_ids=known_song_ids)
        ct_raw  = self._content_candidates(seed_song, k=50) if seed_song else {}

        # Normalize within each pool (make-or-break: prevents scale dominance)
        cf_norm = {}
        if cf_raw:
            vals = np.array(list(cf_raw.values()), dtype=float)
            normed = self._minmax(vals)
            cf_norm = dict(zip(cf_raw.keys(), normed))

        ct_norm = {}
        if ct_raw:
            vals = np.array(list(ct_raw.values()), dtype=float)
            normed = self._minmax(vals)
            ct_norm = dict(zip(ct_raw.keys(), normed))

        # Build blended candidate pool
        all_songs = set(cf_norm) | set(ct_norm)
        rows = []
        for sid in all_songs:
            in_cf = sid in cf_norm
            in_ct = sid in ct_norm
            cf_s  = cf_norm.get(sid, np.nan)
            ct_s  = ct_norm.get(sid, np.nan)

            if in_cf and in_ct:
                source   = 'both'
                h_score  = alpha * cf_s + (1.0 - alpha) * ct_s
            elif in_cf:
                source   = 'cf_only'
                h_score  = alpha * cf_s          # no content contribution
            else:
                source   = 'content_only'
                h_score  = (1.0 - alpha) * ct_s  # no CF contribution

            rows.append({
                'song_id':       sid,
                'cf_score':      cf_s,
                'content_score': ct_s,
                'hybrid_score':  h_score,
                'alpha_used':    alpha,
                'source':        source,
            })

        result = (
            pd.DataFrame(rows)
              .sort_values('hybrid_score', ascending=False)
              .head(k)
              .reset_index(drop=True)
        )
        result = self._attach_metadata(result)
        return result[['song_id', 'title', 'artist_name',
                        'hybrid_score', 'cf_score', 'content_score',
                        'alpha_used', 'source']]

    def _warm_rrf(self, user_id: str, seed_song, k: int, known_song_ids) -> pd.DataFrame:
        """
        Warm-path Reciprocal Rank Fusion.

        Fuses CF ALS ranking and content k-NN ranking without alpha or
        score normalization.  RRF score = 1/(60+rank_cf) + 1/(60+rank_ct).
        k=60 from Cormack et al. (2009).

        alpha_used is NaN (RRF has no alpha); hybrid_score holds the RRF score.
        """
        cf_raw = self._cf_candidates(user_id, k=50, known_song_ids=known_song_ids)
        ct_raw = self._content_candidates(seed_song, k=50) if seed_song else {}

        RRF_K = 60
        cf_ranked = {sid: r for r, sid in enumerate(
            sorted(cf_raw, key=cf_raw.__getitem__, reverse=True), start=1)}
        ct_ranked = {sid: r for r, sid in enumerate(
            sorted(ct_raw, key=ct_raw.__getitem__, reverse=True), start=1)}

        all_songs = set(cf_ranked) | set(ct_ranked)
        rows = []
        for sid in all_songs:
            in_cf, in_ct = sid in cf_ranked, sid in ct_ranked
            rrf = 0.0
            if in_cf: rrf += 1.0 / (RRF_K + cf_ranked[sid])
            if in_ct: rrf += 1.0 / (RRF_K + ct_ranked[sid])
            rows.append({
                'song_id':       sid,
                'cf_score':      float(cf_raw[sid]) if in_cf else np.nan,
                'content_score': float(ct_raw[sid]) if in_ct else np.nan,
                'hybrid_score':  rrf,
                'alpha_used':    np.nan,
                'source': 'both' if (in_cf and in_ct) else
                          ('cf_only' if in_cf else 'content_only'),
            })

        result = (
            pd.DataFrame(rows)
              .sort_values('hybrid_score', ascending=False)
              .head(k)
              .reset_index(drop=True)
        )
        result = self._attach_metadata(result)
        return result[['song_id', 'title', 'artist_name',
                        'hybrid_score', 'cf_score', 'content_score',
                        'alpha_used', 'source']].reset_index(drop=True)

    def _genre_pop_ranked(self, seed_song: str):
        """
        Return (pop_ranked dict {song_id: rank}, fallback_used str).

        If the seed's genre is known, ranks the top-50 MSD-play-count songs
        within that genre (from metadata_catalog).  Falls back to global top-50
        when genre is missing or 'Unknown'.
        """
        fallback_used = 'global'
        pop_ranked    = {}

        if self.popularity_model is None:
            return pop_ranked, fallback_used

        # Genre lookup
        row = self.metadata_catalog.loc[
            self.metadata_catalog['song_id'] == seed_song, 'track_genre'
        ]
        genre = row.iloc[0] if not row.empty else 'Unknown'

        if genre and genre != 'Unknown':
            genre_ids = set(
                self.metadata_catalog.loc[
                    self.metadata_catalog['track_genre'] == genre, 'song_id'
                ].astype(str)
            )
            top_songs = self.popularity_model._top_songs
            genre_top = top_songs[
                top_songs.index.astype(str).isin(genre_ids)
            ].head(50)
            if not genre_top.empty:
                for rank, sid in enumerate(genre_top.index.astype(str), start=1):
                    pop_ranked[sid] = rank
                fallback_used = 'genre'

        if not pop_ranked:  # genre empty or lookup failed
            top50 = self.popularity_model._top_songs.head(50)
            for rank, sid in enumerate(top50.index.astype(str), start=1):
                pop_ranked[sid] = rank
            fallback_used = 'global'

        return pop_ranked, fallback_used

    def _cold_start(self, seed_song: str, k: int) -> pd.DataFrame:
        """
        Reciprocal Rank Fusion cold-start: blend content k-NN with
        genre-conditioned (or global) popularity.

        RRF score = sum_{i in {content, popularity}} 1 / (60 + rank_i(s))
        where rank_i is 1-indexed; missing list contributes 0.

        k=60 constant from Cormack, Clarke & Buettcher (2009) "Reciprocal Rank
        Fusion Outperforms Condorcet and Individual Rank Learning Methods."

        Returns columns: song_id, title, artist_name,
                         rrf_score, content_rank, popularity_rank,
                         source, fallback_used
        (cf_score / content_score / alpha_used are not meaningful with RRF.)
        """
        # ── Content ranking ───────────────────────────────────────────────────
        ct_raw = self._content_candidates(seed_song, k=50)
        ct_ranked = {}
        if ct_raw:
            for rank, sid in enumerate(
                sorted(ct_raw, key=ct_raw.__getitem__, reverse=True), start=1
            ):
                ct_ranked[sid] = rank

        # ── Genre-conditioned popularity ranking ──────────────────────────────
        pop_ranked, fallback_used = self._genre_pop_ranked(seed_song)

        # ── RRF scoring ───────────────────────────────────────────────────────
        RRF_K = 60  # Cormack et al. (2009)
        all_songs = set(ct_ranked) | set(pop_ranked)
        rows = []
        for sid in all_songs:
            in_ct  = sid in ct_ranked
            in_pop = sid in pop_ranked
            rrf    = 0.0
            if in_ct:
                rrf += 1.0 / (RRF_K + ct_ranked[sid])
            if in_pop:
                rrf += 1.0 / (RRF_K + pop_ranked[sid])
            rows.append({
                'song_id':         sid,
                'rrf_score':       rrf,
                'content_rank':    float(ct_ranked[sid])  if in_ct  else np.nan,
                'popularity_rank': float(pop_ranked[sid]) if in_pop else np.nan,
                'fallback_used':   fallback_used,
                'source': 'both' if (in_ct and in_pop) else
                          ('content_only' if in_ct else 'popularity_only'),
            })

        result = (
            pd.DataFrame(rows)
              .sort_values('rrf_score', ascending=False)
              .head(k)
              .reset_index(drop=True)
        )
        result = self._attach_metadata(result)
        return result[['song_id', 'title', 'artist_name',
                        'rrf_score', 'content_rank', 'popularity_rank',
                        'source', 'fallback_used']].reset_index(drop=True)
