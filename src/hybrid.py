import numpy as np
import pandas as pd


class HybridRecommender:
    """
    Combines CF (ALS) and CB (k-NN cosine) with cold-start handling.

    alpha_strategy:
        'adaptive': alpha = sigmoid((log(n_history) - 2) / 1.5)
            -> ~7 interactions:  alpha ~ 0.5  (balanced)
            -> 1 interaction:    alpha ~ 0.25 (lean content)
            -> 100+ interactions: alpha ~ 0.85 (lean CF)
        'fixed':    use the alpha argument directly
        'switching': alpha=1.0 if n_history > 20 else alpha=0.0

    Score normalization: min-max within each candidate pool before blending.
    CF scores and cosine similarities live on different scales; unnormalized
    blending silently lets one model dominate. This is the make-or-break detail.
    """

    def __init__(self, cf_model, content_model, alpha_strategy='adaptive', alpha=0.5):
        self.cf_model = cf_model
        self.content_model = content_model
        self.alpha_strategy = alpha_strategy
        self.alpha = alpha

    def recommend(self, user_id=None, seed_song=None, k=10, return_components=False):
        """
        Not yet implemented -- Block 2 deliverable.

        Plan:
          1. Get top-50 candidates from each available model.
          2. Min-max normalize scores within each pool.
          3. Blend: final = alpha * cf_norm + (1-alpha) * content_norm.
          4. Return top-k by blended score.
          5. Cold-start (no user_id): pure content from seed_song.
          6. Warm user, no seed_song: blend CF top-50 with content recs
             seeded from user's most-played song.
        """
        raise NotImplementedError("HybridRecommender.recommend() not yet implemented (Block 2).")
