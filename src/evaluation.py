"""
Ranking metrics, bootstrap CIs, and paired hypothesis tests.

evaluate_model expects a *callable*, not a model object directly, so any
model can be wrapped cleanly:
    lambda uid, k: model.recommend(uid, k)['song_id'].tolist()

All metric functions share the same signature:
    (recommended: list[str], relevant: set[str], k: int) -> float
where `recommended` is ordered highest-score-first and `relevant` is the
held-out set of song_ids for this user.
"""

import numpy as np
import pandas as pd


# ── Core metrics ─────────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for s in recommended[:k] if s in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for s in recommended[:k] if s in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Binary-relevance NDCG@k.
    DCG  = sum_{i=0}^{k-1}  rel_i / log2(i+2)
    IDCG = sum_{i=0}^{min(|rel|,k)-1}  1 / log2(i+2)
    """
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, s in enumerate(recommended[:k])
        if s in relevant
    )
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    return 1.0 if any(s in relevant for s in recommended[:k]) else 0.0


# ── Batch evaluation ─────────────────────────────────────────────────────────

def evaluate_model(
    recommend_fn,
    test_data: dict,
    k_values=(5, 10, 20),
) -> pd.DataFrame:
    """
    Evaluate a recommendation function across all users in test_data.

    recommend_fn: callable(user_id: str, k: int) -> list[str]
        Returns an ordered list of song_ids (highest score first).
        Exceptions are caught silently and treated as empty recommendations.
    test_data: {user_id: set(held_out_song_ids)}

    Returns long-format DataFrame: [user_id, k, metric, value]
    """
    metric_fns = {
        'precision': precision_at_k,
        'recall':    recall_at_k,
        'ndcg':      ndcg_at_k,
        'hit_rate':  hit_rate_at_k,
    }
    max_k = max(k_values)

    rows = []
    for user_id, relevant in test_data.items():
        try:
            recs = recommend_fn(user_id, max_k)
        except Exception:
            recs = []

        for k in k_values:
            for metric_name, fn in metric_fns.items():
                rows.append({
                    'user_id': user_id,
                    'k':       k,
                    'metric':  metric_name,
                    'value':   fn(recs, relevant, k),
                })

    return pd.DataFrame(rows)


# ── Bootstrap utilities ───────────────────────────────────────────────────────

def bootstrap_ci(
    per_user_scores,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
):
    """
    Bootstrap CI for the mean of per_user_scores.

    Returns: (point_estimate, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(per_user_scores, dtype=float)
    point = float(scores.mean())

    boot_means = np.empty(n_resamples)
    n = len(scores)
    for i in range(n_resamples):
        boot_means[i] = rng.choice(scores, size=n, replace=True).mean()

    alpha = (1.0 - ci) / 2.0
    low  = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return point, low, high


def paired_bootstrap_test(
    scores_a,
    scores_b,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Two-sided paired bootstrap test: H0: E[A - B] = 0.

    Shifts the bootstrap distribution of mean(diff) to be centered at 0,
    then counts how often |boot_mean| >= |observed_mean|.

    Returns p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = a - b
    observed = diffs.mean()
    centered = diffs - observed  # null: E[centered] = 0

    n = len(diffs)
    extreme = sum(
        1
        for _ in range(n_resamples)
        if abs(rng.choice(centered, size=n, replace=True).mean()) >= abs(observed)
    )
    return extreme / n_resamples


# ── Self-verification (run as script) ────────────────────────────────────────

if __name__ == '__main__':
    import math

    rec = ['a', 'b', 'c', 'd', 'e']
    rel = {'a', 'c'}

    # precision
    assert abs(precision_at_k(rec, rel, 3) - 2/3) < 1e-9, "precision@3"
    assert abs(precision_at_k(rec, rel, 5) - 2/5) < 1e-9, "precision@5"

    # recall
    assert abs(recall_at_k(rec, rel, 3) - 1.0) < 1e-9, "recall@3 both in top-3"
    assert abs(recall_at_k(rec, rel, 1) - 0.5) < 1e-9, "recall@1 only a found"

    # hit_rate
    assert hit_rate_at_k(rec, rel, 3) == 1.0,              "hit_rate@3"
    assert hit_rate_at_k(['x', 'y', 'z'], rel, 3) == 0.0,  "hit_rate@3 miss"

    # ndcg -- known analytical result
    # rec=['a','b','c'], rel={'a','c'}, k=3
    # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)              ≈ 1.6309
    dcg_expected  = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg_expected = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    ndcg_expected = dcg_expected / idcg_expected
    got = ndcg_at_k(rec, rel, 3)
    assert abs(got - ndcg_expected) < 1e-9, f"ndcg@3: got {got:.8f}, expected {ndcg_expected:.8f}"

    print(f"All checks passed.")
    print(f"  ndcg@3 = {got:.6f}  (expected ≈ 0.9197)")
