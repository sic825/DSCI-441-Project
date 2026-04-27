import html as _html
import os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from styling import inject, figcap, callout

st.set_page_config(page_title="Live Demo", page_icon="🎧", layout="wide")
inject()

R = ROOT / "results"
M = ROOT / "models"

AUDIO_FEATURES = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness","valence","tempo",
]

# ── Cached data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_warm_recs():
    return pd.read_parquet(R / "precomputed_warm_recs.parquet")

@st.cache_data
def load_history():
    return pd.read_parquet(R / "test_user_history.parquet")

@st.cache_data
def load_user_info():
    return pd.read_parquet(R / "user_info.parquet")

@st.cache_data
def load_metadata():
    return pd.read_parquet(R / "metadata_catalog.parquet")

@st.cache_data
def load_song_features():
    return pd.read_csv(M / "song_features.csv")

@st.cache_data
def load_demo_library():
    return pd.read_parquet(R / "demo_song_library.parquet")

@st.cache_resource
def load_content_model():
    from src.models import ContentModel
    return ContentModel.load(M)

@st.cache_resource
def load_pop_and_content():
    from src.models import ContentModel, PopularityBaseline
    cm      = ContentModel.load(M)
    pop_df  = pd.read_parquet(R / "popularity_top_songs.parquet")
    pop     = PopularityBaseline()
    pop._top_songs = pd.Series(pop_df["total_count"].values, index=pop_df["song_id"])
    meta_cat = load_metadata()
    return cm, pop, meta_cat

@st.cache_resource
def load_cf_and_hybrid():
    from src.models import CFModel, PopularityBaseline
    cf      = CFModel.load(M)
    pop_df  = pd.read_parquet(R / "popularity_top_songs.parquet")
    pop     = PopularityBaseline()
    pop._top_songs = pd.Series(pop_df["total_count"].values, index=pop_df["song_id"])
    meta_cat = load_metadata()
    cm       = load_content_model()
    return cf, cm, pop, meta_cat

# ── Load always-needed data ───────────────────────────────────────────────────
warm_recs  = load_warm_recs()
history_df = load_history()
user_info  = load_user_info()
sf         = load_song_features()

# Metadata coverage per user (CF top-10 known-title count)
cf_known = (
    warm_recs[warm_recs["model"] == "cf"]
    .sort_values(["user_id", "rank"])
    .groupby("user_id")
    .head(10)
    .assign(known=lambda df: df["title"] != "Unknown")
    .groupby("user_id")["known"]
    .sum()
    .reset_index(name="known_top10")
)
user_info = user_info.merge(cf_known, on="user_id", how="left")
user_info["known_top10"]       = user_info["known_top10"].fillna(0).astype(int)
user_info["metadata_coverage"] = user_info["known_top10"] / 10

user_info_sorted = user_info.sort_values("metadata_coverage", ascending=False).reset_index(drop=True)

uid_to_label = {
    row["user_id"]: (
        f"User {i+1}  ({int(row['n_interactions'])} interactions  ·  "
        f"α = {row['adaptive_alpha']:.3f}  ·  "
        f"{int(row['known_top10'])}/10 recs with metadata)"
    )
    for i, (_, row) in enumerate(user_info_sorted.iterrows())
}
label_to_uid = {v: k for k, v in uid_to_label.items()}

# ── BYT helpers ───────────────────────────────────────────────────────────────
QUICK_PICK_TITLES = [
    "Everlong",
    "Basket Case (Album Version)",
    "Everybody Wants To Rule The World",
    "Livin' On A Prayer",
    "Back To Black",
    "Fast Car (LP Version)",
    "Never Gonna Give You Up",
    "Creep (Explicit)",
    "In The End (Live In Texas)",
    "Sweet Caroline",
]


def _get_quick_picks(demo_lib: pd.DataFrame) -> list:
    rows = []
    for title in QUICK_PICK_TITLES:
        match = demo_lib[demo_lib["title"] == title]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    return rows


def _render_song_rows(songs: list, ncols: int = 3):
    """Render a grid of song cards with Add/Remove buttons (keys: byt_{sid})."""
    for row_start in range(0, len(songs), ncols):
        cols = st.columns(ncols)
        for col, song in zip(cols, songs[row_start : row_start + ncols]):
            sid    = song["song_id"]
            is_sel = sid in st.session_state.get("byt_selected", set())
            title  = _html.escape(str(song["title"])[:42])
            artist = _html.escape(str(song["artist_name"])[:32])
            genre  = _html.escape(str(song["track_genre"]))
            bg     = "#EEF3F9" if is_sel else "#FFFFFF"
            border = "2px solid #1F4E79" if is_sel else "1px solid #E0E8F0"
            tick   = ('<span style="float:right;color:#1F4E79;font-weight:700;">✓</span>'
                      if is_sel else "")
            with col:
                st.markdown(
                    f'<div style="background:{bg};border:{border};border-radius:8px;'
                    f'padding:9px 11px;min-height:82px;margin-bottom:2px;">'
                    f'{tick}'
                    f'<div style="font-weight:600;font-size:0.85em;color:#1A1A1A;'
                    f'line-height:1.3;margin-bottom:2px;">{title}</div>'
                    f'<div style="font-size:0.78em;color:#666;margin-bottom:4px;">{artist}</div>'
                    f'<span style="font-size:0.7em;background:#F0F0F0;color:#555;'
                    f'padding:2px 6px;border-radius:3px;">{genre}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                n_cur    = len(st.session_state.get("byt_selected", set()))
                disabled = (not is_sel) and (n_cur >= 10)
                btn_lbl  = "✓ Remove" if is_sel else "+ Add"
                if st.button(btn_lbl, key=f"byt_{sid}", use_container_width=True,
                             disabled=disabled):
                    if is_sel:
                        st.session_state["byt_selected"].discard(sid)
                    else:
                        st.session_state["byt_selected"].add(sid)
                    st.session_state["byt_recs"] = None


def _seed_title(seed_id: str, demo_lib: pd.DataFrame) -> str:
    """Return display title for a seed song from the demo library."""
    if not seed_id:
        return "a selected song"
    m = demo_lib[demo_lib["song_id"] == seed_id]
    return str(m.iloc[0]["title"]) if not m.empty else seed_id[:12] + "…"


def _explain_rec(rec: pd.Series, demo_lib: pd.DataFrame, idx: int) -> str:
    """Return one HTML sentence explaining why a single recommendation surfaced."""
    source    = str(rec.get("source", "cf_only"))
    title_str = _html.escape(str(rec.get("title", "Unknown")))
    raw_ct    = rec.get("raw_content_score", np.nan)
    raw_cf_r  = rec.get("cf_rank", 0)
    seed_id   = str(rec.get("content_seed", ""))

    def _fmt_sim(v):
        try:
            return f"{float(v):.3f}" if not np.isnan(float(v)) else "?"
        except Exception:
            return "?"

    def _fmt_rank(v):
        try:
            return str(int(v)) if v else "?"
        except Exception:
            return "?"

    if source == "both":
        stitle = _html.escape(_seed_title(seed_id, demo_lib))
        return (
            f"<strong>#{idx+1} — {title_str}</strong>: appeared in both candidate pools. "
            f"CF pool rank {_fmt_rank(raw_cf_r)}/50 "
            f"(users with similar taste also liked this); "
            f"cosine similarity {_fmt_sim(raw_ct)} to <em>{stitle}</em>. "
            "Strong hybrid signal from both sources."
        )
    elif source == "content_only":
        stitle = _html.escape(_seed_title(seed_id, demo_lib))
        return (
            f"<strong>#{idx+1} — {title_str}</strong>: audio-similar to "
            f"<em>{stitle}</em> (cosine similarity {_fmt_sim(raw_ct)}). "
            "Did not appear in CF top-50."
        )
    else:
        return (
            f"<strong>#{idx+1} — {title_str}</strong>: collaborative filtering only — "
            f"users who liked your selected songs also liked this "
            f"(CF pool rank {_fmt_rank(raw_cf_r)}/50). Did not appear in content top-50."
        )


@st.fragment
def _byt_tab():
    from rapidfuzz import fuzz

    if "byt_selected" not in st.session_state:
        st.session_state["byt_selected"] = set()
    if "byt_recs" not in st.session_state:
        st.session_state["byt_recs"] = None

    demo_lib = load_demo_library()

    st.markdown("### Build Your Taste")
    st.markdown(
        "Pick 3–10 songs you like and get personalized recommendations via "
        "**CF fold-in inference** and hybrid blending in real time."
    )
    st.markdown(
        '<p style="font-family: Helvetica Neue, Arial, sans-serif; font-size:0.92em; '
        'color:#666; margin-bottom:0.4rem;">'
        "Search for songs or use the quick picks below to build a taste profile "
        "(3–10 songs), then click <strong>Get Recommendations</strong> to see your personalized mix."
        "</p>",
        unsafe_allow_html=True,
    )
    st.info(
        "First click loads the CF model (~340 MB). Subsequent recommendations are fast — model stays cached.",
        icon="ℹ️",
    )

    # ── Search bar ─────────────────────────────────────────────────────────────
    search_query = st.text_input(
        "Search",
        placeholder="Type a song title or artist name…",
        key="byt_search",
        label_visibility="collapsed",
    )

    n_sel = len(st.session_state["byt_selected"])
    q = search_query.strip()

    # ── Selected chips (above quick picks so user sees their selection in context)
    if n_sel > 0:
        sel_color = "#1F4E79" if n_sel >= 3 else "#888"
        st.markdown(
            f'<p style="font-family:Helvetica Neue,Arial,sans-serif;font-size:0.95em;'
            f'color:{sel_color};margin:0.4rem 0 0.3rem 0;">'
            f'<strong>{n_sel}/10 selected</strong>'
            + ("  ✓  Ready to recommend!" if n_sel >= 3 else f"  — pick {3 - n_sel} more")
            + "</p>",
            unsafe_allow_html=True,
        )
        sel_rows = demo_lib[demo_lib["song_id"].isin(
            st.session_state["byt_selected"]
        )].to_dict("records")
        for row_start in range(0, len(sel_rows), 5):
            batch  = sel_rows[row_start : row_start + 5]
            chip_c = st.columns(5)
            for col, song in zip(chip_c, batch):
                sid = song["song_id"]
                with col:
                    if st.button(
                        f'✕  {str(song["title"])[:22]}',
                        key=f"byt_rm_{sid}",
                        use_container_width=True,
                    ):
                        st.session_state["byt_selected"].discard(sid)
                        st.session_state["byt_recs"] = None
        _, clr_col = st.columns([4, 1])
        with clr_col:
            if st.button("Clear all", key="byt_clear"):
                st.session_state["byt_selected"] = set()
                st.session_state["byt_recs"] = None
        st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)

    # ── Search results OR quick picks ──────────────────────────────────────────
    if q:
        scored = []
        for _, row in demo_lib.iterrows():
            candidate = f"{row['title']} {row['artist_name']}"
            score = fuzz.partial_ratio(q.lower(), candidate.lower())
            if score > 45:
                scored.append((score, row.to_dict()))
        scored.sort(key=lambda x: -x[0])
        matches = [r for _, r in scored[:6]]
        if matches:
            st.markdown(
                f'<p style="font-family:Helvetica Neue,Arial,sans-serif;font-size:0.85em;'
                f'color:#888;margin:0.2rem 0 0.5rem 0;">'
                f'{len(matches)} result{"s" if len(matches) != 1 else ""} for '
                f'"{_html.escape(q)}"</p>',
                unsafe_allow_html=True,
            )
            _render_song_rows(matches)
        else:
            st.caption(f'No matches for "{q}". Try another title or artist.')
    elif n_sel < 10:
        # Show quick picks whenever not searching and under the 10-song cap
        st.markdown(
            '<p style="font-family:Helvetica Neue,Arial,sans-serif;font-size:0.85em;'
            'color:#888;margin:0.4rem 0 0.5rem 0;">Quick picks — or search above</p>',
            unsafe_allow_html=True,
        )
        _render_song_rows(_get_quick_picks(demo_lib))
    else:
        st.caption("You've selected 10 songs — that's the cap. Remove one to swap.")

    # ── Get Recommendations ────────────────────────────────────────────────────
    st.markdown("---")
    n_now = len(st.session_state["byt_selected"])
    rec_col, _ = st.columns([2, 3])
    with rec_col:
        clicked = st.button(
            f"Get Recommendations ({n_now} songs selected)"
            if n_now >= 3 else f"Select at least 3 songs ({n_now}/3)",
            key="byt_recommend",
            disabled=n_now < 3,
            type="primary",
            use_container_width=True,
        )

    if clicked:
        with st.spinner("Running fold-in inference and hybrid blending…"):
            cf_byt, cm_byt, pop_byt, meta_byt = load_cf_and_hybrid()
            from src.hybrid import HybridRecommender as HR
            hybrid_byt = HR(cf_byt, cm_byt, meta_byt, pop_byt, alpha_strategy="adaptive")
            recs = hybrid_byt.recommend_for_synthetic_user(
                list(st.session_state["byt_selected"]), k=10
            )
            st.session_state["byt_recs"] = recs

    # ── Recommendation cards ──────────────────────────────────────────────────
    if st.session_state.get("byt_recs") is not None:
        recs = st.session_state["byt_recs"]
        alpha_val = float(recs["alpha_used"].iloc[0]) if len(recs) > 0 else 0.5

        st.markdown("---")
        alpha_note = (
            f'&nbsp;&nbsp;<span style="font-size:0.6em;color:#666;'
            f'font-family:Helvetica Neue,Arial,sans-serif;font-weight:400;">'
            f'α = {alpha_val:.3f} · CF weight</span>'
        )
        st.markdown(
            f'<h3 style="font-family:Helvetica Neue,Arial,sans-serif;color:#1F4E79;'
            f'margin-bottom:0.8rem;">Your Personalized Mix{alpha_note}</h3>',
            unsafe_allow_html=True,
        )

        source_badge = {
            "both":         '<span style="font-size:0.68em;background:#D0E0F0;color:#1F4E79;padding:2px 6px;border-radius:3px;">CF + Content</span>',
            "cf_only":      '<span style="font-size:0.68em;background:#D0E0F0;color:#1F4E79;padding:2px 6px;border-radius:3px;">CF</span>',
            "content_only": '<span style="font-size:0.68em;background:#D5EDD5;color:#2E7D32;padding:2px 6px;border-radius:3px;">Content</span>',
        }

        for row_start in range(0, min(len(recs), 10), 5):
            rec_cols = st.columns(5)
            for i, col in enumerate(rec_cols):
                idx = row_start + i
                if idx >= len(recs):
                    break
                rec   = recs.iloc[idx]
                t     = _html.escape(str(rec.get("title",       "Unknown"))[:42])
                a     = _html.escape(str(rec.get("artist_name", ""))[:32])
                s     = rec.get("source", "cf_only")
                badge = source_badge.get(s, "")
                score = rec.get("hybrid_score", 0)
                with col:
                    st.markdown(
                        f'<div style="background:#FAFCFF;border:1px solid #D0E0F0;'
                        f'border-radius:8px;padding:10px 12px;min-height:110px;'
                        f'box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
                        f'<div style="font-size:0.72em;color:#888;margin-bottom:3px;">#{idx+1}</div>'
                        f'<div style="font-weight:600;font-size:0.85em;color:#1A1A1A;'
                        f'line-height:1.3;margin-bottom:2px;">{t}</div>'
                        f'<div style="font-size:0.78em;color:#555;margin-bottom:6px;">{a}</div>'
                        f'{badge}'
                        f'<div style="font-size:0.7em;color:#aaa;margin-top:4px;">'
                        f'score: {score:.4f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Explanatory sections ──────────────────────────────────────────────
        n_sel_now  = len(st.session_state["byt_selected"])
        content_w  = 1.0 - alpha_val
        n_both_r   = int((recs["source"] == "both").sum())
        n_cf_r     = int((recs["source"] == "cf_only").sum())
        n_ct_r     = int((recs["source"] == "content_only").sum())

        # Section 1
        callout(
            "1. What the system just computed",
            f"<ul style='margin:0;padding-left:1.2em;'>"
            f"<li style='margin-bottom:0.45em;'>You selected <strong>{n_sel_now}</strong> songs. "
            f"Adaptive α = σ((log({n_sel_now}+1) − 2) / 1.5) = <strong>{alpha_val:.3f}</strong>. "
            f"CF weight = {alpha_val:.3f}, Content weight = (1 − {alpha_val:.3f}) = "
            f"<strong>{content_w:.3f}</strong>.</li>"
            f"<li style='margin-bottom:0.45em;'>ALS fold-in computed your user embedding "
            f"<em>u</em> from a 1×{n_sel_now} sparse interaction row in O(K³) time where "
            f"K = 64 (ALS factor count). Confidence weighting applied at α = 40 to match "
            f"training-time scaling.</li>"
            f"<li style='margin-bottom:0.45em;'>CF candidate pool: top-50 by dot product "
            f"⟨<em>u</em>, item_factor⟩. Content candidate pool: union of 20-nearest-neighbors "
            f"of each selected song, merged by max cosine similarity. Both pools min-max "
            f"normalized to [0, 1] independently before blending.</li>"
            f"<li>Final score = {alpha_val:.3f} × cf_norm + {content_w:.3f} × content_norm. "
            f"Songs in only one pool receive a penalized score. "
            f"Of your top-10 results: <strong>{n_both_r}</strong> from both pools, "
            f"<strong>{n_cf_r}</strong> CF-only, <strong>{n_ct_r}</strong> Content-only.</li>"
            f"</ul>",
        )

        # Section 2
        exps = [
            _explain_rec(recs.iloc[i], demo_lib, i)
            for i in range(min(3, len(recs)))
        ]
        callout(
            "2. Why these specific recommendations",
            "<p style='margin:0 0 0.5em 0;'>Top-3 only — showing the signal that brought each song to the surface:</p>"
            "<ul style='margin:0;padding-left:1.2em;'>"
            + "".join(f"<li style='margin-bottom:0.55em;'>{e}</li>" for e in exps)
            + "</ul>",
        )

        # Section 3
        callout(
            "3. How this connects to our findings",
            f"<p style='margin:0 0 0.5em 0;'>Your α = {alpha_val:.3f} (from {n_sel_now} selected songs). "
            f"Real warm users in our test set start at α ≥ 0.593 because the CF training "
            f"pipeline filters to users with at least 20 interactions (MIN_USER_COUNT = 20). "
            f"This structural α bounding — every evaluated user has α ∈ [0.593, 0.946] — "
            f"explains why the warm-user hybrid is statistically equivalent to pure CF "
            f"(Δ = −0.00004, p = 0.72, d = −0.006; Results §1).</p>"
            f"<p style='margin:0;'>The 'CF + Content' badge means a song appeared in both candidate "
            f"pools. In our cold-start evaluation (871 users), only 7.9% of top-10 hybrid "
            f"recommendations had this 'both' label. The remaining ≈90% were single-source, "
            f"consistent with the 12.5% catalog overlap between MSD (98K songs) and the "
            f"Spotify-matched content index (7.6K songs). See Results §2.</p>",
        )

        # Section 4
        st.markdown(
            "**Where this fits in a real system.** "
            "Hybrid recommenders of this form are deployed in production music platforms "
            "(e.g., Spotify's Discover Weekly combines collaborative filtering, content-based "
            "features, and NLP/audio analysis). Our contribution is methodological: we demonstrate "
            "under three evaluation protocols including popularity-bias correction (Steck 2011) "
            "that the hybrid's value differs by deployment scenario — adaptive alpha is "
            "structurally CF-dominated for warm users (p = 0.72, d = −0.006), but the "
            "cold-start hybrid significantly outperforms pure content recommendations "
            "(6.7× relative NDCG, p < 0.001) under bias-corrected evaluation. Production "
            "deployment would likely use this hybrid specifically for new-user onboarding "
            "flows like the one demonstrated above."
        )

    # ── Methodology callout ───────────────────────────────────────────────────
    st.markdown("---")
    callout(
        "Methodology: ALS Fold-In Inference",
        "<p>This demo uses <strong>Alternating Least Squares fold-in</strong> (Hu et al. 2008) "
        "to compute a user vector for someone not in the training data. "
        "Given your liked songs as a binary interaction vector <em>c</em>, "
        "the fold-in formula computes the user embedding as "
        "<em>u</em> = (Y<sup>T</sup>C<sub>u</sub>Y + λI)<sup>−1</sup> "
        "Y<sup>T</sup>C<sub>u</sub><strong>p</strong>, "
        "where Y is the trained item factor matrix, C<sub>u</sub> is a diagonal "
        "confidence matrix (each liked song = 1 implicit play), "
        "and λ is the ALS regularization term. "
        "Recommendations are then blended with content-based k-NN using adaptive alpha: "
        "α = σ((ln(1+<em>n</em>) − 2) / 1.5), where <em>n</em> is your selection count.</p>",
    )


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Live Demo")
st.markdown(
    '<p style="font-family: Helvetica Neue, Arial, sans-serif; font-size:1.0em; '
    'color:#555; margin-bottom:0.2em;">'
    'Three interactive views of the hybrid recommender.'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

st.markdown(
    "This page provides three complementary demonstrations of the system. "
    "The **Build Your Taste** demo is the primary illustration: pick songs you like "
    "and see ALS fold-in inference plus content-based blending generate personalized "
    "recommendations in real time. The other two demos use curated data to make "
    "specific points about the system's behavior."
)

st.markdown(
    "**Build Your Taste** — Search or select from a curated library of recognizable "
    "tracks. Hit Recommend to generate top-10 personalized picks via ALS fold-in "
    "(no retraining required) plus adaptive-alpha hybrid blending. This is the demo "
    "that exercises the full hybrid pipeline against a synthetic user."
)
st.markdown(
    "**Cold-Start Demo** — Pick a single seed song from the matched audio-feature "
    "catalog and see how the system handles a user with zero history. The hybrid "
    "uses Reciprocal Rank Fusion with genre-conditioned popularity. All "
    "recommendations show full metadata because they are drawn from the 7,611-song "
    "matched catalog."
)
st.markdown(
    "**Warm-User Demo** — Pre-computed recommendations for 1,000 test users from our "
    "offline evaluation. Some users will display 'Unknown' for many recommendations "
    "because Collaborative Filtering draws from 98,485 MSD songs while only 12,310 "
    "(12.5%) have matched Spotify metadata. The dropdown is sorted by metadata "
    "coverage so the default user has rich titles, but exploring lower-coverage "
    "users illustrates the structural catalog overlap that constrains warm-user "
    "blending (see Results §1)."
)
st.divider()

tab1, tab2, tab3 = st.tabs([
    "Tab 1: Build Your Taste",
    "Tab 2: Cold-Start Demo",
    "Tab 3: Warm-User Demo",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Build Your Taste
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    _byt_tab()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Cold-Start Demo
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Cold-Start Recommendations")
    st.markdown(
        "Select a seed song from the 7,611-track content catalog. "
        "All three models run live without any user history."
    )

    sf_sorted = sf.sort_values(["artist_name","title"])
    song_options = {
        row["song_id"]: f"{row['title']} — {row['artist_name']} ({row.get('track_genre','?')})"
        for _, row in sf_sorted.iterrows()
    }
    chosen_cs_label = st.selectbox("Select seed song", list(song_options.values()), key="cold_seed")
    chosen_sid      = [k for k, v in song_options.items() if v == chosen_cs_label][0]
    seed_row        = sf[sf["song_id"] == chosen_sid].iloc[0]

    col_info, col_radar = st.columns([1, 1])
    with col_info:
        st.markdown(f"**Seed:** *{seed_row['title']}* — {seed_row['artist_name']}")
        st.caption(
            f"Genre: {seed_row.get('track_genre','Unknown')}  ·  "
            f"Spotify popularity: {seed_row.get('popularity','?')}"
        )
        for feat in AUDIO_FEATURES:
            val = seed_row[feat]
            st.caption(f"{feat}: {val:.3f}")
    with col_radar:
        raw    = seed_row[AUDIO_FEATURES].values.astype(float)
        mins   = sf[AUDIO_FEATURES].min().values
        maxs   = sf[AUDIO_FEATURES].max().values
        normed = np.where(maxs > mins, (raw - mins) / (maxs - mins), 0.5)
        fig = go.Figure(go.Scatterpolar(
            r=list(normed) + [normed[0]], theta=AUDIO_FEATURES + [AUDIO_FEATURES[0]],
            fill="toself", fillcolor="rgba(31,78,121,0.15)",
            line=dict(color="#1F4E79", width=2),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#D0D8E4"),
                       angularaxis=dict(gridcolor="#D0D8E4")),
            showlegend=False, height=270,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Helvetica Neue, Arial, sans-serif", color="#1A1A1A", size=11),
            margin=dict(t=10, b=10, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)
        figcap("Audio feature radar (min-max normalized across catalog).")

    with st.spinner("Loading content and popularity models…"):
        cm_cs, pop_cs, meta_cs = load_pop_and_content()
        from src.hybrid import HybridRecommender
        hybrid_cs = HybridRecommender(
            cf_model=None, content_model=cm_cs,
            metadata_catalog=meta_cs, popularity_model=pop_cs,
            alpha_strategy="adaptive",
        )

    c_ct, c_pop2, c_hy2 = st.columns(3)

    with c_ct:
        st.markdown("**Content k-NN**")
        try:
            ct_recs = cm_cs.recommend(chosen_sid, k=10)
            d = ct_recs[["title","artist_name","track_genre","similarity"]].copy()
            d.columns = ["Title","Artist","Genre","Similarity"]
            d["Similarity"] = d["Similarity"].map("{:.3f}".format)
            st.dataframe(d, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))

    with c_pop2:
        st.markdown("**Popularity (genre-local)**")
        try:
            pop_recs = pop_cs.recommend(k=10)
            pm = pop_recs.merge(meta_cs[["song_id","title","artist_name","track_genre"]],
                                on="song_id", how="left").fillna("Unknown")
            st.dataframe(pm[["title","artist_name","track_genre"]].rename(
                columns={"title":"Title","artist_name":"Artist","track_genre":"Genre"}
            ), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))

    with c_hy2:
        st.markdown("**Hybrid (RRF cold-start)**")
        try:
            hy_recs = hybrid_cs.recommend(seed_song=chosen_sid, k=10)
            cols = [c for c in ["title","artist_name","track_genre","rrf_score","source"]
                    if c in hy_recs.columns]
            d = hy_recs[cols].copy()
            if "rrf_score" in d.columns:
                d["rrf_score"] = d["rrf_score"].map("{:.4f}".format)
            d = d.rename(columns={"title":"Title","artist_name":"Artist",
                                   "track_genre":"Genre","rrf_score":"RRF","source":"Source"})
            st.dataframe(d, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))

    st.info(
        "Hybrid fuses Content k-NN and genre-conditioned Popularity via Reciprocal Rank Fusion "
        "(RRF, *k* = 60; Cormack et al., 2009). Songs appearing in **both** lists receive "
        "contributions from two RRF terms, boosting their final score."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Warm-User Demo
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Warm-User Recommendations")

    callout(
        "Note on Metadata Coverage",
        "<p>CF draws from 98,485 songs in the MSD catalog, but only 12,310 (12.5%) have "
        "matched Spotify metadata. Recommendations for songs outside the matched subset "
        "display as 'Unknown'. Users in the dropdown are sorted by coverage so the demo "
        "defaults to high-metadata users. This 12.5% catalog overlap is itself a structural "
        "finding of our project — see Results §1 for how it caps the hybrid's warm-user "
        "blending influence.</p>",
    )

    st.markdown(
        "Select a test user. Recommendations are precomputed at *k* = 20 "
        "using the full trained models with training interactions withheld."
    )

    chosen_label = st.selectbox("Select user", list(uid_to_label.values()), key="warm_user")
    chosen_uid   = label_to_uid[chosen_label]
    row_ui       = user_info[user_info["user_id"] == chosen_uid].iloc[0]

    st.caption(
        f"Interaction count: **{int(row_ui['n_interactions'])}**  ·  "
        f"Adaptive α: **{row_ui['adaptive_alpha']:.3f}**  ·  "
        f"Most-played in content catalog: *{row_ui['most_played_title']}* "
        f"— {row_ui['most_played_artist']}  ·  "
        f"CF top-10 metadata: {int(row_ui['known_top10'])}/10"
    )

    col_hist, _ = st.columns([2, 1])
    with col_hist:
        st.markdown("**Listening history (top 20 by play count)**")
        hist_user = (history_df[history_df["user_id"] == chosen_uid]
                     .sort_values("play_count", ascending=False).head(20))
        hist_display = hist_user[["rank","title","artist_name","track_genre","play_count"]].copy()
        hist_display.columns = ["#","Title","Artist","Genre","Plays"]
        st.dataframe(hist_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Top-10 Recommendations**")
    c_pop, c_cf, c_hy = st.columns(3)

    for col, model_name, label in [
        (c_pop, "popularity", "Popularity"),
        (c_cf,  "cf",         "CF (ALS)"),
        (c_hy,  "hybrid",     "Hybrid (Adaptive α)"),
    ]:
        recs = (warm_recs[(warm_recs["user_id"] == chosen_uid) & (warm_recs["model"] == model_name)]
                .sort_values("rank").head(10))
        with col:
            st.markdown(f"**{label}**")
            display = recs[["rank","title","artist_name","track_genre","song_id"]].copy()
            display["Title"] = display.apply(
                lambda r: r["title"] if r["title"] != "Unknown" else f"[{r['song_id'][:12]}…]",
                axis=1,
            )
            display["Artist"] = display["artist_name"].replace("Unknown", "—")
            display = display[["rank","Title","Artist","track_genre"]].rename(
                columns={"rank":"#","track_genre":"Genre"}
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

    cf_songs = set(warm_recs[(warm_recs["user_id"]==chosen_uid)&(warm_recs["model"]=="cf")].head(10)["song_id"])
    hy_songs = set(warm_recs[(warm_recs["user_id"]==chosen_uid)&(warm_recs["model"]=="hybrid")].head(10)["song_id"])
    st.info(
        f"Hybrid and CF share **{len(cf_songs & hy_songs)} of 10** recommendations for this user. "
        f"CF metadata coverage for this user: {int(row_ui['known_top10'])}/10 "
        "(CF draws from 98K songs; only ~12K have matched Spotify records — a structural catalog overlap limitation)."
    )
