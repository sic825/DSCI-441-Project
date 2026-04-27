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

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Live Demo")
st.markdown("""
This page provides interactive exploration of the hybrid recommender system.
Three demonstrations are provided below:

- **Build Your Taste** — pick songs from a curated library and receive personalized
  recommendations via CF fold-in inference and hybrid blending in real time.
- **Cold-Start Demo** — select a seed song from the 7,611-track content catalog and
  observe recommendations from the Content k-NN, genre-conditioned Popularity, and
  Hybrid (genre-conditioned RRF) models.
- **Warm-User Demo** — compare Popularity, CF, and Hybrid (adaptive) recommendations
  for any of the 1,000 test users using precomputed top-20 ranked lists.
""")
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
    # Session state
    if "byt_selected" not in st.session_state:
        st.session_state["byt_selected"] = set()
    if "byt_recs" not in st.session_state:
        st.session_state["byt_recs"] = None

    st.markdown("### Build Your Taste")
    st.markdown(
        "Pick 3–10 songs you like and get personalized recommendations from our "
        "hybrid model. The system performs **CF fold-in inference** (no retraining "
        "required) plus content-based blending in real time."
    )
    st.info(
        "First click on this tab loads the full CF model (~340 MB). "
        "Subsequent recommendations are fast — models stay cached.",
        icon="ℹ️",
    )

    demo_lib = load_demo_library()

    # Selection counter row
    n_sel = len(st.session_state["byt_selected"])
    hdr_c1, hdr_c2 = st.columns([4, 1])
    with hdr_c1:
        color = "#1F4E79" if n_sel >= 3 else "#888"
        ready = "  ✓  Ready to recommend!" if n_sel >= 3 else f"  — pick at least {3 - n_sel} more"
        st.markdown(
            f'<p style="font-family: Helvetica Neue, Arial, sans-serif; '
            f'font-size:1.05em; color:{color}; margin:0;">'
            f'<strong>{n_sel}/10</strong> songs selected{ready}'
            f'</p>',
            unsafe_allow_html=True,
        )
    with hdr_c2:
        if st.button("Clear all", key="byt_clear", use_container_width=True):
            st.session_state["byt_selected"] = set()
            st.session_state["byt_recs"] = None

    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)

    # ── Song library grid ─────────────────────────────────────────────────────
    for broad_genre, grp_df in demo_lib.groupby("broad_genre"):
        st.markdown(
            f'<p style="font-family: Helvetica Neue, Arial, sans-serif; '
            f'font-weight:600; font-size:1.0em; color:#1F4E79; '
            f'margin: 1rem 0 0.3rem 0; border-bottom: 1px solid #D0E0F0; '
            f'padding-bottom: 0.2rem;">{_html.escape(broad_genre)}</p>',
            unsafe_allow_html=True,
        )
        songs = grp_df.to_dict("records")
        for row_start in range(0, len(songs), 4):
            row_songs = songs[row_start : row_start + 4]
            cols = st.columns(4)
            for col, song in zip(cols, row_songs):
                sid    = song["song_id"]
                is_sel = sid in st.session_state["byt_selected"]
                title  = _html.escape(str(song["title"])[:46])
                artist = _html.escape(str(song["artist_name"])[:36])
                genre  = _html.escape(str(song["track_genre"]))

                bg     = "#EEF3F9" if is_sel else "#FFFFFF"
                border = "2px solid #1F4E79" if is_sel else "1px solid #E0E8F0"
                tick   = ('<span style="float:right;color:#1F4E79;font-size:1em;'
                          'font-weight:700;margin-top:-2px;">✓</span>') if is_sel else ""

                with col:
                    st.markdown(
                        f'<div style="background:{bg};border:{border};border-radius:8px;'
                        f'padding:10px 12px;min-height:90px;margin-bottom:2px;'
                        f'box-shadow:0 1px 3px rgba(0,0,0,0.07);">'
                        f'{tick}'
                        f'<div style="font-weight:600;font-size:0.85em;color:#1A1A1A;'
                        f'line-height:1.3;margin-bottom:2px;">{title}</div>'
                        f'<div style="font-size:0.78em;color:#666;margin-bottom:5px;">{artist}</div>'
                        f'<span style="font-size:0.7em;color:#555;background:#F0F0F0;'
                        f'padding:2px 6px;border-radius:3px;">{genre}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    btn_label = "✓ Remove" if is_sel else "+ Add"
                    if st.button(btn_label, key=f"byt_{sid}", use_container_width=True):
                        if is_sel:
                            st.session_state["byt_selected"].discard(sid)
                        elif n_sel < 10:
                            st.session_state["byt_selected"].add(sid)
                        st.session_state["byt_recs"] = None

    # ── Recommend button ──────────────────────────────────────────────────────
    st.markdown("---")
    n_now = len(st.session_state["byt_selected"])
    btn_label_rec = (
        f"Get Recommendations  ({n_now} songs selected)"
        if n_now >= 3
        else f"Select at least 3 songs  ({n_now} / 3)"
    )
    rec_col, _ = st.columns([2, 3])
    with rec_col:
        clicked = st.button(
            btn_label_rec,
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

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.get("byt_recs") is not None:
        recs = st.session_state["byt_recs"]
        alpha_val = float(recs["alpha_used"].iloc[0]) if len(recs) > 0 else None

        st.markdown("---")
        alpha_note = (
            f'&nbsp;&nbsp;<span style="font-size:0.6em;color:#666;'
            f'font-family:Helvetica Neue,Arial,sans-serif;font-weight:400;">'
            f'α = {alpha_val:.3f} · CF weight</span>'
        ) if alpha_val is not None else ""
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
                g     = _html.escape(str(rec.get("track_genre", "")))
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
