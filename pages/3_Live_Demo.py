import os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pages._style import inject, figcap

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
    cm      = load_content_model()
    return cf, cm, pop, meta_cat

# ── Load always-needed data ───────────────────────────────────────────────────
warm_recs  = load_warm_recs()
history_df = load_history()
user_info  = load_user_info()
sf         = load_song_features()

uid_list = user_info["user_id"].tolist()
uid_to_label = {
    row["user_id"]: (
        f"User {i+1}  ({row['n_interactions']} interactions  ·  "
        f"most-played: {row['most_played_title'][:32]} — {row['most_played_artist'][:22]}"
        f"  ·  α = {row['adaptive_alpha']:.3f})"
    )
    for i, (_, row) in enumerate(user_info.iterrows())
}
label_to_uid = {v: k for k, v in uid_to_label.items()}

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Live Demo")
st.markdown("""
This page provides interactive exploration of the hybrid recommender system.
Three demonstrations are provided below:

- **Warm-User Demo** — compare Popularity, CF, and Hybrid (adaptive) recommendations
  for any of the 1,000 test users using precomputed top-20 ranked lists.
- **Cold-Start Demo** — select a seed song from the 7,611-track content catalog and
  observe recommendations from the Content k-NN, genre-conditioned Popularity, and
  Hybrid (genre-conditioned RRF) models in real time.
- **Alpha Tuning** — manually override the CF/content blending weight for any warm
  test user and observe how the ranked list changes. The adaptive alpha the system
  would assign is shown alongside the manual setting for comparison.
""")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "Tab 1: Warm-User Demo",
    "Tab 2: Cold-Start Demo",
    "Tab 3: Alpha Tuning",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Warm-User Demo
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Warm-User Recommendations")
    st.markdown(
        "Select a test user. Recommendations are precomputed at *k* = 20 "
        "using the full trained models with training interactions withheld."
    )

    chosen_label = st.selectbox("Select user", list(uid_to_label.values()), key="warm_user")
    chosen_uid   = label_to_uid[chosen_label]
    row_ui       = user_info[user_info["user_id"] == chosen_uid].iloc[0]

    st.caption(
        f"Interaction count: **{row_ui['n_interactions']}**  ·  "
        f"Adaptive α: **{row_ui['adaptive_alpha']:.3f}**  ·  "
        f"Most-played in content catalog: *{row_ui['most_played_title']}* "
        f"— {row_ui['most_played_artist']}"
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

    cf_songs  = set(warm_recs[(warm_recs["user_id"]==chosen_uid)&(warm_recs["model"]=="cf")].head(10)["song_id"])
    hy_songs  = set(warm_recs[(warm_recs["user_id"]==chosen_uid)&(warm_recs["model"]=="hybrid")].head(10)["song_id"])
    known_cf  = (warm_recs[(warm_recs["user_id"]==chosen_uid)&(warm_recs["model"]=="cf")].head(10)["title"] != "Unknown").sum()
    st.info(
        f"Hybrid and CF share **{len(cf_songs & hy_songs)} of 10** recommendations for this user. "
        f"CF: {known_cf}/10 recommendations have Spotify metadata (CF draws from 98K songs; "
        "only ~12K have matched Spotify records — a structural catalog overlap limitation)."
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
# TAB 3: Alpha Tuning
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Alpha Tuning — Manual Blending Override")
    st.markdown(
        "Use the slider to override the CF/content blending weight and observe how "
        "the ranked list shifts. The adaptive alpha the system would assign (based on "
        "the selected user's interaction count) is shown for reference."
    )
    st.info(
        "This tab loads the full CF model (~340 MB). First invocation takes 10–30 s; "
        "subsequent slider interactions are fast (models are cached)."
    )

    col_ctrl, col_info3 = st.columns([2, 1])
    with col_ctrl:
        alpha_slider    = st.slider("Alpha (CF weight → 1.0 = pure CF, 0.0 = pure content)",
                                    0.0, 1.0, 0.5, step=0.05, key="alpha_slider")
        chosen_t3_label = st.selectbox("Select user", list(uid_to_label.values()), key="alpha_user")
    with col_info3:
        chosen_uid_t3 = label_to_uid[chosen_t3_label]
        adaptive_a    = user_info.loc[user_info["user_id"]==chosen_uid_t3, "adaptive_alpha"].iloc[0]
        n_int_t3      = user_info.loc[user_info["user_id"]==chosen_uid_t3, "n_interactions"].iloc[0]
        st.metric("Adaptive α (system default)", f"{adaptive_a:.3f}")
        st.caption(f"n_history = {n_int_t3}")
        delta_a = alpha_slider - adaptive_a
        sign    = "+" if delta_a >= 0 else ""
        st.caption(f"Manual α = {alpha_slider:.2f}  ({sign}{delta_a:.3f} vs adaptive)")

    with st.spinner("Loading CF + content models…"):
        cf_t3, cm_t3, pop_t3, meta_t3 = load_cf_and_hybrid()
        from src.hybrid import HybridRecommender as HR
        hybrid_t3 = HR(cf_t3, cm_t3, meta_t3, alpha_strategy="fixed", alpha=alpha_slider)

    try:
        recs_t3 = hybrid_t3.recommend(user_id=chosen_uid_t3, k=10)
        disp_cols = [c for c in ["song_id","title","artist_name","track_genre",
                                  "hybrid_score","cf_score","content_score","source"]
                     if c in recs_t3.columns]
        d = recs_t3[disp_cols].copy()
        for col in ["hybrid_score","cf_score","content_score"]:
            if col in d.columns:
                d[col] = d[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        d = d.rename(columns={"title":"Title","artist_name":"Artist","track_genre":"Genre",
                               "hybrid_score":"Score","cf_score":"CF","content_score":"Content",
                               "source":"Source","song_id":"Song ID"})
        st.dataframe(d, use_container_width=True, hide_index=True)

        cf_precomp = set(warm_recs[(warm_recs["user_id"]==chosen_uid_t3)
                                   &(warm_recs["model"]=="cf")].head(10)["song_id"])
        live_set   = set(recs_t3["song_id"].tolist())
        st.caption(
            f"Overlap with precomputed CF top-10: **{len(cf_precomp & live_set)}/10**  ·  "
            f"Manual α = {alpha_slider:.2f}  ·  Adaptive α = {adaptive_a:.3f}"
        )
    except Exception as e:
        st.error(f"Recommendation error: {e}")
