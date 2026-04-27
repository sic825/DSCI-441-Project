"""
Build results/demo_song_library.parquet — curated 80-song demo library.

Criteria: song_id in CF catalog AND in metadata_catalog, popularity > 40,
balanced genre distribution across 8 broad groups.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd

ROOT   = Path(__file__).parent.parent
MODELS = ROOT / "models"
RESULT = ROOT / "results"

# ── Load data ─────────────────────────────────────────────────────────────────
idx_to_song = joblib.load(MODELS / "idx_to_song.pkl")
cf_ids      = set(idx_to_song.values())

meta  = pd.read_parquet(RESULT / "metadata_catalog.parquet")
valid = meta[meta["song_id"].isin(cf_ids) & (meta["popularity"] > 40)].copy()
valid = valid.drop_duplicates("song_id").reset_index(drop=True)

# ── Genre → broad group mapping ────────────────────────────────────────────────
GENRE_GROUPS = {
    "Grunge & Alternative": [
        "grunge", "alt-rock", "alternative",
    ],
    "Classic Rock & Hard Rock": [
        "hard-rock", "rock", "rock-n-roll", "rockabilly", "psych-rock",
    ],
    "Electronic & Trip-Hop": [
        "synth-pop", "trip-hop", "breakbeat", "electronic", "trance", "new-age", "ambient",
    ],
    "Metal & Industrial": [
        "metal", "metalcore", "death-metal", "hardcore", "black-metal", "industrial",
    ],
    "Punk & Emo": [
        "punk", "punk-rock", "emo", "ska",
    ],
    "Acoustic, Folk & Country": [
        "acoustic", "folk", "singer-songwriter", "songwriter", "bluegrass", "country",
    ],
    "Soul, R&B & Disco": [
        "soul", "r-n-b", "funk", "groove", "disco", "power-pop",
    ],
    "World & Blues": [
        "blues", "spanish", "swedish", "world-music", "brazil", "mpb", "garage",
    ],
}

# Targets per group (sums to ~80)
TARGETS = {
    "Grunge & Alternative":       12,
    "Classic Rock & Hard Rock":   10,
    "Electronic & Trip-Hop":      12,
    "Metal & Industrial":         10,
    "Punk & Emo":                  8,
    "Acoustic, Folk & Country":    8,
    "Soul, R&B & Disco":          10,
    "World & Blues":               8,
}

# ── Select songs ──────────────────────────────────────────────────────────────
inv_map = {}  # genre_tag -> broad_group
for group, tags in GENRE_GROUPS.items():
    for tag in tags:
        inv_map[tag] = group

valid["broad_genre"] = valid["track_genre"].map(inv_map)
labeled = valid.dropna(subset=["broad_genre"])

# Collapse version variants: "Song (Live)", "Song (Album Version)", etc.
# Keep highest-popularity entry per (clean_title, clean_artist).
import re

def _clean_for_dedup(s: str) -> str:
    s = re.sub(r'\s*[\(\[].*?[\)\]]', '', str(s))  # strip (…) and […]
    s = re.sub(r'\s*[–—-]\s*(feat|ft|with).*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[_]', ' ', s)
    return s.strip().lower()

labeled = labeled.copy()
labeled["_title_key"]  = labeled["title"].apply(_clean_for_dedup)
labeled["_artist_key"] = labeled["artist_name"].apply(_clean_for_dedup)
# Within each group, keep highest-pop entry per (title_key, artist_key)
labeled = (
    labeled.sort_values("popularity", ascending=False)
           .drop_duplicates(subset=["_title_key", "_artist_key"])
)

rows = []
for group, n_target in TARGETS.items():
    pool = labeled[labeled["broad_genre"] == group]
    selected = pool.sort_values("popularity", ascending=False).head(n_target)
    rows.append(selected)

library = pd.concat(rows, ignore_index=True)
library = library[["song_id","title","artist_name","track_genre","popularity","broad_genre"]]
library = library.sort_values(["broad_genre","popularity"], ascending=[True, False]).reset_index(drop=True)

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\nDemo library: {len(library)} songs across {library['broad_genre'].nunique()} groups\n")
for group, grp_df in library.groupby("broad_genre"):
    print(f"[{group}] — {len(grp_df)} songs")
    for _, r in grp_df.iterrows():
        print(f"   {r['title']} — {r['artist_name']}  (pop={r['popularity']})")

library.to_parquet(RESULT / "demo_song_library.parquet", index=False)
print(f"\nSaved → results/demo_song_library.parquet")
