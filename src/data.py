import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# Default data directory: resolved relative to this file so it always points to
# DSCI-441-Project/dataset/ regardless of the working directory.
DATA_DIR = Path(__file__).parent.parent / 'dataset'

AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo',
]


def clean_text(text: str) -> str:
    """Normalize song title / artist name for cross-dataset matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\(.*?\)", "", text)   # drop parenthetical (remixes, versions, etc.)
    text = re.sub(r"\[.*?\]", "", text)   # drop bracketed annotations
    text = re.sub(r"feat\.?.*", "", text) # drop featuring credits
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_msd_triplets(
    data_dir=None,
    min_user_count: int = 20,
    min_song_count: int = 50,
) -> pd.DataFrame:
    """
    Load MSD Taste Profile triplets and apply activity filters.
    Keeps only users who listened to >= min_user_count songs and songs
    with >= min_song_count listeners.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = data_dir / 'taste-profile' / 'train_triplets.txt'
    tp = pd.read_csv(str(path), sep='\t', header=None, names=['uid', 'sid', 'count'])

    user_counts = tp.groupby('uid')['sid'].nunique()
    song_counts = tp.groupby('sid')['uid'].nunique()

    valid_users = user_counts[user_counts >= min_user_count].index
    valid_songs = song_counts[song_counts >= min_song_count].index

    return tp[tp['uid'].isin(valid_users) & tp['sid'].isin(valid_songs)].copy()


def load_msd_metadata(db_path=None) -> pd.DataFrame:
    """
    Load song metadata from the MSD track_metadata SQLite database.
    Returns song_id, title, artist_name, title_clean, artist_clean.
    """
    db_path = db_path or DATA_DIR / 'taste-profile' / 'track_metadata.db'
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql("SELECT song_id, title, artist_name FROM songs", conn)
    conn.close()
    df['title_clean'] = df['title'].apply(clean_text)
    df['artist_clean'] = df['artist_name'].apply(clean_text)
    return df


def load_spotify_tracks(csv_path=None) -> pd.DataFrame:
    """
    Load the maharshipandya/spotify-tracks-dataset CSV.
    Renames track_name -> title, artists -> artist_name, adds *_clean columns.
    """
    csv_path = csv_path or DATA_DIR / 'spotify-tracks' / 'dataset.csv'
    df = pd.read_csv(str(csv_path))
    df = df.rename(columns={'track_name': 'title', 'artists': 'artist_name'})
    df['title_clean'] = df['title'].apply(clean_text)
    df['artist_clean'] = df['artist_name'].apply(clean_text)
    return df


# Columns from Spotify we care about (everything else is dropped for memory)
_SPOTIFY_KEEP = [
    'track_id', 'popularity',
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo',
    'track_genre',
    'title_clean', 'artist_clean',
]


def match_datasets(
    spotify_df: pd.DataFrame,
    msd_meta: pd.DataFrame,
    fuzzy_threshold: int = 90,
) -> pd.DataFrame:
    """
    Match MSD songs to Spotify tracks via exact + fuzzy title/artist matching.

    Strategy:
      1. Exact merge on (title_clean, artist_clean) -> ~11K unique songs.
      2. Fuzzy match on unmatched MSD songs: exact title_clean, then
         rapidfuzz token_set_ratio > fuzzy_threshold on artist_clean -> ~1K extra.
      Union of both gives ~12.3K unique songs vs 11.3K from exact alone.

    Returns a DataFrame with columns:
      song_id, title, artist_name, title_clean, artist_clean,
      track_id, popularity, [9 audio features], track_genre,
      match_type ('exact'|'fuzzy'), fuzzy_score (100 for exact).

    Both input DataFrames must already have title_clean and artist_clean columns
    (added by load_msd_metadata / load_spotify_tracks).
    """
    from rapidfuzz.fuzz import token_set_ratio

    spotify_slim = spotify_df[_SPOTIFY_KEEP].copy()

    # --- Step 1: exact match ---
    exact_merged = msd_meta.merge(
        spotify_slim,
        on=['title_clean', 'artist_clean'],
        how='inner',
    )
    exact_merged = exact_merged.assign(match_type='exact', fuzzy_score=100)

    out_cols = (
        ['song_id', 'title', 'artist_name', 'title_clean', 'artist_clean',
         'track_id', 'popularity'] +
        AUDIO_FEATURES +
        ['track_genre', 'match_type', 'fuzzy_score']
    )
    exact_out = exact_merged[out_cols]
    exact_song_ids = set(exact_out['song_id'])

    # --- Step 2: fuzzy match for unmatched MSD songs ---
    msd_unmatched = msd_meta[~msd_meta['song_id'].isin(exact_song_ids)]

    # Skip MSD songs whose title_clean doesn't appear in Spotify at all.
    spotify_titles = set(spotify_slim['title_clean'])
    msd_candidates = msd_unmatched[msd_unmatched['title_clean'].isin(spotify_titles)]

    if len(msd_candidates) > 0:
        spotify_by_title = spotify_slim.groupby('title_clean')

        fuzzy_rows = []
        for _, row in msd_candidates.iterrows():
            candidates = spotify_by_title.get_group(row['title_clean'])
            for _, cand in candidates.iterrows():
                score = token_set_ratio(row['artist_clean'], cand['artist_clean'])
                if score >= fuzzy_threshold:
                    fuzzy_rows.append({
                        'song_id':          row['song_id'],
                        'title':            row['title'],
                        'artist_name':      row['artist_name'],
                        'title_clean':      row['title_clean'],
                        'artist_clean':     row['artist_clean'],
                        'track_id':         cand['track_id'],
                        'popularity':       cand['popularity'],
                        'danceability':     cand['danceability'],
                        'energy':           cand['energy'],
                        'loudness':         cand['loudness'],
                        'speechiness':      cand['speechiness'],
                        'acousticness':     cand['acousticness'],
                        'instrumentalness': cand['instrumentalness'],
                        'liveness':         cand['liveness'],
                        'valence':          cand['valence'],
                        'tempo':            cand['tempo'],
                        'track_genre':      cand['track_genre'],
                        'match_type':       'fuzzy',
                        'fuzzy_score':      score,
                    })

        if fuzzy_rows:
            fuzzy_out = pd.DataFrame(fuzzy_rows)
            return pd.concat([exact_out, fuzzy_out], ignore_index=True)

    return exact_out.reset_index(drop=True)


def dedup_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate tracks by (title_clean, artist_clean), keeping the entry
    with the highest Spotify popularity.

    This fixes the variant-track contamination bug: songs like
    'I Ran', 'I Ran (L Remix)', 'I Ran (So Far Away)' all clean to the same
    key and would all appear in each other's k-NN results with similarity=1.0.
    After dedup, only the canonical (highest-popularity) version is in the index.
    """
    return (
        df.sort_values('popularity', ascending=False)
          .drop_duplicates(subset=['title_clean', 'artist_clean'])
          .reset_index(drop=True)
    )


def build_metadata_catalog(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-song_id metadata catalog for CF recommendation display.

    Unlike dedup_tracks() (which dedupes by cleaned title/artist for the content
    k-NN index, intentionally collapsing 'I Ran' / 'I Ran (Remix)' to one row),
    this preserves every distinct MSD song_id. CF recommends from the full 98K
    catalog and needs to look up the original song's metadata, not a canonical
    representative.

    For the rare case where one song_id appears multiple times in matched_df
    (matched via both exact and fuzzy paths), keeps the highest-popularity
    Spotify candidate.
    """
    return (
        matched_df.sort_values('popularity', ascending=False)
                  .drop_duplicates(subset=['song_id'])
                  [['song_id', 'title', 'artist_name', 'track_genre', 'popularity']]
                  .reset_index(drop=True)
    )


def build_user_item_matrix(df: pd.DataFrame):
    """
    Build a sparse user-item matrix from triplets DataFrame (uid, sid, count).
    Applies log1p to play counts (standard for implicit feedback with ALS).

    Returns:
        mat       -- csr_matrix of shape (n_users, n_songs)
        mappings  -- dict with user_to_idx, idx_to_user, song_to_idx, idx_to_song
    """
    user_ids = df['uid'].unique()
    song_ids = df['sid'].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    song_to_idx = {s: i for i, s in enumerate(song_ids)}
    idx_to_song = {i: s for s, i in song_to_idx.items()}

    rows = df['uid'].map(user_to_idx).to_numpy()
    cols = df['sid'].map(song_to_idx).to_numpy()
    data = np.log1p(df['count'].to_numpy()).astype(np.float32)

    mat = coo_matrix(
        (data, (rows, cols)),
        shape=(len(user_to_idx), len(song_to_idx)),
    ).tocsr()

    return mat, {
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'song_to_idx': song_to_idx,
        'idx_to_song': idx_to_song,
    }
