# A Hybrid Music Recommendation System: Integrating Collaborative Filtering and Audio-Feature Similarity

## Summary

The goal of this project is to implement a Spotify-style hybrid recommendation system that suggests songs based on user listening and engagement habits. We model personalization using collaborative filtering with matrix factorization trained on the Million Song Dataset (Taste Profile Subset). To improve recommendation relevance, especially for "cold start" scenarios involving new or less active users, we use the Spotify Tracks Dataset to fetch high-dimensional audio features and compare songs by similarity. The final system combines collaborative filtering and content-based similarity into a hybrid recommender capable of generating both personalized and "similar song" recommendations. We will evaluate the model's performance using offline ranking metrics and aim to produce an end-to-end pipeline featuring data visualizations and an interactive web application interface.

### Methodology

For collaborative filtering, we construct a sparse user-item matrix from the Taste Profile Subset, filtering to users with at least 20 interactions and songs with at least 50 listeners (~661K users, ~98K songs, ~40M interactions). We apply a log1p transform to play counts and train an Alternating Least Squares (ALS) model with 64 latent factors, L2 regularization (λ = 0.1), and implicit feedback confidence weighting (α = 40). The model learns latent representations for users and songs, enabling personalized recommendations via dot-product scoring.

For content-based filtering, we match Taste Profile songs to the Spotify Tracks Dataset using exact and fuzzy title/artist matching (~11–12K matched songs). We standardize 9 audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo) via z-score normalization and fit a K-Nearest Neighbors model with cosine distance to retrieve acoustically similar songs.

We evaluate collaborative filtering using a leave-one-out protocol with HitRate@K and Recall@K, and the content-based model using GenreConsistency@K and average cosine similarity. Future work includes combining both into a hybrid recommender and deploying it as an interactive web application.

## Data Sources

- **Taste Profile Subset (Million Song Dataset):** ~48M user-song-play count triplets across ~1M users and ~385K songs. http://millionsongdataset.com/tasteprofile/
- **MSD Track Metadata Database:** SQLite database with song titles, artist names, and metadata for all MSD tracks. http://millionsongdataset.com/pages/getting-dataset/
- **Spotify Tracks Dataset:** ~114K tracks with audio features across 114 genres. https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
