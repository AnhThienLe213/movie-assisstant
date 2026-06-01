"""
data_layer.py
-------------
Bước 1 trong pipeline: load toàn bộ dataset một lần,
cache vào memory, tạo các cấu trúc dữ liệu cần thiết
cho các engine phía sau.

Không module nào khác được đọc CSV trực tiếp —
tất cả đều nhận DataStore từ đây.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    DATA_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)


# ── DataStore ────────────────────────────────────────────────────

@dataclass
class DataStore:
    """
    Container duy nhất chứa toàn bộ data đã được xử lý.

    Được tạo một lần bởi load_data() và truyền xuống
    tất cả các engine trong pipeline.
    """

    # Raw DataFrames
    movies_df:  pd.DataFrame = field(default=None)
    ratings_df: pd.DataFrame = field(default=None)
    tags_df:    pd.DataFrame = field(default=None)

    # Sparse user-item matrix cho CF
    # Shape: (n_users, n_movies) — giá trị 0 = chưa xem
    user_item_matrix: csr_matrix = field(default=None)

    # TF-IDF matrix trên plot cho Content Engine
    # Shape: (n_movies, vocab_size)
    tfidf_matrix:     object = field(default=None)
    tfidf_vectorizer: object = field(default=None)

    # Index mappings — tra cứu O(1)
    movie_id_to_idx: dict = field(default=None)  # movieId  → row index trong matrix
    idx_to_movie_id: dict = field(default=None)  # row index → movieId
    user_id_to_idx:  dict = field(default=None)  # userId   → row index trong matrix
    idx_to_user_id:  dict = field(default=None)  # row index → userId

    # Precomputed stats — tránh group-by lặp lại
    movie_tags:         dict = field(default=None)  # movieId → [tag1, tag2, ...]
    movie_avg_rating:   dict = field(default=None)  # movieId → float avg
    movie_rating_count: dict = field(default=None)  # movieId → int count


# ── Helpers ──────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace khỏi tên cột."""
    df.columns = df.columns.str.strip()
    return df


def _build_user_item_matrix(
    ratings_df:      pd.DataFrame,
    movie_id_to_idx: dict,
    user_id_to_idx:  dict,
    n_users:         int,
    n_movies:        int,
) -> csr_matrix:
    """
    Xây dựng sparse user-item matrix từ ratings_df.

    Chỉ dùng ratings của phim có trong movies_df
    (ratings_df có thể chứa movieId không có plot).
    """
    valid = ratings_df[ratings_df["movieId"].isin(movie_id_to_idx)]

    rows = [user_id_to_idx[uid] for uid in valid["userId"]]
    cols = [movie_id_to_idx[mid] for mid in valid["movieId"]]
    vals = valid["rating"].values.astype(float)

    return csr_matrix(
        (vals, (rows, cols)),
        shape=(n_users, n_movies),
    )


def _build_tfidf_matrix(
    plots: pd.Series,
) -> tuple[TfidfVectorizer, object]:
    """
    Fit TF-IDF vectorizer trên toàn bộ plot.

    sublinear_tf=True: dùng 1 + log(tf) thay vì tf thuần
    → giảm ảnh hưởng của từ lặp nhiều lần trong plot dài.

    ngram_range=(1,2): bắt cụm như "twist ending", "serial killer"
    → content search chính xác hơn với multi-word queries.
    """
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    matrix = vectorizer.fit_transform(plots)
    return vectorizer, matrix


def _build_movie_tag_dict(tags_df: pd.DataFrame) -> dict:
    """movieId → list of tags (lowercase, deduplicated)."""
    return (
        tags_df.groupby("movieId")["tag"]
        .apply(lambda tags: list(set(t.lower() for t in tags)))
        .to_dict()
    )


def _build_rating_stats(ratings_df: pd.DataFrame) -> tuple[dict, dict]:
    """Precompute avg rating và count cho mỗi movie."""
    stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])
    avg   = stats["mean"].round(2).to_dict()
    count = stats["count"].to_dict()
    return avg, count


# ── Public API ───────────────────────────────────────────────────

def load_data(data_dir: Path = DATA_DIR) -> DataStore:
    """
    Load toàn bộ dataset, build các structures cần thiết.

    Gọi một lần khi khởi động — kết quả được cache trong DataStore
    và truyền vào tất cả engine trong pipeline.
    """
    ds = DataStore()

    # ── Load CSVs ────────────────────────────────────────────────
    print("Loading datasets...")

    ds.movies_df  = _normalize_columns(pd.read_csv(data_dir / "movies_with_plots.csv"))
    ds.ratings_df = _normalize_columns(pd.read_csv(data_dir / "ratings.csv"))
    ds.tags_df    = _normalize_columns(pd.read_csv(data_dir / "tags.csv"))

    ds.movies_df["plot"] = ds.movies_df["plot"].fillna("")

    print(f"  Movies : {len(ds.movies_df):,}")
    print(f"  Ratings: {len(ds.ratings_df):,} from {ds.ratings_df['userId'].nunique():,} users")
    print(f"  Tags   : {len(ds.tags_df):,}")

    # ── Index mappings ───────────────────────────────────────────
    movie_ids = ds.movies_df["movieId"].tolist()
    user_ids  = ds.ratings_df["userId"].unique().tolist()

    ds.movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    ds.idx_to_movie_id = {i: mid for mid, i in ds.movie_id_to_idx.items()}
    ds.user_id_to_idx  = {uid: i for i, uid in enumerate(user_ids)}
    ds.idx_to_user_id  = {i: uid for uid, i in ds.user_id_to_idx.items()}

    # ── User-item matrix ─────────────────────────────────────────
    ds.user_item_matrix = _build_user_item_matrix(
        ds.ratings_df,
        ds.movie_id_to_idx,
        ds.user_id_to_idx,
        n_users=len(user_ids),
        n_movies=len(movie_ids),
    )
    print(f"  User-item matrix: {ds.user_item_matrix.shape} "
          f"(density={ds.user_item_matrix.nnz / (ds.user_item_matrix.shape[0] * ds.user_item_matrix.shape[1]):.4f})")

    # ── TF-IDF matrix ────────────────────────────────────────────
    ds.tfidf_vectorizer, ds.tfidf_matrix = _build_tfidf_matrix(ds.movies_df["plot"])
    print(f"  TF-IDF matrix   : {ds.tfidf_matrix.shape}")

    # ── Precomputed stats ────────────────────────────────────────
    ds.movie_tags         = _build_movie_tag_dict(ds.tags_df)
    ds.movie_avg_rating, ds.movie_rating_count = _build_rating_stats(ds.ratings_df)

    print("Data loaded.\n")
    return ds