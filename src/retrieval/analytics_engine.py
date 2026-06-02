"""
analytics_engine.py
-------------------
Analytics Engine: phân tích genre profile của user
và tìm blind spots bằng cách so sánh với dataset distribution.

Dùng cho các query dạng:
- "What's my blind spot?"
- "What genres am I missing?"
- "What kind of movies do I usually like?"

Cơ chế blind spot:
    user_genre_pct   = % mỗi genre trong lịch sử user
    dataset_genre_pct = % mỗi genre trong toàn bộ dataset
    gap = dataset_pct - user_pct

    gap lớn → genre phổ biến trong dataset nhưng user ít xem
    → đây là blind spot tiềm năng

Điểm yếu đã biết:
- Genre là multi-label (1 phim có nhiều genres)
  → tổng % > 100%, dùng để so sánh tương đối chứ không tuyệt đối
- Một số genres rất hiếm (Film-Noir, Western) → gap cao nhưng
  không hẳn là blind spot thực sự
"""

from __future__ import annotations

import pandas as pd

from config import BLIND_SPOT_GAP
from src.data_layer import DataStore


# ── Helpers ──────────────────────────────────────────────────────

def _parse_genres(genres_str: str) -> list[str]:
    """Tách chuỗi genres pipe-separated thành list."""
    return [
        g.strip()
        for g in str(genres_str).split("|")
        if g.strip() and g.strip() != "nan" and g.strip() != "(no genres listed)"
    ]


def _compute_genre_stats(
    user_ratings: pd.DataFrame,
    movies_df:    pd.DataFrame,
) -> dict[str, dict]:
    """
    Tính count và avg_rating theo genre cho user.

    Vì 1 phim có nhiều genre, mỗi genre được count riêng.
    """
    merged = user_ratings.merge(
        movies_df[["movieId", "genres"]],
        on="movieId",
        how="left",
    )

    genre_stats: dict[str, dict] = {}

    for _, row in merged.iterrows():
        rating = float(row["rating"])
        for g in _parse_genres(str(row.get("genres", ""))):
            if g not in genre_stats:
                genre_stats[g] = {"count": 0, "total_rating": 0.0}
            genre_stats[g]["count"]        += 1
            genre_stats[g]["total_rating"] += rating

    # Tính avg_rating
    for g, v in genre_stats.items():
        v["avg_rating"] = round(v["total_rating"] / v["count"], 2)
        del v["total_rating"]

    return genre_stats


def _compute_genre_pct(genre_counts: dict[str, int]) -> dict[str, float]:
    """Chuyển count → % (trên tổng số genre occurrences)."""
    total = sum(genre_counts.values())
    if total == 0:
        return {}
    return {
        g: round(c / total * 100, 1)
        for g, c in genre_counts.items()
    }


def _find_blind_spots(
    user_genre_pct:    dict[str, float],
    dataset_genre_pct: dict[str, float],
    min_gap:           float = BLIND_SPOT_GAP,
) -> list[dict]:
    """
    Tìm genres có gap >= min_gap giữa dataset và user.

    Chỉ xét genres có trong dataset (không xét genres
    user đã xem nhưng không phổ biến trong dataset).
    """
    blind_spots = []

    for g, ds_pct in dataset_genre_pct.items():
        user_pct = user_genre_pct.get(g, 0.0)
        gap      = round(ds_pct - user_pct, 1)
        if gap >= min_gap:
            blind_spots.append({
                "genre":       g,
                "user_pct":    user_pct,
                "dataset_pct": ds_pct,
                "gap":         gap,
            })

    blind_spots.sort(key=lambda x: x["gap"], reverse=True)
    return blind_spots


def _compute_dataset_genre_distribution(movies_df: pd.DataFrame) -> dict[str, float]:
    """
    Tính % mỗi genre trong toàn bộ dataset.
    Được gọi một lần và kết quả ổn định (dataset không thay đổi).
    """
    genre_counts: dict[str, int] = {}

    for genres_str in movies_df["genres"]:
        for g in _parse_genres(str(genres_str)):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    return _compute_genre_pct(genre_counts)


# ── Public API ───────────────────────────────────────────────────

def analytics_engine(user_id: int, ds: DataStore) -> dict:
    """
    Phân tích profile của user:
    - Genre distribution (count + avg rating)
    - So sánh với dataset → tìm blind spots

    Args:
        user_id: ID của user cần phân tích
        ds:      DataStore

    Returns:
        dict với type, n_ratings, avg_rating,
        genre_stats, user_genre_pct, blind_spots

    Ví dụ blind_spots:
        Documentary: user 2% vs dataset 20% → gap = -18%
        → User ít xem Documentary dù genre này phổ biến
    """
    user_ratings = ds.ratings_df[ds.ratings_df["userId"] == user_id]
    if user_ratings.empty:
        return {"error": f"User {user_id} not found in dataset"}

    # ── Genre stats của user ─────────────────────────────────────
    genre_stats = _compute_genre_stats(user_ratings, ds.movies_df)

    user_genre_counts = {g: v["count"] for g, v in genre_stats.items()}
    user_genre_pct    = _compute_genre_pct(user_genre_counts)

    # ── Dataset distribution ─────────────────────────────────────
    dataset_genre_pct = _compute_dataset_genre_distribution(ds.movies_df)

    # ── Blind spots ──────────────────────────────────────────────
    blind_spots = _find_blind_spots(user_genre_pct, dataset_genre_pct)

    # ── High / low rated genres ──────────────────────────────────
    # Chỉ xét genres có ít nhất 3 phim để tránh noise
    high_rated = sorted(
        [(g, v["avg_rating"]) for g, v in genre_stats.items() if v["count"] >= 3],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    low_rated = sorted(
        [(g, v["avg_rating"]) for g, v in genre_stats.items() if v["count"] >= 3],
        key=lambda x: x[1],
    )[:3]

    return {
        "type":             "analytics",
        "n_ratings":        len(user_ratings),
        "avg_rating":       round(float(user_ratings["rating"].mean()), 2),
        "genre_stats":      genre_stats,
        "user_genre_pct":   user_genre_pct,
        "dataset_genre_pct": dataset_genre_pct,
        "blind_spots":      blind_spots[:5],
        "high_rated_genres": high_rated,
        "low_rated_genres":  low_rated,
    }