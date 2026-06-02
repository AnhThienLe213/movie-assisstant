"""
cf_engine.py
------------
Collaborative Filtering Engine dùng User-based CF
với cosine similarity trên mean-centered rating vectors.

Tại sao User-based CF thay vì SVD / Matrix Factorization:
- Interpretable: biết chính xác "User 42 (sim=0.87) rated 5.0"
- Có thể cite trực tiếp vào Explainability Layer
- Không cần training step — chạy ngay trên raw ratings
- SVD cho accuracy cao hơn nhưng latent factors không thể explain

Tại sao Cosine thay vì Pearson:
- Dataset rất sparse (user chỉ rate ~121/5135 phim)
- Cosine chỉ tính trên chiều có giá trị — không bị ảnh hưởng bởi số 0
- Pearson unreliable khi overlap giữa 2 users ít (< 20 phim chung)

Tại sao Mean-centering:
- Loại bỏ bias người hay cho điểm cao / thấp
- User A cho 3/5 = bình thường với họ (mean=3.5)
- User B cho 3/5 = thất vọng với họ (mean=4.2)
- Không center → 2 users này bị coi là giống nhau

Điểm yếu đã biết:
- Cold-start: user < 20 ratings → vector quá thưa → similarity không đáng tin
- Long-tail: phim < 5 ratings → không đủ signal để aggregate
- Scale: O(n_users²) khi tính similarity — ổn với 610 users
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import (
    CF_TOP_K_USERS,
    CF_TOP_N_MOVIES,
    SPARSE_MOVIE_THR,
    SPARSE_USER_THR,
    LOW_CONF_THR,
    MIN_RATING_HIGH,
)
from src.data_layer import DataStore


# ── Helpers ──────────────────────────────────────────────────────

def _get_user_vector(user_id: int, ds: DataStore) -> Optional[np.ndarray]:
    """
    Lấy rating vector của user từ user_item_matrix.
    Trả về None nếu user không tồn tại trong dataset.
    """
    if user_id not in ds.user_id_to_idx:
        return None
    idx = ds.user_id_to_idx[user_id]
    return np.asarray(ds.user_item_matrix[idx].todense()).flatten()


def _mean_center(vec: np.ndarray) -> np.ndarray:
    """
    Trừ mean rating của user (chỉ tính trên các phim đã rate).

    Ví dụ:
        vec      = [5, 0, 4, 0, 3]   (0 = chưa xem)
        rated    = [5, 4, 3]
        mean     = 4.0
        centered = [1, 0, 0, 0, -1]  (0 vẫn là 0 → không xem)
    """
    centered  = vec.copy().astype(float)
    rated_mask = vec != 0
    if rated_mask.sum() == 0:
        return centered
    centered[rated_mask] -= vec[rated_mask].mean()
    return centered


def _cosine_similarity_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vectors. Trả về 0 nếu một trong hai là zero vector."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_confidence_flags(n_rated: int, n_similar: int) -> list[str]:
    """Tạo danh sách cảnh báo dựa trên độ thưa của data."""
    flags = []
    if n_rated < SPARSE_USER_THR:
        flags.append(f"SPARSE_USER: chỉ có {n_rated} ratings — similarity kém tin cậy")
    if n_similar < LOW_CONF_THR:
        flags.append(f"LOW_CONFIDENCE: chỉ tìm được {n_similar} similar users")
    return flags


# ── Core logic ───────────────────────────────────────────────────

def _find_similar_users(
    target_vec_c: np.ndarray,
    target_idx:   int,
    ds:           DataStore,
    top_k:        int,
) -> list[dict]:
    """
    Tìm top-K users có cosine similarity cao nhất với target user.

    Trả về list dict với:
        userId, similarity, n_common (số phim chung), vec (raw vector)
    """
    similar = []

    for idx in range(ds.user_item_matrix.shape[0]):
        if idx == target_idx:
            continue

        other_vec   = np.asarray(ds.user_item_matrix[idx].todense()).flatten()
        other_vec_c = _mean_center(other_vec)

        sim = _cosine_similarity_1d(target_vec_c, other_vec_c)
        if sim <= 0:
            continue

        # Số phim cả 2 đã xem
        n_common = int(((target_vec_c != 0) & (other_vec != 0)).sum())

        similar.append({
            "userId":     ds.idx_to_user_id[idx],
            "similarity": round(sim, 4),
            "n_common":   n_common,
            "vec":        other_vec,   # giữ lại để tính predicted scores
        })

    # Sort theo similarity giảm dần, lấy top_k
    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return similar[:top_k]


def _predict_scores(
    unseen_mask:   np.ndarray,
    similar_users: list[dict],
) -> np.ndarray:
    """
    Weighted average predicted rating cho tất cả phim chưa xem.

    predicted[j] = Σ(sim_u × rating_u_j) / Σ(sim_u)
                   chỉ tính trên users đã rate phim j

    Phim không có similar user nào rate → predicted = 0
    """
    n_movies     = unseen_mask.shape[0]
    weighted_sum = np.zeros(n_movies)
    sim_sum      = np.zeros(n_movies)

    for su in similar_users:
        sv  = su["vec"]
        s   = su["similarity"]
        has_rated = sv != 0
        weighted_sum[has_rated] += s * sv[has_rated]
        sim_sum[has_rated]      += s

    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = np.where(sim_sum > 0, weighted_sum / sim_sum, 0.0)

    # Bỏ phim user đã xem
    predicted[~unseen_mask] = 0.0
    return predicted


# ── Public API ───────────────────────────────────────────────────

def cf_engine(
    user_id:   int,
    ds:        DataStore,
    title_ref: Optional[int] = None,
    top_k:     int = CF_TOP_K_USERS,
    top_n:     int = CF_TOP_N_MOVIES,
) -> dict:
    """
    User-based Collaborative Filtering.

    Hai chế độ hoạt động:
    1. title_ref = None  → Recommendation
       Tìm top-N phim user chưa xem, predicted score cao nhất
       dựa trên weighted average của similar users.

    2. title_ref = movieId → Multi-hop reasoning
       Tìm similar users đã rate phim đó → aggregate
       → dùng cho query "What do similar users think of X?"

    Args:
        user_id:   ID của target user
        ds:        DataStore
        title_ref: movieId nếu cần multi-hop, None nếu recommend
        top_k:     số similar users lấy
        top_n:     số candidate movies trả về (chỉ dùng khi title_ref=None)

    Returns:
        dict với type, similar_users, candidates / ratings, confidence_flags
    """
    user_vec = _get_user_vector(user_id, ds)
    if user_vec is None:
        return {"error": f"User {user_id} not found in dataset"}

    n_rated      = int((user_vec != 0).sum())
    target_idx   = ds.user_id_to_idx[user_id]
    user_vec_c   = _mean_center(user_vec)

    # ── Tìm similar users ────────────────────────────────────────
    similar_users = _find_similar_users(user_vec_c, target_idx, ds, top_k)

    confidence_flags = _build_confidence_flags(n_rated, len(similar_users))

    # Bản tóm tắt similar users (bỏ vec để không expose data thô)
    similar_users_summary = [
        {
            "userId":     su["userId"],
            "similarity": su["similarity"],
            "n_common":   su["n_common"],
        }
        for su in similar_users
    ]

    # ── Chế độ 2: Multi-hop ──────────────────────────────────────
    if title_ref is not None:
        if title_ref not in ds.movie_id_to_idx:
            return {"error": f"Movie ID {title_ref} not found in dataset"}

        movie_idx  = ds.movie_id_to_idx[title_ref]
        movie_row  = ds.movies_df[ds.movies_df["movieId"] == title_ref]
        movie_title = movie_row.iloc[0]["title"] if not movie_row.empty else str(title_ref)

        # Lấy ratings của similar users với phim này
        ratings_from_similar = []
        for su in similar_users:
            r = su["vec"][movie_idx]
            if r > 0:
                ratings_from_similar.append({
                    "userId":     su["userId"],
                    "similarity": su["similarity"],
                    "rating":     float(r),
                })

        # Weighted average
        if ratings_from_similar:
            total_w    = sum(r["similarity"] for r in ratings_from_similar)
            weighted_r = sum(r["similarity"] * r["rating"] for r in ratings_from_similar)
            predicted_score = round(weighted_r / total_w, 2) if total_w > 0 else None
        else:
            predicted_score = None

        return {
            "type":             "multihop",
            "target_movie":     {"movieId": title_ref, "title": movie_title},
            "similar_users":    similar_users_summary,
            "ratings":          ratings_from_similar,
            "predicted_score":  predicted_score,
            "n_rated":          n_rated,
            "confidence_flags": confidence_flags,
        }

    # ── Chế độ 1: Recommendation ─────────────────────────────────
    unseen_mask = user_vec == 0
    predicted   = _predict_scores(unseen_mask, similar_users)

    top_indices = np.argsort(predicted)[::-1][:top_n]

    candidates = []
    for idx in top_indices:
        if predicted[idx] <= 0:
            continue

        mid  = ds.idx_to_movie_id[idx]
        rows = ds.movies_df[ds.movies_df["movieId"] == mid]
        if rows.empty:
            continue

        # Users đã rate phim này trong top similar
        raters = [
            {
                "userId":     su["userId"],
                "similarity": su["similarity"],
                "rating":     float(su["vec"][idx]),
            }
            for su in similar_users
            if su["vec"][idx] > 0
        ]

        n_ratings = ds.movie_rating_count.get(mid, 0)
        sparse_flag = n_ratings < SPARSE_MOVIE_THR

        candidates.append({
            "movieId":         mid,
            "title":           rows.iloc[0]["title"],
            "predicted_score": round(float(predicted[idx]), 2),
            "raters":          raters,
            "n_dataset_ratings": n_ratings,
            "is_sparse_movie": sparse_flag,
        })

    return {
        "type":             "recommendation",
        "similar_users":    similar_users_summary,
        "candidates":       candidates,
        "n_rated":          n_rated,
        "confidence_flags": confidence_flags,
    }