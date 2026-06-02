"""
content_engine.py
-----------------
Content-based search dùng TF-IDF + Cosine Similarity trên plot.

Tại sao TF-IDF thay vì SBERT / OpenAI Embeddings:
- Chạy hoàn toàn local — không cần API call, không có latency thêm
- Explainable: biết chính xác từ nào khớp ("matched terms: twist, psychological")
- Plot dài (~3200 chars) → TF-IDF capture topic tốt ngay cả không có semantic
- SBERT tốt hơn về semantic ("bleak" ~ "dark") nhưng overkill với bài này
- Conscious tradeoff: có thể swap sang SBERT sau nếu cần

Tại sao ngram_range=(1,2):
- Bắt cụm quan trọng: "twist ending", "serial killer", "dark comedy"
- Unigram đơn thuần miss những cụm này

Điểm yếu đã biết:
- Semantic mismatch: "bleak atmosphere" ≠ "dark mood" với TF-IDF
- Query quá ngắn / abstract ("phim hay") → tất cả scores gần 0
- Tên riêng (đạo diễn, diễn viên) không có trong plot → không match
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import CONTENT_TOP_N, SPARSE_MOVIE_THR
from src.data_layer import DataStore


# ── Helpers ──────────────────────────────────────────────────────

def _get_matched_terms(
    query_vec: object,  # sparse matrix 1×vocab
    vectorizer: object,
    top_n: int = 8,
) -> list[str]:
    """
    Lấy top-N terms quan trọng nhất trong query vector.

    Đây là từ có TF-IDF weight cao nhất trong query
    → dùng để explain "matched terms: 'psychological', 'twist'"
    """
    feature_names = vectorizer.get_feature_names_out()
    query_arr     = query_vec.toarray().flatten()
    top_indices   = np.argsort(query_arr)[::-1][:top_n]
    return [feature_names[i] for i in top_indices if query_arr[i] > 0]


def _apply_genre_exclusion(
    scores:         np.ndarray,
    movies_df:      object,  # pd.DataFrame
    exclude_genres: list[str],
) -> np.ndarray:
    """
    Set score = 0 cho phim thuộc genre bị loại trừ.

    Thực hiện in-place trên copy của scores.
    """
    if not exclude_genres:
        return scores

    scores = scores.copy()
    for i, row in movies_df.iterrows():
        genres = str(row.get("genres", ""))
        if any(g.lower() in genres.lower() for g in exclude_genres):
            scores[i] = 0.0

    return scores


def _build_search_query(
    query:     str,
    title_ref: Optional[int],
    ds:        DataStore,
) -> str:
    """
    Xây dựng query cuối cùng để vectorize.

    Nếu có title_ref (ví dụ: "Like Toy Story but no animation"):
        → Mở rộng query với 500 chars đầu của plot phim đó
        → Content search sẽ tìm phim có plot tương tự
    Nếu không:
        → Dùng raw query string
    """
    if title_ref is None:
        return query

    ref_rows = ds.movies_df[ds.movies_df["movieId"] == title_ref]
    if ref_rows.empty:
        return query

    ref_plot = str(ref_rows.iloc[0]["plot"])[:500]
    return f"{query} {ref_plot}"


# ── Public API ───────────────────────────────────────────────────

def content_engine(
    query:          str,
    ds:             DataStore,
    title_ref:      Optional[int] = None,
    exclude_genres: Optional[list[str]] = None,
    top_n:          int = CONTENT_TOP_N,
) -> dict:
    """
    TF-IDF content search trên plot của toàn bộ 5,135 phim.

    Args:
        query:          query string của user
        ds:             DataStore
        title_ref:      movieId nếu query dạng "like X but..."
                        → plot của X sẽ được dùng làm query gốc
        exclude_genres: list genre cần loại trừ khỏi kết quả
        top_n:          số kết quả trả về

    Returns:
        dict với type, matched_terms, matches

    Ví dụ matched_terms:
        Query "dark psychological thriller with a twist"
        → matched_terms = ['psychological', 'thriller', 'twist ending',
                           'dark', 'suspense', ...]
    """
    if exclude_genres is None:
        exclude_genres = []

    # ── Build search query ───────────────────────────────────────
    search_query = _build_search_query(query, title_ref, ds)

    # ── Vectorize query ──────────────────────────────────────────
    # Dùng cùng vectorizer đã fit trên toàn bộ plots
    query_vec = ds.tfidf_vectorizer.transform([search_query])

    # ── Cosine similarity với toàn bộ plot matrix ────────────────
    scores = cosine_similarity(query_vec, ds.tfidf_matrix).flatten()

    # ── Lấy matched terms để explain ────────────────────────────
    matched_terms = _get_matched_terms(query_vec, ds.tfidf_vectorizer)

    # ── Apply constraints ────────────────────────────────────────
    scores = _apply_genre_exclusion(scores, ds.movies_df, exclude_genres)

    # Loại bỏ title_ref khỏi kết quả (không gợi ý lại chính nó)
    if title_ref is not None and title_ref in ds.movie_id_to_idx:
        scores[ds.movie_id_to_idx[title_ref]] = 0.0

    # ── Rank và lấy top N ────────────────────────────────────────
    top_indices = np.argsort(scores)[::-1][:top_n]

    matches = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue

        row = ds.movies_df.iloc[idx]
        mid = int(row["movieId"])

        n_ratings   = ds.movie_rating_count.get(mid, 0)
        avg_rating  = ds.movie_avg_rating.get(mid, None)
        tags        = ds.movie_tags.get(mid, [])
        sparse_flag = n_ratings < SPARSE_MOVIE_THR

        matches.append({
            "movieId":           mid,
            "title":             str(row["title"]),
            "similarity_score":  round(float(scores[idx]), 4),
            "genres":            str(row.get("genres", "")),
            "plot_snippet":      str(row["plot"])[:300],
            "avg_rating":        round(avg_rating, 2) if avg_rating else None,
            "n_ratings":         n_ratings,
            "tags":              tags[:5],
            "is_sparse_movie":   sparse_flag,
        })

    return {
        "type":          "content",
        "matched_terms": matched_terms,
        "matches":       matches,
        "query_used":    search_query[:200],   # để debug
    }