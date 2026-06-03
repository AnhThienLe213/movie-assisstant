"""
lookup_module.py
----------------
Lookup Module: tra cứu thông tin của một phim cụ thể từ dataset.

Dùng cho các query dạng:
- "Explain the plot of Inception"
- "Tell me about Pulp Fiction"
- "What is Interstellar about?"

Điểm khác biệt với Content Engine:
- Content Engine: tìm phim TƯƠNG TỰ query → trả về list
- Lookup Module: tra cứu phim CỤ THỂ đã biết tên → trả về 1 kết quả

Quan trọng — grounding rule:
    LLM chỉ được dùng plot từ dataset để giải thích.
    KHÔNG được dùng LLM knowledge về phim dù phim đó rất nổi tiếng.
    Nếu phim không có trong dataset → trả về error, không hallucinate.

Điểm yếu đã biết:
- Một số phim nổi tiếng vắng mặt (The Matrix, Ocean's Eleven)
  do thiếu plot data trong source → lookup trả về error
- Title matching exact → "inception" match "Inception (2010)" ổn
  nhưng typo như "Inceptionn" sẽ không match
"""

from __future__ import annotations

from config import SPARSE_MOVIE_THR
from src.data_layer import DataStore


# ── Public API ───────────────────────────────────────────────────

def lookup_module(movie_id: int, ds: DataStore) -> dict:
    """
    Tra cứu thông tin đầy đủ của một phim từ dataset.

    Args:
        movie_id: movieId của phim cần tra cứu
                  (được extract bởi Query Classifier từ tên phim)
        ds:       DataStore

    Returns:
        dict với type, movieId, title, year, genres,
        plot (full), avg_rating, n_ratings, tags
        hoặc {"error": "..."} nếu không tìm thấy

    Lưu ý:
        Plot được trả về đầy đủ (không truncate) để
        LLM có đủ thông tin để giải thích nội dung.
        Context Builder sẽ quyết định truncate bao nhiêu
        cho phù hợp với token budget.
    """
    rows = ds.movies_df[ds.movies_df["movieId"] == movie_id]

    if rows.empty:
        return {
            "error": (
                f"Movie ID {movie_id} not found in dataset. "
                "This movie may be absent due to missing plot data in the source."
            )
        }

    row = rows.iloc[0]
    mid = int(row["movieId"])

    avg_rating = ds.movie_avg_rating.get(mid, None)
    n_ratings  = ds.movie_rating_count.get(mid, 0)
    tags       = ds.movie_tags.get(mid, [])

    sparse_flag = n_ratings < SPARSE_MOVIE_THR

    return {
        "type":            "lookup",
        "movieId":         mid,
        "title":           str(row["title"]),
        "year":            str(row.get("year", "N/A")),
        "genres":          str(row.get("genres", "")),
        "plot":            str(row["plot"]),          # full plot cho LLM
        "avg_rating":      round(avg_rating, 2) if avg_rating is not None else None,
        "n_ratings":       n_ratings,
        "tags":            list(set(tags))[:10],      # deduplicated, max 10
        "is_sparse_movie": sparse_flag,
    }