"""
context_builder.py
------------------
Bước 6 trong pipeline: đóng gói tất cả thông tin thành
prompt context gửi vào LLM API.

Nhiệm vụ chính:
    1. Build user profile summary (~200 tokens)
    2. Gom reasoning trail từ Explainability Layer
    3. Thêm constraints (exclude genres, confidence level)
    4. Nếu explain_previous: lấy trail từ Memory thay vì retrieval mới

Token budget (tổng ~1250 tokens để an toàn với 4K context):
    User profile:    ~200 tokens
    Reasoning trail: ~300 tokens
    Candidates:      ~150 tokens × 5 = 750 tokens
    Total:           ~1250 tokens

Tại sao không gửi toàn bộ history vào LLM:
    - Memory chỉ giữ reasoning trail của turn trước
    - Không cần gửi toàn bộ conversation history
    - Tránh token bloat — chỉ gửi đúng những gì LLM cần
      để trả lời query hiện tại
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data_layer import DataStore
from src.memory import Turn


# ── User Profile ─────────────────────────────────────────────────

def _build_user_profile(user_id: int, ds: DataStore) -> str:
    """
    Tóm tắt profile của user:
    - Tổng số ratings, avg rating
    - Top 3 genres (by count)
    - Top 5 phim được rate cao nhất

    Giữ ngắn (~200 tokens) vì chỉ là context nền,
    reasoning trail mới là phần quan trọng.
    """
    user_ratings = ds.ratings_df[ds.ratings_df["userId"] == user_id]

    if user_ratings.empty:
        return f"userId={user_id}: no ratings found in dataset."

    n_ratings  = len(user_ratings)
    avg_rating = round(float(user_ratings["rating"].mean()), 2)

    # Top 5 highest rated movies
    top_rated = (
        user_ratings
        .sort_values("rating", ascending=False)
        .head(5)
        .merge(ds.movies_df[["movieId", "title"]], on="movieId", how="left")
    )
    top_str = ", ".join(
        f"{row['title']} ({row['rating']})"
        for _, row in top_rated.iterrows()
        if pd.notna(row.get("title"))
    )

    # Top 3 genres by count
    genre_counts: dict[str, int] = {}
    merged = user_ratings.merge(
        ds.movies_df[["movieId", "genres"]], on="movieId", how="left"
    )
    for _, row in merged.iterrows():
        for g in str(row.get("genres", "")).split("|"):
            g = g.strip()
            if g and g != "nan":
                genre_counts[g] = genre_counts.get(g, 0) + 1

    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    genre_str  = ", ".join(f"{g} ({c} films)" for g, c in top_genres)

    return (
        f"userId={user_id} | {n_ratings} ratings | avg={avg_rating}\n"
        f"Top genres: {genre_str}\n"
        f"Highest rated: {top_str}"
    )


# ── Constraints section ───────────────────────────────────────────

def _build_constraints_section(constraints: dict) -> str:
    """Format constraints thành text để nhúng vào prompt."""
    if not constraints:
        return ""

    lines = []
    if constraints.get("exclude_genres"):
        genres = ", ".join(constraints["exclude_genres"])
        lines.append(f"Exclude genres: {genres}")

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────

def build_context(
    user_id:          int,
    query:            str,
    intent:           str,
    reasoning_trail:  str,
    constraints:      dict,
    ds:               DataStore,
    previous_turn:    Optional[Turn] = None,
) -> str:
    """
    Đóng gói tất cả thành prompt context cho LLM.

    Cấu trúc output:
        [USER PROFILE]
        ...

        [PREVIOUS RECOMMENDATION]   ← chỉ có khi explain_previous
        ...

        [REASONING TRAIL]
        ...

        [CONSTRAINTS]               ← chỉ có khi có constraints
        ...

    Args:
        user_id:         ID của user
        query:           câu hỏi gốc của user
        intent:          intent từ Query Classifier
        reasoning_trail: text trail từ Explainability Layer
        constraints:     dict từ Query Classifier
        ds:              DataStore
        previous_turn:   Turn gần nhất từ Memory
                         (chỉ dùng khi intent=explain_previous)

    Returns:
        Formatted string, sẵn sàng để gửi vào LLM API
        dưới dạng user message (trước query)
    """
    sections = []

    # ── User Profile ─────────────────────────────────────────────
    sections.append(
        f"[USER PROFILE]\n{_build_user_profile(user_id, ds)}"
    )

    # ── Previous turn (chỉ khi explain_previous) ─────────────────
    if intent == "explain_previous" and previous_turn is not None:
        rec_titles = ", ".join(
            r.get("title", "Unknown")
            for r in previous_turn.recommendations[:3]
        )
        sections.append(
            f"[PREVIOUS RECOMMENDATION]\n"
            f"Previous query: {previous_turn.query}\n"
            f"Intent: {previous_turn.intent}\n"
            f"Recommended: {rec_titles if rec_titles else 'N/A'}"
        )

    # ── Reasoning Trail ──────────────────────────────────────────
    # Nếu explain_previous → dùng trail của turn trước
    # Các intent khác      → dùng trail mới từ Explainability Layer
    if intent == "explain_previous" and previous_turn is not None:
        sections.append(
            f"[REASONING TRAIL FROM PREVIOUS TURN]\n"
            f"{previous_turn.reasoning_trail}"
        )
    else:
        sections.append(
            f"[REASONING TRAIL]\n{reasoning_trail}"
        )

    # ── Constraints ───────────────────────────────────────────────
    constraints_text = _build_constraints_section(constraints)
    if constraints_text:
        sections.append(f"[CONSTRAINTS]\n{constraints_text}")

    return "\n\n".join(sections)