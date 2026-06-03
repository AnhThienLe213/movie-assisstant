"""
query_classifier.py
-------------------
Bước 2 trong pipeline: nhận query string, trả về intent
và constraints để Retrieval Layer biết engine nào cần chạy.

Dùng rule-based keyword matching thay vì LLM classifier vì:
- Predictable 100%: biết chính xác query nào → route nào
- Zero latency, zero API cost
- Dễ debug và thêm keyword mới
- Chỉ có 6 intent categories → không cần LLM

Điểm yếu đã biết: brittle với paraphrase
("folks with my taste" không match "similar users").
Có thể upgrade sang LLM classifier nếu cần.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.data_layer import DataStore


# ── Intent definitions ───────────────────────────────────────────

# Mỗi intent có keyword list riêng.
# Thứ tự ưu tiên khi check: explain_previous > lookup > analytics > CF > hybrid > content

CF_KEYWORDS = [
    "similar users", "people like me", "same taste",
    "others think", "users like me", "similar taste",
    "folks like me", "people with my taste", "think of",
    "think about", "who likes similar","what should i watch", "what to watch", "recommend me",
    "suggest something", "suggest a movie", "what do you recommend",
    "what would you recommend", "what should i see", "what can i watch", "give me a recommendation",
]

ANALYTICS_KEYWORDS = [
    "blind spot", "what genres", "never watched",
    "my profile", "what kind of", "missing",
    "genre am i", "genres am i", "what do i usually",
    "my pattern", "what type", "genres missing",
    "analyze my", "my taste",
]

LOOKUP_KEYWORDS = [
    "what is", "what's", "explain", "tell me about",
    "describe", "plot of", "about the movie",
    "synopsis", "summary of", "nội dung", "giải thích",
    "về phim", "kể về",
]

EXPLAIN_KEYWORDS = [
    "why do you think", "why would i", "why that",
    "explain your", "how do you know", "what makes you think",
    "why recommend", "why did you", "tại sao",
]

# Từ chỉ loại trừ — dùng kết hợp với genre để detect hybrid
EXCLUDE_SIGNAL_WORDS = [
    "no", "not", "tired", "avoid", "without",
    "except", "but not", "no more", "skip",
]

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

# Alias mapping: từ user hay dùng → tên genre chính thức
GENRE_ALIASES: dict[str, str] = {
    "animated":    "Animation",
    "animations":  "Animation",
    "cartoons":    "Animation",
    "cartoon":     "Animation",
    "sci fi":      "Sci-Fi",
    "science fiction": "Sci-Fi",
    "scifi":       "Sci-Fi",
    "docs":        "Documentary",
    "documentaries": "Documentary",
    "romcom":      "Romance",
    "rom-com":     "Romance",
    "musicals":    "Musical",
    "westerns":    "Western",
    "horrors":     "Horror",
}


# ── Result dataclass ─────────────────────────────────────────────

@dataclass
class ClassifierResult:
    """
    Kết quả phân loại query.

    intent: loại query → quyết định engine nào chạy
    constraints: thông tin bổ sung cho engine
      - exclude_genres: list genre cần bỏ qua
      - title_reference: movieId của phim được đề cập
    is_followup: True nếu là câu hỏi tiếp theo về turn trước
    """
    intent:      str
    constraints: dict = field(default_factory=dict)
    is_followup: bool = False


# ── Private helpers ──────────────────────────────────────────────

def _contains_any(text: str, keywords: list[str]) -> bool:
    """Kiểm tra text có chứa bất kỳ keyword nào không."""
    return any(kw in text for kw in keywords)


def _find_title_reference(query: str, ds: DataStore) -> Optional[int]:
    """
    Tìm movieId của phim được nhắc đến trong query.

    Fix so với version cũ:
    - Tăng min length lên 5 để giảm false positive
    - Dùng word boundary \b thay vì substring match
      → "missing" không match phim tên "Miss"
    - Ưu tiên match dài nhất khi nhiều phim cùng match
      → "Toy Story 2" thắng "Toy Story"
    """
    query_lower = query.lower()
    best_id  = None
    best_len = 0

    for _, row in ds.movies_df.iterrows():
        raw_title   = str(row["title"])
        clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title).strip().lower()

        if len(clean_title) <= 3:
            continue

        pattern = r"\b" + re.escape(clean_title) + r"\b"
        if re.search(pattern, query_lower) and len(clean_title) > best_len:
            best_id  = int(row["movieId"])
            best_len = len(clean_title)

    return best_id


def _find_exclude_genres(query: str) -> list[str]:
    """
    Tìm genre cần loại trừ.

    Fix so với version cũ:
    - Thêm GENRE_ALIASES để match "animated" → Animation,
      "sci fi" → Sci-Fi, v.v.
    - Check cả tên chính thức lẫn alias
    """
    q = query.lower()
    if not _contains_any(q, EXCLUDE_SIGNAL_WORDS):
        return []

    found = set()
    # Check tên genre chính thức
    for g in ALL_GENRES:
        if g.lower() in q:
            found.add(g)
    # Check aliases
    for alias, canonical in GENRE_ALIASES.items():
        if alias in q:
            found.add(canonical)
    return list(found)


# ── Public API ───────────────────────────────────────────────────

def classify_query(
    query:            str,
    ds:               DataStore,
    has_previous_turn: bool = False,
) -> ClassifierResult:
    """
    Phân loại query thành một trong 6 intent.

    Thứ tự ưu tiên (từ cao đến thấp):
      1. explain_previous  — chỉ khi có turn trước
      2. lookup            — hỏi về nội dung phim cụ thể
      3. analytics         — hỏi về profile / blind spot
      4. CF                — hỏi về similar users
      5. hybrid            — có title reference + constraint loại trừ
      6. content           — default: tìm phim theo mô tả

    Args:
        query:             câu hỏi của user
        ds:                DataStore để tìm title reference
        has_previous_turn: True nếu conversation đã có ít nhất 1 turn

    Returns:
        ClassifierResult với intent và constraints
    """
    q           = query.lower().strip()
    constraints = {}

    # ── 1. explain_previous ──────────────────────────────────────
    # Chỉ áp dụng khi đã có turn trước
    if has_previous_turn and _contains_any(q, EXPLAIN_KEYWORDS):
        return ClassifierResult(
            intent="explain_previous",
            is_followup=True,
        )

    # ── 2. analytics ─────────────────────────────────────────────
    # Check TRƯỚC title_ref lookup để tránh analytics query
    # bị misroute sang lookup khi có từ ngẫu nhiên match tên phim
    if _contains_any(q, ANALYTICS_KEYWORDS):
        return ClassifierResult(intent="analytics")

    # ── 3. lookup ────────────────────────────────────────────────
    # Phải có cả keyword lookup VÀ tên phim cụ thể
    title_ref = _find_title_reference(query, ds)
    if _contains_any(q, LOOKUP_KEYWORDS) and title_ref is not None:
        constraints["title_reference"] = title_ref
        return ClassifierResult(intent="lookup", constraints=constraints)

    # ── 4. CF ────────────────────────────────────────────────────
    is_cf = _contains_any(q, CF_KEYWORDS)

    # Chuẩn bị constraints chung cho CF và hybrid
    exclude_genres = _find_exclude_genres(query)
    if exclude_genres:
        constraints["exclude_genres"] = exclude_genres
    if title_ref is not None:
        constraints["title_reference"] = title_ref

    # ── 5. hybrid ────────────────────────────────────────────────
    # Có title reference (để lấy plot làm query gốc)
    # VÀ có constraint loại trừ genre
    if title_ref is not None and exclude_genres:
        return ClassifierResult(intent="hybrid", constraints=constraints)

    # ── 4. CF (tiếp) ─────────────────────────────────────────────
    if is_cf:
        return ClassifierResult(intent="CF", constraints=constraints)

    # ── 6. content (default) ─────────────────────────────────────
    return ClassifierResult(intent="content", constraints=constraints)