"""
explainability.py
-----------------
Bước 5 trong pipeline: chuyển raw retrieval results thành
reasoning trail — text có cấu trúc mà Claude sẽ cite.

Tại sao layer này bắt buộc chạy với MỌI intent:
    Không có layer này, Claude nhận được số thô từ retrieval
    ([(movieId=123, score=4.6), ...]) và không biết cite gì
    → Claude tự bịa lý do từ LLM knowledge
    → Vi phạm requirement 3 của đề bài

    Layer này là "firewall" giữa raw data và Claude:
    - Input:  raw dict từ CF / Content / Analytics / Lookup
    - Output: text trail rõ ràng, có thể cite từng câu

Reasoning trail làm được gì:
    CF trail:
        "User 42 (sim=0.87, 23 phim chung) rated Inception 5.0
         3/3 similar users rate ≥ 4.0 → predicted score: 4.6"
    → Claude cite: "Dựa trên User 42 (similarity=0.87)..."

    Content trail:
        "matched terms: 'psychological', 'twist ending', 'thriller'
         Top match: Gone Girl (score=0.73)"
    → Claude cite: "Plot khớp với query ở các từ: psychological, twist..."

    Analytics trail:
        "Action: 45 phim (avg 4.1) — top genre của bạn
         Documentary: 3 phim (-18% vs dataset) — blind spot"
    → Claude cite: "Bạn đã rate 45 phim Action với avg 4.1..."

    Lookup trail:
        "Full plot từ dataset: [text]
         74 users rated, avg = 4.2"
    → Claude cite: "Theo plot trong dataset: ..."

Confidence flags:
    Nếu data thưa → thêm cảnh báo vào trail
    → Claude được phép (và phải) thừa nhận uncertainty
"""

from __future__ import annotations

from config import SPARSE_MOVIE_THR, SPARSE_USER_THR, LOW_CONF_THR
from src.data_layer import DataStore


# ── CF Trail ─────────────────────────────────────────────────────

def _build_cf_recommendation_trail(result: dict, ds: DataStore) -> str:
    """Trail cho intent CF với type=recommendation."""
    lines = []

    # Confidence flags
    for flag in result.get("confidence_flags", []):
        lines.append(f"⚠️  {flag}")

    # Similar users
    sim_users = result.get("similar_users", [])
    lines.append(f"Found {len(sim_users)} similar users (cosine similarity, mean-centered):")
    for su in sim_users[:5]:
        lines.append(
            f"  - User {su['userId']} "
            f"(similarity={su['similarity']}, "
            f"{su['n_common']} movies in common)"
        )

    # Candidate movies
    candidates = result.get("candidates", [])
    if not candidates:
        lines.append("\nNo candidates found — not enough overlap with similar users.")
        return "\n".join(lines)

    lines.append("\nTop candidate movies (weighted predicted score):")
    for c in candidates[:5]:
        # Raters summary
        raters_str = ", ".join(
            f"User {r['userId']}={r['rating']}"
            for r in c["raters"][:3]
        )
        sparse = " ⚠️ SPARSE_MOVIE" if c.get("is_sparse_movie") else ""
        lines.append(
            f"  - {c['title']}: predicted={c['predicted_score']}, "
            f"rated by [{raters_str}]{sparse}"
        )

    return "\n".join(lines)


def _build_cf_multihop_trail(result: dict) -> str:
    """Trail cho intent CF với type=multihop (similar users think of X)."""
    lines = []

    # Confidence flags
    for flag in result.get("confidence_flags", []):
        lines.append(f"⚠️  {flag}")

    movie_title = result.get("target_movie", {}).get("title", "Unknown")
    sim_users   = result.get("similar_users", [])
    ratings     = result.get("ratings", [])

    lines.append(f"Analyzing what similar users think of '{movie_title}':")
    lines.append(f"Total similar users found: {len(sim_users)}")
    lines.append(f"Similar users who rated '{movie_title}': {len(ratings)}")

    if not ratings:
        lines.append(f"\nNone of the similar users have rated '{movie_title}'.")
        lines.append("Cannot make a data-grounded prediction.")
        return "\n".join(lines)

    lines.append("\nRatings from similar users:")
    for r in ratings:
        lines.append(
            f"  - User {r['userId']} "
            f"(similarity={r['similarity']}): "
            f"rating = {r['rating']}"
        )

    predicted = result.get("predicted_score")
    if predicted:
        lines.append(f"\nWeighted predicted score for you: {predicted}/5.0")

    return "\n".join(lines)


def build_cf_trail(result: dict, ds: DataStore) -> str:
    """Dispatcher: chọn trail builder theo type của CF result."""
    if result.get("type") == "multihop":
        return _build_cf_multihop_trail(result)
    return _build_cf_recommendation_trail(result, ds)


# ── Content Trail ─────────────────────────────────────────────────

def build_content_trail(result: dict) -> str:
    """Trail cho Content Engine results."""
    lines = []

    # Matched terms — key để explain
    terms = ", ".join(f"'{t}'" for t in result.get("matched_terms", []))
    lines.append(f"TF-IDF matched terms from query: {terms}")

    matches = result.get("matches", [])
    if not matches:
        lines.append("No content matches found — query may be too vague or abstract.")
        return "\n".join(lines)

    lines.append(f"\nTop {min(5, len(matches))} content matches:")
    for m in matches[:5]:
        sparse = " ⚠️ SPARSE_MOVIE" if m.get("is_sparse_movie") else ""
        avg    = f", dataset avg={m['avg_rating']}" if m.get("avg_rating") else ""
        tags   = f", tags: {', '.join(m['tags'])}" if m.get("tags") else ""

        lines.append(
            f"  - {m['title']} "
            f"(similarity={m['similarity_score']}{avg}{sparse})\n"
            f"    genres: {m['genres']}\n"
            f"    plot snippet: {m['plot_snippet'][:200]}...\n"
            f"    {tags}"
        )

    return "\n".join(lines)


# ── Analytics Trail ───────────────────────────────────────────────

def build_analytics_trail(result: dict) -> str:
    """Trail cho Analytics Engine results."""
    lines = []

    lines.append(
        f"User profile: {result['n_ratings']} ratings, "
        f"avg rating = {result['avg_rating']}"
    )

    # Genre stats — top 8 by count
    genre_stats  = result.get("genre_stats", {})
    user_genre_pct = result.get("user_genre_pct", {})
    sorted_genres  = sorted(
        genre_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )

    lines.append("\nGenre profile (sorted by count):")
    for g, v in sorted_genres[:8]:
        pct = user_genre_pct.get(g, 0)
        lines.append(
            f"  - {g}: {v['count']} films ({pct}%), "
            f"avg rating = {v['avg_rating']}"
        )

    # High / low rated
    high = result.get("high_rated_genres", [])
    low  = result.get("low_rated_genres", [])
    if high:
        lines.append(f"\nHighest rated genres (≥3 films): "
                     f"{', '.join(f'{g} ({r})' for g, r in high[:3])}")
    if low:
        lines.append(f"Lowest rated genres (≥3 films): "
                     f"{', '.join(f'{g} ({r})' for g, r in low[:3])}")

    # Blind spots
    blind_spots = result.get("blind_spots", [])
    if not blind_spots:
        lines.append("\nNo significant blind spots found.")
    else:
        lines.append("\nBlind spots (popular in dataset but underrepresented in history):")
        for bs in blind_spots:
            lines.append(
                f"  - {bs['genre']}: you {bs['user_pct']}% "
                f"vs dataset {bs['dataset_pct']}% "
                f"(gap = -{bs['gap']}%)"
            )

    return "\n".join(lines)


# ── Lookup Trail ──────────────────────────────────────────────────

def build_lookup_trail(result: dict) -> str:
    """Trail cho Lookup Module results."""
    lines = []

    sparse = " ⚠️ SPARSE_MOVIE" if result.get("is_sparse_movie") else ""
    avg    = result.get("avg_rating")
    n      = result.get("n_ratings", 0)
    tags   = result.get("tags", [])

    lines.append(f"Movie: {result['title']} ({result['year']})")
    lines.append(f"Genres: {result['genres']}")

    if avg is not None:
        lines.append(f"Dataset rating: {avg}/5.0 from {n} users{sparse}")
    else:
        lines.append(f"Dataset rating: no ratings yet{sparse}")

    if tags:
        lines.append(f"User-generated tags: {', '.join(tags)}")

    # Full plot — Claude sẽ dùng để giải thích nội dung
    lines.append(f"\nFull plot from dataset:\n{result['plot']}")

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────

def build_reasoning_trail(
    intent:           str,
    retrieval_result: dict,
    ds:               DataStore,
) -> str:
    """
    Entry point duy nhất của Explainability Layer.

    Dispatch đến đúng trail builder theo intent.
    Luôn được gọi trước khi build context cho Claude.

    Args:
        intent:           intent từ Query Classifier
        retrieval_result: raw result từ Retrieval Layer
        ds:               DataStore (cần cho CF trail)

    Returns:
        Formatted reasoning trail string, sẵn sàng để
        nhúng vào Claude prompt dưới [REASONING TRAIL]
    """
    # Error case — retrieval thất bại
    if "error" in retrieval_result:
        return f"[RETRIEVAL ERROR]\n{retrieval_result['error']}"

    if intent == "CF":
        return f"[CF TRAIL]\n{build_cf_trail(retrieval_result, ds)}"

    elif intent == "hybrid":
        # Hybrid có cả CF lẫn Content results
        cf_part      = retrieval_result.get("cf", {})
        content_part = retrieval_result.get("content", {})

        cf_trail      = build_cf_trail(cf_part, ds) if cf_part else "CF: no results"
        content_trail = build_content_trail(content_part) if content_part else "Content: no results"

        return (
            f"[CF TRAIL]\n{cf_trail}\n\n"
            f"[CONTENT TRAIL]\n{content_trail}"
        )

    elif intent == "content":
        return f"[CONTENT TRAIL]\n{build_content_trail(retrieval_result)}"

    elif intent == "analytics":
        return f"[ANALYTICS TRAIL]\n{build_analytics_trail(retrieval_result)}"

    elif intent == "lookup":
        return f"[LOOKUP TRAIL]\n{build_lookup_trail(retrieval_result)}"

    # explain_previous không tạo trail mới — đọc từ Memory
    # (xử lý trong pipeline.py, không phải ở đây)
    return "[NO TRAIL]"