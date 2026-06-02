"""
pipeline.py
-----------
Bước 8: Orchestrator kết nối toàn bộ pipeline.

run_pipeline() là hàm duy nhất mà notebooks cần gọi.
Nó kết nối 7 bước theo đúng thứ tự:

    1. Query Classifier   → xác định intent + constraints
    2. Retrieval Layer    → chạy đúng engine theo intent
    3. Conversation Memory → lưu / đọc reasoning trail
    4. Explainability Layer → chuyển raw results → trail
    5. Context Builder    → đóng gói prompt
    6. Claude API         → gọi Claude với grounding
    7. Save to Memory     → lưu turn hiện tại

Lý do tách pipeline.py khỏi các modules khác:
    Mỗi module (cf_engine, explainability...) chỉ làm
    một việc và không biết gì về flow tổng thể.
    pipeline.py là nơi duy nhất biết thứ tự thực hiện.
    → Dễ thay đổi flow mà không chạm vào logic của modules.

Ví dụ flow với query "What should I watch tonight?":
    Classifier  → intent="CF"
    CF Engine   → similar_users=[...], candidates=[...]
    Memory      → ghi turn mới
    Explainer   → "User 42 (sim=0.87) rated X 5.0..."
    Context     → [USER PROFILE] + [REASONING TRAIL]
    Claude      → "Based on User 42 who has similar taste..."
    Memory.save → lưu trail để explain_previous dùng sau
"""

from __future__ import annotations
from urllib import response

# from src.claude_api       import call_claude
from src.claude_api       import call_chatgpt, call_qwen
from src.context_builder  import build_context
from src.data_layer       import DataStore
from src.explainability   import build_reasoning_trail
from src.memory           import ConversationMemory, Turn
from src.query_classifier import classify_query
from src.retrieval        import (
    analytics_engine,
    cf_engine,
    content_engine,
    lookup_module,
)


# ── Helpers ──────────────────────────────────────────────────────

def _extract_recommendations(intent: str, retrieval_result: dict) -> list[dict]:
    """
    Extract danh sách phim được recommend từ retrieval result.
    Dùng để lưu vào Memory cho intent explain_previous sau này.
    """
    recs = []

    if intent == "CF":
        for c in retrieval_result.get("candidates", [])[:3]:
            recs.append({"movieId": c["movieId"], "title": c["title"]})

    elif intent == "content":
        for m in retrieval_result.get("matches", [])[:3]:
            recs.append({"movieId": m["movieId"], "title": m["title"]})

    elif intent == "hybrid":
        # Ưu tiên lấy từ content matches
        content_part = retrieval_result.get("content", {})
        for m in content_part.get("matches", [])[:3]:
            recs.append({"movieId": m["movieId"], "title": m["title"]})

    elif intent == "lookup":
        if "movieId" in retrieval_result:
            recs.append({
                "movieId": retrieval_result["movieId"],
                "title":   retrieval_result["title"],
            })

    return recs


# ── Public API ───────────────────────────────────────────────────

def run_pipeline(
    user_id: int,
    query:   str,
    ds:      DataStore,
    memory:  ConversationMemory,
    verbose: bool = False,
) -> str:
    """
    Chạy toàn bộ pipeline cho một query.

    Args:
        user_id: ID của user (phải có trong dataset)
        query:   câu hỏi của user (natural language)
        ds:      DataStore đã load từ data_layer.load_data()
        memory:  ConversationMemory của session hiện tại
        verbose: nếu True → print intermediate results để debug

    Returns:
        Response string từ Claude (có citations từ data)
    """
    print(f"\n{'='*60}")
    print(f"User {user_id}: \"{query}\"")
    print("="*60)

    # ── Bước 2: Query Classifier ─────────────────────────────────
    clf = classify_query(
        query,
        ds,
        has_previous_turn=memory.has_previous(),
    )
    print(f"[Classifier] intent={clf.intent} | constraints={clf.constraints}")

    # ── Bước 3: Retrieval Layer ──────────────────────────────────
    retrieval_result: dict = {}

    if clf.intent == "explain_previous":
        # Không chạy retrieval — dùng trail từ Memory
        pass

    elif clf.intent == "CF":
        retrieval_result = cf_engine(
            user_id,
            ds,
            title_ref=clf.constraints.get("title_reference"),
        )

    elif clf.intent == "content":
        retrieval_result = content_engine(
            query,
            ds,
            exclude_genres=clf.constraints.get("exclude_genres", []),
        )

    elif clf.intent == "hybrid":
        cf_result      = cf_engine(user_id, ds)
        content_result = content_engine(
            query,
            ds,
            title_ref=clf.constraints.get("title_reference"),
            exclude_genres=clf.constraints.get("exclude_genres", []),
        )
        retrieval_result = {"cf": cf_result, "content": content_result}

    elif clf.intent == "analytics":
        retrieval_result = analytics_engine(user_id, ds)

    elif clf.intent == "lookup":
        movie_id = clf.constraints.get("title_reference")
        if movie_id is None:
            retrieval_result = {"error": "Could not identify a movie title in the query"}
        else:
            retrieval_result = lookup_module(movie_id, ds)

    if verbose and retrieval_result:
        print(f"[Retrieval] keys={list(retrieval_result.keys())}")

    # ── Bước 4 & 5: Memory + Explainability ─────────────────────
    previous_turn = memory.get_last()

    if clf.intent == "explain_previous":
        if previous_turn is None:
            reasoning_trail = "[No previous turn found in memory]"
        else:
            # Dùng trail đã lưu từ turn trước — không tạo trail mới
            reasoning_trail = previous_turn.reasoning_trail
    else:
        reasoning_trail = build_reasoning_trail(clf.intent, retrieval_result, ds)

    if verbose:
        print(f"[Reasoning Trail]\n{reasoning_trail[:400]}...")

    # ── Bước 6: Context Builder ──────────────────────────────────
    context = build_context(
        user_id         = user_id,
        query           = query,
        intent          = clf.intent,
        reasoning_trail = reasoning_trail,
        constraints     = clf.constraints,
        ds              = ds,
        previous_turn   = previous_turn if clf.intent == "explain_previous" else None,
    )

    if verbose:
        print(f"[Context preview]\n{context[:600]}...")

    # ── Bước 7: Claude API ───────────────────────────────────────
    # response = call_claude(context, query)
    # response = call_chatgpt(context, query)  # Dùng ChatGPT tạm vì Claude đang bị lỗi
    response = call_qwen(context, query)   
    print(f"\n[Assistant]\n{response}")

    # ── Lưu vào Memory (chỉ khi không phải explain_previous) ────
    # explain_previous không tạo turn mới vì không có retrieval mới
    if clf.intent != "explain_previous":
        recommendations = _extract_recommendations(clf.intent, retrieval_result)
        memory.save(Turn(
            turn_id         = len(memory) + 1,
            query           = query,
            intent          = clf.intent,
            recommendations = recommendations,
            reasoning_trail = reasoning_trail,
            retrieval_raw   = retrieval_result,
        ))

    return response