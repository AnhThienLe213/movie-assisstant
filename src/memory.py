"""
memory.py
---------
Bước 4 trong pipeline: lưu reasoning trail giữa các turns
để hỗ trợ intent "explain_previous".

Tại sao cần Conversation Memory:
    Câu hỏi "Why do you think I'd like that?" không có
    ngữ cảnh nếu chỉ nhìn vào query string đơn lẻ.
    Phải biết turn trước đã recommend gì, dựa trên data nào
    → Memory lưu lại reasoning_trail của turn trước
    → Explainability Layer đọc trail đó thay vì chạy retrieval mới

Điều Memory KHÔNG làm:
    - Không lưu toàn bộ conversation history như ChatGPT
    - Không gửi history vào LLM (tránh token waste)
    - Chỉ lưu đủ để trả lời "why" về turn liền trước

Điểm yếu đã biết:
    - Chỉ support "explain turn liền trước" — không explain
      turn cách đây 3 turns ("why did you recommend X earlier?")
    - Memory reset khi tạo ConversationMemory mới
      (in-memory, không persist giữa sessions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Turn dataclass ───────────────────────────────────────────────

@dataclass
class Turn:
    """
    Lưu thông tin của một turn trong conversation.

    reasoning_trail: text đã format từ Explainability Layer
        → đây là thứ LLM cite khi user hỏi "why"
    retrieval_raw: dict thô từ Retrieval Layer
        → giữ lại để Explainability Layer có thể
           đào sâu hơn nếu cần
    recommendations: list phim đã recommend
        → để Context Builder biết turn trước recommend gì
    """
    turn_id:          int
    query:            str
    intent:           str
    recommendations:  list[dict]   # [{"movieId": ..., "title": ...}]
    reasoning_trail:  str          # formatted text, ready to cite
    retrieval_raw:    dict         # raw retrieval result


# ── ConversationMemory ───────────────────────────────────────────

class ConversationMemory:
    """
    Lưu danh sách các turns theo thứ tự thời gian.

    Interface đơn giản: save / get_last / has_previous / reset.
    Pipeline chỉ cần get_last() cho intent explain_previous.
    """

    def __init__(self) -> None:
        self._turns: list[Turn] = []

    def save(self, turn: Turn) -> None:
        """Lưu một turn mới vào cuối danh sách."""
        self._turns.append(turn)

    def get_last(self) -> Optional[Turn]:
        """Trả về turn gần nhất, None nếu chưa có turn nào."""
        return self._turns[-1] if self._turns else None

    def has_previous(self) -> bool:
        """True nếu đã có ít nhất một turn trước đó."""
        return len(self._turns) > 0

    def reset(self) -> None:
        """Xóa toàn bộ history — dùng khi bắt đầu conversation mới."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory({len(self._turns)} turns)"