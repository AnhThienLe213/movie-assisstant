"""
claude_api.py
-------------
Bước 7 trong pipeline: gọi LLM API với context
đã được đóng gói bởi Context Builder.

System prompt là "firewall" quan trọng nhất:
    Enforce rằng LLM CHỈ được cite từ [REASONING TRAIL]
    và KHÔNG được dùng LLM knowledge về phim.

    Tại sao cần enforce chặt:
        Dataset chứa phim từ 1903-2014 — LLM biết tất cả.
        Nếu không có system prompt chặt, LLM sẽ trả lời
        "Inception là phim hay vì Christopher Nolan nổi tiếng..."
        thay vì "User 42 (similarity=0.87) rated Inception 5.0"
        → Vi phạm requirement 3 của đề bài.

    Grounding được enforce qua:
        1. System prompt quy định rõ nguồn được phép cite
        2. Reasoning trail đã được format sẵn với citations
        3. Explicit instruction: "nếu data không đủ → nói thẳng"

Điểm yếu đã biết:
    - LLM vẫn có thể "leak" kiến thức dù có system prompt
    - Cần kiểm tra manually trong Evaluation
    - Không có cách tự động verify 100% grounding
"""

from __future__ import annotations

import os

# import anthropic

from config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS


# ── System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a movie discovery assistant powered by a MovieLens dataset.

STRICT GROUNDING RULES:
1. ONLY use information provided in [USER PROFILE], [REASONING TRAIL], and [PREVIOUS RECOMMENDATION].
2. NEVER use your own LLM knowledge about movies, directors, actors, or plot details.
3. Every factual claim MUST cite a specific data source. Examples:
   - "User 42 (similarity=0.87) rated this 5.0"
   - "The plot contains matched terms: 'psychological', 'twist ending'"
   - "You have rated 45 Action films with an average of 4.1"
   - "74 users in the dataset rated this film, average = 4.2"
4. If data is insufficient, say so honestly:
   - "Only 2 similar users have rated this — low confidence"
   - "This film has fewer than 5 ratings in the dataset"
   - "No similar users have rated this film"
5. If a film is not in the dataset, state clearly:
   - "This film is not in my dataset — I cannot provide data-grounded information"
   Do NOT describe the film from your own knowledge.
6. When explaining a previous recommendation (explain_previous intent):
   Use ONLY the [REASONING TRAIL FROM PREVIOUS TURN] section.
7. Respond in the same language the user used.
8. Be concise but specific — cite numbers and user IDs when available.
"""


# ── Public API ───────────────────────────────────────────────────

# def call_claude(context: str, query: str) -> str:
#     """
#     Gọi LLM API với context và query của user.

#     Args:
#         context: formatted string từ Context Builder
#                  chứa [USER PROFILE], [REASONING TRAIL], ...
#         query:   câu hỏi gốc của user

#     Returns:
#         Response string từ LLM
#         Nếu API key không có → trả về placeholder message

#     Message format:
#         [context sections]

#         [USER QUERY]
#         <query>

#     LLM nhìn thấy context trước query → biết phải
#     dựa vào data nào trước khi trả lời.
#     """
#     api_key = os.environ.get("ANTHROPIC_API_KEY", "")
#     if not api_key:
#         return (
#             "[LLM API key not set]\n"
#             "Set environment variable: export ANTHROPIC_API_KEY=your_key\n\n"
#             f"Context that would be sent:\n{context[:500]}..."
#         )

#     client = anthropic.Anthropic(api_key=api_key)

#     user_message = f"{context}\n\n[USER QUERY]\n{query}"

#     message = client.messages.create(
#         model=CLAUDE_MODEL,
#         max_tokens=CLAUDE_MAX_TOKENS,
#         system=SYSTEM_PROMPT,
#         messages=[
#             {"role": "user", "content": user_message}
#         ],
#     )

#     return message.content[0].text

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# máy yếu → dùng "Qwen/Qwen2.5-1.5B-Instruct"
# load model 1 lần
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# máy yếu → dùng "Qwen/Qwen2.5-1.5B-Instruct"
import os
from openai import OpenAI

OPENROUTER_MODEL = "qwen/qwen3-32b"

def call_qwen(context: str, query: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        return (
            "[OpenRouter API key not set]\n"
            "Set environment variable: export OPENROUTER_API_KEY=your_key\n\n"
            f"Context that would be sent:\n{context[:500]}..."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    user_message = f"{context}\n\n[USER QUERY]\n{query}"

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=0.7,
    )

    return response.choices[0].message.content

### -----------------------------
import os
from google import genai
from google.genai.types import GenerateContentConfig

GEMINI_MODEL = "gemini-3.5-flash"

def call_gemini(context: str, query: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not api_key:
        return (
            "[Gemini API key not set]\n"
            "Set environment variable: export GEMINI_API_KEY=your_key\n\n"
            f"Context that would be sent:\n{context[:500]}..."
        )

    client = genai.Client(api_key=api_key)

    user_message = f"{context}\n\n[USER QUERY]\n{query}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=CLAUDE_MAX_TOKENS,
            temperature=0.7,
        ),
    )

    return response.text