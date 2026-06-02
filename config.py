"""
config.py
---------
Toàn bộ constants dùng chung trong pipeline.
Thay đổi ở đây sẽ ảnh hưởng tất cả modules.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = Path("D:/AnhThienLe/Job/rcm_movie/data/ml-latest-small-filtered")

# ── CF Engine ────────────────────────────────────────────────────
CF_TOP_K_USERS  = 20   # số similar users lấy
CF_TOP_N_MOVIES = 10   # số candidate movies trả về

# ── Content Engine ───────────────────────────────────────────────
CONTENT_TOP_N      = 10     # số kết quả content search trả về
TFIDF_MAX_FEATURES = 15000  # vocab size của TF-IDF
TFIDF_NGRAM_RANGE  = (1, 2) # unigram + bigram

# ── Thresholds ───────────────────────────────────────────────────
MIN_RATING_HIGH  = 4.0  # ngưỡng rating "thích"
SPARSE_USER_THR  = 20   # dưới ngưỡng → cảnh báo sparse user
SPARSE_MOVIE_THR = 5    # dưới ngưỡng → cảnh báo sparse movie
LOW_CONF_THR     = 3    # dưới ngưỡng similar users → low confidence
BLIND_SPOT_GAP   = 5.0  # gap % tối thiểu để coi là blind spot

# ── Claude API ───────────────────────────────────────────────────
CLAUDE_MODEL      = "claude-opus-4-6"
CLAUDE_MAX_TOKENS = 1024