# Movie Discovery Assistant

AI assistant giúp user khám phá phim dựa trên MovieLens dataset.
Không phải search engine — system lập luận về data, kết hợp nhiều signals,
và giải thích recommendation bằng bằng chứng từ lịch sử rating thật.

---

## Mục Lục

- [Dataset](#dataset)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Pipeline](#pipeline)
---

## Dataset

Đặt dataset tại `data/ml-latest-small-filtered/`:

| File | Mô tả |
|------|-------|
| `movies_with_plots.csv` | 5,135 phim — movieId, title, year, genres, plot (~3,200 chars) |
| `ratings.csv` | 74,064 ratings từ 610 users — userId, movieId, rating (0.5–5.0) |
| `tags.csv` | 2,440 user-generated tags — userId, movieId, tag |
| `links.csv` | External links — không dùng trong pipeline |
| `movies.csv` | Basic movie info — không dùng trong pipeline |

**Đặc điểm quan trọng:**
- ~51% phim có dưới 5 ratings → sparse phía phim
- 610 users, avg 121 ratings/user → dày phía user
- Phim 1903–2014. Một số phim nổi tiếng vắng mặt (The Matrix...)

---

## Cấu Trúc Dự Án

```
movie_assistant/
├── data/
│   └── ml-latest-small-filtered/
│       ├── movies_with_plots.csv
│       ├── ratings.csv
│       ├── tags.csv
│       └── links.csv
│
├── src/
│   ├── __init__.py
│   ├── data_layer.py          # Bước 1: load & cache dataset
│   ├── query_classifier.py    # Bước 2: phân loại intent
│   ├── retrieval/
│   │   ├── __init__.py        # export 4 engines
│   │   ├── cf_engine.py       # Collaborative Filtering
│   │   ├── content_engine.py  # TF-IDF content search
│   │   ├── analytics_engine.py# Genre profile & blind spots
│   │   └── lookup_module.py   # Movie info lookup
│   ├── memory.py              # Bước 4: conversation memory
│   ├── explainability.py      # Bước 5: raw results → reasoning trail
│   ├── context_builder.py     # Bước 6: pack Claude prompt
│   ├── llm_api.py             # Bước 7: API call
│   └── pipeline.py            # Orchestrator
│
├── notebooks/
│   ├── demo.ipynb             # Interactive chat
│   └── evaluation.ipynb       # Đánh giá chất lượng
│
├── config.py                  # Constants
├── requirements.txt
└── README.md
```

---

## Cài Đặt

```bash
# 1. Clone / download project
cd movie_assistant

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Set API key
export API_KEY=your_key_here

# 4. Đặt dataset vào đúng folder
# data/ml-latest-small-filtered/

# 5. Chạy demo notebook
jupyter notebook notebooks/demo.ipynb
```

---

## Pipeline

```
User Input (userId + query)
        ↓
┌───────────────────┐
│   Data Layer      │  Load once: user-item matrix, TF-IDF matrix
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Query Classifier  │  Rule-based → intent + constraints
└────────┬──────────┘
         ↓
┌──────────────────────────────────────────────┐
│              Retrieval Layer                  │
│  CF Engine  │ Content │ Analytics │  Lookup  │
└─────────────────────┬────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│          Conversation Memory                 │
│  Lưu reasoning trail cho explain_previous   │
└─────────────────────┬───────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│          Explainability Layer                │
│  Raw results → cited reasoning trail        │
│  (luôn chạy với mọi intent)                 │
└─────────────────────┬───────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│           Context Builder                    │
│  [USER PROFILE] + [REASONING TRAIL]         │
│                                             │
└─────────────────────┬───────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│             Claude API                       │
│  System: "CHỈ cite từ REASONING TRAIL"      │
│  → Recommendation + cited explanation       │
└─────────────────────────────────────────────┘
```
