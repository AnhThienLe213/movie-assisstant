"""
retrieval/__init__.py
---------------------
Export tất cả 4 engines để pipeline.py có thể import
từ một chỗ duy nhất.

Cách dùng trong pipeline.py:
    from src.retrieval import cf_engine, content_engine, analytics_engine, lookup_module

4 engines hoàn toàn độc lập với nhau:
- Không engine nào import engine khác
- Mỗi engine nhận DataStore và trả về dict riêng
- pipeline.py quyết định engine nào chạy tùy theo intent
"""

from .cf_engine        import cf_engine
from .content_engine   import content_engine
from .analytics_engine import analytics_engine
from .lookup_module    import lookup_module

__all__ = [
    "cf_engine",
    "content_engine",
    "analytics_engine",
    "lookup_module",
]