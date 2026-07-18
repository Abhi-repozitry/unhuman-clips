import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from backend.config import BASE_DIR


class ContentCache:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = BASE_DIR / "storage" / "cache.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        cache_key TEXT PRIMARY KEY,
                        response_text TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        model TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stage_model 
                    ON cache (stage, model, model_version)
                """)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def generate_cache_key(
        stage: str,
        prompt: str,
        params: dict,
        model: str,
        model_version: str = "1"
    ) -> str:
        key_data = {
            "stage": stage,
            "prompt": prompt,
            "params": params,
            "model": model,
            "model_version": model_version
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[str]:
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT response_text FROM cache WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

    def set(self, cache_key: str, response_text: str, stage: str, model: str, model_version: str = "1"):
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (cache_key, response_text, stage, model, model_version)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (cache_key, response_text, stage, model, model_version)
                )
                conn.commit()
            finally:
                conn.close()

    def clear(self):
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("DELETE FROM cache")
                conn.commit()
            finally:
                conn.close()


# Global cache instance - lazy initialization to avoid import-time issues
_cache_instance = None


def get_cache():
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ContentCache()
    return _cache_instance


cache = None  # Will be initialized on first use
