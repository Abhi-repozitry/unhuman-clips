#!/usr/bin/env python3
"""Quick smoke test — run locally to verify the pipeline compiles and core modules import.

Usage:
    python scripts/smoke_test.py

This does NOT require ffmpeg, GPU, or network access.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path so backend package is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main() -> int:
    errors: list[str] = []
    passed: list[str] = []

    def check(label: str, fn):
        try:
            fn()
            passed.append(label)
            print(f"  PASS  {label}")
        except Exception as e:
            errors.append(f"{label}: {e}")
            print(f"  FAIL  {label}: {e}")

    print("=== Unhuman Clips Smoke Test ===\n")

    # --- Import checks ---
    print("[1/5] Module imports...")
    check("backend.config", lambda: __import__("backend.config"))
    check("backend.models", lambda: __import__("backend.models"))
    check("backend.ffmpeg_utils", lambda: __import__("backend.ffmpeg_utils"))
    check("backend.logging_config", lambda: __import__("backend.logging_config"))
    check("backend.progress", lambda: __import__("backend.progress"))
    check("backend.output_manager", lambda: __import__("backend.output_manager"))
    check("backend.pipeline.sanitize", lambda: __import__("backend.pipeline.sanitize"))
    check("backend.pipeline.captioner", lambda: __import__("backend.pipeline.captioner"))
    check("backend.pipeline.analyzer", lambda: __import__("backend.pipeline.analyzer"))
    check("backend.pipeline.narration_validator", lambda: __import__("backend.pipeline.narration_validator"))
    check("backend.pipeline.checkpoint", lambda: __import__("backend.pipeline.checkpoint"))
    check("backend.providers.llm", lambda: __import__("backend.providers.llm"))

    # --- Core function checks ---
    print("\n[2/5] Sanitize text...")
    from backend.pipeline.sanitize import sanitize_text
    check("sanitize empty", lambda: _assert(sanitize_text("") == ""))
    check("sanitize unicode", lambda: _assert(sanitize_text("\u201chello\u201d") == '"hello"'))
    check("sanitize banned chars", lambda: _assert("#" not in sanitize_text("a#b")))

    print("\n[3/5] Captioner functions...")
    from backend.pipeline.captioner import (
        _escape_ass_text,
        _format_timestamp,
        _wrap_text_ass,
    )
    check("escape_ass_text", lambda: _assert(_escape_ass_text("a\\b") == "a\\\\b"))
    check("format_timestamp", lambda: _assert(_format_timestamp(65.5) == "0:01:05.50"))
    check("wrap_text_ass", lambda: _assert("\\N" in _wrap_text_ass("a b c d e f g h i j", max_chars=5)))

    print("\n[4/5] Analyzer helpers...")
    from backend.pipeline.analyzer import (
        _compute_group_count_target,
        _extract_json_object,
    )
    check("group_count short", lambda: _assert(_compute_group_count_target(120) == (1, 4)))
    check("group_count long", lambda: _assert(_compute_group_count_target(2400) == (5, 12)))
    check("extract_json", lambda: _assert('"key": "value"' in _extract_json_object('{"key": "value"}')))

    print("\n[5/5] Models validation...")
    from backend.models import VideoJob, ReelPlan, JobStatus
    job = VideoJob(url="https://test.com")
    check("VideoJob creation", lambda: _assert(job.status == JobStatus.QUEUED))
    check("JobStatus enum", lambda: _assert(len(list(JobStatus)) == 11))

    # --- Summary ---
    print(f"\n{'='*40}")
    print(f"Passed: {len(passed)}  |  Failed: {len(errors)}")
    if errors:
        print("\nFAILURES:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll smoke tests passed!")
    return 0


def _assert(condition: bool) -> None:
    if not condition:
        raise AssertionError("Assertion failed")


if __name__ == "__main__":
    sys.exit(main())
