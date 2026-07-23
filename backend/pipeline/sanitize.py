"""Shared text sanitization for ASS subtitles, TTS/SSML, and ffmpeg compatibility.

All text that flows through the pipeline (LLM output, Whisper transcripts,
commentary, narration) MUST pass through sanitize_text() before reaching
captioner.py, tts.py, or compositor.py.
"""
from __future__ import annotations

import re
import unicodedata

__all__ = ["sanitize_text"]

# Characters that break ASS subtitles, TTS/SSML, or ffmpeg filters
_BANNED_CHARS = set('/\\|*#_<>[]{}')


def sanitize_text(text: str) -> str:
    """Sanitize text for ASS subtitles, TTS, and ffmpeg compatibility.

    1. Normalize Unicode (NFKC)
    2. Replace smart quotes, em-dashes, en-dashes, ellipsis with ASCII
    3. Remove banned characters: / \\ | * # _ < > [ ] { }
    4. Collapse whitespace
    5. Strip leading/trailing punctuation artifacts
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Replace unicode equivalents with ASCII
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', ' - ').replace('\u2013', ' - ')
    text = text.replace('\u2026', '...')
    text = text.replace('\u00a0', ' ')
    text = text.replace('\u200b', '')
    text = text.replace('\u200d', '')

    # Remove banned characters (replace with space to avoid joining words)
    for ch in _BANNED_CHARS:
        text = text.replace(ch, ' ')

    # Normalize double hyphens
    text = text.replace('--', ' - ')

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Collapse repeated punctuation
    text = re.sub(r'([,!?;:]){2,}', r'\1', text)

    # Strip leading/trailing punctuation artifacts
    text = text.strip(' ,-;:')

    return text
