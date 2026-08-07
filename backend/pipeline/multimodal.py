"""CPU scene detection and bounded asynchronous hosted OCR enrichment."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backend.config import (
    OCR_MAX_CONCURRENCY,
    OCR_MAX_FRAMES,
    OCR_SAMPLE_INTERVAL_SECONDS,
    SCENE_CUT_THRESHOLD,
    SCENE_SAMPLE_FPS,
    VISION_API_KEY,
    VISION_BASE_URL,
    VISION_MODEL,
    VISION_TIMEOUT_SECONDS,
)
from backend.models import LLMInteraction, MultimodalSignals, OnScreenTextSignal

__all__ = ["detect_scene_cuts", "enrich_multimodal_signals", "select_frame_candidates"]

logger = logging.getLogger(__name__)


def _now_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:12]

_FRAME_MAX_EDGE = 960
_FRAME_JPEG_QUALITY = 82
_EXTRACT_BATCH_SIZE = 10
_NAME_STOP_WORDS = {
    "challenge",
    "contestant",
    "episode",
    "follow",
    "like",
    "round",
    "subscribe",
    "winner",
}


@dataclass(frozen=True)
class _FrameCandidate:
    timestamp: float
    scene_cut_at: float | None


@dataclass(frozen=True)
class _FrameSample:
    candidate: _FrameCandidate
    jpeg: bytes


def detect_scene_cuts(
    source_path: str,
    sample_fps: float = SCENE_SAMPLE_FPS,
    threshold: float = SCENE_CUT_THRESHOLD,
) -> tuple[list[float], float]:
    """Detect hard scene boundaries from small CPU grayscale frames."""
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        logger.warning("Scene detection unavailable: %s", e)
        return [], 0.0

    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        logger.warning("Scene detection could not open source: %s", source_path)
        return [], 0.0

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = frame_count / source_fps if source_fps > 0 else 0.0
        frame_step = max(1, round(source_fps / max(sample_fps, 0.1))) if source_fps > 0 else 1

        scene_cuts: list[float] = []
        previous_gray = None
        previous_hist = None
        frame_index = 0
        while capture.grab():
            if frame_index % frame_step:
                frame_index += 1
                continue
            ok, frame = capture.retrieve()
            timestamp = frame_index / source_fps if source_fps > 0 else 0.0
            frame_index += 1
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if previous_gray is not None and previous_hist is not None:
                pixel_difference = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255.0)
                histogram_difference = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if max(pixel_difference, histogram_difference) >= threshold:
                    scene_cuts.append(round(timestamp, 3))
            previous_gray = gray
            previous_hist = hist
        return scene_cuts, round(duration, 3)
    except Exception as e:
        logger.warning("Scene detection failed for %s: %s", source_path, e)
        return [], 0.0
    finally:
        capture.release()


def select_frame_candidates(
    scene_cut_at: list[float],
    source_duration: float,
    sample_interval_seconds: float = OCR_SAMPLE_INTERVAL_SECONDS,
    max_frames: int = OCR_MAX_FRAMES,
) -> list[_FrameCandidate]:
    """Choose deterministic post-cut and periodic OCR samples within a hard cap."""
    if source_duration <= 0 or max_frames < 1:
        return []

    candidates: list[_FrameCandidate] = []
    for cut_at in sorted(set(scene_cut_at)):
        timestamp = min(source_duration, max(0.0, cut_at + 0.75))
        candidates.append(_FrameCandidate(round(timestamp, 3), cut_at))
    timestamp = 0.0
    while timestamp < source_duration:
        nearest_cut = next((cut for cut in scene_cut_at if abs(cut - timestamp) <= 1.0), None)
        candidates.append(_FrameCandidate(round(timestamp, 3), nearest_cut))
        timestamp += max(sample_interval_seconds, 0.1)

    selected: list[_FrameCandidate] = []
    for candidate in candidates:
        if len(selected) >= max_frames:
            break
        if any(abs(candidate.timestamp - existing.timestamp) < 0.5 for existing in selected):
            continue
        selected.append(candidate)
    return selected


def _extract_frame_batch(source_path: str, candidates: list[_FrameCandidate]) -> list[_FrameSample]:
    try:
        import cv2
    except ImportError:
        return []

    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        return []
    samples: list[_FrameSample] = []
    try:
        for candidate in candidates:
            capture.set(cv2.CAP_PROP_POS_MSEC, candidate.timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, _FRAME_MAX_EDGE / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, _FRAME_JPEG_QUALITY],
            )
            if ok:
                samples.append(_FrameSample(candidate=candidate, jpeg=encoded.tobytes()))
    finally:
        capture.release()
    return samples


def _name_shaped_text(value: str) -> str | None:
    value = value.strip().strip('"\'`')
    if not value or value.casefold() in {"none", "null", "no text"}:
        return None
    words = value.split()
    if not 1 <= len(words) <= 3:
        return None
    if value.casefold() in _NAME_STOP_WORDS:
        return None
    if not all(re.fullmatch(r"[A-Z][A-Za-z'\-]*", word) for word in words):
        return None
    return " ".join(words)


def _parse_ocr_response(content: str | None) -> str | None:
    if not content:
        return None
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            value = payload.get("text") or payload.get("name") or ""
            if isinstance(value, str):
                return _name_shaped_text(value)
    except json.JSONDecodeError:
        pass
    return _name_shaped_text(content)


async def _ocr_samples(
    source_path: str,
    candidates: list[_FrameCandidate],
    interactions: list[LLMInteraction] | None = None,
) -> list[OnScreenTextSignal]:
    import backend.config as _cfg

    signals = [
        OnScreenTextSignal(timestamp=c.timestamp, scene_cut_at=c.scene_cut_at)
        for c in candidates
    ]
    if not candidates or not _cfg.VISION_OCR_ENABLED or _cfg.OCR_MODE == "skip":
        logger.info("OCR skipped: VISION_OCR_ENABLED=%s, OCR_MODE=%s", _cfg.VISION_OCR_ENABLED, _cfg.OCR_MODE)
        if interactions is not None:
            interactions.append(LLMInteraction(
                timestamp=_now_timestamp(),
                type="response",
                role="assistant",
                content=f"OCR skipped (mode={_cfg.OCR_MODE})",
                full_content=f"OCR skipped: VISION_OCR_ENABLED={_cfg.VISION_OCR_ENABLED}, OCR_MODE={_cfg.OCR_MODE}",
                model=VISION_MODEL,
                stage_name="ocr",
            ))
        return signals

    if interactions is not None:
        interactions.append(LLMInteraction(
            timestamp=_now_timestamp(),
            type="prompt",
            role="user",
            content=f"OCR: extracting names from {len(candidates)} frames",
            full_content=f"OCR process started: {len(candidates)} frame candidates, model={VISION_MODEL}",
            model=VISION_MODEL,
            stage_name="ocr",
        ))

    from openai import AsyncOpenAI

    by_timestamp = {signal.timestamp: signal for signal in signals}
    semaphore = asyncio.Semaphore(max(1, OCR_MAX_CONCURRENCY))
    client = AsyncOpenAI(
        base_url=VISION_BASE_URL,
        api_key=VISION_API_KEY or "unused",
        timeout=VISION_TIMEOUT_SECONDS,
        max_retries=0,
    )

    async def ocr_one(sample: _FrameSample) -> None:
        async with semaphore:
            try:
                image = base64.b64encode(sample.jpeg).decode("ascii")
                response = await client.chat.completions.create(
                    model=VISION_MODEL,
                    temperature=0.0,
                    max_tokens=80,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Extract only a person or performer name shown as an on-screen lower-third. "
                                        "Return JSON exactly as {\"text\": \"Title Case Name\"}; use null when absent."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                                },
                            ],
                        }
                    ],
                )
                content = response.choices[0].message.content if response.choices else None
                by_timestamp[sample.candidate.timestamp].text = _parse_ocr_response(content)
            except Exception as e:
                logger.debug("OCR failed at %.2fs: %s", sample.candidate.timestamp, e)

    tasks: list[asyncio.Task[None]] = []
    try:
        for index in range(0, len(candidates), _EXTRACT_BATCH_SIZE):
            batch = candidates[index:index + _EXTRACT_BATCH_SIZE]
            samples = await asyncio.to_thread(_extract_frame_batch, source_path, batch)
            tasks.extend(asyncio.create_task(ocr_one(sample)) for sample in samples)
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        await client.close()

    names_found = sum(1 for s in signals if s.text is not None)
    if interactions is not None:
        interactions.append(LLMInteraction(
            timestamp=_now_timestamp(),
            type="response",
            role="assistant",
            content=f"OCR complete: {names_found} names found in {len(candidates)} frames",
            full_content=f"OCR results: {names_found} name-shaped matches out of {len(candidates)} frames sampled",
            model=VISION_MODEL,
            stage_name="ocr",
        ))

    return signals


async def enrich_multimodal_signals(
    source_path: str,
    interactions: list[LLMInteraction] | None = None,
) -> MultimodalSignals:
    """Build scene-cut and OCR signals without allocating any GPU memory."""
    if not source_path or not Path(source_path).is_file():
        return MultimodalSignals()

    import backend.config as _cfg

    scene_cut_at, source_duration = await asyncio.to_thread(detect_scene_cuts, source_path)

    if _cfg.OCR_MODE == "skip":
        logger.info("enrich_multimodal_signals: OCR_MODE=skip, skipping OCR only")
        if interactions is not None:
            interactions.append(LLMInteraction(
                timestamp=_now_timestamp(),
                type="response",
                role="assistant",
                content=f"OCR skipped (mode={_cfg.OCR_MODE})",
                full_content=f"Skipped OCR only: OCR_MODE={_cfg.OCR_MODE}",
                model=VISION_MODEL,
                stage_name="ocr",
            ))
        return MultimodalSignals(scene_cut_at=scene_cut_at)

    candidates = select_frame_candidates(scene_cut_at, source_duration)
    on_screen_text = await _ocr_samples(source_path, candidates, interactions)
    logger.info(
        "MULTIMODAL: %d scene cuts, %d OCR samples, %d name-shaped OCR matches",
        len(scene_cut_at),
        len(on_screen_text),
        sum(signal.text is not None for signal in on_screen_text),
    )
    return MultimodalSignals(scene_cut_at=scene_cut_at, on_screen_text=on_screen_text)
