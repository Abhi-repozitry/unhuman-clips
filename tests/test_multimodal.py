"""Tests for CPU scene detection and async OCR orchestration."""
from __future__ import annotations

import asyncio

import pytest

from backend.models import ContentIdentity, MultimodalSignals, OnScreenTextSignal
from backend.pipeline import analyzer as analyzer_module
from backend.pipeline import multimodal


def test_select_frame_candidates_prioritizes_scene_cuts_and_respects_cap():
    candidates = multimodal.select_frame_candidates(
        [1.0, 5.0, 9.0],
        source_duration=12.0,
        sample_interval_seconds=4.0,
        max_frames=3,
    )

    assert [candidate.timestamp for candidate in candidates] == [1.75, 5.75, 9.75]
    assert [candidate.scene_cut_at for candidate in candidates] == [1.0, 5.0, 9.0]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"text": "Maya Patel"}', "Maya Patel"),
        ('{"text": null}', None),
        ("Subscribe", None),
        ("maya patel", None),
    ],
)
def test_parse_ocr_response_filters_to_name_shaped_text(content, expected):
    assert multimodal._parse_ocr_response(content) == expected


@pytest.mark.asyncio
async def test_enrichment_returns_scene_and_ocr_signals(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(multimodal, "detect_scene_cuts", lambda path: ([2.0, 8.0], 12.0))

    async def fake_ocr(path, candidates, interactions=None):
        return [
            OnScreenTextSignal(timestamp=c.timestamp, scene_cut_at=c.scene_cut_at, text="Maya Patel")
            for c in candidates
        ]

    monkeypatch.setattr(multimodal, "_ocr_samples", fake_ocr)

    signals = await multimodal.enrich_multimodal_signals(str(source))

    assert signals.scene_cut_at == [2.0, 8.0]
    assert signals.on_screen_text[0].text == "Maya Patel"


def test_preplanning_runs_identifier_then_multimodal(monkeypatch):
    expected_identity = ContentIdentity(
        content_format="contestant challenge",
        detected_genre="game_challenge",
        structure="multi_entity",
        entity_names=["Maya Patel"],
        hook_recommendation="skip",
        planning_notes="Keep contestant moments separate. Preserve each outcome.",
    )
    expected_signals = MultimodalSignals(
        scene_cut_at=[2.0],
        on_screen_text=[OnScreenTextSignal(timestamp=2.75, text="Maya Patel", scene_cut_at=2.0)],
    )
    calls = []

    def fake_identifier(*args, **kwargs):
        calls.append("identifier")
        return expected_identity

    async def fake_enrichment(path, interactions=None):
        calls.append("multimodal")
        await asyncio.sleep(0)
        return expected_signals

    monkeypatch.setattr(analyzer_module, "MULTIMODAL_ENABLED", True)
    monkeypatch.setattr(analyzer_module, "_identify_content", fake_identifier)
    monkeypatch.setattr(multimodal, "enrich_multimodal_signals", fake_enrichment)

    identity, signals = analyzer_module._run_preplanning_enrichment(
        "Title",
        "Description",
        None,
        "source.mp4",
        [],
        None,
        None,
        [],
        None,
    )

    assert identity == expected_identity
    assert signals == expected_signals
    assert calls == ["identifier", "multimodal"]
