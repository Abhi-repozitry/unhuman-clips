"""Tests for deterministic entity segmentation and grouping safeguards."""
from __future__ import annotations

from backend.models import ContentIdentity, MultimodalSignals, OnScreenTextSignal, RichTimeline, RichTimelineSegment
from backend.pipeline.analyzer import SemanticBlock, _segment_entities


def _block(block_id: int, start: float, end: float, text: str, segment_id: int) -> SemanticBlock:
    return SemanticBlock(
        block_id=block_id,
        start=start,
        end=end,
        text=text,
        speech_energy=0.8,
        volume_db=-12.0,
        silence_before=False,
        black_frame=False,
        freeze=False,
        importance=70.0,
        peak_offset=1.0,
        segment_ids=[segment_id],
    )


def _timeline(*speaker_ids: str | None) -> RichTimeline:
    return RichTimeline(segments=[
        RichTimelineSegment(
            segment_id=index,
            start=index * 30.0,
            end=(index + 1) * 30.0,
            duration=30.0,
            speech="",
            speaker_id=speaker_id,
        )
        for index, speaker_id in enumerate(speaker_ids)
    ])


def test_entity_segmentation_fuses_identifier_ocr_and_scene_boundaries():
    blocks = [
        _block(0, 0.0, 30.0, "Maya starts the challenge.", 0),
        _block(1, 30.0, 60.0, "Chris takes the next round.", 1),
        _block(2, 60.0, 90.0, "Jordan gets the final result.", 2),
    ]
    identity = ContentIdentity(
        content_format="contestant challenge",
        detected_genre="game_challenge",
        structure="multi_entity",
        entity_names=["Maya", "Chris", "Jordan"],
        hook_recommendation="skip",
        planning_notes="Keep contestants separate.",
    )
    signals = MultimodalSignals(
        scene_cut_at=[30.0, 60.0],
        on_screen_text=[
            OnScreenTextSignal(timestamp=30.75, scene_cut_at=30.0, text="Chris"),
            OnScreenTextSignal(timestamp=60.75, scene_cut_at=60.0, text="Jordan"),
        ],
    )

    segments, entity_grouped = _segment_entities(blocks, _timeline(None, None, None), signals, identity, 90.0)

    assert entity_grouped is True
    assert [(segment.start, segment.end, segment.entity_name) for segment in segments] == [
        (0.0, 30.0, "Maya"),
        (30.0, 60.0, "Chris"),
        (60.0, 90.0, "Jordan"),
    ]
    assert [block.entity_segment_id for block in blocks] == ["entity-1", "entity-2", "entity-3"]


def test_confirmed_single_narrative_does_not_promote_scene_only_segments():
    blocks = [
        _block(0, 0.0, 20.0, "I begin the story.", 0),
        _block(1, 20.0, 40.0, "Then I continue it.", 1),
        _block(2, 40.0, 60.0, "Finally I finish it.", 2),
    ]
    identity = ContentIdentity(
        content_format="personal story",
        detected_genre="vlog_personal",
        structure="single_narrative",
        entity_names=[],
        hook_recommendation="hook",
        planning_notes="Keep the story chronological.",
    )

    segments, entity_grouped = _segment_entities(
        blocks,
        _timeline(None, None, None),
        MultimodalSignals(scene_cut_at=[20.0, 40.0]),
        identity,
        60.0,
    )

    assert len(segments) == 3
    assert entity_grouped is False


def test_speaker_change_creates_entity_boundary_without_visual_signals():
    blocks = [
        _block(0, 0.0, 30.0, "First person speaks.", 0),
        _block(1, 30.0, 60.0, "Second person responds.", 1),
    ]

    segments, _ = _segment_entities(
        blocks,
        _timeline("speaker-a", "speaker-b"),
        MultimodalSignals(),
        None,
        60.0,
    )

    assert [(segment.entity_name, segment.speaker_ids) for segment in segments] == [
        ("speaker-a", ["speaker-a"]),
        ("speaker-b", ["speaker-b"]),
    ]
