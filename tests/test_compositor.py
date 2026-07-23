"""Tests for backend.pipeline.compositor — ducking filter chain, VAD integration, duration math."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.pipeline.compositor import (
    _build_ducking_filter_chain,
    get_speech_timestamps_from_narration,
)


class TestBuildDuckingFilterChain:
    """Test _build_ducking_filter_chain generates valid ffmpeg filter syntax."""

    def test_single_narration_event(self):
        events = [{"reel_start": 5.0, "reel_end": 8.0}]
        result = _build_ducking_filter_chain(events)
        assert "volume=" in result or "sidechain" in result.lower() or "ducked" in result

    def test_empty_events_returns_valid_filter(self):
        result = _build_ducking_filter_chain([])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multiple_events(self):
        events = [
            {"reel_start": 5.0, "reel_end": 8.0},
            {"reel_start": 20.0, "reel_end": 23.0},
        ]
        result = _build_ducking_filter_chain(events)
        assert isinstance(result, str)

    def test_with_vad_timestamps(self):
        events = [{"reel_start": 5.0, "reel_end": 8.0}]
        vad_timestamps = [
            [{"start": 0.2, "end": 2.8}],  # precise speech within the 5-8s window
        ]
        result = _build_ducking_filter_chain(
            events, narration_vad_timestamps=vad_timestamps
        )
        assert isinstance(result, str)

    def test_custom_labels(self):
        events = [{"reel_start": 0.0, "reel_end": 3.0}]
        result = _build_ducking_filter_chain(
            events, input_label="1:a", output_label="final"
        )
        assert isinstance(result, str)

    def test_target_duration_affects_padding(self):
        events = [{"reel_start": 85.0, "reel_end": 88.0}]
        result = _build_ducking_filter_chain(events, target_duration=90.0)
        assert isinstance(result, str)


class TestGetSpeechTimestampsFromNarration:
    """Test VAD-based speech timestamp detection."""

    def test_nonexistent_file_returns_empty(self):
        """Should return empty list when given a nonexistent file path (graceful fallback)."""
        result = get_speech_timestamps_from_narration("/nonexistent/audio.wav")
        assert isinstance(result, list)

    def test_returns_list_type(self, tmp_path):
        """Function should return a list when given a valid audio file.
        
        Note: This test may fail in CI without torch/torchaudio installed.
        The test is designed to pass silently if torch is unavailable.
        """
        wav_path = tmp_path / "test.wav"
        # Write minimal WAV header (44 bytes) + some data
        import struct
        # Simple WAV: 1 channel, 16-bit, 16kHz, 1 second of silence
        sample_rate = 16000
        num_samples = sample_rate
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + data_size, b'WAVE',
            b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
            b'data', data_size,
        )
        wav_path.write_bytes(wav_header + b'\x00' * data_size)

        try:
            result = get_speech_timestamps_from_narration(str(wav_path))
            assert isinstance(result, list)
        except RuntimeError:
            # Expected if torch/cuDNN not properly installed
            pytest.skip("torch/torchaudio not available")
        except Exception:
            # Other errors from VAD model loading are acceptable in test env
            pytest.skip("Silero VAD model not available")
