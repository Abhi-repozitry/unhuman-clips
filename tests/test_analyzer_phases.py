"""Tests for the structured analyzer phase system.

Covers three layers, each tested at the layer where it actually lives:
  1. `_build_analyzer_phase_plan` — pure branch logic, no I/O.
  2. `ProgressReporter` — the structured state store + broadcast trigger.
  3. `_call_llm` — the single choke point that auto-emits running/done/error
     for every LLM-driven phase. Existing tests in test_analyzer.py monkeypatch
     `_call_llm` itself, which bypasses this wiring entirely — these tests
     monkeypatch one layer lower (`backend.providers.llm.call_llm`) so the
     instrumentation inside `_call_llm` actually runs.
"""
from __future__ import annotations

import pytest

from backend.models import AnalyzerPhaseStatus, VideoJob
from backend.pipeline.analyzer import ANALYZER_PHASE_REGISTRY, _build_analyzer_phase_plan
from backend.progress import ProgressReporter


# ---------------------------------------------------------------------------
# Layer 1: pure phase-plan branch logic
# ---------------------------------------------------------------------------

class TestBuildAnalyzerPhasePlan:
    def _ids(self, plan):
        return [p["id"] for p in plan]

    def test_executor_entity_branch(self):
        ids = self._ids(_build_analyzer_phase_plan(entity_grouped=True, plan_mode="executor"))
        assert ids == [
            "semantic_blocks", "identifier", "multimodal_ocr", "content_classification",
            "entity_group_planner", "moment_beat_matcher", "plan_execution",
            "script_narration_writer", "narration_placement",
            "validation_finalize", "completeness_critic",
        ]

    def test_executor_genre_branch(self):
        ids = self._ids(_build_analyzer_phase_plan(entity_grouped=False, plan_mode="executor"))
        assert "genre_story_planner" in ids
        assert "entity_group_planner" not in ids

    def test_executor_fast_mode_drops_completeness_critic(self):
        """_completeness_critic() returns early under FAST_MODE without calling
        the LLM at all — the phase must not appear in the plan, or it would
        sit at "pending" forever on the frontend."""
        ids = self._ids(_build_analyzer_phase_plan(entity_grouped=True, plan_mode="executor", fast_mode=True))
        assert "completeness_critic" not in ids
        # validation_finalize still runs before the (skipped) critic slot.
        assert ids[-1] == "validation_finalize"

    def test_legacy_branch_order(self):
        ids = self._ids(_build_analyzer_phase_plan(entity_grouped=False, plan_mode="llm"))
        assert ids == [
            "semantic_blocks", "identifier", "multimodal_ocr", "content_classification",
            "structure_planner", "clip_planner", "narration_writer", "critic",
            "validation_finalize",
        ]

    def test_legacy_fast_mode_skips_critic(self):
        ids = self._ids(_build_analyzer_phase_plan(entity_grouped=False, plan_mode="llm", fast_mode=True))
        assert "critic" not in ids
        assert ids[-1] == "validation_finalize"

    def test_every_emitted_id_is_registered(self):
        """Guards against typos in _phase(...) calls — every id must resolve
        against ANALYZER_PHASE_REGISTRY or the frontend gets an unlabeled phase."""
        for entity_grouped in (True, False):
            for plan_mode in ("executor", "llm"):
                for fast_mode in (True, False):
                    plan = _build_analyzer_phase_plan(
                        entity_grouped=entity_grouped, plan_mode=plan_mode, fast_mode=fast_mode,
                    )
                    for phase in plan:
                        assert phase["id"] in ANALYZER_PHASE_REGISTRY
                        assert phase["kind"] in ("llm", "python")
                        assert phase["label"]

    def test_no_duplicate_phase_ids_within_a_plan(self):
        for entity_grouped in (True, False):
            for plan_mode in ("executor", "llm"):
                ids = self._ids(_build_analyzer_phase_plan(entity_grouped=entity_grouped, plan_mode=plan_mode))
                assert len(ids) == len(set(ids)), f"duplicate id in {ids}"


# ---------------------------------------------------------------------------
# Layer 2: ProgressReporter structured phase methods
# ---------------------------------------------------------------------------

@pytest.fixture
def job() -> VideoJob:
    return VideoJob(url="https://example.com/watch?v=x")


@pytest.fixture
def reporter(job) -> ProgressReporter:
    broadcasts: list[VideoJob] = []
    r = ProgressReporter(job, lambda j: broadcasts.append(j))
    r._broadcasts = broadcasts  # type: ignore[attr-defined]
    return r


class TestProgressReporterAnalyzerPhases:
    def test_set_analyzer_phase_plan_replaces_and_defaults_pending(self, reporter, job):
        reporter.set_analyzer_phase_plan([
            {"id": "semantic_blocks", "label": "Semantic Blocks", "kind": "python"},
            {"id": "identifier", "label": "Content Identifier", "kind": "llm"},
        ])
        assert [p.id for p in job.analyzer_phases] == ["semantic_blocks", "identifier"]
        assert all(p.status == AnalyzerPhaseStatus.PENDING for p in job.analyzer_phases)

    def test_append_analyzer_phases_is_order_preserving_and_dedupes(self, reporter, job):
        reporter.set_analyzer_phase_plan([{"id": "semantic_blocks", "label": "Semantic Blocks", "kind": "python"}])
        reporter.append_analyzer_phases([
            {"id": "identifier", "label": "Content Identifier", "kind": "llm"},
            {"id": "semantic_blocks", "label": "Semantic Blocks", "kind": "python"},  # dup, ignored
        ])
        assert [p.id for p in job.analyzer_phases] == ["semantic_blocks", "identifier"]

    def test_update_analyzer_phase_running_then_done_sets_timestamps(self, reporter, job):
        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        reporter.update_analyzer_phase("identifier", "running")
        phase = job.analyzer_phases[0]
        assert phase.status == AnalyzerPhaseStatus.RUNNING
        assert phase.started_at is not None
        assert phase.ended_at is None

        reporter.update_analyzer_phase("identifier", "done", progress=100, detail={"foo": "bar"})
        phase = job.analyzer_phases[0]
        assert phase.status == AnalyzerPhaseStatus.DONE
        assert phase.ended_at is not None
        assert phase.detail == {"foo": "bar"}

    def test_update_analyzer_phase_error_records_message(self, reporter, job):
        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        reporter.update_analyzer_phase("identifier", "running")
        reporter.update_analyzer_phase("identifier", "error", error="LLM timed out")
        phase = job.analyzer_phases[0]
        assert phase.status == AnalyzerPhaseStatus.ERROR
        assert phase.error == "LLM timed out"
        assert phase.ended_at is not None

    def test_retry_after_error_keeps_original_started_at(self, reporter, job):
        """A phase that fails and is retried (e.g. the entity planner's 2-attempt
        loop) should keep its first start time, not reset the clock each retry."""
        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        reporter.update_analyzer_phase("identifier", "running")
        first_start = job.analyzer_phases[0].started_at
        reporter.update_analyzer_phase("identifier", "error", error="malformed json")
        reporter.update_analyzer_phase("identifier", "running")
        assert job.analyzer_phases[0].started_at == first_start

    def test_unknown_phase_id_is_defensively_appended_not_dropped(self, reporter, job):
        """If the registry and the plan ever drift, an update for an
        unannounced phase must still be visible rather than silently lost."""
        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        reporter.update_analyzer_phase("some_new_phase", "running")
        ids = [p.id for p in job.analyzer_phases]
        assert "some_new_phase" in ids

    def test_updates_enqueue_broadcasts_of_the_live_job(self, reporter, job):
        """_broadcast() throttles to one enqueue per 200ms (existing, shared
        behavior — not analyzer-specific), so rapid-fire updates collapse to
        fewer queue entries. That's fine: the job object is mutated in place
        and passed by reference, so whatever broadcast *does* go out carries
        current state. Just confirm at least one fired and it's the same job."""
        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        reporter.update_analyzer_phase("identifier", "running")
        reporter.update_analyzer_phase("identifier", "done")
        assert len(reporter._broadcasts) >= 1
        assert reporter._broadcasts[-1] is job
        assert job.analyzer_phases[0].status == AnalyzerPhaseStatus.DONE


# ---------------------------------------------------------------------------
# Layer 3: _call_llm's automatic instrumentation (the choke point)
# ---------------------------------------------------------------------------

class TestCallLlmPhaseInstrumentation:
    def test_success_marks_running_then_done(self, monkeypatch, reporter, job):
        import backend.pipeline.analyzer as analyzer_module
        import backend.providers.llm as llm_module

        monkeypatch.setattr(analyzer_module, "OPENCODE_API_KEY", "fake-key")
        monkeypatch.setattr(llm_module, "call_llm", lambda **kwargs: '{"ok": true}')

        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        result = analyzer_module._call_llm(
            [{"role": "user", "content": "hi"}], reporter=reporter, stage_name="identifier",
        )

        assert result == '{"ok": true}'
        phase = job.analyzer_phases[0]
        assert phase.status == AnalyzerPhaseStatus.DONE
        assert phase.progress == 100

    def test_provider_failure_marks_error_and_reraises(self, monkeypatch, reporter, job):
        import backend.pipeline.analyzer as analyzer_module
        import backend.providers.llm as llm_module

        monkeypatch.setattr(analyzer_module, "OPENCODE_API_KEY", "fake-key")

        def boom(**kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(llm_module, "call_llm", boom)

        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        with pytest.raises(RuntimeError, match="provider down"):
            analyzer_module._call_llm(
                [{"role": "user", "content": "hi"}], reporter=reporter, stage_name="identifier",
            )

        phase = job.analyzer_phases[0]
        assert phase.status == AnalyzerPhaseStatus.ERROR
        assert "provider down" in phase.error

    def test_empty_response_marks_error_not_silent_done(self, monkeypatch, reporter, job):
        """A response with no JSON in it is a real failure for this phase —
        it must not be reported as done just because the HTTP call succeeded."""
        import backend.pipeline.analyzer as analyzer_module
        import backend.providers.llm as llm_module

        monkeypatch.setattr(analyzer_module, "OPENCODE_API_KEY", "fake-key")
        monkeypatch.setattr(llm_module, "call_llm", lambda **kwargs: "")

        reporter.set_analyzer_phase_plan([{"id": "identifier", "label": "Content Identifier", "kind": "llm"}])
        with pytest.raises(ValueError):
            analyzer_module._call_llm(
                [{"role": "user", "content": "hi"}], reporter=reporter, stage_name="identifier",
            )

        assert job.analyzer_phases[0].status == AnalyzerPhaseStatus.ERROR
