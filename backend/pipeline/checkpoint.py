"""Pipeline checkpoint management — persists stage results to disk for resumability.

Each stage writes a JSON file in the job's working directory so that if the pipeline
crashes or is restarted, completed stages can be skipped on retry.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["PipelineCheckpoint"]

logger = logging.getLogger(__name__)


class PipelineCheckpoint:
    """Manages JSON checkpoint files for pipeline stages.

    Checkpoints are stored as ``{working_dir}/checkpoint_{stage}.json``.
    Each checkpoint is a small JSON document containing the stage's output data.

    Example::

        ckpt = PipelineCheckpoint("/path/to/working/job123")
        ckpt.save_stage("download", {"source_path": "/path/to/video.mp4"})
        data = ckpt.load_stage("download")  # -> {"source_path": "..."}
        if ckpt.has_stage("download"):
            print("Already downloaded")
    """

    def __init__(self, working_dir: Path | str):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def _stage_path(self, stage: str) -> Path:
        """Return the file path for a given stage checkpoint."""
        # Sanitize stage name to be filesystem-safe
        safe_name = stage.replace("/", "_").replace("\\", "_")
        return self.working_dir / f"checkpoint_{safe_name}.json"

    def save_stage(self, stage: str, data: dict[str, Any]) -> None:
        """Save checkpoint data for a pipeline stage.

        Writes atomically via a temp file + rename to avoid partial writes.

        Args:
            stage: Stage name (e.g., "download", "transcribe", "group_0_clips").
            data: JSON-serializable data to persist.
        """
        path = self._stage_path(stage)
        tmp_path = path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
            tmp_path.replace(path)
            logger.debug("Checkpoint saved: %s", path.name)
        except Exception as e:
            logger.warning("Failed to save checkpoint %s: %s", stage, e)

    def load_stage(self, stage: str) -> dict[str, Any] | None:
        """Load checkpoint data for a pipeline stage.

        Args:
            stage: Stage name.

        Returns:
            The saved data dict, or None if no checkpoint exists or is unreadable.
        """
        path = self._stage_path(stage)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.debug("Checkpoint loaded: %s", path.name)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load checkpoint %s: %s", stage, e)
            return None

    def has_stage(self, stage: str) -> bool:
        """Check whether a checkpoint exists for the given stage.

        Args:
            stage: Stage name.

        Returns:
            True if the checkpoint file exists and is readable.
        """
        path = self._stage_path(stage)
        if not path.exists():
            return False
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def clear_stage(self, stage: str) -> None:
        """Delete a checkpoint file if it exists.

        Args:
            stage: Stage name.
        """
        path = self._stage_path(stage)
        if path.exists():
            try:
                path.unlink()
                logger.debug("Checkpoint cleared: %s", path.name)
            except OSError as e:
                logger.warning("Failed to clear checkpoint %s: %s", stage, e)

    def cleanup(self) -> int:
        """Delete all checkpoint files in the working directory.

        Returns:
            Number of checkpoint files removed.
        """
        count = 0
        for path in self.working_dir.glob("checkpoint_*.json"):
            try:
                path.unlink()
                count += 1
            except OSError as e:
                logger.warning("Failed to remove checkpoint %s: %s", path.name, e)
        # Also clean up any leftover tmp files
        for path in self.working_dir.glob("checkpoint_*.json.tmp"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        logger.info("Cleaned up %d checkpoint files from %s", count, self.working_dir)
        return count

    def list_stages(self) -> list[str]:
        """Return a list of stage names that have checkpoints.

        Returns:
            Sorted list of stage names.
        """
        stages = []
        for path in self.working_dir.glob("checkpoint_*.json"):
            name = path.stem.removeprefix("checkpoint_")
            stages.append(name)
        return sorted(stages)
