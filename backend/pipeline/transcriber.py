"""Video transcription module — speech-to-text via Faster-Whisper (CTranslate2).

Provides transcribe_video() with lazy model loading and CUDA fallback logic.
Handles CUDA runtime library preloading for NVIDIA GPU support.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Callable

__all__ = ["transcribe_video"]

logger = logging.getLogger(__name__)


def _is_windows() -> bool:
    """Return True if running on Windows."""
    return sys.platform.startswith("win")


def _prepare_cuda_runtime_libraries() -> list[str]:
    """Make CUDA runtime libraries visible to CTranslate2/Faster-Whisper.

    On Windows, DLLs are added to PATH and loaded via add_dll_directory.
    On Linux, .so files are added to LD_LIBRARY_PATH and loaded with RTLD_GLOBAL.

    Returns:
        List of library paths that were successfully loaded.
    """
    here = Path(__file__).resolve()
    backend_dir = here.parents[1]
    nvidia_lib_subdir = "bin" if _is_windows() else "lib"
    nvidia_lib_ext = ".dll" if _is_windows() else ".so"

    # On Windows, use the standard site-packages layout (Lib\site-packages)
    if _is_windows():
        candidate_site_packages = []
        for venv_name in (".venv", "venv"):
            candidate_site_packages.append(
                backend_dir / venv_name / "Lib" / "site-packages"
            )
        candidate_site_packages.extend(
            Path(p) for p in sys.path if p.endswith("site-packages")
        )
    else:
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidate_site_packages = []
        for venv_name in (".venv", "venv"):
            candidate_site_packages.append(
                backend_dir / venv_name / "lib" / pyver / "site-packages"
            )
        candidate_site_packages.extend(
            Path(p) for p in sys.path if p.endswith("site-packages")
        )

    lib_dirs: list[Path] = []
    for site_packages in candidate_site_packages:
        nvidia_dir = site_packages / "nvidia"
        if not nvidia_dir.exists():
            continue
        for subdir in ("cublas", "cuda_nvrtc", "cudnn", "cuda_runtime"):
            lib_dir = nvidia_dir / subdir / nvidia_lib_subdir
            if lib_dir.exists() and lib_dir not in lib_dirs:
                lib_dirs.append(lib_dir)

    if _is_windows():
        # On Windows, add DLL directories to PATH so ctypes can find them
        existing = [p for p in os.environ.get("PATH", "").split(";") if p]
        merged = [str(p) for p in lib_dirs] + existing
        os.environ["PATH"] = ";".join(dict.fromkeys(merged))
        # Also use add_dll_directory if available (Python 3.8+)
        if hasattr(os, "add_dll_directory"):
            for d in lib_dirs:
                try:
                    os.add_dll_directory(str(d))
                except Exception:
                    pass
    else:
        # Keep this useful for subprocesses spawned by the app as well.
        existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
        merged = [str(p) for p in lib_dirs] + existing
        if merged:
            os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(merged))

    loaded: list[str] = []
    if _is_windows():
        preferred_names = (
            "cudart64_12.dll",
            "nvrtc64_120_0.dll",
            "cublasLt64_12.dll",
            "cublas64_12.dll",
            "cudnn64_9.dll",
        )
    else:
        preferred_names = (
            "libcudart.so.12",
            "libnvrtc.so.12",
            "libcublasLt.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
        )
    for name in preferred_names:
        for lib_dir in lib_dirs:
            lib_path = lib_dir / name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(str(lib_path))
                break
            except OSError as exc:
                logger.warning(f"Could not preload CUDA library {lib_path}: {exc}")

    if loaded:
        logger.info("Preloaded CUDA runtime libraries for Faster-Whisper")
    return loaded


_prepare_cuda_runtime_libraries()

import faster_whisper
from backend.config import WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE_CUDA, WHISPER_COMPUTE_TYPE_CPU

_model = None
_model_loaded = False


def _load_model():
    global _model
    if _model is not None:
        return _model

    allow_cpu_fallback = os.environ.get("ALLOW_CPU_WHISPER_FALLBACK") == "1"
    errors = []
    attempts = [
        ("cuda", WHISPER_COMPUTE_TYPE_CUDA, f"CUDA ({WHISPER_COMPUTE_TYPE_CUDA})"),
        ("cuda", "float16", "CUDA (float16)"),
        ("cuda", "int8", "CUDA (int8)")
    ]
    if allow_cpu_fallback:
        attempts.append(("cpu", WHISPER_COMPUTE_TYPE_CPU, f"CPU ({WHISPER_COMPUTE_TYPE_CPU})"))

    for device, compute_type, label in attempts:
        try:
            logger.info(f"Loading Whisper model on {label} (compute_type={compute_type})...")
            _model = faster_whisper.WhisperModel(
                WHISPER_MODEL_SIZE, device=device, compute_type=compute_type
            )
            logger.info(f"Whisper model loaded on {label} successfully")
            return _model
        except Exception as e:
            msg = f"Failed to load on {label}: {e}"
            logger.warning(f"{msg}")
            errors.append(msg)
            continue

    hint = " Set ALLOW_CPU_WHISPER_FALLBACK=1 to permit CPU fallback." if not allow_cpu_fallback else ""
    raise RuntimeError("Could not load Whisper model on CUDA.\n  " + "\n  ".join(errors) + hint)


def _ensure_model():
    """Lazily load the Whisper model on first transcription call."""
    global _model, _model_loaded
    if _model_loaded:
        return
    try:
        _load_model()
        _model_loaded = True
    except Exception as e:
        logger.error(f"Whisper model initialization failed: {e}")
        logger.info("Transcription will fail at runtime - check CUDA/cuDNN installation")


def transcribe_video(video_path: str, progress_cb: Callable[[str, float], None] | None = None) -> list[dict]:
    _ensure_model()
    if _model is None:
        raise RuntimeError("Whisper CUDA model failed to load. Check CUDA/cuDNN installation or ensure sufficient GPU memory.")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        segments, info = _model.transcribe(video_path, word_timestamps=True)
        result = []
        seg_list = list(segments)
        total = len(seg_list)
        for i, seg in enumerate(seg_list):
            text = seg.text.strip()
            if text:
                words_info = []
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        words_info.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 2),
                            "end": round(w.end, 2)
                        })
                result.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                    "words": words_info
                })
            if progress_cb:
                progress_cb(f"Transcribing segment {i+1}/{total}", ((i + 1) / total) * 100)
        return result
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e) or "out of memory" in str(e).lower():
            raise RuntimeError(
                "CUDA out of memory or GPU error during transcription. "
                "Try a smaller Whisper model (e.g., 'tiny' instead of 'base') or free GPU memory."
            ) from e
        raise RuntimeError(f"Transcription failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
