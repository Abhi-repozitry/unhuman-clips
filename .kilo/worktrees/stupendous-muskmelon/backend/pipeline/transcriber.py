import ctypes
import os
import sys
from pathlib import Path


def _prepare_cuda_runtime_libraries() -> list[str]:
    """
    Make CUDA runtime libraries installed by NVIDIA pip wheels visible to
    CTranslate2/Faster-Whisper before the model is used.

    WSL exposes the NVIDIA driver (`libcuda.so`) through /usr/lib/wsl/lib, but
    CTranslate2 still needs runtime libraries such as `libcublas.so.12` and
    cuDNN. Those may live inside the project virtualenv instead of a system CUDA
    install. Loading them with RTLD_GLOBAL avoids runtime failures like:
    "Library libcublas.so.12 is not found or cannot be loaded".
    """
    here = Path(__file__).resolve()
    backend_dir = here.parents[1]
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
            lib_dir = nvidia_dir / subdir / "lib"
            if lib_dir.exists() and lib_dir not in lib_dirs:
                lib_dirs.append(lib_dir)

    # Keep this useful for subprocesses spawned by the app as well.
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged = [str(p) for p in lib_dirs] + existing
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(merged))

    loaded: list[str] = []
    # Load lower-level dependencies first, then cuBLAS/cuDNN front libraries.
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
                print(f"[WARN] Could not preload CUDA library {lib_path}: {exc}")

    if loaded:
        print("[INFO] Preloaded CUDA runtime libraries for Faster-Whisper")
    return loaded


_prepare_cuda_runtime_libraries()

import faster_whisper
from backend.config import WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE_CUDA, WHISPER_COMPUTE_TYPE_CPU
from typing import Callable, Optional

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    allow_cpu_fallback = os.environ.get("ALLOW_CPU_WHISPER_FALLBACK") == "1"
    errors = []
    attempts = [("cuda", WHISPER_COMPUTE_TYPE_CUDA, "CUDA")]
    if allow_cpu_fallback:
        attempts.append(("cpu", WHISPER_COMPUTE_TYPE_CPU, "CPU"))

    for device, compute_type, label in attempts:
        try:
            print(f"[INFO] Loading Whisper model on {label} (compute_type={compute_type})...")
            _model = faster_whisper.WhisperModel(
                WHISPER_MODEL_SIZE, device=device, compute_type=compute_type
            )
            print(f"[INFO] Whisper model loaded on {label} successfully")
            return _model
        except Exception as e:
            msg = f"Failed to load on {label}: {e}"
            print(f"[WARN] {msg}")
            errors.append(msg)
            continue

    hint = " Set ALLOW_CPU_WHISPER_FALLBACK=1 to permit CPU fallback." if not allow_cpu_fallback else ""
    raise RuntimeError("Could not load Whisper model on CUDA.\n  " + "\n  ".join(errors) + hint)


# Initialize at module level, catching errors so the app doesn't crash on import
try:
    _load_model()
except Exception as e:
    print(f"[ERROR] Whisper model initialization failed: {e}")
    print("[INFO] Transcription will fail at runtime - check CUDA/cuDNN installation")


def transcribe_video(video_path: str, progress_cb: Optional[Callable[[str, float], None]] = None) -> list[dict]:
    if _model is None:
        raise RuntimeError("Whisper CUDA model failed to load. Check CUDA/cuDNN installation or ensure sufficient GPU memory.")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        segments, info = _model.transcribe(video_path, word_timestamps=False)
        result = []
        seg_list = list(segments)
        total = len(seg_list)
        for i, seg in enumerate(seg_list):
            text = seg.text.strip()
            if text:
                result.append({"start": seg.start, "end": seg.end, "text": text})
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
