import ctypes
import os
import sys
from pathlib import Path


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _prepare_cuda_runtime_libraries() -> list[str]:
    """
    Make CUDA runtime libraries installed by NVIDIA pip wheels visible to
    CTranslate2/Faster-Whisper before the model is used.

    WSL exposes the NVIDIA driver (`libcuda.so`) through /usr/lib/wsl/lib, but
    CTranslate2 still needs runtime libraries such as `libcublas.so.12` and
    cuDNN. Those may live inside the project virtualenv instead of a system CUDA
    install. Loading them with RTLD_GLOBAL avoids runtime failures like:
    "Library libcublas.so.12 is not found or cannot be loaded".

    On Windows, the DLLs live in `bin` subdirectories and need to be added to
    the `PATH` or loaded explicitly. On Linux, the `.so` files live in `lib`
    subdirectories.
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
    attempts = [
        ("cuda", WHISPER_COMPUTE_TYPE_CUDA, f"CUDA ({WHISPER_COMPUTE_TYPE_CUDA})"),
        ("cuda", "float16", "CUDA (float16)"),
        ("cuda", "int8", "CUDA (int8)")
    ]
    if allow_cpu_fallback:
        attempts.append(("cpu", WHISPER_COMPUTE_TYPE_CPU, f"CPU ({WHISPER_COMPUTE_TYPE_CPU})"))

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
