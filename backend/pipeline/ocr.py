import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import json
import os
from backend.ffmpeg_utils import get_ffmpeg

# Module-level EasyOCR reader cache (expensive to instantiate)
_easyocr_reader = None


def _get_easyocr_reader():
    """Get or create the EasyOCR reader (cached at module level)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"])
    return _easyocr_reader


def _extract_frame(video_path: str, timestamp_seconds: float, output_path: str) -> bool:
    """Extract a single frame at timestamp from video using ffmpeg."""
    try:
        cmd = [
            get_ffmpeg(), "-loglevel", "error",
            "-ss", str(timestamp_seconds),
            "-i", str(video_path),
            "-vframes", "1",
            "-y", str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _try_easyocr(image_path: str) -> List[Dict[str, Any]]:
    """Try to detect text using EasyOCR. Returns list of {text, confidence, bbox, language}."""
    try:
        reader = _get_easyocr_reader()
        result = reader.readtext(str(image_path), detail=1, paragraph=False)
        return [
            {
                "text": " ".join([r[1] for r in result]),
                "confidence": np.mean([r[2] for r in result]) if result else 0.0,
                "bbox": [r[0].tolist() for r in result] if result else [],
                "language": "en",
            }
        ]
    except (ImportError, Exception):
        return []


def _try_pytesseract(image_path: str) -> List[Dict[str, Any]]:
    """Try to detect text using pytesseract. Returns list of {text, confidence, bbox, language}."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(str(image_path))
        text = pytesseract.image_to_string(img)
        if text.strip():
            return [{"text": text.strip(), "confidence": 0.7, "bbox": [], "language": "en"}]
    except (ImportError, Exception):
        pass
    return []


def _try_ocr_engine(image_path: str) -> List[Dict[str, Any]]:
    """Try OCR engines in order: EasyOCR -> pytesseract -> empty."""
    results = _try_easyocr(image_path)
    if results:
        return results
    results = _try_pytesseract(image_path)
    if results:
        return results
    return []


def _crop_bottom_region(image_path: str, output_path: str, bottom_percentage: float = 0.3) -> bool:
    """Crop image to bottom region (typical caption zone) and save."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        height, width = img.shape[:2]
        crop_height = int(height * bottom_percentage)
        y_start = height - crop_height
        cropped = img[y_start:height, 0:width]
        cv2.imwrite(str(output_path), cropped)
        return True
    except Exception:
        return False


def _extract_and_crop_frame(
    video_path: str,
    timestamp_seconds: float,
    output_path: str,
    bottom_percentage: float = 0.3
) -> bool:
    """Extract a frame and crop to bottom region."""
    temp_path = output_path + ".temp.jpg"
    try:
        if not _extract_frame(video_path, timestamp_seconds, temp_path):
            return False
        if not _crop_bottom_region(temp_path, output_path, bottom_percentage):
            return False
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return True
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return False
