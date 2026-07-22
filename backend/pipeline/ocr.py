import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import subprocess
import json
import os


def _extract_frame(video_path: str, timestamp_seconds: float, output_path: str) -> bool:
    """Extract a single frame at timestamp from video using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-loglevel", "error",
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
        import easyocr
        reader = easyocr.Reader(["en"])
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


def detect_frame_text(
    video_path: str,
    clip_windows: List[Dict[str, float]],
    working_dir: str,
    progress_cb: Optional[Any] = None,
    job_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Detect text on keyframes for each clip window.
    Returns list of OCR results per clip.
    """
    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)

    ocr_results = []
    total = len(clip_windows)

    for i, window in enumerate(clip_windows):
        clip_idx = i
        start = window["start"]
        end = window["end"]
        midpoint = start + (end - start) / 2

        frame_path = working_path / f"ocr_frame_{clip_idx}.jpg"

        if progress_cb:
            progress_cb(f"OCR: Extracting frame for clip {i+1}/{total}...", (i / total) * 50)

        if not _extract_frame(video_path, midpoint, str(frame_path)):
            if progress_cb:
                progress_cb(f"OCR: Frame extraction failed for clip {i+1}", (i / total) * 100)
            ocr_results.append({
                "clip_idx": clip_idx,
                "frame_time": midpoint,
                "text": "",
                "confidence": 0.0,
                "bbox": [],
                "language": "",
                "engine": "none",
                "error": "frame_extraction_failed"
            })
            continue

        if progress_cb:
            progress_cb(f"OCR: Running OCR on clip {i+1}/{total}...", 50 + (i / total) * 50)

        try:
            results = _try_ocr_engine(str(frame_path))
            if results:
                for r in results:
                    ocr_results.append({
                        "clip_idx": clip_idx,
                        "frame_time": midpoint,
                        **r,
                        "engine": "easyocr" if "easyocr" in str(_try_easyocr.__module__) else "pytesseract"
                    })
            else:
                ocr_results.append({
                    "clip_idx": clip_idx,
                    "frame_time": midpoint,
                    "text": "",
                    "confidence": 0.0,
                    "bbox": [],
                    "language": "",
                    "engine": "none",
                    "error": "no_text_detected"
                })
        except Exception as e:
            ocr_results.append({
                "clip_idx": clip_idx,
                "frame_time": midpoint,
                "text": "",
                "confidence": 0.0,
                "bbox": [],
                "language": "",
                "engine": "none",
                "error": str(e)
            })

        try:
            os.remove(str(frame_path))
        except OSError:
            pass

    return ocr_results


def detect_existing_captions(
    video_path: str,
    clip_windows: List[Dict[str, float]],
    working_dir: str,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    min_text_length: int = 8,
    min_confidence: float = 0.5,
    samples_per_clip: int = 3,
    bottom_region_percentage: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Detect if each clip window has existing burned-in captions.
    
    Samples multiple frames per clip from the bottom region (typical caption zone).
    Returns list of dicts with detailed detection results per clip.
    
    Args:
        video_path: Path to the video file
        clip_windows: List of clip windows with start/end times
        working_dir: Directory for temporary files
        progress_cb: Optional progress callback
        min_text_length: Minimum characters to consider as meaningful text
        min_confidence: Minimum OCR confidence threshold
        samples_per_clip: Number of frames to sample per clip
        bottom_region_percentage: Percentage of frame height to check (from bottom)
    
    Returns:
        List[Dict]: Per-clip results with keys:
            - has_captions: bool
            - detections: List[Dict] with text, confidence, timestamp
            - samples_checked: int
            - samples_with_text: int
    """
    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_clips = len(clip_windows)
    
    for i, window in enumerate(clip_windows):
        start = window["start"]
        end = window["end"]
        clip_duration = end - start
        
        clip_result = {
            "has_captions": False,
            "detections": [],
            "samples_checked": 0,
            "samples_with_text": 0
        }
        
        if clip_duration <= 0:
            results.append(clip_result)
            continue
        
        # Sample multiple timestamps across the clip
        if samples_per_clip == 1:
            timestamps = [start + clip_duration / 2]
        else:
            timestamps = [
                start + clip_duration * (j + 1) / (samples_per_clip + 1)
                for j in range(samples_per_clip)
            ]
        
        for sample_idx, ts in enumerate(timestamps):
            frame_path = working_path / f"caption_detect_{i}_{sample_idx}.jpg"
            
            if progress_cb:
                overall_progress = ((i * samples_per_clip + sample_idx) / (total_clips * samples_per_clip)) * 100
                progress_cb(
                    f"Checking clip {i+1}/{total_clips} sample {sample_idx+1}/{samples_per_clip}...",
                    overall_progress
                )
            
            try:
                if not _extract_and_crop_frame(
                    video_path, ts, str(frame_path), bottom_region_percentage
                ):
                    continue
                
                clip_result["samples_checked"] += 1
                
                ocr_results = _try_ocr_engine(str(frame_path))
                
                for r in ocr_results:
                    text = r.get("text", "").strip()
                    confidence = r.get("confidence", 0.0)
                    
                    if len(text) >= min_text_length and confidence >= min_confidence:
                        clip_result["detections"].append({
                            "text": text,
                            "confidence": round(confidence, 2),
                            "timestamp": round(ts, 2)
                        })
                        clip_result["samples_with_text"] += 1
                        break
                
                try:
                    os.remove(str(frame_path))
                except OSError:
                    pass
                    
            except Exception:
                try:
                    if os.path.exists(str(frame_path)):
                        os.remove(str(frame_path))
                except OSError:
                    pass
                continue
        
        # Require at least 2 samples to have text to reduce false positives
        clip_result["has_captions"] = clip_result["samples_with_text"] >= 2
        results.append(clip_result)
    
    return results


def summarize_ocr(ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize OCR results for stage_data."""
    total = len(ocr_results)
    detected = sum(1 for r in ocr_results if r.get("text", "").strip())
    avg_confidence = np.mean([r.get("confidence", 0.0) for r in ocr_results]) if ocr_results else 0.0
    engines_used = list(set(r.get("engine", "none") for r in ocr_results))

    return {
        "ocr_detected": detected,
        "ocr_total_clips": total,
        "ocr_avg_confidence": round(avg_confidence, 2),
        "ocr_engines_used": engines_used,
        "ocr_texts": [
            {
                "clip_idx": r["clip_idx"],
                "text": r.get("text", "")[:200],
                "confidence": round(r.get("confidence", 0.0), 2),
                "frame_time": round(r.get("frame_time", 0.0), 2)
            }
            for r in ocr_results
        ]
    }


def summarize_caption_detection(
    clip_windows: List[Dict[str, float]],
    caption_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarize caption detection results for stage_data."""
    total = len(caption_results)
    detected = sum(1 for r in caption_results if r.get("has_captions", False))
    skipped = detected
    generated = total - detected
    
    per_clip = []
    for i, r in enumerate(caption_results):
        per_clip.append({
            "clip_idx": i,
            "start": clip_windows[i]["start"],
            "end": clip_windows[i]["end"],
            "has_existing_captions": r.get("has_captions", False),
            "detections": r.get("detections", []),
            "samples_checked": r.get("samples_checked", 0),
            "samples_with_text": r.get("samples_with_text", 0)
        })
    
    return {
        "caption_detection_total": total,
        "caption_detection_detected": detected,
        "caption_detection_skipped": skipped,
        "caption_detection_generated": generated,
        "caption_detection_per_clip": per_clip
    }


def extract_visual_scene_summary(video_path: str, duration: float, working_dir: str, sample_interval: float = 15.0) -> List[Dict[str, Any]]:
    """
    Sample keyframes at intervals across the video duration and detect text/graphics
    to provide multimodal visual cues for LLM planning.
    """
    if duration <= 0:
        return []

    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)
    
    timestamps = [round(t, 1) for t in np.arange(0, duration, max(10.0, sample_interval))]
    visual_events = []

    for idx, ts in enumerate(timestamps[:30]):  # Cap at 30 keyframe samples max
        frame_path = str(working_path / f"keyframe_{idx}.jpg")
        if not _extract_frame(video_path, ts, frame_path):
            continue

        ocr_res = _try_ocr_engine(frame_path)
        extracted_text = ""
        if ocr_res and ocr_res[0].get("text"):
            extracted_text = ocr_res[0]["text"][:100].strip()

        visual_events.append({
            "timestamp": ts,
            "has_screen_text": bool(extracted_text),
            "screen_text": extracted_text
        })

        try:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        except OSError:
            pass

    return visual_events
