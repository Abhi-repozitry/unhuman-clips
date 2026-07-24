"""Test EasyOCR .tolist() fix with real detections."""
import easyocr
import tempfile, subprocess, os, json
import logging, sys
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("ocr_test")

from backend.ffmpeg_utils import get_ffmpeg
VIDEO = r"C:\Projects\unhuman-clips\backend\storage\working\0449350f-7669-4b85-a333-d1202f75dcba\downloads\fKoAOWQHP0o.webm"

# Extract a frame
tmp_fd, tmp_img = tempfile.mkstemp(suffix=".jpg", prefix="ocr_test_")
os.close(tmp_fd)
ffmpeg = get_ffmpeg()
subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", "5.0", "-i", VIDEO, "-vframes", "1", tmp_img], timeout=30)
logger.info(f"Frame extracted: {os.path.getsize(tmp_img)} bytes")

# Initialize EasyOCR
reader = easyocr.Reader(["en"], gpu=False)
result = reader.readtext(tmp_img, detail=1, paragraph=False)
logger.info(f"EasyOCR returned {len(result)} detections")

# Inspect raw return types
for i, r in enumerate(result[:5]):
    bbox = r[0]
    text = r[1]
    conf = r[2]
    bbox_type = type(bbox).__name__
    bbox_0_type = type(bbox[0]).__name__ if bbox else "N/A"
    has_tolist = hasattr(bbox, "tolist")
    logger.info(f"  [{i}] bbox_type={bbox_type}, bbox[0]_type={bbox_0_type}, hasattr_tolist={has_tolist}, text={text!r}, conf={conf:.3f}")

# Now test the fixed OCR function
from backend.pipeline.ocr import _try_easyocr
ocr_results = _try_easyocr(tmp_img)
logger.info(f"_try_easyocr returned {len(ocr_results)} results")
for r in ocr_results:
    logger.info(f"  text={r['text']!r}  confidence={r['confidence']:.3f}  bbox_len={len(r['bbox'])}")
    if r["bbox"]:
        logger.info(f"  bbox[0]={r['bbox'][0]}")

os.unlink(tmp_img)
logger.info("OCR .tolist() fix VERIFIED - no AttributeError")
