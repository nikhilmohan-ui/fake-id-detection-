#VASISTA'S CODE
"""
Document/Identity Preprocessing Pipeline
For fake ID / document screening systems.

Typical flow:
  raw upload -> validation -> deskew/crop -> enhancement ->
  quality checks -> feature extraction -> ready for OCR/classifier
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import hashlib
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------
# 1. Basic validation
# ---------------------------------------------------------------------

@dataclass
class ValidationResult:
    is_valid: bool
    reasons: list = field(default_factory=list)


def validate_upload(path: str, max_size_mb: int = 15) -> ValidationResult:
    reasons = []
    try:
        img = Image.open(path)
        img.verify()  # catches truncated/corrupt files
    except Exception as e:
        return ValidationResult(False, [f"corrupt_or_unreadable: {e}"])

    import os
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_size_mb:
        reasons.append("file_too_large")

    img = Image.open(path)  # reopen after verify()
    if img.format not in ("JPEG", "PNG", "HEIF", "TIFF"):
        reasons.append(f"unsupported_format:{img.format}")

    w, h = img.size
    if w < 600 or h < 400:
        reasons.append("resolution_too_low")

    return ValidationResult(len(reasons) == 0, reasons)


def strip_and_read_exif(path: str) -> dict:
    """Capture EXIF before stripping (useful metadata for fraud signals,
    e.g. re-saved/edited images, missing camera info, mismatched timestamps)."""
    img = Image.open(path)
    exif_data = {}
    raw_exif = img._getexif() if hasattr(img, "_getexif") else None
    if raw_exif:
        for tag_id, value in raw_exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif_data[tag] = value
    return exif_data


# ---------------------------------------------------------------------
# 2. Geometric normalization (deskew, crop, perspective correction)
# ---------------------------------------------------------------------

def load_image_cv(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def find_document_contour(img: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest 4-point contour, assumed to be the document edge."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def perspective_correct(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def deskew_and_crop(img: np.ndarray) -> np.ndarray:
    contour = find_document_contour(img)
    if contour is not None:
        return perspective_correct(img, contour)
    return img  # fall back to original if no clean quadrilateral found


# ---------------------------------------------------------------------
# 3. Image quality checks (blur, glare, lighting) — gate before OCR/model
# ---------------------------------------------------------------------

def blur_score(img: np.ndarray) -> float:
    """Variance of Laplacian; lower = blurrier."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def glare_ratio(img: np.ndarray, thresh: int = 240) -> float:
    """Fraction of near-pure-white pixels — proxy for flash glare over ID."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > thresh)
    return bright_pixels / gray.size


def brightness_stats(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"mean": float(gray.mean()), "std": float(gray.std())}


@dataclass
class QualityReport:
    blur: float
    glare: float
    brightness_mean: float
    brightness_std: float
    passed: bool
    issues: list


def run_quality_checks(
    img: np.ndarray,
    blur_min: float = 100.0,
    glare_max: float = 0.15,
    brightness_range=(40, 220),
) -> QualityReport:
    issues = []
    b = blur_score(img)
    g = glare_ratio(img)
    stats = brightness_stats(img)

    if b < blur_min:
        issues.append("too_blurry")
    if g > glare_max:
        issues.append("excessive_glare")
    if not (brightness_range[0] <= stats["mean"] <= brightness_range[1]):
        issues.append("poor_lighting")

    return QualityReport(
        blur=b, glare=g,
        brightness_mean=stats["mean"], brightness_std=stats["std"],
        passed=len(issues) == 0, issues=issues,
    )


# ---------------------------------------------------------------------
# 4. Enhancement (for OCR / downstream model input)
# ---------------------------------------------------------------------

def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return denoised


def normalize_size(img: np.ndarray, target_width: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------
# 5. Forensic / tamper-signal features (feed into a classifier, not a verdict)
# ---------------------------------------------------------------------

def error_level_analysis(path: str, quality: int = 90) -> np.ndarray:
    """
    ELA: re-save at known JPEG quality, diff against original.
    Regions edited after the fact often show different compression error levels.
    """
    original = Image.open(path).convert("RGB")
    tmp_path = "/tmp/_ela_resave.jpg"
    original.save(tmp_path, "JPEG", quality=quality)
    resaved = Image.open(tmp_path).convert("RGB")

    diff = np.array(original).astype(int) - np.array(resaved).astype(int)
    diff = np.abs(diff)
    # scale for visibility, but keep raw values for downstream feature use
    scale = 255.0 / (diff.max() if diff.max() != 0 else 1)
    return (diff * scale).astype(np.uint8)


def noise_consistency_map(img: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Local noise variance per block. Splicing/edited regions often
    show noise-level discontinuities vs. the rest of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    heat = np.zeros((h // block_size, w // block_size), dtype=np.float32)
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            heat[i // block_size, j // block_size] = np.var(block)
    return heat


def perceptual_hash(path: str) -> str:
    """For duplicate/reuse detection across submissions (template ID reuse fraud)."""
    img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    dct = cv2.dct(pixels)
    dct_low = dct[:8, :8]
    med = np.median(dct_low)
    bits = (dct_low > med).flatten()
    return hashlib.sha256(np.packbits(bits).tobytes()).hexdigest()


# ---------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------

@dataclass
class PreprocessOutput:
    cropped: np.ndarray
    ocr_ready: np.ndarray
    quality: QualityReport
    phash: str
    exif: dict
    ready_for_model: bool


def preprocess_document(path: str) -> PreprocessOutput:
    validation = validate_upload(path)
    if not validation.is_valid:
        raise ValueError(f"Upload failed validation: {validation.reasons}")

    exif = strip_and_read_exif(path)
    img = load_image_cv(path)
    cropped = deskew_and_crop(img)
    cropped = normalize_size(cropped)

    quality = run_quality_checks(cropped)
    ocr_ready = enhance_for_ocr(cropped)
    phash = perceptual_hash(path)

    return PreprocessOutput(
        cropped=cropped,
        ocr_ready=ocr_ready,
        quality=quality,
        phash=phash,
        exif=exif,
        ready_for_model=quality.passed,
    )


if __name__ == "__main__":
    result = preprocess_document("sample_id.jpg")
    print("Quality passed:", result.quality.passed)
    print("Issues:", result.quality.issues)
    print("Perceptual hash:", result.phash)
    cv2.imwrite("cropped_output.jpg", result.cropped)
    cv2.imwrite("ocr_ready_output.jpg", result.ocr_ready)