"""
backend/modules/ocr_module.py
Extracts text, bounding boxes, and confidence scores from document images
using PaddleOCR with defensive type checking, format normalization, and robust error handling.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import re
import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

logger = logging.getLogger("xai_docguard.ocr")


class OCRResult:
    def __init__(
        self,
        raw_text: str,
        structured_fields: Dict[str, Any],
        tokens: List[Dict[str, Any]],
        mean_confidence: float,
        is_valid_format: bool,
        flags: List[str]
    ):
        self.raw_text = raw_text
        self.structured_fields = structured_fields
        self.tokens = tokens
        self.mean_confidence = mean_confidence
        self.is_valid_format = is_valid_format
        self.flags = flags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "structured_fields": self.structured_fields,
            "mean_confidence": round(self.mean_confidence, 4),
            "is_valid_format": self.is_valid_format,
            "token_count": len(self.tokens),
            "flags": self.flags
        }


class OCRModule:
    # Common document header words to exclude from ID matching
    DOCUMENT_STOPWORDS = {
        "IDENTITY", "IDENTIFICATION", "PASSPORT", "GOVERNMENT",
        "REPUBLIC", "NATIONAL", "DEPARTMENT", "SIGNATURE",
        "PERMANENT", "ACCOUNT", "NUMBER", "OFFICIAL", "REPUBLIC"
    }

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        confidence_threshold: float = 0.65
    ):
        self.confidence_threshold = confidence_threshold
        # Non-capturing groups ensure re.findall returns the full matched string
        self.date_pattern = re.compile(
            r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b"
        )
        # Requires at least one numeric digit to avoid matching plain English words
        self.id_pattern = re.compile(
            r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{6,14}\b"
        )

        self.engine = None
        if PaddleOCR is None:
            logger.error("PaddleOCR is not installed. Please install requirements.")
            return

        # Safe initialization with automatic GPU -> CPU fallback
        try:
            self.engine = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info("PaddleOCR initialized (use_gpu=%s).", use_gpu)
        except Exception as err:
            if use_gpu:
                logger.warning("GPU initialization failed (%s). Retrying with CPU...", err)
                try:
                    self.engine = PaddleOCR(
                        use_angle_cls=use_angle_cls,
                        lang=lang,
                        use_gpu=False,
                        show_log=False
                    )
                    logger.info("PaddleOCR initialized successfully on CPU fallback.")
                except Exception as fallback_err:
                    logger.error("Failed to initialize PaddleOCR on CPU: %s", fallback_err)
            else:
                logger.error("PaddleOCR initialization failed: %s", err)

    def normalize_image(self, image: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Validates dimensions and normalizes channels/dtypes for PaddleOCR.
        PaddleOCR requires uint8 BGR/RGB images with shape (H, W, 3).
        """
        if image is None:
            return None, "Input image is None."
        if not isinstance(image, np.ndarray):
            return None, f"Expected numpy.ndarray, got {type(image)}."
        if image.size == 0:
            return None, "Image array is empty."

        # Handle floating-point images scaled 0.0 - 1.0
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Dimension checks
        if len(image.shape) == 2:  # Grayscale (H, W) -> 3-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] != 3:
                return None, f"Unsupported channel count: {image.shape[2]}"
        else:
            return None, f"Invalid image shape: {image.shape}"

        h, w = image.shape[:2]
        if h < 32 or w < 32:
            return None, f"Image dimensions too small: {w}x{h}"

        return image, None

    def _polygon_to_bbox(self, polygon: Any, img_shape: Tuple[int, ...]) -> Optional[List[int]]:
        """
        Converts PaddleOCR's 4-point polygon to a clamped bounding box [x, y, w, h].
        """
        try:
            pts = np.array(polygon, dtype=np.float32)
            if pts.shape != (4, 2) or np.isnan(pts).any():
                return None

            min_x = int(np.floor(np.min(pts[:, 0])))
            max_x = int(np.ceil(np.max(pts[:, 0])))
            min_y = int(np.floor(np.min(pts[:, 1])))
            max_y = int(np.ceil(np.max(pts[:, 1])))

            h, w = img_shape[:2]
            x = max(0, min(min_x, w - 1))
            y = max(0, min(min_y, h - 1))
            box_w = max(1, min(max_x - min_x, w - x))
            box_h = max(1, min(max_y - min_y, h - y))

            return [x, y, box_w, box_h]
        except Exception:
            return None

    def run_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs inference and guards against malformed or empty OCR results.
        """
        if self.engine is None:
            logger.error("PaddleOCR engine is unavailable.")
            return []

        tokens: List[Dict[str, Any]] = []
        try:
            ocr_output = self.engine.ocr(image, cls=True)

            # PaddleOCR returns None or empty structure when no text is found
            if not ocr_output or not isinstance(ocr_output, list) or len(ocr_output) == 0:
                return []

            first_page = ocr_output[0]
            if not first_page or not isinstance(first_page, list):
                return []

            for item in first_page:
                if not item or len(item) < 2:
                    continue

                polygon, rec_tuple = item[0], item[1]
                if not rec_tuple or len(rec_tuple) < 2:
                    continue

                text, confidence = rec_tuple[0], rec_tuple[1]
                text_clean = str(text).strip()
                if not text_clean:
                    continue

                bbox = self._polygon_to_bbox(polygon, image.shape)
                if bbox is None:
                    continue

                try:
                    conf_float = float(confidence)
                except (ValueError, TypeError):
                    conf_float = 0.0

                tokens.append({
                    "text": text_clean,
                    "confidence": conf_float,
                    "bbox": bbox,
                    "polygon": polygon if isinstance(polygon, list) else []
                })

        except Exception as e:
            logger.error("PaddleOCR inference error: %s", str(e), exc_info=True)

        return tokens

    def is_noise_token(self, text: str) -> bool:
        """Filters out fragmented single letters, isolated single digits, and noise symbols."""
        clean = text.strip()
        if len(clean) <= 1:
            return True
        if clean.isdigit() and len(clean) == 1:
            return True
        if re.match(r"^[^a-zA-Z0-9]+$", clean):
            return True
        return False

    def consolidate_tokens(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidates adjacent word tokens into full information lines and phrases."""
        if not tokens:
            return []
        
        # Sort top-to-bottom, then left-to-right
        sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][1] // 18, t["bbox"][0]))
        lines: List[List[Dict[str, Any]]] = []
        current_line: List[Dict[str, Any]] = []

        for token in sorted_tokens:
            if not current_line:
                current_line.append(token)
                continue

            last = current_line[-1]
            last_y_mid = last["bbox"][1] + last["bbox"][3] / 2
            tok_y_mid = token["bbox"][1] + token["bbox"][3] / 2
            same_line = abs(last_y_mid - tok_y_mid) <= max(last["bbox"][3], token["bbox"][3]) * 0.75
            x_gap = token["bbox"][0] - (last["bbox"][0] + last["bbox"][2])

            if same_line and -10 <= x_gap <= 45:
                current_line.append(token)
            else:
                lines.append(current_line)
                current_line = [token]

        if current_line:
            lines.append(current_line)

        consolidated: List[Dict[str, Any]] = []
        for line in lines:
            text = " ".join(t["text"] for t in line).strip()
            min_x = min(t["bbox"][0] for t in line)
            min_y = min(t["bbox"][1] for t in line)
            max_x = max(t["bbox"][0] + t["bbox"][2] for t in line)
            max_y = max(t["bbox"][1] + t["bbox"][3] for t in line)
            avg_conf = sum(t["confidence"] for t in line) / len(line)

            consolidated.append({
                "text": text,
                "confidence": round(avg_conf, 4),
                "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                "polygon": [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            })

        return consolidated

    def parse_fields(self, tokens: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Parses full information entities: Name, DOB, Place, Date, Authority, and ID number.
        Rejects single letters, single numbers, and header stopwords.
        """
        flags: List[str] = []
        
        # Step 1: Filter noise tokens
        cleaned_tokens = [t for t in tokens if not self.is_noise_token(t.get("text", ""))]
        # Step 2: Consolidate into full lines
        consolidated = self.consolidate_tokens(cleaned_tokens)
        full_text = " ".join(t.get("text", "") for t in consolidated)

        fields: Dict[str, Any] = {
            "name": None,
            "dob": None,
            "place": None,
            "date": None,
            "authority": None,
            "id_number": None,
            "dates_found": []
        }

        # 1. Dates (DOB and Document Date)
        dates = self.date_pattern.findall(full_text)
        if dates:
            fields["dob"] = dates[0]
            fields["dates_found"] = dates
            if len(dates) > 1:
                fields["date"] = dates[1]
            else:
                fields["date"] = dates[0]
        else:
            flags.append("MISSING_DOB")

        # 2. Document ID
        id_candidates = self.id_pattern.findall(full_text)
        valid_id = None
        for cand in id_candidates:
            if cand.upper() not in self.DOCUMENT_STOPWORDS and not cand.isdigit():
                valid_id = cand
                break
        if not valid_id and id_candidates:
            valid_id = id_candidates[0]

        if valid_id:
            fields["id_number"] = valid_id
        else:
            flags.append("MISSING_ID_NUMBER")

        # 3. Name, Place, Authority heuristics from consolidated lines
        for item in consolidated:
            line_txt = item["text"].strip()
            upper = line_txt.upper()

            # Name extraction
            if not fields["name"]:
                name_match = re.search(r"(?:NAME|CARDHOLDER|HOLDER|FULL NAME)[:\s]+([A-Za-z\s.'-]{3,40})", line_txt, re.I)
                if name_match:
                    fields["name"] = name_match.group(1).strip()
                elif re.match(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}$", line_txt):
                    if not any(sw in upper for sw in self.DOCUMENT_STOPWORDS):
                        fields["name"] = line_txt

            # Place extraction
            if not fields["place"]:
                place_match = re.search(r"(?:PLACE|LOCATION|CITY|ADDRESS|BIRTHPLACE|STATE)[:\s]+([A-Za-z0-9\s,.-]{3,50})", line_txt, re.I)
                if place_match:
                    fields["place"] = place_match.group(1).strip()

            # Authority extraction
            if not fields["authority"]:
                auth_match = re.search(r"(?:AUTHORITY|ISSUED BY|GOVERNMENT OF|DEPARTMENT OF|REPUBLIC OF)[:\s]+([A-Za-z0-9\s,.-]{3,60})", line_txt, re.I)
                if auth_match:
                    fields["authority"] = auth_match.group(0).strip()
                elif any(kw in upper for kw in ["REPUBLIC OF", "GOVERNMENT OF", "DEPARTMENT OF", "TRANSPORT AUTHORITY"]):
                    fields["authority"] = line_txt

        return fields, flags

    def process(self, raw_image: np.ndarray) -> OCRResult:
        """Processes an image end-to-end with validation."""
        normalized_img, err_msg = self.normalize_image(raw_image)
        if normalized_img is None:
            return OCRResult(
                raw_text="",
                structured_fields={},
                tokens=[],
                mean_confidence=0.0,
                is_valid_format=False,
                flags=[f"INPUT_ERROR: {err_msg}"]
            )

        tokens = self.run_ocr(normalized_img)
        flags: List[str] = []

        confidences = [t["confidence"] for t in tokens]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        if confidences and mean_conf < self.confidence_threshold:
            flags.append(f"LOW_OVERALL_CONFIDENCE ({round(mean_conf, 2)})")

        structured_fields, parse_flags = self.parse_fields(tokens)
        flags.extend(parse_flags)

        is_valid_format = ("MISSING_ID_NUMBER" not in flags) and (mean_conf >= self.confidence_threshold)

        return OCRResult(
            raw_text=" ".join(t["text"] for t in tokens),
            structured_fields=structured_fields,
            tokens=tokens,
            mean_confidence=mean_conf,
            is_valid_format=is_valid_format,
            flags=flags
        )
