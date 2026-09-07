"""
XAI-DocGuard Backend Modules Package
"""

from .face_match import (
    compare_faces,
    extract_face_and_bbox,
    get_face_embedding,
    calculate_face_risk,
    validate_image_and_faces
)

__all__ = [
    "compare_faces",
    "extract_face_and_bbox",
    "get_face_embedding",
    "calculate_face_risk",
    "validate_image_and_faces"
]
