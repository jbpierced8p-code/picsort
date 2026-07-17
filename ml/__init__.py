"""
PicSort AI - ML Module
Facial recognition, face grouping, and ML utilities.
"""

__version__ = "0.2.0"

from .face_recognition import (
    FaceEncoding,
    FaceGroup,
    detect_faces,
    compute_face_encodings,
    cluster_faces_by_person,
    batch_process_faces,
    generate_face_thumbnail,
    init_face_db,
    save_face_group,
    get_face_groups,
    get_faces_for_group,
    clear_face_cache,
)

__all__ = [
    "FaceEncoding",
    "FaceGroup",
    "detect_faces",
    "compute_face_encodings",
    "cluster_faces_by_person",
    "batch_process_faces",
    "generate_face_thumbnail",
    "init_face_db",
    "save_face_group",
    "get_face_groups",
    "get_faces_for_group",
    "clear_face_cache",
]