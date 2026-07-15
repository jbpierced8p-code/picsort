"""
PicSort AI - Engine Package
Core scanning engine, metadata extraction, perceptual hashing, and database index.
"""

from .scanner import (
    MediaFileInfo,
    DuplicateGroup,
    ScanResult,
    ScanPhase,
    MediaType,
    scan_directory,
    find_exact_duplicates,
    find_near_duplicates,
    find_all_duplicates,
    run_scan,
    quick_scan,
    is_supported_media,
    compute_sha256,
    extract_metadata,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
)

from .exif import extract_exif
from .hashing import (
    compute_phash,
    compute_phash_hex,
    compute_dhash,
    compute_whash,
    compute_colorhash,
    compute_all_hashes,
    compute_video_phash,
    hamming_distance,
    find_near_duplicates_by_phash,
    find_video_near_duplicates,
    find_all_near_duplicates,
    sample_video_frames,
)

from .db import (
    get_connection,
    init_db,
    get_stats,
    search_files,
    get_recent_scans,
    get_media_by_type,
    close,
)

from .tiers import set_tier, get_tier, has_feature, AppTier, TIER_CONFIG

from .face_detection import (
    DetectedFace,
    FaceGroup,
    detect_faces,
    compute_face_encodings,
    cluster_faces_by_person,
    find_face_groups_in_directory,
    init_face_db,
    save_face_group,
    get_face_groups,
    get_faces_for_group,
)

__version__ = "0.4.0"

__all__ = [
    # Scanner
    "MediaFileInfo",
    "DuplicateGroup",
    "ScanResult",
    "ScanPhase",
    "MediaType",
    "scan_directory",
    "find_exact_duplicates",
    "find_near_duplicates",
    "find_all_duplicates",
    "run_scan",
    "quick_scan",
    "is_supported_media",
    "compute_sha256",
    "extract_metadata",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    # EXIF
    "extract_exif",
    # Hashing
    "compute_phash",
    "compute_phash_hex",
    "compute_dhash",
    "compute_whash",
    "compute_colorhash",
    "compute_all_hashes",
    "compute_video_phash",
    "hamming_distance",
    "find_near_duplicates_by_phash",
    "find_video_near_duplicates",
    "find_all_near_duplicates",
    "sample_video_frames",
    # Database
    "get_connection",
    "init_db",
    "get_stats",
    "search_files",
    "get_recent_scans",
    "get_media_by_type",
    "close",
    # Tiers
    "set_tier",
    "get_tier",
    "has_feature",
    "AppTier",
    "TIER_CONFIG",
    # Face Detection
    "DetectedFace",
    "FaceGroup",
    "detect_faces",
    "compute_face_encodings",
    "cluster_faces_by_person",
    "find_face_groups_in_directory",
    "init_face_db",
    "save_face_group",
    "get_face_groups",
    "get_faces_for_group",
]