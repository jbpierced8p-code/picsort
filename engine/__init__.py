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
]