"""
PicSort AI - Engine Package
Core scanning engine, metadata extraction, and database index.
"""

from .scanner import (
    MediaFileInfo,
    DuplicateGroup,
    ScanResult,
    ScanPhase,
    MediaType,
    scan_directory,
    find_exact_duplicates,
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
from .db import (
    get_connection,
    init_db,
    get_stats,
    search_files,
    get_recent_scans,
    get_media_by_type,
    close,
)

__all__ = [
    # Scanner
    "MediaFileInfo",
    "DuplicateGroup",
    "ScanResult",
    "ScanPhase",
    "MediaType",
    "scan_directory",
    "find_exact_duplicates",
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
    # Database
    "get_connection",
    "init_db",
    "get_stats",
    "search_files",
    "get_recent_scans",
    "get_media_by_type",
    "close",
]