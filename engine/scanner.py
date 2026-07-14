"""
PicSort AI - Scanning Engine
Core library for scanning media files, extracting metadata, and building a searchable index.
"""

import hashlib
import mimetypes
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.exif import extract_exif
from engine.db import (
    get_connection,
    init_db,
    create_scan,
    complete_scan,
    upsert_media_file,
    get_stats,
    close,
    find_near_duplicates_by_hash,
    find_duplicate_groups,
)
from engine.tiers import has_feature

__version__ = "0.3.0"


class ScanPhase(Enum):
    WALKING = "walking"
    EXTRACTING = "extracting"
    HASHING = "hashing"
    INDEXING = "indexing"
    DONE = "done"


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass
class MediaFileInfo:
    """Rich metadata about a scanned media file."""
    path: str
    filename: str
    extension: str
    media_type: str = "unknown"
    size: int = 0
    created: Optional[float] = None
    modified: Optional[float] = None
    sha256: Optional[str] = None

    # Perceptual hashes
    phash: Optional[str] = None  # average hash
    dhash: Optional[str] = None  # difference hash
    whash: Optional[str] = None  # wavelet hash

    # Image metadata
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    date_taken: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    has_exif: bool = False

    # Video metadata
    duration_sec: Optional[float] = None
    video_codec: Optional[str] = None
    video_hash: Optional[str] = None  # frame-sampled hash


@dataclass
class DuplicateGroup:
    files: List[MediaFileInfo] = field(default_factory=list)
    algorithm: str = "exact"
    confidence: float = 1.0


@dataclass
class ScanResult:
    total_files: int = 0
    total_size: int = 0
    images: int = 0
    videos: int = 0
    with_exif: int = 0
    total_duplicates: int = 0
    storage_reclaimable: int = 0
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    scan_duration_ms: float = 0.0
    scan_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)

    @property
    def formatted_duration(self) -> str:
        seconds = self.scan_duration_ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"


# Supported file extensions
IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".tiff", ".tif", ".heic", ".heif", ".avif", ".svg",
}

VIDEO_EXTENSIONS: Set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv",
}

SUPPORTED_EXTENSIONS: Set[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def classify_media(extension: str) -> str:
    """Classify a file extension as image, video, or unknown."""
    ext = extension.lower()
    if ext in IMAGE_EXTENSIONS:
        return MediaType.IMAGE.value
    elif ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO.value
    return MediaType.UNKNOWN.value


def is_supported_media(path: str) -> bool:
    """Check if a file has a supported media extension."""
    ext = Path(path).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def compute_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def get_file_dates(filepath: str) -> Tuple[Optional[float], Optional[float]]:
    """Get file creation and modification timestamps."""
    try:
        stat = Path(filepath).stat()
        return stat.st_ctime, stat.st_mtime
    except OSError:
        return None, None


def extract_metadata(filepath: str) -> MediaFileInfo:
    """
    Extract all available metadata from a media file.
    Combines file system info with EXIF metadata for images.
    """
    p = Path(filepath)
    created, modified = get_file_dates(filepath)
    extension = p.suffix.lower()
    media_type = classify_media(extension)
    size = p.stat().st_size if p.exists() else 0

    info = MediaFileInfo(
        path=str(p.resolve()),
        filename=p.name,
        extension=extension,
        media_type=media_type,
        size=size,
        created=created,
        modified=modified,
    )

    # Extract EXIF for images
    if media_type == MediaType.IMAGE.value:
        try:
            exif = extract_exif(filepath)
            info.has_exif = exif.get("has_exif", False)
            info.image_width = exif.get("image_width")
            info.image_height = exif.get("image_height")
            info.camera_make = exif.get("camera_make")
            info.camera_model = exif.get("camera_model")
            info.date_taken = exif.get("date_taken")
            info.gps_latitude = exif.get("gps_latitude")
            info.gps_longitude = exif.get("gps_longitude")
        except Exception:
            pass

        # Compute perceptual hashes for images (Premium feature)
        if has_feature("perceptual_hashing"):
            try:
                from engine.hashing import compute_phash_hex, compute_dhash, compute_whash
                info.phash = compute_phash_hex(filepath)
                info.dhash = compute_dhash(filepath)
                info.whash = compute_whash(filepath)
            except Exception:
                pass

    # Compute perceptual hash for videos (Premium feature, uses ffmpeg)
    elif media_type == MediaType.VIDEO.value:
        if has_feature("perceptual_hashing"):
            try:
                from engine.hashing import compute_video_phash_ffmpeg
                info.video_hash = compute_video_phash_ffmpeg(filepath)
            except Exception:
                pass

    # For videos, try to get dimensions from ffmpeg probe or keep basic info
    # TODO: Add ffprobe-based video metadata extraction

    return info


def scan_directory(
    root_path: str,
    recursive: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[MediaFileInfo]:
    """
    Scan a directory for supported media files with rich metadata extraction.
    Returns a list of MediaFileInfo objects.
    """
    files: List[MediaFileInfo] = []
    root = Path(root_path).expanduser().resolve()

    if not root.exists():
        return files

    # Collect all candidate paths first
    candidates: List[Path] = []
    if recursive:
        candidates = list(root.rglob("*"))
    else:
        candidates = list(root.glob("*"))

    total = len(candidates)
    processed = 0
    error_count = 0

    for p in candidates:
        processed += 1
        if not p.is_file():
            continue
        if not is_supported_media(str(p)):
            continue

        if progress_callback:
            progress_callback(str(p), processed, total)

        try:
            info = extract_metadata(str(p))
            files.append(info)
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                pass  # Silently skip problematic files

    return files


def find_exact_duplicates(files: List[MediaFileInfo]) -> List[DuplicateGroup]:
    """
    Find exact duplicates by comparing file size and SHA-256 hash.
    Uses size as a fast pre-filter before computing hashes.
    """
    # Group by size first (fast pre-filter)
    size_groups: Dict[int, List[MediaFileInfo]] = {}
    for f in files:
        size_groups.setdefault(f.size, []).append(f)

    groups: List[DuplicateGroup] = []
    for size, candidates in size_groups.items():
        if len(candidates) < 2:
            continue

        # Compute hashes for size-matched candidates
        hash_groups: Dict[str, List[MediaFileInfo]] = {}
        for f in candidates:
            try:
                f.sha256 = compute_sha256(f.path)
                hash_groups.setdefault(f.sha256, []).append(f)
            except (IOError, OSError):
                continue  # Skip files we can't read

        for h, matches in hash_groups.items():
            if len(matches) >= 2:
                groups.append(DuplicateGroup(
                    files=matches,
                    algorithm="exact",
                    confidence=1.0,
                ))

    return groups


def find_near_duplicates(
    files: List[MediaFileInfo],
    phash_threshold: int = 5,
    video_threshold: int = 20,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[DuplicateGroup]:
    """
    Find near-duplicate images and videos using perceptual hashing.

    For images: uses average hash (pHash) with Hamming distance.
    For videos: uses frame-sampling comparison.

    Args:
        files: List of MediaFileInfo objects.
        phash_threshold: Max Hamming distance for image near-duplicates (0-64).
        video_threshold: Max distance for video near-duplicates.
        progress_callback: Optional progress callback.
    """
    from engine.hashing import find_all_near_duplicates
    return find_all_near_duplicates(
        files,
        phash_threshold=phash_threshold,
        video_threshold=video_threshold,
        progress_callback=progress_callback,
    )


def find_all_duplicates(
    files: List[MediaFileInfo],
    phash_threshold: int = 5,
    video_threshold: int = 20,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    premium: bool = False,
) -> List[DuplicateGroup]:
    """
    Find all duplicates: exact (SHA-256) + optional near (perceptual hash).

    Near-duplicate detection (perceptual hashing) is a Premium-tier feature.
    When premium=False, only exact SHA-256 duplicates are returned.

    Args:
        files: List of MediaFileInfo objects to check.
        phash_threshold: Max Hamming distance for image near-duplicates.
        video_threshold: Max distance for video near-duplicates.
        progress_callback: Optional progress callback.
        premium: If True, includes perceptual near-duplicate detection.

    Returns:
        Combined list of DuplicateGroup objects.
    """
    groups: List[DuplicateGroup] = []

    # Phase 1: Exact duplicates (available on all tiers)
    if progress_callback:
        progress_callback({"phase": "exact", "total": len(files), "current": 0, "percentage": 0})
    exact = find_exact_duplicates(files)
    groups.extend(exact)

    # Phase 2: Near-duplicate detection (Premium tier only)
    if premium:
        if progress_callback:
            progress_callback({"phase": "near_image", "total": len(files), "current": 0, "percentage": 0})
        near = find_near_duplicates(files, phash_threshold=phash_threshold, video_threshold=video_threshold)
        groups.extend(near)

    if progress_callback:
        progress_callback({"phase": "done", "total": len(files), "current": len(files), "percentage": 100})

    return groups


def run_scan(
    paths: List[str],
    db_path: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ScanResult:
    """
    Run a full scan: discover files, extract metadata, find duplicates,
    and persist results to the SQLite index.
    """
    start = time.time()
    errors: List[str] = []

    # Initialize database
    conn = None
    scan_id = None
    if db_path is not None:
        conn = get_connection(db_path)
        init_db(conn)
        scan_id = create_scan(conn, paths)

    # Phase 1: Walk directories and extract metadata
    all_files: List[MediaFileInfo] = []
    total_paths = 0
    all_candidates: List[Path] = []

    for p in paths:
        root = Path(p).expanduser().resolve()
        if not root.exists():
            errors.append(f"Path does not exist: {p}")
            continue
        all_candidates.extend(root.rglob("*"))

    total_paths = len(all_candidates)

    if progress_callback:
        progress_callback({
            "phase": ScanPhase.WALKING.value,
            "total": total_paths,
            "current": 0,
            "percentage": 0,
        })

    for idx, candidate in enumerate(all_candidates):
        if not candidate.is_file():
            continue
        if not is_supported_media(str(candidate)):
            continue

        try:
            info = extract_metadata(str(candidate))
            all_files.append(info)
        except Exception as e:
            errors.append(f"Error processing {candidate}: {e}")
            continue

        if progress_callback and idx % 50 == 0:
            progress_callback({
                "phase": ScanPhase.EXTRACTING.value,
                "total": total_paths,
                "current": idx + 1,
                "percentage": int((idx + 1) / total_paths * 100) if total_paths > 0 else 0,
            })

    # Phase 2: Persist to database
    if conn is not None and scan_id is not None:
        if progress_callback:
            progress_callback({
                "phase": ScanPhase.INDEXING.value,
                "total": len(all_files),
                "current": 0,
                "percentage": 0,
            })

        for idx, file_info in enumerate(all_files):
            file_dict = {
                "path": file_info.path,
                "filename": file_info.filename,
                "extension": file_info.extension,
                "media_type": file_info.media_type,
                "size": file_info.size,
                "created": file_info.created,
                "modified": file_info.modified,
                "sha256": file_info.sha256,
                "phash": file_info.phash,
                "dhash": file_info.dhash,
                "whash": file_info.whash,
                "video_hash": file_info.video_hash,
                "image_width": file_info.image_width,
                "image_height": file_info.image_height,
                "camera_make": file_info.camera_make,
                "camera_model": file_info.camera_model,
                "date_taken": file_info.date_taken,
                "gps_latitude": file_info.gps_latitude,
                "gps_longitude": file_info.gps_longitude,
                "duration_sec": file_info.duration_sec,
                "video_codec": file_info.video_codec,
                "has_exif": file_info.has_exif,
            }
            upsert_media_file(conn, scan_id, file_dict)

            if progress_callback and idx % 100 == 0:
                progress_callback({
                    "phase": ScanPhase.INDEXING.value,
                    "total": len(all_files),
                    "current": idx + 1,
                    "percentage": int((idx + 1) / len(all_files) * 100) if all_files else 0,
                })

    # Phase 3: Find all duplicates (exact + perceptual)
    if progress_callback:
        progress_callback({
            "phase": ScanPhase.HASHING.value,
            "total": len(all_files),
            "current": 0,
            "percentage": 0,
        })

    duplicates = find_all_duplicates(
        all_files,
        phash_threshold=5,
        video_threshold=20,
        progress_callback=lambda p: progress_callback(p) if progress_callback else None,
    )

    # Compute summary
    total_dup = sum(len(g.files) - 1 for g in duplicates)
    reclaimable = sum(g.files[0].size * (len(g.files) - 1) for g in duplicates)
    image_count = sum(1 for f in all_files if f.media_type == MediaType.IMAGE.value)
    video_count = sum(1 for f in all_files if f.media_type == MediaType.VIDEO.value)
    exif_count = sum(1 for f in all_files if f.has_exif)
    total_size = sum(f.size for f in all_files)
    elapsed = (time.time() - start) * 1000

    # Finalize database
    if conn is not None and scan_id is not None:
        complete_scan(conn, scan_id, len(all_files), total_size)
        close(conn)

    if progress_callback:
        progress_callback({
            "phase": ScanPhase.DONE.value,
            "total": len(all_files),
            "current": len(all_files),
            "percentage": 100,
        })

    return ScanResult(
        total_files=len(all_files),
        total_size=total_size,
        images=image_count,
        videos=video_count,
        with_exif=exif_count,
        total_duplicates=total_dup,
        storage_reclaimable=reclaimable,
        duplicate_groups=duplicates,
        scan_duration_ms=elapsed,
        scan_id=scan_id,
        errors=errors,
    )


def quick_scan(paths: List[str]) -> ScanResult:
    """
    Quick scan without database persistence.
    Good for "preview" scans before committing to indexing.
    """
    import time
    start = time.time()

    all_files: List[MediaFileInfo] = []
    for p in paths:
        all_files.extend(scan_directory(p))

    duplicates = find_exact_duplicates(all_files)

    total_dup = sum(len(g.files) - 1 for g in duplicates)
    reclaimable = sum(g.files[0].size * (len(g.files) - 1) for g in duplicates)
    image_count = sum(1 for f in all_files if f.media_type == MediaType.IMAGE.value)
    video_count = sum(1 for f in all_files if f.media_type == MediaType.VIDEO.value)
    exif_count = sum(1 for f in all_files if f.has_exif)
    total_size = sum(f.size for f in all_files)
    elapsed = (time.time() - start) * 1000

    return ScanResult(
        total_files=len(all_files),
        total_size=total_size,
        images=image_count,
        videos=video_count,
        with_exif=exif_count,
        total_duplicates=total_dup,
        storage_reclaimable=reclaimable,
        duplicate_groups=duplicates,
        scan_duration_ms=elapsed,
    )