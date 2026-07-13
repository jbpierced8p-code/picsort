"""
PicSort AI - Scanning Engine
Core library for scanning media files, computing hashes, and detecting duplicates.
"""

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Tuple

__version__ = "0.1.0"


class ScanPhase(Enum):
    SCANNING = "scanning"
    HASHING = "hashing"
    COMPARING = "comparing"
    DONE = "done"


@dataclass
class MediaFile:
    path: str
    size: int
    modified: float
    created: float
    extension: str
    sha256: Optional[str] = None
    phash: Optional[str] = None


@dataclass
class DuplicateGroup:
    files: List[MediaFile] = field(default_factory=list)
    algorithm: str = "exact"
    confidence: float = 1.0


@dataclass
class ScanResult:
    total_files: int = 0
    total_duplicates: int = 0
    storage_reclaimable: int = 0
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    scan_duration_ms: float = 0.0


SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".tiff", ".tif", ".heic", ".heif",
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
}


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


def scan_directory(root_path: str, recursive: bool = True) -> List[MediaFile]:
    """
    Scan a directory for supported media files.
    Returns a list of MediaFile objects.
    """
    files: List[MediaFile] = []
    root = Path(root_path).expanduser().resolve()

    if not root.exists():
        return files

    paths: List[Path] = []
    if recursive:
        paths = list(root.rglob("*"))
    else:
        paths = list(root.glob("*"))

    for p in paths:
        if not p.is_file():
            continue
        if not is_supported_media(str(p)):
            continue

        stat = p.stat()
        media = MediaFile(
            path=str(p),
            size=stat.st_size,
            modified=stat.st_mtime,
            created=stat.st_ctime,
            extension=p.suffix.lower(),
        )
        files.append(media)

    return files


def find_exact_duplicates(files: List[MediaFile]) -> List[DuplicateGroup]:
    """
    Find exact duplicates by comparing file size first, then SHA-256 hash.
    """
    # Group by size first (fast pre-filter)
    size_groups: dict[int, List[MediaFile]] = {}
    for f in files:
        size_groups.setdefault(f.size, []).append(f)

    groups: List[DuplicateGroup] = []
    for size, candidates in size_groups.items():
        if len(candidates) < 2:
            continue

        # Compute hashes for candidates
        hash_groups: dict[str, List[MediaFile]] = {}
        for f in candidates:
            f.sha256 = compute_sha256(f.path)
            hash_groups.setdefault(f.sha256, []).append(f)

        for h, matches in hash_groups.items():
            if len(matches) >= 2:
                groups.append(DuplicateGroup(
                    files=matches,
                    algorithm="exact",
                    confidence=1.0,
                ))

    return groups


def run_scan(paths: List[str]) -> ScanResult:
    """
    Run a full scan: discover files, hash them, find duplicates.
    """
    import time
    start = time.time()

    all_files: List[MediaFile] = []
    for p in paths:
        all_files.extend(scan_directory(p))

    duplicates = find_exact_duplicates(all_files)

    total_dup = sum(len(g.files) - 1 for g in duplicates)
    reclaimable = sum(
        g.files[0].size * (len(g.files) - 1) for g in duplicates
    )

    elapsed = (time.time() - start) * 1000

    return ScanResult(
        total_files=len(all_files),
        total_duplicates=total_dup,
        storage_reclaimable=reclaimable,
        duplicate_groups=duplicates,
        scan_duration_ms=elapsed,
    )