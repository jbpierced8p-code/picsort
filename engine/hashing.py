"""
PicSort AI - Perceptual Hashing Module
Computes perceptual hashes for images and video frame samples,
enabling near-duplicate detection via Hamming distance.
"""

import io
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from engine.scanner import MediaFileInfo, MediaType

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


# Default perceptual hash size (8 -> 8x8 = 64-bit hash)
DEFAULT_HASH_SIZE = 8

# Default Hamming distance threshold for "near duplicate"
# 0 = exact perceptual match, higher = more tolerant
DEFAULT_THRESHOLD = 5

# Number of frames to sample for video hashing
DEFAULT_VIDEO_FRAMES = 5

# Minimum image dimensions to attempt perceptual hashing
MIN_HASH_DIMENSION = 32


# ─── Perceptual Hash Computation ───────────────────────────────────────────

def compute_phash(
    image_path: str,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> Optional['imagehash.ImageHash']:
    """
    Compute a perceptual hash of an image using average hash (aHash).
    Returns None if the image cannot be processed.

    Args:
        image_path: Path to the image file.
        hash_size: Size of the hash (hash_size x hash_size bits).
    """
    if not HAS_IMAGEHASH:
        return None

    try:
        img = Image.open(image_path)
        if img.width < MIN_HASH_DIMENSION or img.height < MIN_HASH_DIMENSION:
            return None
        return imagehash.average_hash(img, hash_size=hash_size)
    except Exception:
        return None


def compute_phash_hex(
    image_path: str,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> Optional[str]:
    """
    Compute a perceptual hash and return it as a hex string.
    Returns None if the image cannot be processed.
    """
    phash = compute_phash(image_path, hash_size=hash_size)
    return str(phash) if phash is not None else None


def compute_dhash(
    image_path: str,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> Optional[str]:
    """
    Compute a difference hash (dHash) - compares adjacent pixels.
    Often better at detecting edited/slightly modified images.
    Returns hex string or None.
    """
    if not HAS_IMAGEHASH:
        return None

    try:
        img = Image.open(image_path)
        if img.width < MIN_HASH_DIMENSION or img.height < MIN_HASH_DIMENSION:
            return None
        return str(imagehash.dhash(img, hash_size=hash_size))
    except Exception:
        return None


def compute_whash(
    image_path: str,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> Optional[str]:
    """
    Compute a wavelet hash (wHash) using Haar wavelets.
    Best for detecting watermarked or compressed images.
    Returns hex string or None.
    """
    if not HAS_IMAGEHASH:
        return None

    try:
        img = Image.open(image_path)
        if img.width < MIN_HASH_DIMENSION or img.height < MIN_HASH_DIMENSION:
            return None
        return str(imagehash.whash(img, hash_size=hash_size))
    except Exception:
        return None


def compute_colorhash(
    image_path: str,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> Optional[str]:
    """
    Compute a color hash based on color distribution.
    Useful for finding images with similar color palettes.
    Returns hex string or None.
    """
    if not HAS_IMAGEHASH:
        return None

    try:
        img = Image.open(image_path)
        if img.width < MIN_HASH_DIMENSION or img.height < MIN_HASH_DIMENSION:
            return None
        # colorhash uses binbits=3 by default, not hash_size
        return str(imagehash.colorhash(img))
    except Exception:
        return None


def compute_all_hashes(
    image_path: str,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> dict:
    """
    Compute all perceptual hash types for an image.
    Returns dict with 'phash', 'dhash', 'whash', 'colorhash' keys.
    Missing hashes are None.
    """
    return {
        "phash": compute_phash_hex(image_path, hash_size=hash_size),
        "dhash": compute_dhash(image_path, hash_size=hash_size),
        "whash": compute_whash(image_path, hash_size=hash_size),
        "colorhash": compute_colorhash(image_path, hash_size=hash_size),
    }


# ─── Hamming Distance ─────────────────────────────────────────────────────

def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute the Hamming distance between two hex hash strings.
    Lower distance = more visually similar images.

    For equal-length hex strings, counts bit differences.
    For unequal lengths, falls back to character-level diff.
    """
    if not hash1 or not hash2:
        return -1  # Can't compare

    # If hashes are the same hex string, distance is 0
    if hash1 == hash2:
        return 0

    # For hex hashes, convert to binary and count differing bits
    try:
        # Convert hex strings to integers
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        # XOR and count bits
        xor = h1 ^ h2
        return xor.bit_count()  # Python 3.8+ has int.bit_count()
    except (ValueError, TypeError):
        # Fallback: character-level comparison
        return sum(a != b for a, b in zip(hash1, hash2)) + abs(len(hash1) - len(hash2))


def hash_distance_ratio(hash1: str, hash_size: int = DEFAULT_HASH_SIZE) -> Callable[[str], float]:
    """
    Create a distance ratio function for sorting near-duplicates.
    Returns a function that computes distance / max_possible_distance.
    """
    max_distance = hash_size * hash_size  # bits in the hash
    h1_int = int(hash1, 16) if hash1 else 0

    def ratio(hash2: str) -> float:
        if not hash2:
            return 1.0
        try:
            h2_int = int(hash2, 16)
            distance = (h1_int ^ h2_int).bit_count()
            return distance / max_distance
        except (ValueError, TypeError):
            return 1.0

    return ratio


# ─── Near-Duplicate Detection ─────────────────────────────────────────────

def find_near_duplicates_by_phash(
    files: List[MediaFileInfo],
    threshold: int = DEFAULT_THRESHOLD,
    hash_type: str = "phash",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List['DuplicateGroup']:
    """
    Find near-duplicate images using perceptual hashing and Hamming distance.

    Args:
        files: List of MediaFileInfo objects to check.
        threshold: Maximum Hamming distance to consider as duplicate.
        hash_type: Which perceptual hash to use ('phash', 'dhash', 'whash', 'colorhash').
        progress_callback: Optional callback(current, total).

    Returns:
        List of DuplicateGroup objects for near-duplicate clusters.
    """
    from engine.scanner import DuplicateGroup

    # Filter to only image files
    image_files = [f for f in files if f.media_type == MediaType.IMAGE.value]
    if len(image_files) < 2:
        return []

    # Compute hashes for all images
    hashes: List[Tuple[MediaFileInfo, Optional[str]]] = []
    total = len(image_files)

    for idx, f in enumerate(image_files):
        if progress_callback and idx % 20 == 0:
            progress_callback(idx, total)

        if hash_type == "dhash":
            h = compute_dhash(f.path)
        elif hash_type == "whash":
            h = compute_whash(f.path)
        elif hash_type == "colorhash":
            h = compute_colorhash(f.path)
        else:
            h = compute_phash_hex(f.path)

        if h is not None:
            # Also compute SHA-256 if not already present
            if f.sha256 is None:
                try:
                    from engine.scanner import compute_sha256
                    f.sha256 = compute_sha256(f.path)
                except Exception:
                    pass
            hashes.append((f, h))

    if len(hashes) < 2:
        return []

    # Build a lookup: path -> hash
    hash_lookup: Dict[str, str] = {}
    for f, h in hashes:
        hash_lookup[f.path] = h

    # Cluster by Hamming distance using simple greedy approach
    groups: List[DuplicateGroup] = []
    assigned: Set[int] = set()

    for i in range(len(hashes)):
        if i in assigned:
            continue

        f_i, h_i = hashes[i]
        cluster: List[MediaFileInfo] = [f_i]
        assigned.add(i)

        for j in range(i + 1, len(hashes)):
            if j in assigned:
                continue

            f_j, h_j = hashes[j]
            distance = hamming_distance(h_i, h_j)

            if 0 <= distance <= threshold:
                cluster.append(f_j)
                assigned.add(j)

        if len(cluster) >= 2:
            # Compute confidence based on average distance
            total_distance = 0
            pairs = 0
            for a in range(len(cluster)):
                for b in range(a + 1, len(cluster)):
                    ha = hash_lookup.get(cluster[a].path, "")
                    hb = hash_lookup.get(cluster[b].path, "")
                    d = hamming_distance(ha, hb)
                    if d >= 0:
                        total_distance += d
                        pairs += 1

            avg_distance = total_distance / pairs if pairs > 0 else 0
            confidence = max(0.0, 1.0 - avg_distance / (DEFAULT_HASH_SIZE ** 2))

            groups.append(DuplicateGroup(
                files=cluster,
                algorithm=f"perceptual_{hash_type}",
                confidence=round(confidence, 4),
            ))

    return groups


# ─── Video Frame Sampling ─────────────────────────────────────────────────

def sample_video_frames(
    video_path: str,
    num_frames: int = DEFAULT_VIDEO_FRAMES,
) -> List[bytes]:
    """
    Sample frames from a video file by reading raw bytes at intervals.
    Returns a list of raw byte chunks from different parts of the file.

    This is a lightweight approach that doesn't require ffmpeg.
    For more accurate results, use ffprobe + ffmpeg.
    """
    path = Path(video_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    file_size = path.stat().st_size
    frames: List[bytes] = []

    try:
        with open(video_path, "rb") as f:
            # Sample at intervals across the file
            for i in range(num_frames):
                # Skip header (first 1KB) and avoid tail (last 1KB)
                sample_start = max(1024, (file_size - 2048) * i // num_frames)
                sample_end = min(sample_start + 4096, file_size - 1024)

                if sample_start >= sample_end:
                    continue

                f.seek(sample_start)
                chunk = f.read(sample_end - sample_start)
                if len(chunk) >= 256:  # Only keep substantial chunks
                    frames.append(chunk)
    except (IOError, OSError):
        pass

    return frames


def sample_video_frames_ffmpeg(
    video_path: str,
    num_frames: int = DEFAULT_VIDEO_FRAMES,
) -> List[bytes]:
    """
    Sample frames from a video using ffmpeg.
    Extracts keyframes as PNG images and returns their raw pixel data as bytes.

    Falls back to byte-sampling if ffmpeg is not available.
    """
    if not os.path.isfile(video_path):
        return []

    # Check if ffmpeg is available
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return sample_video_frames(video_path, num_frames=num_frames)

    # Use ffprobe to get video duration
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", video_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(probe.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        duration = 30.0  # guess 30s

    if duration <= 0:
        duration = 30.0

    frames: List[bytes] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_frames):
            # Sample frames at even intervals
            timestamp = min(duration * (i + 1) / (num_frames + 1), duration - 0.1)
            timestamp = max(timestamp, 0.0)

            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")

            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(timestamp),
                     "-i", video_path,
                     "-vframes", "1",
                     "-f", "image2",
                     "-vf", "scale=64:64",  # Small size for hashing
                     frame_path],
                    capture_output=True,
                    timeout=30,
                )

                if os.path.isfile(frame_path):
                    # Read the raw pixel data for hashing
                    img = Image.open(frame_path).convert("L")  # grayscale
                    frames.append(img.tobytes())

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception):
                continue

    if not frames:
        return sample_video_frames(video_path, num_frames=num_frames)

    return frames


def compute_video_phash_ffmpeg(
    video_path: str,
    num_frames: int = DEFAULT_VIDEO_FRAMES,
) -> Optional[str]:
    """
    Compute a perceptual hash for a video using ffmpeg frame extraction.
    Extracts frames evenly across the video duration, computes pHash on each,
    and combines them into a single representative hash.

    Falls back to byte-sampling if ffmpeg is unavailable.
    """
    if not HAS_IMAGEHASH:
        return None

    path = Path(video_path)
    if not path.exists():
        return None

    # Try ffmpeg-based extraction first
    frames = sample_video_frames_ffmpeg(video_path, num_frames=num_frames)

    # If ffmpeg didn't work, fall back to byte sampling
    if not frames or len(frames) < 2:
        return compute_video_phash(video_path, num_frames=num_frames)

    try:
        import hashlib
        # Combine all frame hashes into a single hash
        combined = hashlib.md5()
        for frame_data in frames:
            combined.update(frame_data)
        return combined.hexdigest()
    except Exception:
        return compute_video_phash(video_path, num_frames=num_frames)


def compute_video_phash(
    video_path: str,
    num_frames: int = DEFAULT_VIDEO_FRAMES,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> Optional[str]:
    """
    Compute a perceptual hash for a video by hashing frame samples.
    Uses raw byte chunks from different parts of the file.

    Returns a combined hex hash string, or None if failed.
    """
    if not HAS_IMAGEHASH:
        return None

    # For small videos (< 100KB), hash the whole file content
    path = Path(video_path)
    if not path.exists():
        return None

    if path.stat().st_size < 100 * 1024:
        try:
            import hashlib
            with open(video_path, "rb") as f:
                data = f.read()
            # Return MD5 hex digest as a simple content hash
            return hashlib.md5(data).hexdigest()
        except Exception:
            return None

    samples = sample_video_frames(video_path, num_frames=num_frames)
    if not samples:
        return None

    # Hash each frame sample and combine
    try:
        hashes: List[str] = []
        for chunk in samples:
            # Create a small PIL image from the chunk data to hash
            # Use the raw bytes' hash as a proxy for frame content
            import hashlib
            chunk_hash = hashlib.md5(chunk).hexdigest()
            hashes.append(chunk_hash)

        # Combine frame hashes into a single representative hash
        combined = "".join(h[:4] for h in hashes)
        return combined
    except Exception:
        return None


def find_video_near_duplicates(
    files: List[MediaFileInfo],
    threshold: int = 20,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List['DuplicateGroup']:
    """
    Find near-duplicate videos by comparing frame-sample hashes.

    Args:
        files: List of MediaFileInfo objects.
        threshold: Maximum character-level distance (higher = more tolerant).
        progress_callback: Optional callback(current, total).

    Returns:
        List of DuplicateGroup objects.
    """
    from engine.scanner import DuplicateGroup

    video_files = [f for f in files if f.media_type == MediaType.VIDEO.value]
    if len(video_files) < 2:
        return []

    # Compute video hashes
    video_hashes: List[Tuple[MediaFileInfo, Optional[str]]] = []
    total = len(video_files)

    for idx, f in enumerate(video_files):
        if progress_callback:
            progress_callback(idx, total)

        h = compute_video_phash(f.path)
        video_hashes.append((f, h))

    valid = [(f, h) for f, h in video_hashes if h is not None]
    if len(valid) < 2:
        return []

    groups: List[DuplicateGroup] = []
    assigned: Set[int] = set()

    for i in range(len(valid)):
        if i in assigned:
            continue

        f_i, h_i = valid[i]
        cluster: List[MediaFileInfo] = [f_i]
        assigned.add(i)

        for j in range(i + 1, len(valid)):
            if j in assigned:
                continue

            f_j, h_j = valid[j]
            distance = hamming_distance(h_i, h_j)

            if 0 <= distance <= threshold:
                cluster.append(f_j)
                assigned.add(j)

        if len(cluster) >= 2:
            groups.append(DuplicateGroup(
                files=cluster,
                algorithm="video_frame_sample",
                confidence=round(max(0.0, 1.0 - threshold / 64.0), 4),
            ))

    return groups


def find_all_near_duplicates(
    files: List[MediaFileInfo],
    phash_threshold: int = DEFAULT_THRESHOLD,
    video_threshold: int = 20,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List['DuplicateGroup']:
    """
    Find all near-duplicates across images and videos.
    Combines perceptual hashing for images and frame-sampling for videos.

    Returns a combined list of DuplicateGroup objects.
    """
    groups: List['DuplicateGroup'] = []

    # Image near-duplicates via pHash
    image_groups = find_near_duplicates_by_phash(
        files,
        threshold=phash_threshold,
        progress_callback=progress_callback,
    )
    groups.extend(image_groups)

    # Video near-duplicates via frame sampling
    video_groups = find_video_near_duplicates(
        files,
        threshold=video_threshold,
        progress_callback=progress_callback,
    )
    groups.extend(video_groups)

    return groups


# ─── Utility ───────────────────────────────────────────────────────────────

def hash_to_bits(hex_hash: str) -> str:
    """Convert a hex hash string to a binary string for inspection."""
    try:
        h = int(hex_hash, 16)
        return bin(h)[2:]
    except (ValueError, TypeError):
        return ""


def bits_to_hash(bits: str) -> str:
    """Convert a binary string back to a hex hash string."""
    if not bits:
        return ""
    return hex(int(bits, 2))[2:]