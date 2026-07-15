"""
PicSort AI - Face Detection & Recognition Module
Detects faces in photos using OpenCV Haar cascades, computes face encodings
(perceptual hashes of face regions), and clusters faces by person.

Gated behind Premium tier — Free tier has no access to face grouping.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from engine.hashing import compute_phash_hex, hamming_distance
from engine.tiers import has_feature

# Lazy-load OpenCV to avoid import overhead when not needed
_cv2 = None
_face_cascade = None


def _get_cv2():
    """Lazy import of OpenCV."""
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_face_cascade():
    """Get the Haar cascade classifier for face detection (lazy load)."""
    global _face_cascade
    if _face_cascade is None:
        cv2 = _get_cv2()
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.isfile(cascade_path):
            _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


@dataclass
class DetectedFace:
    """A single face detected in an image."""
    image_path: str
    x: int
    y: int
    width: int
    height: int
    encoding: Optional[str] = None  # perceptual hash of the face region
    confidence: float = 1.0


@dataclass
class FaceGroup:
    """A group of faces belonging to the same person."""
    person_id: str
    label: str = "Unknown"
    faces: List[DetectedFace] = field(default_factory=list)
    image_count: int = 0
    sample_encoding: Optional[str] = None


def detect_faces(image_path: str) -> List[DetectedFace]:
    """
    Detect all faces in an image using OpenCV Haar cascade.

    Args:
        image_path: Path to the image file.

    Returns:
        List of DetectedFace objects, empty list if no faces found
        or if the feature is not available (Premium tier).
    """
    if not has_feature("facial_recognition"):
        return []

    if not os.path.isfile(image_path):
        return []

    cascade = _get_face_cascade()
    if cascade is None:
        return []

    cv2 = _get_cv2()

    try:
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces with tuned parameters
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        results: List[DetectedFace] = []
        for (x, y, w, h) in faces:
            detected = DetectedFace(
                image_path=image_path,
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
            )
            # Compute perceptual hash of the face region for encoding
            encoding = _compute_face_encoding(image_path, x, y, w, h)
            if encoding:
                detected.encoding = encoding
            results.append(detected)

        return results
    except Exception:
        return []


def _compute_face_encoding(
    image_path: str, x: int, y: int, w: int, h: int
) -> Optional[str]:
    """
    Compute a perceptual hash (encoding) of a detected face region.
    Uses the pHash of the cropped face area.

    Returns hex string of the perceptual hash, or None on failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        # Crop to face region with a small margin
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(img.width, x + w + margin_x)
        bottom = min(img.height, y + h + margin_y)

        face_img = img.crop((left, top, right, bottom))
        # Save to temp file for hashing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            face_img.save(tmp_path, format="PNG")

        try:
            encoding = compute_phash_hex(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return encoding
    except Exception:
        return None


def compute_face_encodings(
    image_paths: List[str],
    progress_callback: Optional[callable] = None,
) -> Dict[str, List[DetectedFace]]:
    """
    Detect faces and compute encodings for a list of images.

    Args:
        image_paths: List of image file paths.
        progress_callback: Optional callback for progress reporting.

    Returns:
        Dict mapping image_path -> list of DetectedFace objects.
    """
    if not has_feature("facial_recognition"):
        return {}

    result: Dict[str, List[DetectedFace]] = {}
    total = len(image_paths)

    for i, path in enumerate(image_paths):
        if progress_callback:
            progress_callback({
                "phase": "face_detection",
                "current": i + 1,
                "total": total,
                "percentage": int((i + 1) / total * 100),
            })

        faces = detect_faces(path)
        if faces:
            result[path] = faces

    return result


def cluster_faces_by_person(
    face_map: Dict[str, List[DetectedFace]],
    encoding_threshold: int = 8,
) -> List[FaceGroup]:
    """
    Cluster detected faces into person groups by comparing encoding hashes.

    Uses Hamming distance between perceptual hashes of face regions.
    Faces with distance <= encoding_threshold are considered the same person.

    Args:
        face_map: Dict mapping image_path -> list of DetectedFace.
        encoding_threshold: Max Hamming distance for same-person match.

    Returns:
        List of FaceGroup objects, sorted by size (largest first).
    """
    if not face_map:
        return []

    # Collect all faces with valid encodings
    all_faces: List[DetectedFace] = []
    for faces in face_map.values():
        all_faces.extend(f for f in faces if f.encoding is not None)

    if not all_faces:
        return []

    # Greedy clustering by encoding similarity
    assigned: Set[int] = set()
    groups: List[FaceGroup] = []

    for i, face in enumerate(all_faces):
        if i in assigned:
            continue

        group = FaceGroup(
            person_id=f"person_{len(groups):04d}",
            faces=[face],
            image_count=1,
            sample_encoding=face.encoding,
        )
        assigned.add(i)

        for j, other in enumerate(all_faces):
            if j in assigned or i == j:
                continue
            if face.encoding and other.encoding:
                dist = hamming_distance(face.encoding, other.encoding)
                if dist <= encoding_threshold:
                    group.faces.append(other)
                    group.image_count += 1
                    assigned.add(j)

        groups.append(group)

    # Sort by group size (largest first)
    groups.sort(key=lambda g: len(g.faces), reverse=True)

    return groups


def find_face_groups_in_directory(
    directory: str,
    recursive: bool = True,
    progress_callback: Optional[callable] = None,
) -> List[FaceGroup]:
    """
    Scan a directory for images, detect faces, and cluster by person.

    This is a convenience function that combines scanning, face detection,
    and clustering into one call.

    Args:
        directory: Path to directory to scan.
        recursive: Whether to scan subdirectories.
        progress_callback: Optional progress callback.

    Returns:
        List of FaceGroup objects.
    """
    if not has_feature("facial_recognition"):
        return []

    # Collect image files
    from engine.scanner import IMAGE_EXTENSIONS, is_supported_media

    image_paths: List[str] = []
    root = Path(directory)
    if not root.is_dir():
        return []

    if recursive:
        for f in root.rglob("*"):
            if f.is_file() and is_supported_media(str(f)):
                image_paths.append(str(f))
    else:
        for f in root.iterdir():
            if f.is_file() and is_supported_media(str(f)):
                image_paths.append(str(f))

    if not image_paths:
        return []

    # Detect faces
    face_map = compute_face_encodings(image_paths, progress_callback)

    # Cluster
    groups = cluster_faces_by_person(face_map)

    return groups


# ─── Database helpers ────────────────────────────────────────────────────

def init_face_db(conn) -> None:
    """Create face-related tables in the database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS face_groups (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id       TEXT NOT NULL UNIQUE,
            label           TEXT NOT NULL DEFAULT 'Unknown',
            sample_encoding TEXT,
            face_count      INTEGER DEFAULT 0,
            image_count     INTEGER DEFAULT 0,
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS face_detections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id   INTEGER NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
            face_group_id   INTEGER REFERENCES face_groups(id) ON DELETE SET NULL,
            x               INTEGER NOT NULL,
            y               INTEGER NOT NULL,
            width           INTEGER NOT NULL,
            height          INTEGER NOT NULL,
            encoding        TEXT,
            confidence      REAL DEFAULT 1.0
        );

        CREATE INDEX IF NOT EXISTS idx_face_detections_media
            ON face_detections(media_file_id);
        CREATE INDEX IF NOT EXISTS idx_face_detections_group
            ON face_detections(face_group_id);
    """)


def save_face_group(
    conn, group: FaceGroup, scan_id: int, media_map: Dict[str, int]
) -> Optional[int]:
    """
    Save a face group and its face detections to the database.

    Args:
        conn: Database connection.
        group: FaceGroup to save.
        scan_id: Scan session ID.
        media_map: Dict mapping file path -> media_file_id.

    Returns:
        face_group_id if saved, None otherwise.
    """
    import time
    now = time.time()

    # Insert or update face group
    cur = conn.execute(
        """INSERT INTO face_groups (person_id, label, sample_encoding, face_count, image_count, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(person_id) DO UPDATE SET
               label = excluded.label,
               sample_encoding = excluded.sample_encoding,
               face_count = excluded.face_count,
               image_count = excluded.image_count,
               updated_at = excluded.updated_at""",
        (
            group.person_id,
            group.label,
            group.sample_encoding,
            len(group.faces),
            group.image_count,
            now,
            now,
        ),
    )
    group_id = cur.lastrowid

    # Insert face detections
    for face in group.faces:
        media_id = media_map.get(face.image_path)
        if media_id is None:
            continue
        conn.execute(
            """INSERT INTO face_detections
               (media_file_id, face_group_id, x, y, width, height, encoding, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                media_id,
                group_id,
                face.x,
                face.y,
                face.width,
                face.height,
                face.encoding,
                face.confidence,
            ),
        )

    return group_id


def get_face_groups(conn) -> List[Dict]:
    """Get all face groups from the database."""
    rows = conn.execute(
        """SELECT * FROM face_groups ORDER BY face_count DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_faces_for_group(conn, group_id: int) -> List[Dict]:
    """Get all face detections for a face group."""
    rows = conn.execute(
        """SELECT fd.*, mf.path as image_path
           FROM face_detections fd
           JOIN media_files mf ON fd.media_file_id = mf.id
           WHERE fd.face_group_id = ?
           ORDER BY fd.id""",
        (group_id,),
    ).fetchall()
    return [dict(r) for r in rows]