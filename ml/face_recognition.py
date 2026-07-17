"""
PicSort AI - Face Recognition Module
Detects faces in photos, computes 128-d face encodings (via face_recognition
or OpenCV fallback), clusters faces by person using DBSCAN, and stores
results in the database with thumbnail generation.

Built for the `ml/` directory with proper schema: faces, face_groups, face_group_members.
Gated behind Premium tier.
"""

import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from engine.tiers import has_feature

# ─── Lazy imports ───────────────────────────────────────────────────────

_FACE_RECOGNITION = None
_CV2 = None
_FACE_CASCADE = None
_SKLEARN_DBSCAN = None


def _try_import_face_recognition():
    """Try to import face_recognition; return None if unavailable."""
    global _FACE_RECOGNITION
    if _FACE_RECOGNITION is None:
        try:
            import face_recognition as fr
            # Quick test to verify it actually works
            _ = fr.face_locations
            _FACE_RECOGNITION = fr
        except (ImportError, SystemExit, Exception):
            _FACE_RECOGNITION = False  # sentinel
    return _FACE_RECOGNITION if _FACE_RECOGNITION is not False else None


def _get_cv2():
    """Lazy import of OpenCV."""
    global _CV2
    if _CV2 is None:
        import cv2
        _CV2 = cv2
    return _CV2


def _get_face_cascade():
    """Get the Haar cascade classifier (lazy load)."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cv2 = _get_cv2()
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.isfile(cascade_path):
            _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _get_dbscan():
    """Lazy import of DBSCAN."""
    global _SKLEARN_DBSCAN
    if _SKLEARN_DBSCAN is None:
        from sklearn.cluster import DBSCAN
        _SKLEARN_DBSCAN = DBSCAN
    return _SKLEARN_DBSCAN


# ─── Data Classes ───────────────────────────────────────────────────────

@dataclass
class FaceEncoding:
    """A face detected in an image with its encoding."""
    image_path: str
    x: int
    y: int
    width: int
    height: int
    encoding: Optional[np.ndarray] = None  # 128-d vector (face_recognition) or None
    encoding_hex: Optional[str] = None      # perceptual hash fallback
    confidence: float = 1.0
    thumbnail_path: Optional[str] = None


@dataclass
class FaceGroup:
    """A group of faces belonging to the same person."""
    group_id: int = 0
    label: str = "Unknown"
    faces: List[FaceEncoding] = field(default_factory=list)
    face_count: int = 0
    image_count: int = 0
    thumbnail_path: Optional[str] = None
    sample_encoding: Optional[bytes] = None  # serialized 128-d vector


# ─── Face Detection ─────────────────────────────────────────────────────

def detect_faces(image_path: str) -> List[FaceEncoding]:
    """
    Detect faces in an image.

    Uses face_recognition if available, otherwise falls back to
    OpenCV Haar cascade detection.

    Args:
        image_path: Path to the image file.

    Returns:
        List of FaceEncoding objects, empty list if no faces or
        feature not available (Premium tier).
    """
    if not has_feature("facial_recognition"):
        return []

    if not os.path.isfile(image_path):
        return []

    fr = _try_import_face_recognition()
    if fr is not None:
        return _detect_faces_fr(fr, image_path)
    else:
        return _detect_faces_opencv(image_path)


def _detect_faces_fr(fr, image_path: str) -> List[FaceEncoding]:
    """Detect faces using face_recognition library (128-d encodings)."""
    try:
        image = fr.load_image_file(image_path)
        locations = fr.face_locations(image)
        if not locations:
            return []

        encodings = fr.face_encodings(image, locations)

        results: List[FaceEncoding] = []
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            fe = FaceEncoding(
                image_path=image_path,
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                encoding=encoding,  # 128-d numpy array
            )
            results.append(fe)
        return results
    except Exception:
        return []


def _detect_faces_opencv(image_path: str) -> List[FaceEncoding]:
    """Detect faces using OpenCV Haar cascade (perceptual hash fallback)."""
    cascade = _get_face_cascade()
    if cascade is None:
        return []

    cv2 = _get_cv2()
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
        )

        results: List[FaceEncoding] = []
        for (x, y, w, h) in faces:
            encoding_hex = _compute_face_phash(image_path, x, y, w, h)
            fe = FaceEncoding(
                image_path=image_path,
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                encoding_hex=encoding_hex,
            )
            results.append(fe)
        return results
    except Exception:
        return []


def _compute_face_phash(image_path: str, x: int, y: int, w: int, h: int) -> Optional[str]:
    """Compute perceptual hash of a face region (OpenCV fallback encoding)."""
    try:
        from engine.hashing import compute_phash_hex
        img = Image.open(image_path).convert("RGB")
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(img.width, x + w + margin_x)
        bottom = min(img.height, y + h + margin_y)
        face_img = img.crop((left, top, right, bottom))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            face_img.save(tmp_path, format="PNG")
        try:
            return compute_phash_hex(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        return None


# ─── Face Encoding → Feature Vectors ────────────────────────────────────

def _face_to_vector(fe: FaceEncoding) -> Optional[np.ndarray]:
    """Convert a FaceEncoding to a feature vector for clustering."""
    if fe.encoding is not None:
        return fe.encoding  # 128-d from face_recognition
    if fe.encoding_hex is not None:
        # Convert perceptual hash hex to bits for distance computation
        from engine.hashing import hash_to_bits
        bits = hash_to_bits(fe.encoding_hex)
        # Pad to 64 bits
        bits = bits.zfill(64)
        return np.array([int(b) for b in bits], dtype=np.float64)
    return None


# ─── Batch Processing ───────────────────────────────────────────────────

def compute_face_encodings(
    image_paths: List[str],
    batch_size: int = 50,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, List[FaceEncoding]]:
    """
    Detect faces and compute encodings for a list of images in batches.

    Args:
        image_paths: List of image file paths.
        batch_size: Number of images to process per batch.
        progress_callback: Optional callback for progress reporting.

    Returns:
        Dict mapping image_path -> list of FaceEncoding objects.
    """
    if not has_feature("facial_recognition"):
        return {}

    result: Dict[str, List[FaceEncoding]] = {}
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch = image_paths[i:i + batch_size]
        for j, path in enumerate(batch):
            if progress_callback:
                progress_callback({
                    "phase": "face_detection",
                    "current": i + j + 1,
                    "total": total,
                    "percentage": int((i + j + 1) / total * 100),
                })
            faces = detect_faces(path)
            if faces:
                result[path] = faces

    return result


# ─── DBSCAN Clustering ──────────────────────────────────────────────────

def cluster_faces_by_person(
    face_map: Dict[str, List[FaceEncoding]],
    eps: float = 0.5,
    min_samples: int = 1,
) -> List[FaceGroup]:
    """
    Cluster faces into person groups using DBSCAN.

    Args:
        face_map: Dict mapping image_path -> list of FaceEncoding.
        eps: Maximum distance between samples for DBSCAN.
        min_samples: Minimum samples in a neighborhood for DBSCAN.

    Returns:
        List of FaceGroup objects, sorted by size (largest first).
    """
    if not face_map:
        return []

    # Build feature vectors
    all_faces: List[FaceEncoding] = []
    vectors: List[np.ndarray] = []

    for faces in face_map.values():
        for fe in faces:
            vec = _face_to_vector(fe)
            if vec is not None:
                all_faces.append(fe)
                vectors.append(vec)

    if not vectors:
        return []

    X = np.array(vectors)

    # If using perceptual hash (binary vectors), use Hamming distance
    # If using 128-d face_recognition encodings, use Euclidean
    if all(f.encoding is not None for f in all_faces):
        # 128-d vectors - use Euclidean distance
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import euclidean_distances
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X)
    else:
        # Binary vectors - use Hamming distance
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import hamming_distances
        # Convert Hamming distance to a similarity metric DBSCAN understands
        # DBSCAN uses `eps` as the maximum distance, so we compute pairwise
        # and use a precomputed distance matrix
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X, metric="hamming")
        clustering = DBSCAN(
            eps=eps / 64.0,  # Normalize threshold for Hamming distance on 64-bit vectors
            min_samples=min_samples,
            metric="precomputed",
        ).fit(D)

    labels = clustering.labels_

    # Group faces by cluster label
    label_to_faces: Dict[int, List[FaceEncoding]] = {}
    label_to_paths: Dict[int, Set[str]] = {}
    for face, label in zip(all_faces, labels):
        if label == -1:
            continue  # noise
        if label not in label_to_faces:
            label_to_faces[label] = []
            label_to_paths[label] = set()
        label_to_faces[label].append(face)
        label_to_paths[label].add(face.image_path)

    # Build FaceGroup objects
    groups: List[FaceGroup] = []
    for label, faces in label_to_faces.items():
        # Get sample encoding
        sample_enc = None
        for fe in faces:
            if fe.encoding is not None:
                sample_enc = fe.encoding.tobytes()
                break
            elif fe.encoding_hex is not None:
                sample_enc = fe.encoding_hex.encode()
                break

        group = FaceGroup(
            label=f"Person {label + 1}",
            faces=faces,
            face_count=len(faces),
            image_count=len(label_to_paths[label]),
            sample_encoding=sample_enc,
        )
        groups.append(group)

    # Sort by size (largest first)
    groups.sort(key=lambda g: g.face_count, reverse=True)

    return groups


# ─── Thumbnail Generation ───────────────────────────────────────────────

def generate_face_thumbnail(
    face: FaceEncoding,
    output_dir: str,
    size: Tuple[int, int] = (150, 150),
) -> Optional[str]:
    """
    Generate a thumbnail image of a detected face.

    Args:
        face: FaceEncoding with image path and bounding box.
        output_dir: Directory to save the thumbnail.
        size: Desired thumbnail size (width, height).

    Returns:
        Path to the generated thumbnail, or None on failure.
    """
    if not os.path.isfile(face.image_path):
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(face.image_path).convert("RGB")
        # Crop face region with margin
        margin_x = int(face.width * 0.2)
        margin_y = int(face.height * 0.2)
        left = max(0, face.x - margin_x)
        top = max(0, face.y - margin_y)
        right = min(img.width, face.x + face.width + margin_x)
        bottom = min(img.height, face.y + face.height + margin_y)

        face_img = img.crop((left, top, right, bottom))
        face_img.thumbnail(size, Image.LANCZOS)

        thumb_filename = f"face_{os.path.basename(face.image_path)}_{face.x}_{face.y}.jpg"
        thumb_path = os.path.join(output_dir, thumb_filename)
        face_img.save(thumb_path, "JPEG", quality=85)
        return thumb_path
    except Exception:
        return None


# ─── Directory Scanning ─────────────────────────────────────────────────

def find_face_groups_in_directory(
    directory: str,
    recursive: bool = True,
    batch_size: int = 50,
    progress_callback: Optional[Callable] = None,
) -> List[FaceGroup]:
    """
    Scan a directory for images, detect faces, and cluster by person.

    Args:
        directory: Path to directory to scan.
        recursive: Whether to scan subdirectories.
        batch_size: Images per batch.
        progress_callback: Optional progress callback.

    Returns:
        List of FaceGroup objects.
    """
    if not has_feature("facial_recognition"):
        return []

    # Collect image files
    from engine.scanner import is_supported_media

    image_paths: List[str] = []
    root = Path(directory)
    if not root.is_dir():
        return []

    pattern = root.rglob if recursive else root.iterdir
    for f in pattern("*"):
        if f.is_file() and is_supported_media(str(f)):
            image_paths.append(str(f))

    if not image_paths:
        return []

    # Detect faces
    face_map = compute_face_encodings(image_paths, batch_size, progress_callback)

    # Cluster
    groups = cluster_faces_by_person(face_map)

    # Generate thumbnails
    thumb_dir = os.path.join(str(Path.home()), ".picsort", "thumbnails", "faces")
    for group in groups:
        if group.faces:
            best_face = group.faces[0]  # First face is best
            thumb = generate_face_thumbnail(best_face, thumb_dir)
            if thumb:
                group.thumbnail_path = thumb
                best_face.thumbnail_path = thumb

    return groups


def batch_process_faces(
    image_paths: List[str],
    batch_size: int = 50,
    progress_callback: Optional[Callable] = None,
) -> List[FaceGroup]:
    """
    Process images in batches, detect faces, cluster, and generate thumbnails.

    Args:
        image_paths: List of image file paths.
        batch_size: Images per batch.
        progress_callback: Optional progress callback.

    Returns:
        List of FaceGroup objects.
    """
    if not has_feature("facial_recognition"):
        return []

    face_map = compute_face_encodings(image_paths, batch_size, progress_callback)
    groups = cluster_faces_by_person(face_map)

    thumb_dir = os.path.join(str(Path.home()), ".picsort", "thumbnails", "faces")
    for group in groups:
        if group.faces:
            best_face = group.faces[0]
            thumb = generate_face_thumbnail(best_face, thumb_dir)
            if thumb:
                group.thumbnail_path = thumb
                best_face.thumbnail_path = thumb

    return groups


# ─── Database Schema & Helpers ──────────────────────────────────────────

def init_face_db(conn) -> None:
    """Create face-related tables in the database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS face_groups (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            label           TEXT NOT NULL DEFAULT 'Unknown',
            face_count      INTEGER DEFAULT 0,
            image_count     INTEGER DEFAULT 0,
            thumbnail_path  TEXT,
            sample_encoding BLOB,
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS faces (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            media_file_id   INTEGER NOT NULL REFERENCES media_files(id) ON DELETE CASCADE,
            face_group_id   INTEGER REFERENCES face_groups(id) ON DELETE SET NULL,
            x               INTEGER NOT NULL,
            y               INTEGER NOT NULL,
            width           INTEGER NOT NULL,
            height          INTEGER NOT NULL,
            encoding        BLOB,
            encoding_hex    TEXT,
            confidence      REAL DEFAULT 1.0,
            thumbnail_path  TEXT
        );

        CREATE TABLE IF NOT EXISTS face_group_members (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id        INTEGER NOT NULL REFERENCES face_groups(id) ON DELETE CASCADE,
            face_id         INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
            UNIQUE(group_id, face_id)
        );

        CREATE INDEX IF NOT EXISTS idx_faces_media ON faces(media_file_id);
        CREATE INDEX IF NOT EXISTS idx_faces_group ON faces(face_group_id);
        CREATE INDEX IF NOT EXISTS idx_face_group_members_group ON face_group_members(group_id);
        CREATE INDEX IF NOT EXISTS idx_face_group_members_face ON face_group_members(face_id);
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
    now = time.time()

    # Serialize sample encoding
    sample_bytes = group.sample_encoding

    # Insert face group
    cur = conn.execute(
        """INSERT INTO face_groups
           (label, face_count, image_count, thumbnail_path, sample_encoding, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            group.label,
            group.face_count,
            group.image_count,
            group.thumbnail_path,
            sample_bytes,
            now,
            now,
        ),
    )
    group_id = cur.lastrowid
    group.group_id = group_id

    # Insert faces and group members
    for face in group.faces:
        media_id = media_map.get(face.image_path)
        if media_id is None:
            continue

        # Serialize encoding
        enc_bytes = face.encoding.tobytes() if face.encoding is not None else None

        cur = conn.execute(
            """INSERT INTO faces
               (media_file_id, x, y, width, height, encoding, encoding_hex, confidence, thumbnail_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                media_id,
                face.x,
                face.y,
                face.width,
                face.height,
                enc_bytes,
                face.encoding_hex,
                face.confidence,
                face.thumbnail_path,
            ),
        )
        face_id = cur.lastrowid

        conn.execute(
            """INSERT INTO face_group_members (group_id, face_id) VALUES (?, ?)""",
            (group_id, face_id),
        )

    return group_id


def get_face_groups(conn) -> List[Dict]:
    """Get all face groups from the database."""
    rows = conn.execute(
        "SELECT * FROM face_groups ORDER BY face_count DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_faces_for_group(conn, group_id: int) -> List[Dict]:
    """Get all faces for a face group with image paths."""
    rows = conn.execute(
        """SELECT f.*, mf.path as image_path
           FROM faces f
           JOIN media_files mf ON f.media_file_id = mf.id
           JOIN face_group_members fgm ON f.id = fgm.face_id
           WHERE fgm.group_id = ?
           ORDER BY f.id""",
        (group_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def clear_face_cache(conn) -> None:
    """Clear all face data from the database."""
    conn.execute("DELETE FROM face_group_members")
    conn.execute("DELETE FROM faces")
    conn.execute("DELETE FROM face_groups")