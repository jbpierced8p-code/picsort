"""
PicSort AI - Database Index
SQLite-based persistent index for scanned media files.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional


# Default database path
DEFAULT_DB_PATH = str(Path.home() / ".picsort" / "media_index.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent access."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scans (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at    REAL NOT NULL,
            completed_at  REAL,
            total_files   INTEGER DEFAULT 0,
            total_size    INTEGER DEFAULT 0,
            status        TEXT NOT NULL DEFAULT 'in_progress'
        );

        CREATE TABLE IF NOT EXISTS scan_folders (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id   INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
            folder    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS media_files (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id         INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
            path            TEXT NOT NULL UNIQUE,
            filename        TEXT NOT NULL,
            extension       TEXT NOT NULL,
            media_type      TEXT NOT NULL DEFAULT 'unknown',
            size            INTEGER NOT NULL DEFAULT 0,
            created         REAL,
            modified        REAL,
            sha256          TEXT,
            phash           TEXT,
            dhash           TEXT,
            whash           TEXT,
            video_hash      TEXT,
            image_width     INTEGER,
            image_height    INTEGER,
            camera_make     TEXT,
            camera_model    TEXT,
            date_taken      TEXT,
            gps_latitude    REAL,
            gps_longitude   REAL,
            duration_sec    REAL,
            video_codec     TEXT,
            has_exif        INTEGER DEFAULT 0,
            last_scanned    REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_media_files_path
            ON media_files(path);
        CREATE INDEX IF NOT EXISTS idx_media_files_extension
            ON media_files(extension);
        CREATE INDEX IF NOT EXISTS idx_media_files_media_type
            ON media_files(media_type);
        CREATE INDEX IF NOT EXISTS idx_media_files_sha256
            ON media_files(sha256);
        CREATE INDEX IF NOT EXISTS idx_media_files_phash
            ON media_files(phash);
        CREATE INDEX IF NOT EXISTS idx_media_files_scan_id
            ON media_files(scan_id);
        CREATE INDEX IF NOT EXISTS idx_media_files_date_taken
            ON media_files(date_taken);
    """)
    conn.commit()


def create_scan(conn: sqlite3.Connection, folders: List[str]) -> int:
    """Create a new scan session and return its ID."""
    cursor = conn.execute(
        "INSERT INTO scans (started_at, status) VALUES (?, ?)",
        (time.time(), "in_progress"),
    )
    scan_id = cursor.lastrowid

    for folder in folders:
        conn.execute(
            "INSERT INTO scan_folders (scan_id, folder) VALUES (?, ?)",
            (scan_id, folder),
        )

    conn.commit()
    return scan_id


def complete_scan(
    conn: sqlite3.Connection,
    scan_id: int,
    total_files: int,
    total_size: int,
) -> None:
    """Mark a scan session as completed."""
    conn.execute(
        """UPDATE scans
           SET completed_at = ?, total_files = ?, total_size = ?, status = 'completed'
           WHERE id = ?""",
        (time.time(), total_files, total_size, scan_id),
    )
    conn.commit()


def upsert_media_file(conn: sqlite3.Connection, scan_id: int, file_data: Dict) -> None:
    """Insert or update a media file record."""
    conn.execute(
        """INSERT OR REPLACE INTO media_files (
            scan_id, path, filename, extension, media_type,
            size, created, modified, sha256,
            phash, dhash, whash, video_hash,
            image_width, image_height, camera_make, camera_model,
            date_taken, gps_latitude, gps_longitude,
            duration_sec, video_codec, has_exif, last_scanned
        ) VALUES (
            :scan_id, :path, :filename, :extension, :media_type,
            :size, :created, :modified, :sha256,
            :phash, :dhash, :whash, :video_hash,
            :image_width, :image_height, :camera_make, :camera_model,
            :date_taken, :gps_latitude, :gps_longitude,
            :duration_sec, :video_codec, :has_exif, :last_scanned
        )""",
        {
            "scan_id": scan_id,
            "path": file_data["path"],
            "filename": file_data.get("filename", ""),
            "extension": file_data.get("extension", ""),
            "media_type": file_data.get("media_type", "unknown"),
            "size": file_data.get("size", 0),
            "created": file_data.get("created"),
            "modified": file_data.get("modified"),
            "sha256": file_data.get("sha256"),
            "phash": file_data.get("phash"),
            "dhash": file_data.get("dhash"),
            "whash": file_data.get("whash"),
            "video_hash": file_data.get("video_hash"),
            "image_width": file_data.get("image_width"),
            "image_height": file_data.get("image_height"),
            "camera_make": file_data.get("camera_make"),
            "camera_model": file_data.get("camera_model"),
            "date_taken": file_data.get("date_taken"),
            "gps_latitude": file_data.get("gps_latitude"),
            "gps_longitude": file_data.get("gps_longitude"),
            "duration_sec": file_data.get("duration_sec"),
            "video_codec": file_data.get("video_codec"),
            "has_exif": 1 if file_data.get("has_exif") else 0,
            "last_scanned": time.time(),
        },
    )


def get_media_count(conn: sqlite3.Connection) -> int:
    """Get total number of indexed media files."""
    row = conn.execute("SELECT COUNT(*) as count FROM media_files").fetchone()
    return row["count"] if row else 0


def get_total_size(conn: sqlite3.Connection) -> int:
    """Get total size of all indexed media files in bytes."""
    row = conn.execute("SELECT COALESCE(SUM(size), 0) as total FROM media_files").fetchone()
    return row["total"] if row else 0


def get_media_by_type(conn: sqlite3.Connection, media_type: str) -> List[Dict]:
    """Get all media files of a specific type (image/video)."""
    rows = conn.execute(
        "SELECT * FROM media_files WHERE media_type = ? ORDER BY path",
        (media_type,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_scans(conn: sqlite3.Connection, limit: int = 10) -> List[Dict]:
    """Get the most recent scan sessions."""
    rows = conn.execute(
        "SELECT * FROM scans ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def search_files(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 100,
) -> List[Dict]:
    """Search media files by filename or path."""
    rows = conn.execute(
        """SELECT * FROM media_files
           WHERE filename LIKE ? OR path LIKE ?
           ORDER BY size DESC LIMIT ?""",
        (f"%{query}%", f"%{query}%", limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats(conn: sqlite3.Connection) -> Dict:
    """Get summary statistics from the index."""
    row = conn.execute(
        """SELECT
            COUNT(*) as total_files,
            COUNT(DISTINCT scan_id) as total_scans,
            COALESCE(SUM(CASE WHEN media_type = 'image' THEN 1 ELSE 0 END), 0) as images,
            COALESCE(SUM(CASE WHEN media_type = 'video' THEN 1 ELSE 0 END), 0) as videos,
            COALESCE(SUM(size), 0) as total_size,
            COALESCE(SUM(CASE WHEN has_exif = 1 THEN 1 ELSE 0 END), 0) as with_exif,
            COALESCE(SUM(CASE WHEN sha256 IS NOT NULL THEN 1 ELSE 0 END), 0) as hashed
        FROM media_files"""
    ).fetchone()
    return dict(row) if row else {}


def close(conn: sqlite3.Connection) -> None:
    """Close the database connection."""
    conn.close()