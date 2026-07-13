"""
Tests for the PicSort AI scanning engine, EXIF extraction, and database index.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from engine.scanner import (
    MediaFileInfo,
    MediaType,
    DuplicateGroup,
    ScanResult,
    is_supported_media,
    compute_sha256,
    scan_directory,
    find_exact_duplicates,
    extract_metadata,
    classify_media,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    quick_scan,
    run_scan,
)

from engine.exif import extract_exif
from engine.db import get_connection, init_db, get_stats, search_files


# ─── Basic File Support Tests ──────────────────────────────────────────────

class TestIsSupportedMedia:
    def test_supported_image(self):
        assert is_supported_media("photo.jpg") is True
        assert is_supported_media("photo.jpeg") is True
        assert is_supported_media("photo.png") is True
        assert is_supported_media("photo.gif") is True
        assert is_supported_media("photo.webp") is True
        assert is_supported_media("photo.heic") is True
        assert is_supported_media("photo.avif") is True

    def test_supported_video(self):
        assert is_supported_media("video.mp4") is True
        assert is_supported_media("video.mov") is True
        assert is_supported_media("video.avi") is True
        assert is_supported_media("video.mkv") is True
        assert is_supported_media("video.webm") is True
        assert is_supported_media("video.m4v") is True

    def test_unsupported(self):
        assert is_supported_media("document.pdf") is False
        assert is_supported_media("script.py") is False
        assert is_supported_media("archive.zip") is False
        assert is_supported_media("noextension") is False

    def test_extension_constants(self):
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS
        assert all(ext in SUPPORTED_EXTENSIONS for ext in [".jpg", ".mp4", ".png"])


class TestClassifyMedia:
    def test_classify_image(self):
        assert classify_media(".jpg") == MediaType.IMAGE.value
        assert classify_media(".PNG") == MediaType.IMAGE.value
        assert classify_media(".webp") == MediaType.IMAGE.value

    def test_classify_video(self):
        assert classify_media(".mp4") == MediaType.VIDEO.value
        assert classify_media(".MOV") == MediaType.VIDEO.value

    def test_classify_unknown(self):
        assert classify_media(".pdf") == MediaType.UNKNOWN.value
        assert classify_media(".txt") == MediaType.UNKNOWN.value


# ─── Hash Tests ────────────────────────────────────────────────────────────

class TestComputeSha256:
    def test_known_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            assert compute_sha256(path) == expected
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            assert compute_sha256(path) == expected
        finally:
            os.unlink(path)


# ─── Directory Scanning Tests ──────────────────────────────────────────────

class TestScanDirectory:
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = scan_directory(tmpdir)
            assert len(files) == 0

    def test_supported_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "photo.jpg").touch()
            Path(tmpdir, "video.mp4").touch()
            Path(tmpdir, "readme.txt").touch()

            files = scan_directory(tmpdir)
            assert len(files) == 2
            extensions = {f.extension for f in files}
            assert ".jpg" in extensions
            assert ".mp4" in extensions

    def test_recursive_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir, "sub")
            subdir.mkdir()
            Path(subdir, "nested.png").touch()

            files = scan_directory(tmpdir, recursive=True)
            assert len(files) == 1
            assert files[0].extension == ".png"

    def test_files_have_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir, "test.jpg")
            test_file.write_text("test data")

            files = scan_directory(tmpdir)
            assert len(files) == 1
            info = files[0]
            assert info.filename == "test.jpg"
            assert info.size > 0
            assert info.created is not None
            assert info.modified is not None
            assert info.media_type == MediaType.IMAGE.value


class TestExtractMetadata:
    def test_text_file_basic_metadata(self):
        """Test that even non-image files get basic file metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir, "test.txt")
            p.write_text("hello")
            info = extract_metadata(str(p))
            assert info.filename == "test.txt"
            assert info.extension == ".txt"
            assert info.media_type == MediaType.UNKNOWN.value
            assert info.size == 5

    def test_image_metadata(self):
        """Test that image files get proper classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir, "photo.jpg")
            p.write_text("fake jpeg data")
            info = extract_metadata(str(p))
            assert info.media_type == MediaType.IMAGE.value
            assert info.filename == "photo.jpg"

    def test_video_metadata(self):
        """Test that video files get proper classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir, "video.mp4")
            p.write_text("fake mp4 data")
            info = extract_metadata(str(p))
            assert info.media_type == MediaType.VIDEO.value
            assert info.filename == "video.mp4"


# ─── Duplicate Detection Tests ─────────────────────────────────────────────

class TestFindExactDuplicates:
    def test_no_duplicates(self):
        files = [
            MediaFileInfo(path="/tmp/a.jpg", filename="a.jpg", extension=".jpg", size=100),
            MediaFileInfo(path="/tmp/b.jpg", filename="b.jpg", extension=".jpg", size=200),
        ]
        groups = find_exact_duplicates(files)
        assert len(groups) == 0

    def test_same_size_different_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir, "a.jpg")
            p2 = Path(tmpdir, "b.jpg")
            p1.write_text("content_a")
            p2.write_text("content_b")

            files = scan_directory(tmpdir)
            groups = find_exact_duplicates(files)
            assert len(groups) == 0

    def test_exact_duplicate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = Path(tmpdir, "original.jpg")
            duplicate = Path(tmpdir, "duplicate.jpg")
            content = "exact same content here"
            original.write_text(content)
            duplicate.write_text(content)

            files = scan_directory(tmpdir)
            groups = find_exact_duplicates(files)
            assert len(groups) == 1
            assert len(groups[0].files) == 2
            assert groups[0].algorithm == "exact"
            assert groups[0].confidence == 1.0

    def test_multiple_duplicate_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Group A: 3 copies of same content
            for name in ["a1.jpg", "a2.jpg", "a3.jpg"]:
                Path(tmpdir, name).write_text("group_a_content")
            # Group B: 2 copies of different content
            for name in ["b1.png", "b2.png"]:
                Path(tmpdir, name).write_text("group_b_content")

            files = scan_directory(tmpdir)
            groups = find_exact_duplicates(files)
            assert len(groups) == 2
            # Group A should have 3 files, Group B should have 2
            group_sizes = sorted(len(g.files) for g in groups)
            assert group_sizes == [2, 3]


# ─── Quick Scan Integration Tests ──────────────────────────────────────────

class TestQuickScan:
    def test_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = quick_scan([tmpdir])
            assert result.total_files == 0
            assert result.total_duplicates == 0
            assert result.storage_reclaimable == 0

    def test_mixed_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pic1.jpg").write_text("photo")
            Path(tmpdir, "pic2.jpg").write_text("photo")
            Path(tmpdir, "vid1.mp4").write_text("video")
            Path(tmpdir, "document.txt").write_text("text")

            result = quick_scan([tmpdir])
            assert result.total_files == 3  # 2 jpg + 1 mp4
            assert result.images == 2
            assert result.videos == 1
            assert result.total_size > 0

    def test_duplicates_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "orig.jpg").write_text("same_content")
            Path(tmpdir, "copy.jpg").write_text("same_content")
            Path(tmpdir, "unique.jpg").write_text("different")

            result = quick_scan([tmpdir])
            assert result.total_duplicates == 1  # one extra copy
            assert result.storage_reclaimable > 0
            assert len(result.duplicate_groups) == 1


# ─── Database Index Tests ──────────────────────────────────────────────────

class TestDatabase:
    def test_init_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = get_connection(db_path)
            init_db(conn)
            # Check tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = [t["name"] for t in tables]
            assert "media_files" in table_names
            assert "scans" in table_names
            assert "scan_folders" in table_names
            conn.close()
        finally:
            os.unlink(db_path)

    def test_get_stats_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = get_connection(db_path)
            init_db(conn)
            stats = get_stats(conn)
            assert stats["total_files"] == 0
            assert stats["images"] == 0
            assert stats["videos"] == 0
            assert stats["total_size"] == 0
            conn.close()
        finally:
            os.unlink(db_path)

    def test_full_scan_with_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_fd, db_path = tempfile.mkstemp(suffix=".db")
            os.close(db_fd)

            # Create some test files
            Path(tmpdir, "pic1.jpg").write_text("photo1")
            Path(tmpdir, "pic2.jpg").write_text("photo1")  # duplicate
            Path(tmpdir, "unique.jpg").write_text("unique")
            Path(tmpdir, "video.mp4").write_text("video")

            result = run_scan([tmpdir], db_path=db_path)
            assert result.total_files == 4
            assert result.images == 3
            assert result.videos == 1
            assert result.total_duplicates == 1
            assert result.scan_id is not None

            # Verify database
            conn = get_connection(db_path)
            stats = get_stats(conn)
            assert stats["total_files"] == 4
            assert stats["images"] == 3
            assert stats["videos"] == 1

            # Test search
            results = search_files(conn, "pic")
            assert len(results) >= 2

            conn.close()
            os.unlink(db_path)


# ─── EXIF Extraction Tests ────────────────────────────────────────────────

class TestExifExtraction:
    def test_non_existent_file(self):
        """Should not crash on non-existent files."""
        exif = extract_exif("/nonexistent/file.jpg")
        assert exif["has_exif"] is False

    def test_non_image_file(self):
        """Should not crash on non-image files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            path = f.name
        try:
            exif = extract_exif(path)
            assert exif["has_exif"] is False
        finally:
            os.unlink(path)

    def test_text_file_no_exif(self):
        """Text files have no EXIF but should return basic dimensions."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", mode="wb", delete=False) as f:
            f.write(b"not really a jpg")
            path = f.name
        try:
            exif = extract_exif(path)
            # Corrupted image, so no EXIF
            assert exif["has_exif"] is False
        finally:
            os.unlink(path)


# ─── ScanResult Tests ──────────────────────────────────────────────────────

class TestScanResult:
    def test_formatted_duration_seconds(self):
        result = ScanResult(scan_duration_ms=2500)
        assert "2.5s" in result.formatted_duration

    def test_formatted_duration_minutes(self):
        result = ScanResult(scan_duration_ms=125000)
        assert "2m" in result.formatted_duration
        assert "5s" in result.formatted_duration

    def test_defaults(self):
        result = ScanResult()
        assert result.total_files == 0
        assert result.errors == []
        assert result.duplicate_groups == []