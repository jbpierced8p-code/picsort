"""
Tests for the PicSort AI scanning engine.
"""

import pytest
import tempfile
import os
from pathlib import Path

from engine.scanner import (
    is_supported_media,
    compute_sha256,
    scan_directory,
    find_exact_duplicates,
    MediaFile,
)


class TestIsSupportedMedia:
    def test_supported_image(self):
        assert is_supported_media("photo.jpg") is True
        assert is_supported_media("photo.jpeg") is True
        assert is_supported_media("photo.png") is True
        assert is_supported_media("photo.gif") is True
        assert is_supported_media("photo.webp") is True

    def test_supported_video(self):
        assert is_supported_media("video.mp4") is True
        assert is_supported_media("video.mov") is True
        assert is_supported_media("video.avi") is True

    def test_unsupported(self):
        assert is_supported_media("document.pdf") is False
        assert is_supported_media("script.py") is False
        assert is_supported_media("archive.zip") is False
        assert is_supported_media("noextension") is False


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


class TestFindExactDuplicates:
    def test_no_duplicates(self):
        files = [
            MediaFile(path="/tmp/a.jpg", size=100, modified=0, created=0, extension=".jpg"),
            MediaFile(path="/tmp/b.jpg", size=200, modified=0, created=0, extension=".jpg"),
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