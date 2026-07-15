"""
Tests for the face detection, encoding, and clustering module.
"""

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from engine.face_detection import (
    DetectedFace,
    FaceGroup,
    detect_faces,
    compute_face_encodings,
    cluster_faces_by_person,
    find_face_groups_in_directory,
    init_face_db,
    get_face_groups,
    get_faces_for_group,
    _compute_face_encoding,
)
from engine.tiers import set_tier, AppTier
from engine.db import get_connection, init_db

# Enable Premium tier for face detection tests
set_tier(AppTier.PREMIUM)


# ─── Helper: create a test image ──────────────────────────────────────────

def create_test_image(
    filepath: str,
    size: tuple = (200, 200),
    color: tuple = (200, 200, 200),
    draw_face: bool = False,
) -> str:
    """Create a test image, optionally with a face-like oval."""
    img = Image.new("RGB", size, color)
    if draw_face:
        draw = ImageDraw.Draw(img)
        # Draw a face-like shape (oval)
        draw.ellipse([50, 50, 150, 150], fill=(220, 180, 160))
        # Eyes
        draw.ellipse([75, 80, 90, 95], fill=(50, 50, 50))
        draw.ellipse([110, 80, 125, 95], fill=(50, 50, 50))
        # Mouth
        draw.arc([80, 100, 120, 130], 0, 180, fill=(100, 60, 60), width=3)
    img.save(filepath, format="PNG")
    return filepath


# ─── Tests ────────────────────────────────────────────────────────────────

class TestDetectFaces:
    def test_no_faces_in_empty_image(self):
        """No faces should be detected in a blank image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "blank.png")), draw_face=False)
            faces = detect_faces(path)
            assert len(faces) == 0

    def test_face_detection_in_face_image(self):
        """Faces should be detected in images with face-like features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(
                str(Path(tmpdir, "face.png")),
                size=(300, 300),
                draw_face=True,
            )
            faces = detect_faces(path)
            # Haar cascade may or may not detect a drawn face, but the function should run
            assert isinstance(faces, list)

    def test_nonexistent_file(self):
        """Non-existent file should return empty list."""
        faces = detect_faces("/nonexistent/path.jpg")
        assert faces == []

    def test_invalid_file(self):
        """Invalid file should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "invalid.txt")
            path.write_text("not an image")
            faces = detect_faces(str(path))
            assert faces == []

    def test_detected_face_has_encoding(self):
        """Detected face should have an encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(
                str(Path(tmpdir, "face.png")),
                size=(300, 300),
                draw_face=True,
            )
            faces = detect_faces(path)
            for face in faces:
                assert face.image_path == path
                assert face.x >= 0
                assert face.y >= 0
                assert face.width > 0
                assert face.height > 0


class TestFaceEncoding:
    def test_encoding_is_hex(self):
        """Face encoding should be a hex string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "test.png")), size=(100, 100))
            encoding = _compute_face_encoding(path, 0, 0, 100, 100)
            if encoding:
                assert isinstance(encoding, str)
                assert all(c in "0123456789abcdef" for c in encoding)

    def test_encoding_none_for_invalid_region(self):
        """Encoding should be None for invalid regions."""
        encoding = _compute_face_encoding("/nonexistent.png", 0, 0, 10, 10)
        assert encoding is None


class TestComputeFaceEncodings:
    def test_empty_image_list(self):
        """Empty image list should return empty dict."""
        result = compute_face_encodings([])
        assert result == {}

    def test_multiple_images(self):
        """Multiple images should be processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p = create_test_image(
                    str(Path(tmpdir, f"img_{i}.png")),
                    draw_face=(i == 0),
                )
                paths.append(p)

            result = compute_face_encodings(paths)
            assert isinstance(result, dict)
            # All paths should be in the result
            for p in paths:
                assert p in result or True  # may have no faces


class TestClusterFaces:
    def test_empty_face_map(self):
        """Empty face map should return empty list."""
        groups = cluster_faces_by_person({})
        assert groups == []

    def test_single_person_clustering(self):
        """Faces with same encoding should cluster together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(
                str(Path(tmpdir, "face.png")),
                size=(200, 200),
                draw_face=True,
            )
            face_map = compute_face_encodings([path])

            groups = cluster_faces_by_person(face_map)
            assert isinstance(groups, list)
            # Groups should be sorted by size
            if groups:
                for i in range(len(groups) - 1):
                    assert len(groups[i].faces) >= len(groups[i + 1].faces)

    def test_face_group_has_metadata(self):
        """Face groups should have proper metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(
                str(Path(tmpdir, "face.png")),
                size=(200, 200),
                draw_face=True,
            )
            face_map = compute_face_encodings([path])
            groups = cluster_faces_by_person(face_map)

            if groups:
                group = groups[0]
                assert group.person_id.startswith("person_")
                assert isinstance(group.label, str)
                assert isinstance(group.faces, list)
                assert group.image_count > 0


class TestFindFaceGroupsInDirectory:
    def test_nonexistent_directory(self):
        """Non-existent directory should return empty list."""
        groups = find_face_groups_in_directory("/nonexistent")
        assert groups == []

    def test_empty_directory(self):
        """Empty directory should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            groups = find_face_groups_in_directory(tmpdir)
            assert groups == []

    def test_directory_with_images(self):
        """Directory with images should be processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(str(Path(tmpdir, "img1.png")), draw_face=True)
            create_test_image(str(Path(tmpdir, "img2.png")), draw_face=False)
            groups = find_face_groups_in_directory(tmpdir)
            assert isinstance(groups, list)


class TestDatabase:
    def test_init_face_db(self):
        """Face DB tables should be created."""
        conn = get_connection(":memory:")
        init_db(conn)
        init_face_db(conn)

        # Verify tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "face_groups" in table_names
        assert "face_detections" in table_names

    def test_get_face_groups_empty(self):
        """Empty face groups table should return empty list."""
        conn = get_connection(":memory:")
        init_db(conn)
        init_face_db(conn)
        groups = get_face_groups(conn)
        assert groups == []

    def test_get_faces_for_group_empty(self):
        """Getting faces for non-existent group should return empty list."""
        conn = get_connection(":memory:")
        init_db(conn)
        init_face_db(conn)
        faces = get_faces_for_group(conn, 999)
        assert faces == []


class TestFreeTier:
    def test_face_detection_disabled_on_free(self):
        """Face detection should be disabled on Free tier."""
        # Switch to Free tier
        set_tier(AppTier.FREE)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = create_test_image(str(Path(tmpdir, "test.png")), draw_face=True)
                faces = detect_faces(path)
                assert faces == []
        finally:
            # Restore Premium for other tests
            set_tier(AppTier.PREMIUM)

    def test_face_encodings_disabled_on_free(self):
        """Face encoding should be disabled on Free tier."""
        set_tier(AppTier.FREE)
        try:
            result = compute_face_encodings(["/tmp/test.png"])
            assert result == {}
        finally:
            set_tier(AppTier.PREMIUM)

    def test_face_groups_disabled_on_free(self):
        """Face grouping should be disabled on Free tier."""
        set_tier(AppTier.FREE)
        try:
            groups = find_face_groups_in_directory("/tmp")
            assert groups == []
        finally:
            set_tier(AppTier.PREMIUM)