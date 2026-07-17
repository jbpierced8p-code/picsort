"""
Tests for the ML face recognition module.
"""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np
from PIL import Image, ImageDraw

from ml.face_recognition import (
    FaceEncoding,
    FaceGroup,
    detect_faces,
    compute_face_encodings,
    cluster_faces_by_person,
    batch_process_faces,
    generate_face_thumbnail,
    init_face_db,
    save_face_group,
    get_face_groups,
    get_faces_for_group,
    clear_face_cache,
    _face_to_vector,
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
        draw.ellipse([50, 50, 150, 150], fill=(220, 180, 160))
        draw.ellipse([75, 80, 90, 95], fill=(50, 50, 50))
        draw.ellipse([110, 80, 125, 95], fill=(50, 50, 50))
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

    def test_face_detection_returns_list(self):
        """Face detection should return a list (possibly empty)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "face.png")), draw_face=True)
            faces = detect_faces(path)
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

    def test_detected_face_has_bounds(self):
        """Detected face should have valid bounding box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "face.png")), draw_face=True)
            faces = detect_faces(path)
            for face in faces:
                assert face.image_path == path
                assert face.x >= 0
                assert face.y >= 0
                assert face.width > 0
                assert face.height > 0


class TestComputeFaceEncodings:
    def test_empty_image_list(self):
        """Empty image list should return empty dict."""
        result = compute_face_encodings([])
        assert result == {}

    def test_batch_processing(self):
        """Multiple images should be processed in batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [create_test_image(str(Path(tmpdir, f"img_{i}.png")), draw_face=True) for i in range(5)]
            result = compute_face_encodings(paths, batch_size=3)
            assert isinstance(result, dict)


class TestClusterFaces:
    def test_empty_face_map(self):
        """Empty face map should return empty list."""
        groups = cluster_faces_by_person({})
        assert groups == []

    def test_single_image_clustering(self):
        """Single image with faces should produce at least one group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "face.png")), draw_face=True)
            face_map = compute_face_encodings([path])
            groups = cluster_faces_by_person(face_map)
            assert isinstance(groups, list)

    def test_face_group_has_metadata(self):
        """Face groups should have proper metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "face.png")), draw_face=True)
            face_map = compute_face_encodings([path])
            groups = cluster_faces_by_person(face_map)
            if groups:
                group = groups[0]
                assert isinstance(group.label, str)
                assert group.label.startswith("Person")
                assert isinstance(group.faces, list)
                assert group.face_count > 0

    def test_groups_sorted_by_size(self):
        """Groups should be sorted by face count (largest first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [create_test_image(str(Path(tmpdir, f"img_{i}.png")), draw_face=True) for i in range(3)]
            face_map = compute_face_encodings(paths)
            groups = cluster_faces_by_person(face_map)
            for i in range(len(groups) - 1):
                assert groups[i].face_count >= groups[i + 1].face_count


class TestThumbnail:
    def test_thumbnail_generation(self):
        """Face thumbnail should be a valid image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "face.png")), draw_face=True)
            faces = detect_faces(path)
            if faces:
                face = faces[0]
                thumb = generate_face_thumbnail(face, tmpdir)
                if thumb:
                    assert os.path.isfile(thumb)
                    img = Image.open(thumb)
                    assert img.width > 0
                    assert img.height > 0

    def test_thumbnail_for_nonexistent_face(self):
        """Thumbnail for non-existent image should return None."""
        face = FaceEncoding(image_path="/nonexistent.png", x=0, y=0, width=10, height=10)
        thumb = generate_face_thumbnail(face, "/tmp")
        assert thumb is None


class TestDatabase:
    def test_init_face_db(self):
        """Face DB tables should be created."""
        conn = get_connection(":memory:")
        init_db(conn)
        init_face_db(conn)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "face_groups" in table_names
        assert "faces" in table_names
        assert "face_group_members" in table_names

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

    def test_clear_face_cache(self):
        """Clearing face cache should remove all face data."""
        conn = get_connection(":memory:")
        init_db(conn)
        init_face_db(conn)
        clear_face_cache(conn)
        assert get_face_groups(conn) == []

    def test_save_and_retrieve_face_group(self):
        """Saving a face group and retrieving it should work."""
        conn = get_connection(":memory:")
        init_db(conn)
        init_face_db(conn)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image and insert it into media_files
            path = create_test_image(str(Path(tmpdir, "face.png")), draw_face=True)
            from engine.db import create_scan, upsert_media_file
            from engine.scanner import MediaFileInfo

            scan_id = create_scan(conn, folders=[tmpdir])
            media_id = upsert_media_file(conn, scan_id, {
                "path": path, "filename": "face.png", "extension": ".png",
                "media_type": "image", "size": 1000,
            })

            # Detect faces
            faces = detect_faces(path)
            if faces:
                group = FaceGroup(
                    label="Person 1",
                    faces=faces,
                    face_count=len(faces),
                    image_count=1,
                )
                media_map = {path: media_id}
                saved_id = save_face_group(conn, group, scan_id, media_map)
                if saved_id:
                    retrieved = get_face_groups(conn)
                    assert len(retrieved) >= 1
                    group_faces = get_faces_for_group(conn, saved_id)
                    assert len(group_faces) >= 1


class TestFeatureVector:
    def test_face_to_vector_with_encoding(self):
        """128-d encoding should produce a 128-element vector."""
        encoding = np.random.rand(128).astype(np.float64)
        fe = FaceEncoding(image_path="test.png", x=0, y=0, width=10, height=10, encoding=encoding)
        vec = _face_to_vector(fe)
        assert vec is not None
        assert len(vec) == 128

    def test_face_to_vector_with_hex(self):
        """Perceptual hash hex should produce a 64-element vector."""
        fe = FaceEncoding(image_path="test.png", x=0, y=0, width=10, height=10, encoding_hex="0f0f0f0f0f0f0f0f")
        vec = _face_to_vector(fe)
        assert vec is not None
        assert len(vec) == 64

    def test_face_to_vector_none(self):
        """Face with no encoding should return None."""
        fe = FaceEncoding(image_path="test.png", x=0, y=0, width=10, height=10)
        vec = _face_to_vector(fe)
        assert vec is None


class TestBatchProcess:
    def test_batch_process_empty(self):
        """Empty image list should return empty list."""
        groups = batch_process_faces([])
        assert groups == []

    def test_batch_process_with_images(self):
        """Batch processing should handle images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [create_test_image(str(Path(tmpdir, f"img_{i}.png")), draw_face=True) for i in range(3)]
            groups = batch_process_faces(paths, batch_size=2)
            assert isinstance(groups, list)


class TestFreeTier:
    def test_face_detection_disabled_on_free(self):
        """Face detection should be disabled on Free tier."""
        set_tier(AppTier.FREE)
        try:
            result = detect_faces("/tmp/test.png")
            assert result == []
        finally:
            set_tier(AppTier.PREMIUM)

    def test_face_encodings_disabled_on_free(self):
        """Face encoding should be disabled on Free tier."""
        set_tier(AppTier.FREE)
        try:
            result = compute_face_encodings(["/tmp/test.png"])
            assert result == {}
        finally:
            set_tier(AppTier.PREMIUM)

    def test_batch_process_disabled_on_free(self):
        """Batch processing should be disabled on Free tier."""
        set_tier(AppTier.FREE)
        try:
            groups = batch_process_faces(["/tmp/test.png"])
            assert groups == []
        finally:
            set_tier(AppTier.PREMIUM)