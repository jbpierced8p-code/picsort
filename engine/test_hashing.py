"""
Tests for the perceptual hashing, near-duplicate detection, and video frame sampling.
"""

import os
import struct
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from engine.hashing import (
    compute_phash,
    compute_phash_hex,
    compute_dhash,
    compute_whash,
    compute_colorhash,
    compute_all_hashes,
    hamming_distance,
    hash_distance_ratio,
    find_near_duplicates_by_phash,
    find_video_near_duplicates,
    find_all_near_duplicates,
    sample_video_frames,
    sample_video_frames_ffmpeg,
    compute_video_phash_ffmpeg,
    compute_video_phash,
    hash_to_bits,
    HAS_IMAGEHASH,
)
from engine.scanner import (
    MediaFileInfo,
    MediaType,
    DuplicateGroup,
    find_near_duplicates,
    find_all_duplicates,
)
from engine.tiers import set_tier, AppTier

# Enable Premium tier for all perceptual hashing tests
set_tier(AppTier.PREMIUM)


# ─── Helper: create a test image ──────────────────────────────────────────

def create_test_image(
    filepath: str,
    size: tuple = (100, 100),
    color: tuple = (128, 128, 128),
    format: str = "PNG",
) -> str:
    """Create a simple solid-color test image."""
    img = Image.new("RGB", size, color)
    img.save(filepath, format=format)
    return filepath


def create_gradient_image(
    filepath: str,
    size: tuple = (100, 100),
    format: str = "PNG",
    offset: int = 0,
) -> str:
    """Create a gradient test image with an offset for variation."""
    img = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            r = (x * 2 + offset) % 256
            g = (y * 2 + offset * 2) % 256
            b = (x + y + offset) % 256
            img.putpixel((x, y), (r, g, b))
    img.save(filepath, format=format)
    return filepath


# ─── Perceptual Hash Computation Tests ────────────────────────────────────

@pytest.mark.skipif(not HAS_IMAGEHASH, reason="imagehash not installed")
class TestComputePhash:
    def test_valid_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "test.png")))
            phash = compute_phash(path)
            assert phash is not None
            assert len(str(phash)) == 16  # 64-bit hash = 16 hex chars

    def test_hex_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "test.png")))
            hex_hash = compute_phash_hex(path)
            assert hex_hash is not None
            assert isinstance(hex_hash, str)
            assert len(hex_hash) == 16

    def test_nonexistent_file(self):
        phash = compute_phash("/nonexistent/image.jpg")
        assert phash is None

    def test_same_image_same_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = create_test_image(str(Path(tmpdir, "img1.png")), color=(100, 150, 200))
            path2 = create_test_image(str(Path(tmpdir, "img2.png")), color=(100, 150, 200))
            hash1 = compute_phash_hex(path1)
            hash2 = compute_phash_hex(path2)
            assert hash1 == hash2

    def test_different_images_different_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = create_gradient_image(str(Path(tmpdir, "grad1.png")), offset=0)
            path2 = create_gradient_image(str(Path(tmpdir, "grad2.png")), offset=100)
            hash1 = compute_phash_hex(path1)
            hash2 = compute_phash_hex(path2)
            assert hash1 != hash2

    def test_hash_size_variation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "test.png")))
            phash8 = compute_phash(path, hash_size=8)
            phash16 = compute_phash(path, hash_size=16)
            assert phash8 is not None
            assert phash16 is not None
            assert len(str(phash8)) == 16  # 8x8 = 64 bits = 16 hex
            assert len(str(phash16)) == 64  # 16x16 = 256 bits = 64 hex

    def test_tiny_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "tiny.png")), size=(16, 16))
            phash = compute_phash(path)
            # Below minimum dimension, should return None
            assert phash is None


@pytest.mark.skipif(not HAS_IMAGEHASH, reason="imagehash not installed")
class TestComputeAllHashTypes:
    def test_all_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "test.png")))
            hashes = compute_all_hashes(path)
            assert "phash" in hashes
            assert "dhash" in hashes
            assert "whash" in hashes
            assert "colorhash" in hashes
            assert hashes["phash"] is not None
            assert hashes["dhash"] is not None
            assert hashes["whash"] is not None
            assert hashes["colorhash"] is not None

    def test_dhash_different_from_phash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_gradient_image(str(Path(tmpdir, "test.png")), offset=42)
            hashes = compute_all_hashes(path)
            # Different hash types should give different values
            assert hashes["phash"] != hashes["dhash"]

    def test_whash_computation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_gradient_image(str(Path(tmpdir, "gradient.png")))
            whash = compute_whash(path)
            assert whash is not None
            assert len(whash) == 16


# ─── Hamming Distance Tests ───────────────────────────────────────────────

class TestHammingDistance:
    def test_same_hash(self):
        assert hamming_distance("abc123", "abc123") == 0

    def test_completely_different(self):
        # These are different hex strings
        dist = hamming_distance("0000000000000000", "ffffffffffffffff")
        assert dist > 0
        assert dist == 64  # All 64 bits differ

    def test_one_bit_difference(self):
        # 64-bit: 0...0 vs 0...1
        h1 = "0000000000000000"
        h2 = "0000000000000001"
        dist = hamming_distance(h1, h2)
        assert dist == 1

    def test_empty_strings(self):
        assert hamming_distance("", "abc") == -1
        assert hamming_distance("abc", "") == -1
        assert hamming_distance("", "") == -1

    def test_none_input(self):
        assert hamming_distance(None, "abc") == -1  # type: ignore
        assert hamming_distance("abc", None) == -1  # type: ignore

    def test_varied_distance(self):
        """Test that larger visual differences give larger distances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two very different images
            path1 = create_gradient_image(str(Path(tmpdir, "grad1.png")), offset=0)
            path2 = create_gradient_image(str(Path(tmpdir, "grad2.png")), offset=200)
            h1 = compute_phash_hex(path1)
            h2 = compute_phash_hex(path2)
            assert h1 is not None and h2 is not None
            dist = hamming_distance(h1, h2)
            # Very different gradients should have non-zero distance
            assert dist > 0

    def test_hash_distance_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "test.png")))
            h = compute_phash_hex(path)
            assert h is not None
            ratio_fn = hash_distance_ratio(h)
            # Same hash should give ratio 0
            assert ratio_fn(h) == 0.0


# ─── Near-Duplicate Detection Tests ───────────────────────────────────────

@pytest.mark.skipif(not HAS_IMAGEHASH, reason="imagehash not installed")
class TestFindNearDuplicatesByPhash:
    def test_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = create_gradient_image(str(Path(tmpdir, "grad1.png")), offset=0)
            p2 = create_gradient_image(str(Path(tmpdir, "grad2.png")), offset=50)
            p3 = create_gradient_image(str(Path(tmpdir, "grad3.png")), offset=150)

            files = [
                MediaFileInfo(path=p1, filename="grad1.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p2, filename="grad2.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p3, filename="grad3.png", extension=".png", media_type="image"),
            ]

            groups = find_near_duplicates_by_phash(files, threshold=0)
            # With strict threshold, different gradients should not match
            assert len(groups) == 0

    def test_exact_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (100, 150, 200)
            p1 = create_test_image(str(Path(tmpdir, "a.png")), color=content)
            p2 = create_test_image(str(Path(tmpdir, "b.png")), color=content)
            p3 = create_test_image(str(Path(tmpdir, "c.png")), color=content)

            files = [
                MediaFileInfo(path=p1, filename="a.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p2, filename="b.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p3, filename="c.png", extension=".png", media_type="image"),
            ]

            groups = find_near_duplicates_by_phash(files, threshold=0)
            assert len(groups) == 1
            assert len(groups[0].files) == 3
            assert groups[0].algorithm == "perceptual_phash"

    def test_with_non_image_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = create_test_image(str(Path(tmpdir, "img.png")))
            # Non-image files should be filtered out
            files = [
                MediaFileInfo(path=p1, filename="img.png", extension=".png", media_type="image"),
                MediaFileInfo(path="/tmp/vid.mp4", filename="vid.mp4", extension=".mp4", media_type="video"),
                MediaFileInfo(path="/tmp/doc.txt", filename="doc.txt", extension=".txt", media_type="unknown"),
            ]
            groups = find_near_duplicates_by_phash(files)
            assert len(groups) == 0  # only one image file

    def test_mixed_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Two identical gradient images and one different
            p1 = create_gradient_image(str(Path(tmpdir, "a.png")), offset=0)
            p2 = create_gradient_image(str(Path(tmpdir, "b.png")), offset=0)
            p3 = create_gradient_image(str(Path(tmpdir, "c.png")), offset=100)

            files = [
                MediaFileInfo(path=p1, filename="a.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p2, filename="b.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p3, filename="c.png", extension=".png", media_type="image"),
            ]

            groups = find_near_duplicates_by_phash(files, threshold=0)
            assert len(groups) == 1
            assert len(groups[0].files) == 2  # a.png and b.png are identical

    def test_confidence_scoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = create_test_image(str(Path(tmpdir, "orig.png")), color=(100, 150, 200))
            p2 = create_test_image(str(Path(tmpdir, "copy.png")), color=(100, 150, 200))

            files = [
                MediaFileInfo(path=p1, filename="orig.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p2, filename="copy.png", extension=".png", media_type="image"),
            ]
            groups = find_near_duplicates_by_phash(files, threshold=0)
            assert len(groups) == 1
            # Exact perceptual match should have high confidence
            assert groups[0].confidence >= 0.9


# ─── Video Frame Sampling Tests ───────────────────────────────────────────

class TestSampleVideoFrames:
    def test_nonexistent_video(self):
        frames = sample_video_frames("/nonexistent/video.mp4")
        assert frames == []

    def test_empty_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            frames = sample_video_frames(path)
            assert frames == []  # empty file = no frames
        finally:
            os.unlink(path)

    def test_small_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x1c" * 1000)  # 4KB of data
            path = f.name
        try:
            frames = sample_video_frames(path, num_frames=3)
            assert len(frames) >= 1
        finally:
            os.unlink(path)

    def test_large_video_multiple_frames(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            # 100KB of data
            f.write(b"\xff" * 100 * 1024)
            path = f.name
        try:
            frames = sample_video_frames(path, num_frames=5)
            assert len(frames) >= 3  # should get multiple samples
            # Each frame should be a bytes chunk
            assert all(isinstance(f, bytes) for f in frames)
        finally:
            os.unlink(path)


class TestComputeVideoPhash:
    def test_nonexistent_file(self):
        vhash = compute_video_phash("/nonexistent/video.mp4")
        assert vhash is None

    def test_small_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"test video data here")
            path = f.name
        try:
            vhash = compute_video_phash(path)
            assert vhash is not None
            assert isinstance(vhash, str)
        finally:
            os.unlink(path)

    def test_large_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\xab" * 200 * 1024)  # 200KB
            path = f.name
        try:
            vhash = compute_video_phash(path)
            assert vhash is not None
            assert isinstance(vhash, str)
            # Should be a combined hash of frame samples
            assert len(vhash) >= 4
        finally:
            os.unlink(path)


class TestFindVideoNearDuplicates:
    def test_no_videos(self):
        files = [
            MediaFileInfo(path="/tmp/img.jpg", filename="img.jpg", extension=".jpg", media_type="image"),
        ]
        groups = find_video_near_duplicates(files)
        assert groups == []

    def test_single_video(self):
        files = [
            MediaFileInfo(path="/tmp/vid.mp4", filename="vid.mp4", extension=".mp4", media_type="video"),
        ]
        groups = find_video_near_duplicates(files)
        assert groups == []

    def test_identical_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = b"\xcd" * 50 * 1024  # 50KB
            p1 = Path(tmpdir, "vid1.mp4")
            p2 = Path(tmpdir, "vid2.mp4")
            p1.write_bytes(content)
            p2.write_bytes(content)

            files = [
                MediaFileInfo(path=str(p1), filename="vid1.mp4", extension=".mp4", media_type="video"),
                MediaFileInfo(path=str(p2), filename="vid2.mp4", extension=".mp4", media_type="video"),
            ]
            # Use generous threshold since hash is content-based
            groups = find_video_near_duplicates(files, threshold=50)
            assert len(groups) >= 1


# ─── Integration Tests ────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_IMAGEHASH, reason="imagehash not installed")
class TestFindAllNearDuplicates:
    def test_mixed_media(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Two identical images
            p1 = create_test_image(str(Path(tmpdir, "a.png")), color=(200, 100, 50))
            p2 = create_test_image(str(Path(tmpdir, "b.png")), color=(200, 100, 50))
            # Two identical small videos
            vid_content = b"\xef" * 30 * 1024
            v1 = Path(tmpdir, "v1.mp4")
            v2 = Path(tmpdir, "v2.mp4")
            v1.write_bytes(vid_content)
            v2.write_bytes(vid_content)

            files = [
                MediaFileInfo(path=p1, filename="a.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p2, filename="b.png", extension=".png", media_type="image"),
                MediaFileInfo(path=str(v1), filename="v1.mp4", extension=".mp4", media_type="video"),
                MediaFileInfo(path=str(v2), filename="v2.mp4", extension=".mp4", media_type="video"),
            ]

            groups = find_all_near_duplicates(files, phash_threshold=0, video_threshold=50)
            assert len(groups) >= 1

    def test_no_near_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = create_gradient_image(str(Path(tmpdir, "grad1.png")), offset=0)
            p2 = create_gradient_image(str(Path(tmpdir, "grad2.png")), offset=100)

            files = [
                MediaFileInfo(path=p1, filename="grad1.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p2, filename="grad2.png", extension=".png", media_type="image"),
            ]
            groups = find_all_near_duplicates(files, phash_threshold=0)
            assert len(groups) == 0


@pytest.mark.skipif(not HAS_IMAGEHASH, reason="imagehash not installed")
class TestFindAllDuplicates:
    def test_combined_exact_and_near(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Exact duplicate pair (same bytes)
            p1 = Path(tmpdir, "exact1.jpg")
            p2 = Path(tmpdir, "exact2.jpg")
            p1.write_text("exact same content")
            p2.write_text("exact same content")
            # Near-duplicate pair (same visual content - same gradient offset)
            p3 = create_gradient_image(str(Path(tmpdir, "near1.png")), offset=0)
            p4 = create_gradient_image(str(Path(tmpdir, "near2.png")), offset=0)

            files = [
                MediaFileInfo(path=str(p1), filename="exact1.jpg", extension=".jpg", media_type="image"),
                MediaFileInfo(path=str(p2), filename="exact2.jpg", extension=".jpg", media_type="image"),
                MediaFileInfo(path=p3, filename="near1.png", extension=".png", media_type="image"),
                MediaFileInfo(path=p4, filename="near2.png", extension=".png", media_type="image"),
            ]

            groups = find_all_duplicates(files, phash_threshold=0, video_threshold=20)
            assert len(groups) >= 2  # At least one exact + one near group


# ─── Scanner Integration Tests ────────────────────────────────────────────

@pytest.mark.skipif(not HAS_IMAGEHASH, reason="imagehash not installed")
class TestScannerMetadataIntegration:
    def test_extract_metadata_includes_phash(self):
        """Verify that extract_metadata computes perceptual hashes."""
        from engine.scanner import extract_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_test_image(str(Path(tmpdir, "test.png")))
            info = extract_metadata(path)
            assert info.phash is not None
            assert len(info.phash) == 16
            assert info.dhash is not None
            assert info.whash is not None

    def test_video_metadata_includes_hash(self):
        """Verify that extract_metadata computes video hash."""
        from engine.scanner import extract_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir, "test.mp4"))
            with open(path, "wb") as f:
                f.write(b"\xab" * 50 * 1024)
            info = extract_metadata(path)
            assert info.video_hash is not None
            assert info.media_type == "video"

    def test_scan_directory_computes_perceptual_hashes(self):
        """Verify scan_directory computes perceptual hashes for all images."""
        from engine.scanner import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(str(Path(tmpdir, "img1.png")), color=(100, 100, 100))
            create_test_image(str(Path(tmpdir, "img2.png")), color=(200, 200, 200))

            files = scan_directory(tmpdir)
            assert len(files) == 2
            for f in files:
                assert f.phash is not None


# ─── Hash Utility Tests ───────────────────────────────────────────────────

class TestHashToBits:
    def test_known_hash(self):
        bits = hash_to_bits("0f")
        assert bits == "1111"  # 0x0F = 0b1111

    def test_longer_hash(self):
        bits = hash_to_bits("ff")
        assert bits == "11111111"

    def test_empty(self):
        assert hash_to_bits("") == ""
        assert hash_to_bits(None) == ""  # type: ignore