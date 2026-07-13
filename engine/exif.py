"""
PicSort AI - EXIF Metadata Extractor
Extracts EXIF metadata from image files using Pillow.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ExifTags


def extract_exif(filepath: str) -> Dict[str, Any]:
    """
    Extract EXIF metadata from an image file.
    Returns a dict with extracted fields (or empty dict if no EXIF).
    """
    result: Dict[str, Any] = {
        "has_exif": False,
        "image_width": None,
        "image_height": None,
        "camera_make": None,
        "camera_model": None,
        "date_taken": None,
        "gps_latitude": None,
        "gps_longitude": None,
        "orientation": None,
        "software": None,
        "iso": None,
        "focal_length": None,
        "aperture": None,
        "shutter_speed": None,
        "flash": None,
    }

    try:
        img = Image.open(filepath)
        result["image_width"] = img.width
        result["image_height"] = img.height

        exif_data = img.getexif()
        if not exif_data:
            return result

        result["has_exif"] = True

        # Build a human-readable tag map
        exif_dict = {}
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            exif_dict[tag_name] = value

        # Camera info
        result["camera_make"] = str(exif_dict.get("Make", "")) or None
        result["camera_model"] = str(exif_dict.get("Model", "")) or None
        result["software"] = str(exif_dict.get("Software", "")) or None

        # Orientation
        result["orientation"] = exif_dict.get("Orientation")

        # Date taken - parse various date formats
        date_str = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTimeDigitized") or exif_dict.get("DateTime")
        if date_str:
            try:
                # EXIF date format: "YYYY:MM:DD HH:MM:SS"
                dt = datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
                result["date_taken"] = dt.isoformat()
            except (ValueError, TypeError):
                result["date_taken"] = str(date_str)

        # GPS coordinates
        gps_info = exif_data.get_ifd(0x8825)  # GPSInfo IFD
        if gps_info:
            gps_tags = {v: k for k, v in ExifTags.GPSTAGS.items()}
            lat_ref = gps_info.get(gps_tags.get("GPSLatitudeRef"))
            lat_data = gps_info.get(gps_tags.get("GPSLatitude"))
            lon_ref = gps_info.get(gps_tags.get("GPSLongitudeRef"))
            lon_data = gps_info.get(gps_tags.get("GPSLongitude"))

            if lat_data and lon_data:
                result["gps_latitude"] = _dms_to_decimal(lat_data, lat_ref == "S" if lat_ref else False)
                result["gps_longitude"] = _dms_to_decimal(lon_data, lon_ref == "W" if lon_ref else False)

        # Photo settings
        result["iso"] = exif_dict.get("ISOSpeedRatings")
        result["focal_length"] = _extract_focal_length(exif_dict.get("FocalLength"))
        result["aperture"] = _extract_aperture(exif_dict.get("FNumber"))
        result["shutter_speed"] = _extract_shutter_speed(exif_dict.get("ExposureTime"))
        flash_val = exif_dict.get("Flash")
        if flash_val is not None:
            result["flash"] = bool(int(flash_val) & 0x1)

    except Exception:
        # If we can't read the file (corrupted, unsupported format, etc.)
        # just return basic dimensions if we got them
        pass

    return result


def _dms_to_decimal(dms: Tuple[float, ...], negative: bool = False) -> float:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    if len(dms) >= 3:
        degrees = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
    elif len(dms) == 2:
        degrees = float(dms[0]) + float(dms[1]) / 60.0
    elif len(dms) == 1:
        degrees = float(dms[0])
    else:
        return 0.0
    return -degrees if negative else degrees


def _extract_focal_length(value: Any) -> Optional[float]:
    """Extract focal length from EXIF rational value."""
    if value is None:
        return None
    try:
        if isinstance(value, tuple):
            return float(value[0]) / float(value[1]) if value[1] else None
        return float(value)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _extract_aperture(value: Any) -> Optional[float]:
    """Extract aperture/FNumber from EXIF rational value."""
    if value is None:
        return None
    try:
        if isinstance(value, tuple):
            return round(float(value[0]) / float(value[1]), 1) if value[1] else None
        return round(float(value), 1)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _extract_shutter_speed(value: Any) -> Optional[str]:
    """Extract shutter speed from EXIF rational value."""
    if value is None:
        return None
    try:
        if isinstance(value, tuple):
            numerator = float(value[0])
            denominator = float(value[1])
            if denominator == 0:
                return None
            if numerator >= denominator:
                return f"{numerator / denominator:.1f}s"
            else:
                return f"{int(denominator / numerator)}s" if numerator >= 1 else f"{numerator/denominator:.3f}s"
        return str(value)
    except (ValueError, ZeroDivisionError, TypeError):
        return None