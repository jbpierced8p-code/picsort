"""
PicSort AI - Tier Configuration
Defines feature access for free vs premium tiers.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict


class AppTier(Enum):
    FREE = "free"
    PREMIUM = "premium"


@dataclass
class TierLimits:
    max_scan_size: int  # bytes, 0 = unlimited
    perceptual_hashing: bool
    facial_recognition: bool
    auto_cleanup: bool
    cross_device_sync: bool
    cloud_backup: bool


TIER_CONFIG: Dict[AppTier, TierLimits] = {
    AppTier.FREE: TierLimits(
        max_scan_size=5 * 1024 * 1024 * 1024,  # 5 GB
        perceptual_hashing=False,
        facial_recognition=False,
        auto_cleanup=False,
        cross_device_sync=False,
        cloud_backup=False,
    ),
    AppTier.PREMIUM: TierLimits(
        max_scan_size=0,  # unlimited
        perceptual_hashing=True,
        facial_recognition=True,
        auto_cleanup=True,
        cross_device_sync=True,
        cloud_backup=True,
    ),
}


def is_feature_allowed(tier: AppTier, feature: str) -> bool:
    """Check if a feature is available for the given tier."""
    config = TIER_CONFIG.get(tier, TIER_CONFIG[AppTier.FREE])
    return getattr(config, feature, False)


# Global tier setting (defaults to free, changed by app)
_active_tier: AppTier = AppTier.FREE


def set_tier(tier: AppTier) -> None:
    """Set the active tier for the current session."""
    global _active_tier
    _active_tier = tier


def get_tier() -> AppTier:
    """Get the currently active tier."""
    return _active_tier


def has_feature(feature: str) -> bool:
    """Check if the active tier has access to a feature."""
    return is_feature_allowed(_active_tier, feature)