from typing import Protocol
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SyncStrategy(Protocol):
    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        """
        Compute time offsets for each camera relative to the reference camera (cam_ids[0]).
        Returns:
            dict mapping cam_id -> offset_seconds (positive = this cam is LATE)
        """
        ...

class FeatureSyncStrategy:
    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using Feature-Based (CV) Synchronization Strategy")
        try:
            from app.services.feature_based_approach.wrapper import compute_feature_offsets
            return compute_feature_offsets(chunk_dir, cam_ids)
        except Exception as e:
            logger.error(
                "Feature Sync failed. Falling back to zero offsets. "
                "Check the feature-based sync pipeline and source videos.",
                exc_info=True,
            )
            return {cam_id: 0.0 for cam_id in cam_ids}

class SeSynNetSyncStrategy:
    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using SeSyn-Net (Pose-Based) Synchronization Strategy")
        try:
            from app.services.sesyn_net_approach.wrapper import compute_sesyn_offsets
            return compute_sesyn_offsets(chunk_dir, cam_ids)
        except Exception as e:
            logger.error(
                "SeSyn-Net Sync failed. Falling back to zero offsets. "
                "Ensure the SeSyn-Net repository, model weights, and video inputs are available.",
                exc_info=True,
            )
            return {cam_id: 0.0 for cam_id in cam_ids}

class AutoSyncStrategy:
    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using Auto Sync Strategy: Trying methods from fastest to most robust.")
            
        # 1. Try Feature-Based / Multividsynch (Fast CV matching)
        try:
            logger.info("AutoSync [1/2]: Attempting Feature-based (CV) sync...")
            from app.services.feature_based_approach.wrapper import compute_feature_offsets
            return compute_feature_offsets(chunk_dir, cam_ids)
        except Exception as e:
            logger.warning(
                "AutoSync: Feature-based sync failed. Trying SeSyn-Net next.",
                exc_info=True,
            )

        # 2. Try SeSyn-Net Pose-Based (Slower but robust for human activity)
        try:
            logger.info("AutoSync [2/2]: Attempting SeSyn-Net (Pose-Based) sync...")
            from app.services.sesyn_net_approach.wrapper import compute_sesyn_offsets
            return compute_sesyn_offsets(chunk_dir, cam_ids)
        except Exception as e:
            logger.error(
                "AutoSync: SeSyn-Net sync failed. Falling back to zero offsets for all cameras.",
                exc_info=True,
            )
            
        logger.error("AutoSync: All synchronization strategies failed. Falling back to zero offsets for all cameras.")
        return {cam_id: 0.0 for cam_id in cam_ids}

def get_sync_strategy(name: str) -> SyncStrategy:
    name_lower = name.lower()
    if name_lower in ["feature", "cv", "feature_based", "multividsynch"]:
        return FeatureSyncStrategy()
    elif name_lower in ["sesyn", "sesyn_net", "pose"]:
        return SeSynNetSyncStrategy()
    elif name_lower in ["auto", "default"]:
        return AutoSyncStrategy()
    else:
        logger.warning(f"Unknown sync strategy '{name}', falling back to AutoSyncStrategy")
        return AutoSyncStrategy()
