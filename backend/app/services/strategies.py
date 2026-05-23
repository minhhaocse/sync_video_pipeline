from typing import Protocol
from pathlib import Path
import logging
import math
import shutil
import subprocess

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VISUAL_AGREEMENT_TOLERANCE_SECONDS = 0.75
VALIDATION_SCORE_MIN_MARGIN = 0.005
FINE_WINDOW_SECONDS = 6.0
MIN_FINE_WINDOW_SECONDS = 4.0
MAX_FINE_RESIDUAL_SECONDS = 2.0
MAX_GLOBAL_VISUAL_COARSE_SECONDS = 5.0
EDGE_OF_SEARCH_RATIO = 0.85

class SyncStrategy(Protocol):
    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        """
        Compute time offsets for each camera relative to the reference camera (cam_ids[0]).
        Returns:
            dict mapping cam_id -> offset_seconds.
            Positive values are trimmed; negative values are padded/delayed.
        """
        ...


def _normalize_offsets(offsets: dict[str, float], cam_ids: list[str]) -> dict[str, float]:
    """
    Validate and complete a strategy result without inventing successful syncs.

    Offsets are stored relative to cam_ids[0]. Positive values mean the camera
    has leading material that should be trimmed; negative values mean the camera
    should be delayed/padded relative to the reference.
    """
    if not cam_ids:
        return {}
    missing = [cam_id for cam_id in cam_ids if cam_id not in offsets]
    if missing:
        raise ValueError(f"Sync strategy did not produce offsets for cameras: {missing}")

    normalized: dict[str, float] = {}
    ref_offset = float(offsets.get(cam_ids[0], 0.0))
    for cam_id in cam_ids:
        value = float(offsets[cam_id]) - ref_offset
        if not math.isfinite(value):
            raise ValueError(f"Non-finite offset for camera {cam_id}: {offsets[cam_id]}")
        normalized[cam_id] = value
    normalized[cam_ids[0]] = 0.0
    return normalized


class FeatureSyncStrategy:
    last_method = "feature_based"

    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using Feature-Based (CV) Synchronization Strategy")
        from app.services.feature_based_approach.wrapper import compute_feature_offsets
        return _normalize_offsets(compute_feature_offsets(chunk_dir, cam_ids), cam_ids)

class SeSynNetSyncStrategy:
    last_method = "sesyn_net"
    last_errors: list[str] = []
    last_details: dict = {}

    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using SeSyn-Net (Pose-Based) Synchronization Strategy")
        from app.services.sesyn_net_approach import wrapper

        self.last_errors = []
        offsets = _normalize_offsets(wrapper.compute_sesyn_offsets(chunk_dir, cam_ids), cam_ids)
        self.last_details = {
            "mode": "sesyn_net",
            **getattr(wrapper, "LAST_SYNC_DETAILS", {}),
        }
        return offsets

class AudioSyncStrategy:
    last_method = "audio"

    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using Audio Cross-Correlation Synchronization Strategy")
        from app.services.offset import compute_offsets
        return _normalize_offsets(compute_offsets(chunk_dir, cam_ids), cam_ids)


class MultiSyncVideoStrategy:
    last_method = "multisyncvideo"
    last_errors: list[str] = []
    last_details: dict = {}

    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        """
        General-purpose coarse synchronizer.

        It first tries audio cross-correlation, then falls back to global/CV
        frame features. This is intentionally the fast, broad alignment stage
        used directly in MultiSyncVideo mode and as Auto's coarse pass.
        """
        logger.info("Using MultiSyncVideo Strategy: audio first, then global visual features.")
        attempts = [
            ("audio", AudioSyncStrategy()),
            ("global_visual", FeatureSyncStrategy()),
        ]
        errors: list[str] = []
        self.last_errors = []
        self.last_details = {
            "mode": "multisyncvideo",
            "pipeline_stages": [],
            "selection_reason": "",
        }

        for method, strategy in attempts:
            try:
                offsets = strategy.compute_offsets(chunk_dir, cam_ids)
                normalized = _normalize_offsets(offsets, cam_ids)
                if method == "global_visual":
                    largest_offset = max(abs(float(value)) for value in normalized.values())
                    if largest_offset >= MAX_GLOBAL_VISUAL_COARSE_SECONDS * EDGE_OF_SEARCH_RATIO:
                        raise ValueError(
                            f"Global visual sync returned {largest_offset:.3f}s, which is near the "
                            f"{MAX_GLOBAL_VISUAL_COARSE_SECONDS:.1f}s search boundary. This usually "
                            "means no reliable global sync marker was found."
                        )
                self.last_method = "multisyncvideo"
                self.last_errors = errors
                self.last_details = {
                    "mode": "multisyncvideo",
                    "pipeline_stages": [f"MultiSyncVideo_{method}"],
                    "coarse_method": method,
                    "selection_reason": (
                        "Audio cross-correlation was used for coarse sync."
                        if method == "audio"
                        else "Audio was unavailable or unreliable; global visual features were used for coarse sync."
                    ),
                }
                logger.info(f"MultiSyncVideo selected {method}: {normalized}")
                return normalized
            except Exception as exc:
                errors.append(f"{method}: {exc}")
                self.last_errors = errors
                logger.warning(f"MultiSyncVideo {method} stage failed.", exc_info=True)

        raise RuntimeError("MultiSyncVideo failed: " + " | ".join(errors))


def _find_video_path(chunk_dir: Path, cam_id: str) -> Path:
    for ext in [".mp4", ".webm", ".mov", ".mkv"]:
        exact = chunk_dir / f"{cam_id}{ext}"
        if exact.exists():
            return exact
        matches = list(chunk_dir.glob(f"*{cam_id}*{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No video file found for camera {cam_id} in {chunk_dir}")


def _video_duration_seconds(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Could not read duration for {path}")
    return float(frame_count / fps)


def _read_thumb_at(path: Path, time_seconds: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
    thumb = cv2.equalizeHist(thumb)
    return thumb.astype(np.float32) / 255.0


def _trim_offsets_for_overlap(offsets: dict[str, float]) -> dict[str, float]:
    min_offset = min(offsets.values()) if offsets else 0.0
    return {cam_id: float(offset - min_offset) for cam_id, offset in offsets.items()}


def _score_candidate_offsets(
    chunk_dir: Path,
    cam_ids: list[str],
    offsets: dict[str, float],
    samples: int = 8,
) -> float:
    """
    Lower is better. Scores how visually similar sampled frames look after
    applying candidate offsets to the short sync clips.
    """
    paths = {cam_id: _find_video_path(chunk_dir, cam_id) for cam_id in cam_ids}
    durations = {cam_id: _video_duration_seconds(path) for cam_id, path in paths.items()}
    trims = _trim_offsets_for_overlap(offsets)
    overlap_duration = min(durations[cam_id] - trims.get(cam_id, 0.0) for cam_id in cam_ids)
    if overlap_duration <= 1.0:
        return float("inf")

    ref_cam = cam_ids[0]
    sample_times = np.linspace(0.25, max(0.3, overlap_duration - 0.25), num=samples)
    diffs: list[float] = []

    for t in sample_times:
        ref_thumb = _read_thumb_at(paths[ref_cam], float(t + trims.get(ref_cam, 0.0)))
        if ref_thumb is None:
            continue
        for cam_id in cam_ids[1:]:
            thumb = _read_thumb_at(paths[cam_id], float(t + trims.get(cam_id, 0.0)))
            if thumb is None:
                continue
            diffs.append(float(np.mean(np.abs(ref_thumb - thumb))))

    if not diffs:
        return float("inf")
    return float(np.median(diffs))


def _max_offset_disagreement(
    left: dict[str, float],
    right: dict[str, float],
    cam_ids: list[str],
) -> float:
    return max(abs(float(left[cam_id]) - float(right[cam_id])) for cam_id in cam_ids)


def _average_offsets(candidates: list[dict[str, float]], cam_ids: list[str]) -> dict[str, float]:
    averaged = {
        cam_id: float(sum(float(candidate[cam_id]) for candidate in candidates) / len(candidates))
        for cam_id in cam_ids
    }
    averaged[cam_ids[0]] = 0.0
    return averaged


def _create_coarse_aligned_fine_window(
    chunk_dir: Path,
    cam_ids: list[str],
    coarse_offsets: dict[str, float],
) -> Path:
    """
    Build a short set of roughly aligned clips for SeSyn-Net fine tuning.

    Coarse offsets are converted to nonnegative trims, so all generated clips
    begin at the coarse-aligned overlap. SeSyn-Net then only has to estimate a
    small residual correction instead of searching the full timeline.
    """
    fine_dir = chunk_dir / "_sesyn_fine_window"
    if fine_dir.exists():
        shutil.rmtree(fine_dir)
    fine_dir.mkdir(parents=True, exist_ok=True)

    paths = {cam_id: _find_video_path(chunk_dir, cam_id) for cam_id in cam_ids}
    durations = {cam_id: _video_duration_seconds(path) for cam_id, path in paths.items()}
    trims = _trim_offsets_for_overlap(coarse_offsets)
    overlap_duration = min(durations[cam_id] - trims.get(cam_id, 0.0) for cam_id in cam_ids)
    window_duration = min(FINE_WINDOW_SECONDS, overlap_duration)
    if window_duration < MIN_FINE_WINDOW_SECONDS:
        raise ValueError(
            f"Coarse-aligned overlap is too short for SeSyn-Net fine tuning "
            f"({window_duration:.3f}s; need at least {MIN_FINE_WINDOW_SECONDS:.1f}s)."
        )

    for cam_id in cam_ids:
        output_path = fine_dir / f"{cam_id}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{trims.get(cam_id, 0.0):.6f}",
            "-t", f"{window_duration:.6f}",
            "-i", str(paths[cam_id]),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Could not create SeSyn fine window for {cam_id}: {result.stderr[-1000:]}"
            )

    return fine_dir


class HybridVisualSyncStrategy:
    last_method = "visual_hybrid"
    last_errors: list[str] = []
    last_details: dict = {}

    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using Hybrid Visual Synchronization Strategy: Feature-Based + SeSyn-Net")
        attempts = [
            ("feature_based", FeatureSyncStrategy()),
            ("sesyn_net", SeSynNetSyncStrategy()),
        ]
        candidates: dict[str, dict[str, float]] = {}
        scores: dict[str, float] = {}
        errors: list[str] = []

        self.last_errors = []
        self.last_details = {
            "mode": "visual_hybrid",
            "agreement_tolerance_seconds": VISUAL_AGREEMENT_TOLERANCE_SECONDS,
            "validation_score_min_margin": VALIDATION_SCORE_MIN_MARGIN,
            "selection_reason": "",
            "selection_confidence": "unknown",
            "candidates": {},
            "candidate_scores": {},
        }

        for method, strategy in attempts:
            try:
                offsets = strategy.compute_offsets(chunk_dir, cam_ids)
                candidates[method] = _normalize_offsets(offsets, cam_ids)
                logger.info(f"Hybrid visual candidate {method}: {candidates[method]}")
            except Exception as exc:
                errors.append(f"{method}: {exc}")
                logger.warning(f"Hybrid visual candidate {method} failed.", exc_info=True)

        self.last_errors = errors
        self.last_details["candidates"] = candidates

        if not candidates:
            raise RuntimeError("Both visual synchronization methods failed: " + " | ".join(errors))

        if len(candidates) == 1:
            method, offsets = next(iter(candidates.items()))
            self.last_method = method
            self.last_details["selection_reason"] = f"Only {method} produced usable offsets."
            self.last_details["selection_confidence"] = "medium"
            return offsets

        feature_offsets = candidates["feature_based"]
        sesyn_offsets = candidates["sesyn_net"]
        disagreement = _max_offset_disagreement(feature_offsets, sesyn_offsets, cam_ids)
        self.last_details["max_disagreement_seconds"] = float(disagreement)

        if disagreement <= VISUAL_AGREEMENT_TOLERANCE_SECONDS:
            averaged = _average_offsets([feature_offsets, sesyn_offsets], cam_ids)
            self.last_method = "visual_consensus"
            self.last_details["selection_reason"] = (
                "Feature-Based and SeSyn-Net agreed, so their offsets were averaged."
            )
            self.last_details["selection_confidence"] = "high"
            return averaged

        for method, offsets in candidates.items():
            try:
                scores[method] = _score_candidate_offsets(chunk_dir, cam_ids, offsets)
            except Exception as exc:
                scores[method] = float("inf")
                errors.append(f"{method} validation: {exc}")
                logger.warning(f"Could not score {method} candidate offsets.", exc_info=True)

        self.last_errors = errors
        self.last_details["candidate_scores"] = scores
        finite_scores = {method: score for method, score in scores.items() if math.isfinite(score)}

        if finite_scores:
            sorted_scores = sorted(finite_scores.items(), key=lambda item: item[1])
            selected_method = sorted_scores[0][0]
            score_margin = (
                sorted_scores[1][1] - sorted_scores[0][1]
                if len(sorted_scores) > 1
                else float("inf")
            )
            self.last_details["score_margin"] = float(score_margin)
            if len(sorted_scores) > 1 and score_margin < VALIDATION_SCORE_MIN_MARGIN:
                selected_method = "sesyn_net" if "sesyn_net" in candidates else selected_method
                self.last_method = selected_method
                self.last_details["selection_confidence"] = "low"
                self.last_details["selection_reason"] = (
                    "Feature-Based and SeSyn-Net disagreed, and validation scores were too close "
                    "to be decisive; SeSyn-Net was used as the tie-breaker."
                )
                logger.warning(
                    f"Hybrid visual low-confidence tie; selected {selected_method}. "
                    f"disagreement={disagreement:.3f}s, scores={scores}"
                )
                return candidates[selected_method]

            self.last_method = selected_method
            self.last_details["selection_confidence"] = "high"
            self.last_details["selection_reason"] = (
                "Feature-Based and SeSyn-Net disagreed, so the candidate with the better "
                "aligned-frame validation score was used."
            )
            logger.info(
                f"Hybrid visual selected {selected_method}; disagreement={disagreement:.3f}s, "
                f"scores={scores}"
            )
            return candidates[selected_method]

        self.last_method = "sesyn_net"
        self.last_details["selection_confidence"] = "low"
        self.last_details["selection_reason"] = (
            "Feature-Based and SeSyn-Net disagreed, but validation scoring failed; "
            "falling back to SeSyn-Net."
        )
        return sesyn_offsets

class AutoSyncStrategy:
    last_method = "auto"
    last_errors: list[str] = []
    last_details: dict = {}

    def compute_offsets(self, chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
        logger.info("Using Auto Sync Strategy: MultiSyncVideo coarse, then SeSyn-Net fine.")

        errors: list[str] = []
        self.last_errors = []
        self.last_method = "auto"
        self.last_details = {
            "mode": "auto_coarse_to_fine",
            "pipeline_stages": [],
            "selection_reason": "",
            "selection_confidence": "unknown",
        }

        coarse_strategy = MultiSyncVideoStrategy()
        try:
            coarse_offsets = coarse_strategy.compute_offsets(chunk_dir, cam_ids)
        except Exception as exc:
            errors.append(f"MultiSyncVideo_Coarse: {exc}")
            logger.warning("AutoSync: MultiSyncVideo coarse stage failed; trying SeSyn-Net standalone.", exc_info=True)
            try:
                fallback_strategy = SeSynNetSyncStrategy()
                fallback_offsets = fallback_strategy.compute_offsets(chunk_dir, cam_ids)
                fallback_offsets = _normalize_offsets(fallback_offsets, cam_ids)
                self.last_method = "sesyn_net"
                self.last_errors = errors
                self.last_details = {
                    "mode": "auto_pose_fallback",
                    "pipeline_stages": ["SeSynNet_Fallback"],
                    "selection_confidence": "medium",
                    "sesyn_details": getattr(fallback_strategy, "last_details", {}),
                    "selection_reason": (
                        "MultiSyncVideo could not find a reliable audio/global marker, so Auto used "
                        "SeSyn-Net directly on the sync clips."
                    ),
                }
                return fallback_offsets
            except Exception as fallback_exc:
                errors.append(f"SeSynNet_Fallback: {fallback_exc}")
                self.last_errors = errors
                raise RuntimeError("Auto sync failed: " + " | ".join(errors)) from fallback_exc

        self.last_details.update({
            "pipeline_stages": ["MultiSyncVideo_Coarse"],
            "coarse_offsets": coarse_offsets,
            "coarse_method": coarse_strategy.last_details.get("coarse_method"),
        })
        errors.extend(coarse_strategy.last_errors)

        try:
            fine_dir = _create_coarse_aligned_fine_window(chunk_dir, cam_ids, coarse_offsets)
            fine_strategy = SeSynNetSyncStrategy()
            residual_offsets = fine_strategy.compute_offsets(fine_dir, cam_ids)
            residual_offsets = _normalize_offsets(residual_offsets, cam_ids)
            max_residual = max(abs(float(value)) for value in residual_offsets.values())
            if max_residual > MAX_FINE_RESIDUAL_SECONDS:
                raise ValueError(
                    f"SeSyn-Net residual was too large for a fine-tune pass "
                    f"({max_residual:.3f}s > {MAX_FINE_RESIDUAL_SECONDS:.3f}s)."
                )

            final_offsets = {
                cam_id: float(coarse_offsets[cam_id] + residual_offsets[cam_id])
                for cam_id in cam_ids
            }
            final_offsets = _normalize_offsets(final_offsets, cam_ids)
            self.last_method = "auto_coarse_to_fine"
            self.last_errors = errors
            self.last_details.update({
                "pipeline_stages": ["MultiSyncVideo_Coarse", "SeSynNet_Fine"],
                "fine_method": "sesyn_net",
                "fine_residual_offsets": residual_offsets,
                "fine_details": getattr(fine_strategy, "last_details", {}),
                "max_fine_residual_seconds": float(max_residual),
                "selection_confidence": "high",
                "selection_reason": (
                    "MultiSyncVideo produced the coarse timeline, then SeSyn-Net refined it "
                    "inside the coarse-aligned human-motion window."
                ),
            })
            logger.info(
                f"Auto coarse-to-fine offsets: coarse={coarse_offsets}, "
                f"residual={residual_offsets}, final={final_offsets}"
            )
            return final_offsets
        except Exception as exc:
            errors.append(f"SeSynNet_Fine: {exc}")
            self.last_errors = errors
            self.last_method = "multisyncvideo"
            self.last_details.update({
                "pipeline_stages": ["MultiSyncVideo_Coarse"],
                "selection_confidence": "medium",
                "selection_reason": (
                    "SeSyn-Net fine tuning failed or was not reliable, so Auto fell back "
                    "to the MultiSyncVideo coarse offsets."
                ),
            })
            logger.warning("AutoSync: SeSyn-Net fine stage failed; using MultiSyncVideo coarse offsets.", exc_info=True)
            return coarse_offsets

def get_sync_strategy(name: str) -> SyncStrategy:
    name_lower = name.lower()
    if name_lower in ["feature", "cv", "feature_based"]:
        return FeatureSyncStrategy()
    if name_lower in ["multisyncvideo", "multisync", "multividsynch", "multividsync"]:
        return MultiSyncVideoStrategy()
    elif name_lower in ["sesyn", "sesyn_net", "pose"]:
        return SeSynNetSyncStrategy()
    elif name_lower in ["audio", "audio_cross_correlation"]:
        return AudioSyncStrategy()
    elif name_lower in ["hybrid", "visual_hybrid", "combined", "ensemble"]:
        return HybridVisualSyncStrategy()
    elif name_lower in ["auto", "default"]:
        return AutoSyncStrategy()
    else:
        logger.warning(f"Unknown sync strategy '{name}', falling back to AutoSyncStrategy")
        return AutoSyncStrategy()
