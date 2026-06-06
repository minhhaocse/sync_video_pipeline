import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

from app.services.feature_based_approach.utils import get_total_frames
from app.services.feature_based_approach.OTP import (
    detect_features,
    extract_features_from_frame,
    match_features,
    construct_trajectories,
    compute_fundamental_matrix,
    filter_trajectories,
    match_trajectories
)

logger = logging.getLogger(__name__)

MAX_FEATURE_OFFSET_SECONDS = 15.0
MIN_MATCHED_TRAJECTORIES = 8
FRAME_SIMILARITY_SAMPLE_RATE = 3.0
MIN_FRAME_SIMILARITY_OVERLAP_SECONDS = 3.0
FRAME_SIMILARITY_EDGE_WEIGHT = 0.35
FRAME_SIMILARITY_MOTION_WEIGHT = 0.45
FEATURE_MOTION_SAMPLE_RATE = 3.0
FEATURE_MOTION_MIN_OVERLAP_SECONDS = 3.0
FEATURE_MOTION_COUNT_WEIGHT = 0.25
FEATURE_MOTION_DISPLACEMENT_WEIGHT = 0.75


def _safe_fundamental_matrix(*args):
    try:
        return compute_fundamental_matrix(*args)
    except cv2.error as exc:
        logger.warning(f"OpenCV rejected fundamental-matrix inputs: {exc}")
        return None, [], args[0], args[2]


def _estimate_offsets_by_frame_similarity(
    capture_files: dict[str, Path],
    cam_ids: list[str],
    fps: float,
    sample_rate: float = FRAME_SIMILARITY_SAMPLE_RATE,
    max_shift_seconds: float = MAX_FEATURE_OFFSET_SECONDS,
) -> dict[str, float]:
    """
    Lightweight visual fallback for silent clips.

    It compares small grayscale frame thumbnails over a bounded lag window.
    This is less ambitious than trajectory matching, but it gives us a stable
    offset estimate for same-scene clips instead of crashing on weak features.
    """
    def _zscore(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        std = float(np.std(values))
        if std < 1e-6:
            return values - float(np.mean(values))
        return (values - float(np.mean(values))) / std

    def _series_distance(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 2 or right.size < 2:
            return float("inf")
        return float(np.mean(np.abs(_zscore(left) - _zscore(right))))

    def load_series(path: Path) -> dict[str, list[np.ndarray] | np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
        step = max(1, int(round(src_fps / sample_rate)))
        thumbs: list[np.ndarray] = []
        edges: list[float] = []
        motions: list[float] = []
        prev_thumb: np.ndarray | None = None
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thumb = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
                thumb = cv2.equalizeHist(thumb).astype(np.float32) / 255.0
                edge = cv2.Canny((thumb * 255).astype(np.uint8), 80, 160)
                edges.append(float(np.mean(edge > 0)))
                if prev_thumb is None:
                    motions.append(0.0)
                else:
                    motions.append(float(np.mean(np.abs(thumb - prev_thumb))))
                prev_thumb = thumb
                thumbs.append(thumb.reshape(-1))
            frame_idx += 1
        cap.release()
        return {
            "thumbs": thumbs,
            "edges": np.asarray(edges, dtype=np.float32),
            "motions": np.asarray(motions, dtype=np.float32),
        }

    series_by_cam = {cam_id: load_series(capture_files[cam_id]) for cam_id in cam_ids}
    ref_series = series_by_cam[cam_ids[0]]
    ref_thumbs = ref_series["thumbs"]
    if len(ref_thumbs) < 3:
        raise ValueError("Frame-similarity fallback needs at least 3 sampled reference frames.")

    max_lag = int(round(max_shift_seconds * sample_rate))
    shortest_series = min(len(series_by_cam[cam_id]["thumbs"]) for cam_id in cam_ids)
    min_overlap = min(
        shortest_series,
        max(3, int(round(MIN_FRAME_SIMILARITY_OVERLAP_SECONDS * sample_rate))),
    )
    offsets = {cam_ids[0]: 0.0}

    for cam_id in cam_ids[1:]:
        cam_series = series_by_cam[cam_id]
        cam_thumbs = cam_series["thumbs"]
        if len(cam_thumbs) < 3:
            raise ValueError(f"Frame-similarity fallback needs at least 3 sampled frames for {cam_id}.")

        best_lag = 0
        best_score = float("inf")
        second_best_score = float("inf")
        for lag in range(-max_lag, max_lag + 1):
            ref_start = max(0, -lag)
            cam_start = max(0, lag)
            overlap = min(len(ref_thumbs) - ref_start, len(cam_thumbs) - cam_start)
            if overlap < min_overlap:
                continue

            diffs = [
                float(np.mean(np.abs(ref_thumbs[ref_start + i] - cam_thumbs[cam_start + i])))
                for i in range(overlap)
            ]
            thumb_score = float(np.median(diffs))
            ref_slice = slice(ref_start, ref_start + overlap)
            cam_slice = slice(cam_start, cam_start + overlap)
            edge_score = _series_distance(
                ref_series["edges"][ref_slice],
                cam_series["edges"][cam_slice],
            )
            motion_score = _series_distance(
                ref_series["motions"][ref_slice],
                cam_series["motions"][cam_slice],
            )

            if not np.isfinite(edge_score):
                edge_score = thumb_score
            if not np.isfinite(motion_score):
                motion_score = thumb_score

            overlap_seconds = overlap / sample_rate
            short_overlap_penalty = 1.0 + max(0.0, MIN_FRAME_SIMILARITY_OVERLAP_SECONDS - overlap_seconds) * 0.1
            score = (
                thumb_score
                + FRAME_SIMILARITY_EDGE_WEIGHT * edge_score
                + FRAME_SIMILARITY_MOTION_WEIGHT * motion_score
            ) * short_overlap_penalty
            if score < best_score:
                second_best_score = best_score
                best_score = score
                best_lag = lag
            elif score < second_best_score:
                second_best_score = score

        if not np.isfinite(best_score):
            raise ValueError(f"Frame-similarity fallback could not compare {cam_id}.")

        offsets[cam_id] = float(best_lag / sample_rate)
        boundary_ratio = abs(best_lag) / max_lag if max_lag else 0.0
        score_margin = second_best_score - best_score if np.isfinite(second_best_score) else float("inf")
        logger.info(
            f"Frame-similarity fallback offset for {cam_id}: "
            f"{offsets[cam_id]:.3f}s (lag={best_lag}, score={best_score:.4f}, "
            f"margin={score_margin:.4f}, boundary={boundary_ratio:.2f})"
        )

    return offsets


def _estimate_offsets_by_feature_motion(
    capture_files: dict[str, Path],
    cam_ids: list[str],
    fps: float,
    sample_rate: float = FEATURE_MOTION_SAMPLE_RATE,
    max_shift_seconds: float = MAX_FEATURE_OFFSET_SECONDS,
) -> dict[str, float]:
    """
    Estimate temporal offsets from feature-motion signatures.

    This keeps the useful MultiVidSync idea (feature motion over time), but the
    final offset is computed by an explicit temporal lag search. Positive
    offsets follow the pipeline convention: trim that camera.
    """
    def _zscore(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        std = float(np.std(values))
        if std < 1e-6:
            return values - float(np.mean(values))
        return (values - float(np.mean(values))) / std

    def _signature_distance(
        ref_motion: np.ndarray,
        cam_motion: np.ndarray,
        ref_counts: np.ndarray,
        cam_counts: np.ndarray,
    ) -> float:
        if ref_motion.size < 2 or cam_motion.size < 2:
            return float("inf")
        motion_distance = float(np.mean(np.abs(_zscore(ref_motion) - _zscore(cam_motion))))
        count_distance = float(np.mean(np.abs(_zscore(ref_counts) - _zscore(cam_counts))))
        return (
            FEATURE_MOTION_DISPLACEMENT_WEIGHT * motion_distance
            + FEATURE_MOTION_COUNT_WEIGHT * count_distance
        )

    def load_signature(path: Path) -> dict[str, np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
        step = max(1, int(round(src_fps / sample_rate)))

        sampled_keypoints = []
        sampled_descriptors = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                keypoints, descriptors = detect_features(frame)
                sampled_keypoints.append(keypoints or [])
                sampled_descriptors.append(descriptors)
            frame_idx += 1
        cap.release()

        counts = np.asarray([len(kp) for kp in sampled_keypoints], dtype=np.float32)
        motion_values: list[float] = [0.0]
        for idx in range(1, len(sampled_keypoints)):
            prev_kp = sampled_keypoints[idx - 1]
            curr_kp = sampled_keypoints[idx]
            matches = match_features(sampled_descriptors[idx - 1], sampled_descriptors[idx])
            displacements = []
            for match in matches:
                if match.queryIdx >= len(prev_kp) or match.trainIdx >= len(curr_kp):
                    continue
                prev_pt = np.asarray(prev_kp[match.queryIdx].pt, dtype=np.float32)
                curr_pt = np.asarray(curr_kp[match.trainIdx].pt, dtype=np.float32)
                displacements.append(float(np.linalg.norm(curr_pt - prev_pt)))
            motion_values.append(float(np.median(displacements)) if displacements else 0.0)

        return {
            "motion": np.asarray(motion_values, dtype=np.float32),
            "counts": counts,
        }

    signatures = {cam_id: load_signature(capture_files[cam_id]) for cam_id in cam_ids}
    ref_signature = signatures[cam_ids[0]]
    ref_motion = ref_signature["motion"]
    ref_counts = ref_signature["counts"]
    if len(ref_motion) < 3:
        raise ValueError("Feature-motion sync needs at least 3 sampled reference frames.")

    max_lag = int(round(max_shift_seconds * sample_rate))
    shortest_series = min(len(signatures[cam_id]["motion"]) for cam_id in cam_ids)
    min_overlap = min(
        shortest_series,
        max(3, int(round(FEATURE_MOTION_MIN_OVERLAP_SECONDS * sample_rate))),
    )
    offsets = {cam_ids[0]: 0.0}

    for cam_id in cam_ids[1:]:
        cam_signature = signatures[cam_id]
        cam_motion = cam_signature["motion"]
        cam_counts = cam_signature["counts"]
        if len(cam_motion) < 3:
            raise ValueError(f"Feature-motion sync needs at least 3 sampled frames for {cam_id}.")

        best_lag = 0
        best_score = float("inf")
        second_best_score = float("inf")
        for lag in range(-max_lag, max_lag + 1):
            ref_start = max(0, -lag)
            cam_start = max(0, lag)
            overlap = min(len(ref_motion) - ref_start, len(cam_motion) - cam_start)
            if overlap < min_overlap:
                continue

            ref_slice = slice(ref_start, ref_start + overlap)
            cam_slice = slice(cam_start, cam_start + overlap)
            score = _signature_distance(
                ref_motion[ref_slice],
                cam_motion[cam_slice],
                ref_counts[ref_slice],
                cam_counts[cam_slice],
            )
            if score < best_score:
                second_best_score = best_score
                best_score = score
                best_lag = lag
            elif score < second_best_score:
                second_best_score = score

        if not np.isfinite(best_score):
            raise ValueError(f"Feature-motion sync could not compare {cam_id}.")

        offsets[cam_id] = float(best_lag / sample_rate)
        boundary_ratio = abs(best_lag) / max_lag if max_lag else 0.0
        score_margin = second_best_score - best_score if np.isfinite(second_best_score) else float("inf")
        logger.info(
            f"Feature-motion offset for {cam_id}: {offsets[cam_id]:.3f}s "
            f"(lag={best_lag}, score={best_score:.4f}, margin={score_margin:.4f}, "
            f"boundary={boundary_ratio:.2f})"
        )

    return offsets

def extract_representative_frames(video_path: Path, segment_duration_seconds: float = 10.0, fps: float = 30.0):
    """
    Extract frames from the first and last segments of a video.
    
    Args:
        video_path: Path to video file
        segment_duration_seconds: Duration of segment to extract from start and end (default 10s)
        fps: Frames per second (default 30fps)
    
    Returns:
        list of frames from first segment + last segment
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    
    segment_frame_count = int(segment_duration_seconds * cap_fps)
    frames = []
    
    # Extract first segment (frames 0 to segment_frame_count)
    logger.info(f"Extracting first {segment_duration_seconds}s ({segment_frame_count} frames) from {video_path.name}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(segment_frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Extract last segment (frames total_frames-segment_frame_count to total_frames)
    # if total_frames > segment_frame_count * 2:  # Ensure segments don't overlap
    #     logger.info(f"Extracting last {segment_duration_seconds}s ({segment_frame_count} frames) from {video_path.name}")
    #     last_segment_start = max(0, total_frames - segment_frame_count)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, last_segment_start)
    #     for i in range(segment_frame_count):
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frames.append(frame)
    # else:
    #     logger.warning(f"Video {video_path.name} is too short to extract separate first and last segments")
    
    cap.release()
    return frames

def compute_feature_offsets(chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
    """
    Computes offsets using a feature-based (CV) alignment approach.
    This method analyzes visual landmarks across cameras to determine temporal shifts.
    Since the core algorithm calculates offset in frames, we convert it to seconds
    based on the video's framerate.
    
    OPTIMIZATION: The full pipeline passes pre-cropped sync clips here, so this
    method works on a bounded number of frames instead of full-length videos.
    """
    if not cam_ids:
        return {}

    logger.info("Feature-based offset computation: using short sync clips")

    capture_files: dict[str, Path] = {}
    for cam_id in cam_ids:
        # Try multiple extensions
        video_path = None
        for ext in [".webm", ".mp4", ".mov", ".mkv"]:
            test_path = (chunk_dir / f"{cam_id}{ext}").resolve()
            if test_path.exists():
                video_path = test_path
                break
        
        if not video_path:
            raise FileNotFoundError(f"No input file found for camera {cam_id} in {chunk_dir}")
        capture_files[cam_id] = video_path

    if len(capture_files) < 2:
        raise ValueError("Feature sync requires at least two valid video files.")
    
    # Fallback FPS to 30.0, will try to read from actual video
    fps = 30.0
    ref_path = capture_files[cam_ids[0]]
    cap_for_fps = cv2.VideoCapture(str(ref_path))
    if cap_for_fps.isOpened():
        fps = cap_for_fps.get(cv2.CAP_PROP_FPS) or 30.0
    cap_for_fps.release()

    total_frames_by_cam = {
        cam_id: get_total_frames(str(capture_files[cam_id])) for cam_id in cam_ids
    }
    if any(frame_count <= 0 for frame_count in total_frames_by_cam.values()):
        raise ValueError(f"Feature sync found empty/unreadable videos: {total_frames_by_cam}")

    search_frames = min(30, min(total_frames_by_cam.values()))

    first_frames = []
    first_frames_keypoints = []
    first_frames_descriptors = []
    trajectories_data = {}

    for i, cam_name in enumerate(cam_ids):
        video_path = capture_files[cam_name]
        
        cap_for_metadata = cv2.VideoCapture(str(video_path))
        height = int(cap_for_metadata.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap_for_metadata.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_for_metadata.release()
        if height <= 0 or width <= 0:
            raise ValueError(f"Could not read dimensions for {video_path}")

        roi_start = (0, 0)
        roi_size = (height, width)

        best_frame = None
        best_kp = None
        best_desc = None
        max_kp = -1

        temp_cap = cv2.VideoCapture(str(video_path))
        for f_idx in range(search_frames):
            ret, frame = temp_cap.read()
            if not ret:
                break
            kp, desc = detect_features(frame)
            if kp and len(kp) > max_kp:
                max_kp = len(kp)
                best_frame = frame.copy()
                best_kp = kp
                best_desc = desc
        temp_cap.release()
        
        if best_frame is None:
            temp_cap = cv2.VideoCapture(str(video_path))
            ret, best_frame = temp_cap.read()
            if not ret or best_frame is None:
                temp_cap.release()
                raise ValueError(f"Could not read any frames from {video_path}")
            best_kp, best_desc = detect_features(best_frame)
            temp_cap.release()

        first_frames.append(best_frame)
        first_frames_keypoints.append(best_kp)
        first_frames_descriptors.append(best_desc)

        trajectories = {}
        match_map = {} 

        p0 = best_kp
        desc0 = best_desc

        # The caller provides a short sync clip, so this stays bounded even for
        # long source videos.
        representative_frames = extract_representative_frames(video_path, segment_duration_seconds=20.0, fps=fps)
        logger.info(f"Processing {len(representative_frames)} representative frames for {cam_name}")
        
        for frame in tqdm(representative_frames, desc=f"Analyzing {cam_name}"):
            p1, desc1 = extract_features_from_frame(frame, roi_start, roi_size)
            matches = match_features(desc0, desc1)
            
            if len(matches) > 1:
                trajectories, match_map = construct_trajectories(matches, p0, p1, trajectories, match_map)

            p0 = p1
            desc0 = desc1

        if i > 0:
            F, fund_matches, p1, p2 = _safe_fundamental_matrix(first_frames_keypoints[0], first_frames_descriptors[0], first_frames_keypoints[i], first_frames_descriptors[i])
            if F is None or F.shape != (3, 3):
                logger.warning(f"Failed to compute fundamental matrix for {cam_name}")
                filtered_trajectories = filter_trajectories(list(trajectories.values()), None)
            else:
                filtered_trajectories = filter_trajectories(list(trajectories.values()), F)
        else:
            filtered_trajectories = filter_trajectories(list(trajectories.values()), None)

        trajectories_data[cam_name] = filtered_trajectories

    ref_name = cam_ids[0]
    if len(trajectories_data) > 1:
        other_cams = [name for name in trajectories_data.keys() if name != ref_name]
        if other_cams:
            target_cam = other_cams[0]
            target_idx = cam_ids.index(target_cam)
            F, fund_matches, p1, p2 = _safe_fundamental_matrix(first_frames_keypoints[0], first_frames_descriptors[0], first_frames_keypoints[target_idx], first_frames_descriptors[target_idx])
            if F is not None and F.shape == (3, 3):
                trajectories_data[ref_name] = filter_trajectories(trajectories_data[ref_name], F)

    if not trajectories_data.get(ref_name):
        logger.warning(f"Feature sync could not build reference trajectories for {ref_name}; trying frame-similarity fallback.")
        return _estimate_offsets_by_frame_similarity(capture_files, cam_ids, fps)

    sync_dict = {ref_name: 0.0}

    # Pre-compute feature-motion offsets ONCE for all cameras before the loop.
    # Previously this was called inside the per-camera loop, causing a full
    # video scan of all cameras to repeat N-1 times redundantly (9x for 10 cams).
    try:
        cached_motion_offsets = _estimate_offsets_by_feature_motion(capture_files, cam_ids, fps)
    except Exception:
        logger.warning(
            "Feature-motion temporal lag search failed; using frame-similarity fallback for all cameras.",
            exc_info=True,
        )
        return _estimate_offsets_by_frame_similarity(capture_files, cam_ids, fps)

    for i in range(1, len(cam_ids)):
        cam_name = cam_ids[i]
        if cam_name not in trajectories_data or not trajectories_data[cam_name]:
            logger.warning(f"Feature sync could not build trajectories for {cam_name}; trying frame-similarity fallback.")
            return _estimate_offsets_by_frame_similarity(capture_files, cam_ids, fps)

        matched_trajectories = match_trajectories(trajectories_data[ref_name], trajectories_data[cam_name])
        
        if not matched_trajectories:
            logger.warning(f"Feature sync found no matched trajectories for {cam_name}; trying frame-similarity fallback.")
            return _estimate_offsets_by_frame_similarity(capture_files, cam_ids, fps)
        if len(matched_trajectories) < MIN_MATCHED_TRAJECTORIES:
            logger.warning(
                f"Feature sync found only {len(matched_trajectories)} matched trajectories for {cam_name}; "
                "trying frame-similarity fallback."
            )
            return _estimate_offsets_by_frame_similarity(capture_files, cam_ids, fps)

        logger.info(
            f"Feature sync matched {len(matched_trajectories)} trajectories for {cam_name}; "
            "using cached feature-motion offset."
        )
        offset_seconds = float(cached_motion_offsets[cam_name])
        if abs(offset_seconds) > MAX_FEATURE_OFFSET_SECONDS:
            raise ValueError(
                f"Feature sync produced implausible offset for {cam_name}: "
                f"{offset_seconds:.3f}s."
            )

        sync_dict[cam_name] = offset_seconds

    return sync_dict
