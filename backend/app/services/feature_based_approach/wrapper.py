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
    match_trajectories,
    synchronize_videos
)

logger = logging.getLogger(__name__)

MAX_FEATURE_OFFSET_SECONDS = 5.0
MIN_MATCHED_TRAJECTORIES = 8


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
    sample_rate: float = 2.0,
    max_shift_seconds: float = MAX_FEATURE_OFFSET_SECONDS,
) -> dict[str, float]:
    """
    Lightweight visual fallback for silent clips.

    It compares small grayscale frame thumbnails over a bounded lag window.
    This is less ambitious than trajectory matching, but it gives us a stable
    offset estimate for same-scene clips instead of crashing on weak features.
    """
    def load_series(path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
        step = max(1, int(round(src_fps / sample_rate)))
        series: list[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thumb = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
                thumb = cv2.equalizeHist(thumb).astype(np.float32) / 255.0
                series.append(thumb.reshape(-1))
            frame_idx += 1
        cap.release()
        return series

    series_by_cam = {cam_id: load_series(capture_files[cam_id]) for cam_id in cam_ids}
    ref_series = series_by_cam[cam_ids[0]]
    if len(ref_series) < 3:
        raise ValueError("Frame-similarity fallback needs at least 3 sampled reference frames.")

    max_lag = int(round(max_shift_seconds * sample_rate))
    offsets = {cam_ids[0]: 0.0}

    for cam_id in cam_ids[1:]:
        cam_series = series_by_cam[cam_id]
        if len(cam_series) < 3:
            raise ValueError(f"Frame-similarity fallback needs at least 3 sampled frames for {cam_id}.")

        best_lag = 0
        best_score = float("inf")
        for lag in range(-max_lag, max_lag + 1):
            ref_start = max(0, -lag)
            cam_start = max(0, lag)
            overlap = min(len(ref_series) - ref_start, len(cam_series) - cam_start)
            if overlap < 3:
                continue

            diffs = [
                float(np.mean(np.abs(ref_series[ref_start + i] - cam_series[cam_start + i])))
                for i in range(overlap)
            ]
            score = float(np.median(diffs))
            if score < best_score:
                best_score = score
                best_lag = lag

        if not np.isfinite(best_score):
            raise ValueError(f"Frame-similarity fallback could not compare {cam_id}.")

        offsets[cam_id] = float(best_lag / sample_rate)
        logger.info(
            f"Frame-similarity fallback offset for {cam_id}: "
            f"{offsets[cam_id]:.3f}s (lag={best_lag}, score={best_score:.4f})"
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

        left_percent = 0.15 
        roi_height = height
        roi_start_x = int(width * left_percent)
        roi_width = width - roi_start_x
        roi_start = (0, roi_start_x)
        roi_size = (roi_height, roi_width)

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
        representative_frames = extract_representative_frames(video_path, segment_duration_seconds=10.0, fps=fps)
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

        offsets = synchronize_videos(matched_trajectories)
        offset_array = np.asarray(offsets, dtype=np.float64).reshape(-1)
        if offset_array.size < 2 or not np.all(np.isfinite(offset_array[:2])):
            logger.warning(f"Feature sync solver returned invalid offsets for {cam_name}: {offsets}; trying frame-similarity fallback.")
            return _estimate_offsets_by_frame_similarity(capture_files, cam_ids, fps)

        ref_offset = offset_array[0]
        adjusted_offset = offset_array[1] - ref_offset
        offset_seconds = float(adjusted_offset / fps)
        if abs(offset_seconds) > MAX_FEATURE_OFFSET_SECONDS:
            logger.warning(
                f"Feature sync produced implausible offset for {cam_name}: {offset_seconds:.3f}s; "
                "trying frame-similarity fallback."
            )
            return _estimate_offsets_by_frame_similarity(capture_files, cam_ids, fps)

        sync_dict[cam_name] = offset_seconds

    return sync_dict
