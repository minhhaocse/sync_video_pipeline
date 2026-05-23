import sys
import subprocess
import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MIN_HUMAN_FRAME_RATIO = 0.15
MIN_KEYPOINT_CONFIDENCE = 0.20
POSE_MOTION_MAX_OFFSET_SECONDS = 9.0
POSE_MOTION_MIN_OVERLAP_SECONDS = 1.5
POSE_MOTION_MIN_CORRELATION = 0.30
POSE_MOTION_OVERRIDE_DISAGREEMENT_SECONDS = 1.0

# Module-level cache so the YOLO and GCN models are only loaded once per process
_sesyn_dir_cache: Path | None = None
_gcn_model_cache = None
_pose_model_cache = None
LAST_SYNC_DETAILS: dict = {}


def _patch_cuda_noops_for_cpu(torch_module) -> None:
    """
    The upstream SeSyn-Net files call .cuda() directly in a few places. In this
    app we support CPU-only Docker images, so make those calls harmless when
    CUDA is unavailable instead of failing before inference can run.
    """
    cuda_compiled = bool(getattr(torch_module.backends, "cuda", None) and torch_module.backends.cuda.is_built())
    if cuda_compiled and torch_module.cuda.is_available():
        return
    if getattr(torch_module, "_videosync_cuda_noop_patch", False):
        return

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        return self

    def module_cuda(self, device=None):
        return self

    torch_module.Tensor.cuda = tensor_cuda
    torch_module.nn.Module.cuda = module_cuda
    torch_module._videosync_cuda_noop_patch = True
    logger.info("Applied CPU compatibility patch for SeSyn-Net CUDA calls.")


def _select_torch_device(torch_module):
    cuda_compiled = bool(getattr(torch_module.backends, "cuda", None) and torch_module.backends.cuda.is_built())
    if cuda_compiled and torch_module.cuda.is_available():
        return torch_module.device("cuda")
    _patch_cuda_noops_for_cpu(torch_module)
    return torch_module.device("cpu")


def setup_sesyn_net() -> Path:
    """
    Ensures the Sync-Camera repository is cloned and its modules are in the Python path.
    Returns the path to the SeSyn-Net source directory.

    Raises:
        RuntimeError: if cloning fails
        FileNotFoundError: if the expected directory structure is not found
    """
    global _sesyn_dir_cache
    if _sesyn_dir_cache is not None:
        return _sesyn_dir_cache

    base_dir = Path(__file__).resolve().parent
    repo_dir = base_dir / "Sync-Camera"

    if not repo_dir.exists():
        logger.info("Sync-Camera repository not found locally. Attempting to clone...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/Cocobaut/Sync-Camera.git", str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone Sync-Camera repository: {e.stderr.decode('utf-8', errors='replace')}")
            raise RuntimeError(
                "Could not clone Sync-Camera repository. Ensure git is installed and network is available."
            ) from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timed out while cloning Sync-Camera repository.")

    # Flexibly locate the SeSyn-Net source — handles both flat and nested layouts
    if (repo_dir / "SeSyn-Net-main" / "network").exists():
        sesyn_main_dir = repo_dir / "SeSyn-Net-main"
    elif (repo_dir / "network").exists():
        sesyn_main_dir = repo_dir
    else:
        subdirs = [d for d in repo_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        found = [d for d in subdirs if (d / "network").exists()]
        if found:
            sesyn_main_dir = found[0]
        else:
            raise FileNotFoundError(
                f"Could not find SeSyn-Net source code in {repo_dir}. "
                "Expected a 'network' sub-directory."
            )

    if str(sesyn_main_dir) not in sys.path:
        sys.path.insert(0, str(sesyn_main_dir))

    _sesyn_dir_cache = sesyn_main_dir
    return sesyn_main_dir


def _get_models(sesyn_dir: Path):
    """
    Load and cache the YOLO-pose and Adjusted_GCN models.
    Models are loaded only once per process for performance.
    """
    global _gcn_model_cache, _pose_model_cache
    import torch
    from ultralytics import YOLO
    device = _select_torch_device(torch)

    if _pose_model_cache is None:
        yolo_weights = sesyn_dir / "yolo11s-pose.pt"
        if not yolo_weights.exists():
            yolo_weights = Path("yolo11s-pose.pt")  # Let ultralytics auto-download
        logger.info("Initializing YOLO-pose model (first use)...")
        _pose_model_cache = YOLO(str(yolo_weights))

    if _gcn_model_cache is None:
        from network.adjusted_stgcn import Adjusted_GCN

        # Try internal repo path first, then fallback to global services path
        weights_path = sesyn_dir / "model" / "cmu_syn.pth"
        if not weights_path.exists():
            # Fallback: check the parent directory where the wrapper lives
            weights_path = Path(__file__).resolve().parent / "model" / "cmu_syn.pth"
            
        if not weights_path.exists():
            raise FileNotFoundError(
                f"SeSyn-Net model weights not found. Please place cmu_syn.pth at: "
                f"{Path(__file__).resolve().parent}/model/cmu_syn.pth"
            )

        logger.info(f"Loading SeSyn-Net Adjusted_GCN model onto {device} (first use)...")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        gcn_model = Adjusted_GCN(
            in_channels=3,
            layout="coco",
            strategy="spatial",
            edge_importance_weighting=True,
        )
        gcn_model.load_state_dict(checkpoint["model"])
        _gcn_model_cache = gcn_model.to(device).eval()

    return _pose_model_cache, _gcn_model_cache


def extract_keypoints_for_video(video_path: Path, model) -> np.ndarray:
    """
    Extracts pose keypoints using YOLO-pose for every frame in a video.

    Returns:
        np.ndarray of shape (3, 17, T, 1) — (xy+conf, joints, frames, persons)
    """
    import torch
    device_obj = _select_torch_device(torch)
    device = "cuda:0" if device_obj.type == "cuda" else "cpu"
    results = model(str(video_path), stream=True, device=device, verbose=False)

    all_frames_data = []
    for result in results:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            kpts = result.keypoints.data[0].cpu().numpy()  # (17, 3)
        else:
            kpts = np.zeros((17, 3), dtype=np.float32)
        all_frames_data.append(kpts)

    if not all_frames_data:
        raise ValueError(f"No frames could be extracted from {video_path}")

    data = np.array(all_frames_data, dtype=np.float32)  # (T, 17, 3)
    data = np.transpose(data, (2, 1, 0))                # (3, 17, T)
    data = np.expand_dims(data, axis=-1)                 # (3, 17, T, 1)
    return data


def _human_presence_ratio(pose_data: np.ndarray) -> float:
    confidences = pose_data[2, :, :, 0]
    if confidences.size == 0:
        return 0.0
    frame_has_human = np.max(confidences, axis=0) >= MIN_KEYPOINT_CONFIDENCE
    return float(np.mean(frame_has_human))


def _pose_motion_signal(pose_data: np.ndarray) -> np.ndarray:
    """
    Build a view-tolerant motion signal from detected skeletons.

    SeSyn-Net's GCN matcher is good at comparing local pose windows, but the
    upstream helper drops the sign and can under-estimate large start gaps.
    This signal keeps the signed global search simple: compare how much the
    body is moving over time, not where it is in the image.
    """
    xy = pose_data[:2, :, :, 0]
    conf = pose_data[2, :, :, 0]
    values: list[float] = []

    for frame_idx in range(1, xy.shape[2]):
        valid = (
            (conf[:, frame_idx] >= MIN_KEYPOINT_CONFIDENCE)
            & (conf[:, frame_idx - 1] >= MIN_KEYPOINT_CONFIDENCE)
        )
        if int(np.sum(valid)) < 4:
            values.append(0.0)
            continue

        deltas = xy[:, valid, frame_idx] - xy[:, valid, frame_idx - 1]
        values.append(float(np.median(np.linalg.norm(deltas, axis=0))))

    signal = np.asarray(values, dtype=np.float64)
    if signal.size < 3:
        return signal

    # A tiny smoothing pass reduces detector jitter without erasing gestures.
    kernel = np.ones(3, dtype=np.float64) / 3.0
    signal = np.convolve(signal, kernel, mode="same")
    return signal


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - float(np.mean(values))) / (float(np.std(values)) + 1e-6)


def _estimate_offsets_from_pose_motion(
    cam_data: dict[str, np.ndarray],
    fps_by_cam: dict[str, float],
    cam_ids: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Estimate signed offsets by cross-correlating skeleton motion magnitude.

    Positive offsets use the app convention: trim that camera because it
    started earlier than the anchor.
    """
    ref_cam = cam_ids[0]
    ref_fps = float(fps_by_cam.get(ref_cam, 30.0) or 30.0)
    signals = {cam_id: _pose_motion_signal(cam_data[cam_id]) for cam_id in cam_ids}
    ref_signal = signals[ref_cam]
    if len(ref_signal) < int(POSE_MOTION_MIN_OVERLAP_SECONDS * ref_fps):
        raise ValueError("Pose-motion sync needs a longer reference motion signal.")

    offsets = {ref_cam: 0.0}
    confidence = {ref_cam: 1.0}
    max_lag = int(round(POSE_MOTION_MAX_OFFSET_SECONDS * ref_fps))
    min_overlap = max(6, int(round(POSE_MOTION_MIN_OVERLAP_SECONDS * ref_fps)))

    for cam_id in cam_ids[1:]:
        cam_signal = signals[cam_id]
        if len(cam_signal) < min_overlap:
            raise ValueError(f"Pose-motion sync needs a longer motion signal for {cam_id}.")

        best: tuple[float, float, int, int] | None = None
        for lag in range(-max_lag, max_lag + 1):
            ref_start = max(0, -lag)
            cam_start = max(0, lag)
            overlap = min(
                len(ref_signal) - ref_start,
                len(cam_signal) - cam_start,
            )
            if overlap < min_overlap:
                continue

            ref_window = _zscore(ref_signal[ref_start:ref_start + overlap])
            cam_window = _zscore(cam_signal[cam_start:cam_start + overlap])
            corr = float(np.mean(ref_window * cam_window))
            score = float(np.mean(np.square(ref_window - cam_window)))
            if best is None or score < best[0]:
                best = (score, corr, lag, overlap)

        if best is None:
            raise ValueError(f"Pose-motion sync could not compare {cam_id}.")

        score, corr, lag, overlap = best
        if corr < POSE_MOTION_MIN_CORRELATION:
            raise ValueError(
                f"Pose-motion sync confidence for {cam_id} was too low "
                f"(corr={corr:.3f}, score={score:.3f}, overlap={overlap} frames)."
            )

        offsets[cam_id] = float(lag / ref_fps)
        confidence[cam_id] = corr
        logger.info(
            f"Pose-motion signed offset for {cam_id}: {offsets[cam_id]:.3f}s "
            f"(lag={lag} frames, corr={corr:.3f}, overlap={overlap} frames)"
        )

    return offsets, confidence


def compute_sesyn_offsets(chunk_dir: Path, cam_ids: list[str]) -> dict[str, float]:
    """
    Compute temporal offsets using the SeSyn-Net GCN pose-based approach.

    Args:
        chunk_dir: directory containing video files named {cam_id}*.mp4
        cam_ids: ordered list of camera IDs; cam_ids[0] is the reference

    Returns:
        dict mapping cam_id -> offset_seconds (reference cam = 0.0)
    """
    import torch
    global LAST_SYNC_DETAILS
    LAST_SYNC_DETAILS = {
        "internal_method": "sesyn_gcn",
        "selection_reason": "",
    }
    _select_torch_device(torch)

    logger.info("Setting up SeSyn-Net environment...")
    sesyn_dir = setup_sesyn_net()

    try:
        from test_model import solve_least_squares_general
        from matching import corresponding
    except ImportError as e:
        logger.error(f"Failed to import SeSyn-Net modules: {e}")
        raise

    pose_model, gcn_model = _get_models(sesyn_dir)
    device = next(gcn_model.parameters()).device

    # ── Extract keypoints for every camera ─────────────────────────────────────
    cam_data: dict[str, np.ndarray] = {}
    fps_by_cam: dict[str, float] = {}

    for cid in cam_ids:
        # Find the video file (support multiple extensions)
        video_files = []
        for ext in [".mp4", ".webm", ".mov", ".mkv"]:
            video_files += list(chunk_dir.glob(f"*{cid}*{ext}"))

        if not video_files:
            logger.warning(f"No video file found for camera {cid} — skipping.")
            continue

        video_path = video_files[0]

        cap = cv2.VideoCapture(str(video_path))
        fps_by_cam[cid] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        logger.info(f"Extracting keypoints for camera {cid} ({video_path.name})...")
        cam_data[cid] = extract_keypoints_for_video(video_path, pose_model)
        human_ratio = _human_presence_ratio(cam_data[cid])
        logger.info(f"SeSyn-Net human presence ratio for {cid}: {human_ratio:.3f}")
        if human_ratio < MIN_HUMAN_FRAME_RATIO:
            raise ValueError(
                f"SeSyn-Net could not detect reliable human pose in {cid} "
                f"(human frame ratio={human_ratio:.3f}, required={MIN_HUMAN_FRAME_RATIO:.3f})."
            )

    if len(cam_data) < 2:
        raise ValueError(
            f"SeSyn-Net requires at least 2 cameras with video files; "
            f"only found: {list(cam_data.keys())}"
        )
    missing = [cid for cid in cam_ids if cid not in cam_data]
    if missing:
        raise ValueError(f"SeSyn-Net could not find videos for cameras: {missing}")

    # ── Sliding window GCN inference ────────────────────────────────────────────
    window_size = 120
    stride = 30
    root_id = cam_ids[0]

    # Use the shortest available sequence to bound the window loop
    total_frames = min(d.shape[2] for d in cam_data.values())
    if total_frames < window_size:
        raise ValueError(
            f"Videos are too short ({total_frames} frames) for SeSyn-Net "
            f"(requires at least {window_size} frames)."
        )

    # Aggregate measurements across ALL windows (mean is more robust than last-write)
    raw_measurements: dict[tuple, list[float]] = defaultdict(list)

    logger.info(f"Running sliding-window GCN inference ({total_frames} frames, window={window_size}, stride={stride})...")
    for start in range(0, total_frames - window_size + 1, stride):
        end = start + window_size

        for i, cid1 in enumerate(cam_ids):
            for j, cid2 in enumerate(cam_ids):
                if i >= j or cid1 not in cam_data or cid2 not in cam_data:
                    continue

                # Slice window: (3, 17, T, 1) -> (1, 3, T, 17, 1) [B, C, T, V, M]
                sub_d1 = cam_data[cid1][:, :, start:end, :]
                sub_d2 = cam_data[cid2][:, :, start:end, :]

                sub_d1 = np.expand_dims(np.transpose(sub_d1, (0, 2, 1, 3)), axis=0)
                sub_d2 = np.expand_dims(np.transpose(sub_d2, (0, 2, 1, 3)), axis=0)

                tensor1 = torch.tensor(sub_d1, dtype=torch.float32).to(device)
                tensor2 = torch.tensor(sub_d2, dtype=torch.float32).to(device)

                with torch.no_grad():
                    out1 = gcn_model(tensor1)
                    out2 = gcn_model(tensor2)

                label = torch.zeros(1).to(device)
                predicted_frames = corresponding(out1, out2, label)
                raw_measurements[(cid1, cid2)].append(predicted_frames.item())

    if not raw_measurements:
        raise ValueError("SeSyn-Net: no sliding window measurements could be computed.")

    # Median across windows for robustness against outlier frames
    measurements = {
        pair: float(np.median(vals)) for pair, vals in raw_measurements.items()
    }

    logger.info("Solving least-squares for global offsets...")
    opt_offsets = solve_least_squares_general(measurements, cam_ids)

    ref_frames = float(opt_offsets.get(cam_ids[0], 0.0))
    ref_fps = fps_by_cam.get(cam_ids[0], 30.0) or 30.0

    # Convert frames -> seconds and normalize to the first camera.
    final_offsets = {
        cid: float((float(opt_offsets[cid]) - ref_frames) / (fps_by_cam.get(cid) or ref_fps))
        for cid in cam_ids
        if cid in opt_offsets
    }
    final_offsets[cam_ids[0]] = 0.0
    gcn_offsets = dict(final_offsets)

    try:
        motion_offsets, motion_confidence = _estimate_offsets_from_pose_motion(
            cam_data,
            fps_by_cam,
            cam_ids,
        )
        max_disagreement = max(
            abs(float(final_offsets[cid]) - float(motion_offsets[cid]))
            for cid in cam_ids
        )
        if max_disagreement > POSE_MOTION_OVERRIDE_DISAGREEMENT_SECONDS:
            logger.warning(
                "SeSyn-Net GCN offsets disagreed with signed pose-motion sync "
                f"by {max_disagreement:.3f}s. Using signed pose-motion offsets. "
                f"gcn={final_offsets}, pose_motion={motion_offsets}, "
                f"confidence={motion_confidence}"
            )
            final_offsets = motion_offsets
            LAST_SYNC_DETAILS = {
                "internal_method": "signed_pose_motion",
                "gcn_offsets": gcn_offsets,
                "pose_motion_offsets": motion_offsets,
                "pose_motion_confidence": motion_confidence,
                "max_internal_disagreement_seconds": float(max_disagreement),
                "selection_reason": (
                    "The SeSyn-Net GCN local-window estimate disagreed with the signed "
                    "skeleton-motion timeline search, so the signed pose-motion offset was used."
                ),
            }
        else:
            logger.info(
                "SeSyn-Net GCN offsets agree with signed pose-motion validation "
                f"(max disagreement={max_disagreement:.3f}s)."
            )
            LAST_SYNC_DETAILS = {
                "internal_method": "sesyn_gcn",
                "gcn_offsets": gcn_offsets,
                "pose_motion_offsets": motion_offsets,
                "pose_motion_confidence": motion_confidence,
                "max_internal_disagreement_seconds": float(max_disagreement),
                "selection_reason": (
                    "The SeSyn-Net GCN estimate agreed with signed skeleton-motion validation."
                ),
            }
    except Exception as exc:
        logger.warning(
            f"Signed pose-motion validation failed; keeping GCN offsets: {exc}",
            exc_info=True,
        )
        LAST_SYNC_DETAILS = {
            "internal_method": "sesyn_gcn",
            "gcn_offsets": gcn_offsets,
            "selection_reason": (
                "Signed skeleton-motion validation failed, so the SeSyn-Net GCN estimate was kept."
            ),
            "validation_error": str(exc),
        }

    logger.info(f"SeSyn-Net final offsets (seconds): {final_offsets}")
    return final_offsets
