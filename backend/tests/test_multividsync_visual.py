from pathlib import Path

import cv2
import numpy as np

from app.services.feature_based_approach.wrapper import _estimate_offsets_by_frame_similarity
from app.services.sync_pipeline import _sync_clip_duration_for_strategy


FPS = 30
SIZE = (320, 180)


def _make_motion_frames(seconds: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for idx in range(FPS * seconds):
        frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        x = 24 + (idx * 3) % 270
        y = 90 + int(38 * np.sin(idx / 17))
        line_y = (idx * 4) % SIZE[1]
        cv2.line(frame, (0, line_y), (SIZE[0] - 1, (line_y + 60) % SIZE[1]), (150, 150, 150), 3)
        cv2.circle(frame, (x, y), 17, (255, 255, 255), -1)
        cv2.rectangle(frame, (215, 35 + (idx // 4) % 90), (260, 75 + (idx // 4) % 90), (70, 170, 255), -1)
        frames.append(frame)
    return frames


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for frame in frames:
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0


def _estimate_for_shift(tmp_path: Path, *, ref_start_seconds: int, cam_start_seconds: int) -> float:
    frames = _make_motion_frames(30)
    ref_path = tmp_path / "cam1.mp4"
    cam_path = tmp_path / "cam2.mp4"
    clip_frames = 20 * FPS
    _write_video(ref_path, frames[ref_start_seconds * FPS : ref_start_seconds * FPS + clip_frames])
    _write_video(cam_path, frames[cam_start_seconds * FPS : cam_start_seconds * FPS + clip_frames])

    offsets = _estimate_offsets_by_frame_similarity(
        {"cam1": ref_path, "cam2": cam_path},
        ["cam1", "cam2"],
        FPS,
    )
    return offsets["cam2"]


def test_frame_similarity_handles_camera_missing_first_5_seconds(tmp_path):
    assert abs(_estimate_for_shift(tmp_path, ref_start_seconds=0, cam_start_seconds=5) + 5.0) <= 0.5


def test_frame_similarity_handles_reference_missing_first_7_seconds(tmp_path):
    assert abs(_estimate_for_shift(tmp_path, ref_start_seconds=7, cam_start_seconds=0) - 7.0) <= 0.5


def test_multisync_uses_longer_discovery_clips_without_slowing_sesyn_direct():
    assert _sync_clip_duration_for_strategy("multividsynch") == 20.0
    assert _sync_clip_duration_for_strategy("auto") == 20.0
    assert _sync_clip_duration_for_strategy("sesyn_net") == 10.0
