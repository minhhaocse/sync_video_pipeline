import json
import logging
from pathlib import Path

import ffmpeg
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_audio_wav(video_path: Path, out_wav: Path) -> None:
    """Extract mono 16kHz audio from a video file using FFmpeg."""
    video_path = video_path.resolve()
    out_wav = out_wav.resolve()
    (
        ffmpeg
        .input(str(video_path))
        .output(str(out_wav), ar=16000, ac=1, format="wav")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def compute_offsets(chunk_dir: Path, cam_ids: list[str], reference_cam: str = None) -> dict[str, float]:
    """
    Compute time offsets for each camera relative to the reference camera.

    Args:
        chunk_dir: directory containing {cam_id}.mp4 files
        cam_ids: list of camera IDs (e.g. ["camA", "camB", "camC"])
        reference_cam: the reference camera (default: first in list)

    Returns:
        dict mapping cam_id -> offset_seconds.
        Positive values mean trim this camera; negative values mean pad/delay it.

    Raises:
        FileNotFoundError: if a video file is missing
        ValueError: if audio is silent/missing (triggers AutoSync fallback)
    """
    if reference_cam is None:
        reference_cam = cam_ids[0]

    audio_data: dict[str, np.ndarray] = {}
    sample_rate: int = 16000
    wav_paths_to_cleanup: list[Path] = []

    try:
        for cam_id in cam_ids:
            # Try multiple extensions: .webm (sim), .mp4 (live), .mov, etc.
            video_path = None
            for ext in [".webm", ".mp4", ".mov", ".mkv"]:
                test_path = (chunk_dir / f"{cam_id}{ext}").resolve()
                if test_path.exists():
                    video_path = test_path
                    break

            if not video_path:
                error_msg = f"No input file found for camera {cam_id} in {chunk_dir} (tried .webm, .mp4, etc.)"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            wav_path = chunk_dir / f"{cam_id}_audio.wav"
            wav_paths_to_cleanup.append(wav_path)
            _extract_audio_wav(video_path, wav_path)
            sr, data = wavfile.read(str(wav_path))
            sample_rate = sr

            # Normalize to float32 — handle both integer and float PCM formats
            if np.issubdtype(data.dtype, np.integer):
                audio_norm = data.astype(np.float32) / np.iinfo(data.dtype).max
            else:
                # Already float (e.g. float32 PCM from some containers)
                audio_norm = data.astype(np.float32)

            # Validate audio is not silent — silent audio gives unreliable correlations
            rms = float(np.sqrt(np.mean(audio_norm ** 2)))
            if rms < 1e-6:
                raise ValueError(
                    f"Audio for camera {cam_id} appears to be silent (RMS={rms:.2e}). "
                    "Cannot reliably compute audio-based offsets."
                )

            audio_data[cam_id] = audio_norm

        ref_audio = audio_data[reference_cam]
        offsets: dict[str, float] = {}

        for cam_id in cam_ids:
            if cam_id == reference_cam:
                offsets[cam_id] = 0.0
                continue

            cam_audio = audio_data[cam_id]

            # Cross-correlate
            correlation = correlate(ref_audio, cam_audio, mode="full")
            lag_samples = int(correlation.argmax()) - (len(cam_audio) - 1)
            # scipy's lag is positive when cam_audio must shift right to match
            # ref_audio. align_chunk uses the opposite convention: positive
            # offsets trim leading material from the camera file.
            offset_seconds = -lag_samples / sample_rate

            offsets[cam_id] = float(offset_seconds)
            logger.info(f"Offset {cam_id} vs {reference_cam}: {offset_seconds:.4f}s")

        return offsets

    finally:
        # Always clean up temp WAV files to prevent disk leaks
        for wav_path in wav_paths_to_cleanup:
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except OSError as e:
                logger.warning(f"Could not remove temp WAV file {wav_path}: {e}")


def save_offsets(offsets: dict[str, float], session_dir: Path) -> Path:
    """Persist offsets to a JSON file in the session directory."""
    offset_file = session_dir / "offset.json"
    offset_file.write_text(json.dumps(offsets, indent=2))
    return offset_file


def load_offsets(session_dir: Path) -> dict[str, float]:
    """Load offsets from the session's offset.json."""
    offset_file = session_dir / "offset.json"
    if not offset_file.exists():
        raise FileNotFoundError(f"No offset.json found in {session_dir}")
    return json.loads(offset_file.read_text())
