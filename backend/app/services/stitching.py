import logging
from pathlib import Path
from enum import Enum

import ffmpeg
from app.config import get_settings
from app.diag_logger import log_diag

logger = logging.getLogger(__name__)


def has_audio(input_path: Path) -> bool:
    """Check if a video file has an audio stream."""
    if not input_path.exists():
        return False
    try:
        probe = ffmpeg.probe(str(input_path))
        return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))
    except ffmpeg.Error:
        return False


class StitchLayout(str, Enum):
    HSTACK = "hstack"      # All cameras side by side horizontally
    VSTACK = "vstack"      # All cameras stacked vertically
    GRID_2x2 = "grid_2x2"  # 2x2 grid (up to 4 cameras)


def stitch_chunks(
    aligned_paths: dict[str, Path],
    output_path: Path,
    layout: StitchLayout = StitchLayout.HSTACK,
) -> Path:
    """
    Stitch multiple aligned video chunks into a single output video.

    Args:
        aligned_paths: ordered dict of cam_id -> aligned file path
        output_path: destination path for stitched video
        layout: how to arrange the camera feeds

    Returns:
        Path to the stitched output file

    Raises:
        ValueError: if an unknown layout is specified
        RuntimeError: if FFmpeg fails
    """
    storage_base = Path(get_settings().storage_base).resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cam_ids = list(aligned_paths.keys())
    n = len(cam_ids)

    # Ensure all aligned paths are absolute
    resolved_paths = {cam: Path(p).resolve() for cam, p in aligned_paths.items()}

    # Standard tile resolution — 1280x720 (HD, compatible with all players)
    T_WIDTH = 1280
    T_HEIGHT = 720

    video_streams = []
    audio_streams = []

    for cam_id in cam_ids:
        input_file = str(resolved_paths[cam_id])
        inp = ffmpeg.input(input_file)

        # Scale to tile size preserving aspect ratio, then pad with black bars
        v = (
            inp.video
            .filter("scale", T_WIDTH, T_HEIGHT, force_original_aspect_ratio="decrease")
            .filter("pad", T_WIDTH, T_HEIGHT, "(ow-iw)/2", "(oh-ih)/2", color="black")
        )
        video_streams.append(v)

        if has_audio(resolved_paths[cam_id]):
            audio_streams.append(inp.audio)

    # ── Layout assembly ────────────────────────────────────────────────────────
    if n == 1:
        combined_video = video_streams[0]
    elif layout == StitchLayout.HSTACK:
        combined_video = ffmpeg.filter(video_streams, "hstack", inputs=n)
    elif layout == StitchLayout.VSTACK:
        combined_video = ffmpeg.filter(video_streams, "vstack", inputs=n)
    elif layout == StitchLayout.GRID_2x2:
        active_streams = video_streams[:4]
        # Pad to exactly 4 tiles with a black synthetic source
        while len(active_streams) < 4:
            placeholder = (
                ffmpeg
                .input(f"color=c=black:s={T_WIDTH}x{T_HEIGHT}:r=30", f="lavfi")
                .video
            )
            active_streams.append(placeholder)
        top = ffmpeg.filter([active_streams[0], active_streams[1]], "hstack", inputs=2)
        bottom = ffmpeg.filter([active_streams[2], active_streams[3]], "hstack", inputs=2)
        combined_video = ffmpeg.filter([top, bottom], "vstack", inputs=2)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # ── Encoding ───────────────────────────────────────────────────────────────
    out_kwargs = {
        "vcodec": "libx264",
        "preset": "fast",
        "crf": 23,
        "pix_fmt": "yuv420p",  # Maximum player compatibility
        "shortest": None,      # End encoding when the shortest stream ends
    }

    if audio_streams:
        combined_audio = ffmpeg.filter(audio_streams, "amix", inputs=len(audio_streams), duration="shortest")
        out_kwargs["acodec"] = "aac"
        stream = ffmpeg.output(combined_video, combined_audio, str(output_path), **out_kwargs)
    else:
        stream = ffmpeg.output(combined_video, str(output_path), **out_kwargs)

    try:
        stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else "Unknown ffmpeg error"
        logger.error(f"FFmpeg Error in stitching:\n{stderr}")
        log_file = storage_base / "ffmpeg_error.log"
        with open(log_file, "a") as f:
            f.write(f"=== STITCHING ERROR ({output_path.name}) ===\n{stderr}\n\n")
        raise RuntimeError(f"FFmpeg stitching failed: {stderr[:500]}...")

    logger.info(f"Stitched {n} cameras -> {output_path.name} (layout={layout})")
    return output_path
