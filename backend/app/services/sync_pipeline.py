"""
Orchestrates the full sync pipeline for a single chunk set.
Called by the Celery task after all cameras have uploaded.
"""
import logging
import shutil
import subprocess
from pathlib import Path

from app.config import get_settings
from app.services.offset import save_offsets, load_offsets
from app.services.alignment import align_all_chunks, align_chunk
from app.services.stitching import stitch_chunks, StitchLayout
from app.services.strategies import get_sync_strategy

logger = logging.getLogger(__name__)
settings = get_settings()


def _has_audio_stream(path: Path) -> bool:
    """
    Probe a media file and return True if it contains at least one audio stream.
    Browser MediaRecorder chunks sometimes lack audio (denied mic permission,
    screen-share without audio, the very first chunk before the audio track
    starts), so we have to check before assuming `[i:a]` is mappable.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and "audio" in result.stdout


def _repair_binary_concat(raw_combined_path: Path, repaired_path: Path, has_audio: bool) -> bool:
    """Try binary concatenation + ffmpeg repair if concat filter fails."""
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+igndts+discardcorrupt",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-i", str(raw_combined_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")

    cmd.extend(["-movflags", "+faststart", str(repaired_path)])
    logger.info(
        f"Attempting binary concat repair for {raw_combined_path.name} -> {repaired_path.name}"
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            f"Binary concat repair failed: {result.stderr[-1000:]}"
        )
        return False
    return repaired_path.exists() and repaired_path.stat().st_size > 0


def _remux_to_mp4(input_path: Path, output_path: Path) -> bool:
    has_audio = _has_audio_stream(input_path)
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+igndts+discardcorrupt",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", str(output_path)])

    logger.info(f"Remuxing direct input {input_path.name} -> {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Remux failed: {result.stderr[-1000:]}")
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _concat_camera_chunks(session_dir: Path, cam_id_list: list[str]) -> dict[str, Path]:
    """
    Concatenate all chunks for each camera into a single full video.

    We use the FFmpeg 'concat' filter (not the concat demuxer) because it
    decodes all chunks into a raw stream and then re-encodes them, which
    flattens any timestamp resets from broken MKV headers in browser
    recordings.

    If any chunk for a camera is missing an audio stream, we fall back to
    a video-only concat for that camera — the alternative is FFmpeg
    aborting with "Stream specifier ':a' matches no streams".

    When no chunk directories exist, the pipeline can also process direct
    full video inputs from the session directory.

    Returns dict cam_id -> full_video_path
    """
    full_videos: dict[str, Path] = {}

    direct_video_paths: dict[str, Path] = {}
    for cam_id in cam_id_list:
        for ext in [".mp4", ".mkv", ".mov", ".webm"]:
            candidate = session_dir / f"{cam_id}{ext}"
            if candidate.exists():
                direct_video_paths[cam_id] = candidate
                break

    if len(direct_video_paths) == len(cam_id_list):
        logger.info("Direct full-video inputs detected for all requested cameras. Skipping chunk concatenation.")
        for cam_id, video_path in direct_video_paths.items():
            if video_path.suffix.lower() == ".mp4":
                full_videos[cam_id] = video_path
                continue

            repaired_path = session_dir / f"full_{cam_id}.mp4"
            if _remux_to_mp4(video_path, repaired_path):
                logger.info(f"✅ Remux succeeded for {cam_id} -> {repaired_path}")
                full_videos[cam_id] = repaired_path
            else:
                logger.error(f"❌ Remux failed for {cam_id}. Skipping this camera.")
        return full_videos

    chunk_dirs = sorted(
        [d for d in session_dir.glob("chunk_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if chunk_dirs:
        for cam_id in cam_id_list:
            chunk_paths: list[Path] = []
            for chunk_dir in chunk_dirs:
                # Check for .mkv (live) or .mp4 (manual upload)
                chunk_file = chunk_dir / f"{cam_id}.mkv"
                if not chunk_file.exists():
                    chunk_file = chunk_dir / f"{cam_id}.mp4"
                if chunk_file.exists():
                    chunk_paths.append(chunk_file)

            if not chunk_paths:
                logger.warning(f"No chunks found for cam {cam_id}")
                continue

            n = len(chunk_paths)
            repaired_path = session_dir / f"full_{cam_id}.mp4"
            logger.info(f"[{cam_id}] Starting concat-filter for {n} chunks")

            # Only include audio if EVERY input has it — concat filter requires
            # every input to expose every mapped stream type.
            all_have_audio = all(_has_audio_stream(p) for p in chunk_paths)
            if not all_have_audio:
                logger.warning(
                    f"[{cam_id}] One or more chunks lack an audio stream; "
                    f"falling back to video-only concat"
                )

            cmd: list[str] = [
                "ffmpeg", "-y",
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-analyzeduration", "100M",
                "-probesize", "100M",
            ]
            for p in chunk_paths:
                cmd.extend(["-i", str(p)])

            if all_have_audio:
                inputs = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
                filter_complex = f"{inputs}concat=n={n}:v=1:a=1[v][a]"
                maps = ["-map", "[v]", "-map", "[a]"]
                audio_args = ["-c:a", "aac", "-b:a", "128k"]
            else:
                inputs = "".join(f"[{i}:v:0]" for i in range(n))
                filter_complex = f"{inputs}concat=n={n}:v=1:a=0[v]"
                maps = ["-map", "[v]"]
                audio_args = ["-an"]

            cmd.extend([
                "-filter_complex", filter_complex,
                *maps,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                *audio_args,
                "-movflags", "+faststart",
                str(repaired_path),
            ])

            logger.info(
                f"[{cam_id}] Running concat filter ({n} inputs, audio={all_have_audio})"
            )
            res = subprocess.run(cmd, capture_output=True, text=True)

            if (
                res.returncode == 0
                and repaired_path.exists()
                and repaired_path.stat().st_size > 0
            ):
                logger.info(f"✅ Concat filter succeeded for {cam_id} -> {repaired_path}")
                full_videos[cam_id] = repaired_path
                continue

            logger.error(
                f"Concat filter failed for {cam_id} (returncode={res.returncode})\n"
                f"STDERR (last 3000 chars):\n{res.stderr[-3000:]}"
            )

            raw_combined_path = session_dir / f"raw_combined_{cam_id}.mp4"
            logger.info(f"[{cam_id}] Falling back to binary concat repair")
            with open(raw_combined_path, "wb") as outfile:
                for chunk in chunk_paths:
                    with open(chunk, "rb") as infile:
                        outfile.write(infile.read())

            if _repair_binary_concat(raw_combined_path, repaired_path, all_have_audio):
                logger.info(f"✅ Binary concat repair succeeded for {cam_id} -> {repaired_path}")
                raw_combined_path.unlink(missing_ok=True)
                full_videos[cam_id] = repaired_path
            else:
                logger.error(f"[{cam_id}] Binary concat fallback also failed")

        return full_videos

    logger.info(f"No chunk directories found in {session_dir}. Checking for direct full session files.")
    for cam_id in cam_id_list:
        video_path = None
        for ext in [".mp4", ".mkv", ".mov", ".webm"]:
            candidate = session_dir / f"{cam_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if not video_path:
            logger.warning(f"No direct full video found for cam {cam_id}")
            continue

        if video_path.suffix.lower() == ".mp4":
            full_videos[cam_id] = video_path
            continue

        repaired_path = session_dir / f"full_{cam_id}.mp4"
        if _remux_to_mp4(video_path, repaired_path):
            logger.info(f"✅ Remux succeeded for {cam_id} -> {repaired_path}")
            full_videos[cam_id] = repaired_path
        else:
            logger.error(f"❌ Remux failed for {cam_id}. Skipping this camera.")

    return full_videos


def run_full_sync_pipeline(
    session_id: str,
    cam_ids: list[str],
    layout: StitchLayout = StitchLayout.HSTACK,
    strategy_name: str = "auto",
) -> Path:
    """
    Full pipeline for full videos: concat chunks -> compute offsets -> align -> stitch.
    """
    from app.ws.redis_bridge import publish_event_sync

    storage = Path(settings.storage_base).resolve()
    logger.info(f"[{session_id}] Resolved storage base: {storage}")

    session_dir = storage / "raw" / session_id
    aligned_dir = (session_dir / "aligned").resolve()
    synced_dir = (storage / "synced" / session_id).resolve()

    synced_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    output_path = synced_dir / "synced_full.mp4"

    # Step 1: Concatenate all chunks for each camera
    logger.info(f"[{session_id}] Concatenating chunks for all cameras...")
    publish_event_sync({
        "type": "concatenating",
        "session_id": session_id,
        "message": "Combining video chunks into full videos...",
    })
    full_videos = _concat_camera_chunks(session_dir, cam_ids)
    if not full_videos:
        raise RuntimeError(
            f"[{session_id}] No full videos created during concat step. "
            f"Cameras requested: {cam_ids}. See logs above for FFmpeg stderr."
        )

    # Use only cameras that successfully produced a full video
    valid_cam_ids = list(full_videos.keys())

    # Rename full_<cam>.mp4 -> <cam>.mp4 so the offset strategy and alignment
    # step can find them with their canonical name. replace() is atomic on
    # POSIX and (unlike rename()) overwrites an existing target, which matters
    # on retries where a stale file may already be there.
    canonical_paths: dict[str, Path] = {}
    for cam_id, path in full_videos.items():
        new_path = session_dir / f"{cam_id}.mp4"
        if path != new_path:
            logger.debug(f"[{session_id}] Renaming {path.name} -> {new_path.name}")
            if new_path.exists():
                new_path.unlink(missing_ok=True)
            path.replace(new_path)
        canonical_paths[cam_id] = new_path

    # Step 2: Compute offsets using full videos
    logger.info(
        f"[{session_id}] Computing offsets from full videos "
        f"using {strategy_name} strategy..."
    )
    publish_event_sync({
        "type": "computing_offsets",
        "session_id": session_id,
        "message": f"Computing offsets using {strategy_name} strategy...",
    })
    strategy = get_sync_strategy(strategy_name)
    offsets = strategy.compute_offsets(session_dir, valid_cam_ids)
    save_offsets(offsets, session_dir)
    logger.info(f"[{session_id}] Offsets saved: {offsets}")

    # Step 3: Align full videos
    logger.info(f"[{session_id}] Aligning full videos...")
    publish_event_sync({
        "type": "aligning",
        "session_id": session_id,
        "message": "Trimming and aligning video streams...",
    })

    # Clear stale aligned files to avoid FFmpeg in-place conflict on retry
    if aligned_dir.exists():
        shutil.rmtree(aligned_dir)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    aligned_paths: dict[str, Path] = {}
    for cam_id, offset in offsets.items():
        input_path = canonical_paths.get(cam_id, session_dir / f"{cam_id}.mp4")
        aligned_file_path = aligned_dir / f"{cam_id}_aligned.mp4"
        if input_path.exists():
            # chunk_index=0 keeps repair logic happy for the full-video case;
            # see align_chunk for details.
            align_chunk(input_path, aligned_file_path, offset, chunk_index=0)
            if aligned_file_path.exists() and aligned_file_path.stat().st_size > 0:
                aligned_paths[cam_id] = aligned_file_path
            else:
                logger.error(
                    f"[{session_id}] align_chunk produced no output for {cam_id}"
                )
        else:
            logger.warning(
                f"[{session_id}] Could not find input path for alignment: {input_path}"
            )

    if not aligned_paths:
        raise RuntimeError(
            f"[{session_id}] Alignment step produced no outputs. "
            f"Cameras requested: {list(offsets.keys())}. "
            f"See logger output above for FFmpeg stderr."
        )

    # Step 4: Stitch
    logger.info(f"[{session_id}] Stitching with layout={layout}...")
    publish_event_sync({
        "type": "stitching",
        "session_id": session_id,
        "message": "Stitching videos into combined layout...",
    })
    stitch_chunks(aligned_paths, output_path, layout)

    logger.info(f"[{session_id}] ✅ Full sync done → {output_path}")
    return output_path


def run_sync_pipeline(
    session_id: str,
    chunk_index: int,
    cam_ids: list[str],
    layout: StitchLayout = StitchLayout.HSTACK,
    strategy_name: str = "auto",
) -> Path:
    """
    Full pipeline: offset discovery (chunk 0) → align → stitch.

    Returns:
        Path to the final synced output video.
    """
    from app.ws.redis_bridge import publish_event_sync

    storage = Path(settings.storage_base).resolve()
    logger.info(f"[{session_id}] Resolved storage base: {storage}")

    session_dir = storage / "raw" / session_id
    chunk_dir = (session_dir / f"chunk_{chunk_index}").resolve()
    aligned_dir = (session_dir / f"chunk_{chunk_index}_aligned").resolve()
    synced_dir = (storage / "synced" / session_id).resolve()

    logger.info(f"[{session_id}] Chunk dir: {chunk_dir}")
    synced_dir.mkdir(parents=True, exist_ok=True)

    output_path = synced_dir / f"synced_chunk_{chunk_index}.mp4"

    # Step 1: Compute offsets — only on the first chunk
    if chunk_index == 0:
        logger.info(
            f"[{session_id}] Computing offsets from chunk_0 "
            f"using {strategy_name} strategy..."
        )
        publish_event_sync({
            "type": "computing_offsets",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "message": f"Computing offsets using {strategy_name} strategy...",
        })
        strategy = get_sync_strategy(strategy_name)
        offsets = strategy.compute_offsets(chunk_dir, cam_ids)
        save_offsets(offsets, session_dir)
        logger.info(f"[{session_id}] Offsets saved: {offsets}")
    else:
        offsets = load_offsets(session_dir)
        logger.info(f"[{session_id}] Loaded existing offsets: {offsets}")

    # Step 2: Align chunks
    logger.info(f"[{session_id}] Aligning chunk_{chunk_index}...")
    publish_event_sync({
        "type": "aligning",
        "session_id": session_id,
        "chunk_index": chunk_index,
        "message": "Trimming and aligning video streams...",
    })

    # Clear stale aligned files for this chunk to avoid in-place conflicts on retry
    if aligned_dir.exists():
        shutil.rmtree(aligned_dir)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    aligned_paths = align_all_chunks(chunk_dir, aligned_dir, offsets)

    # Step 3: Stitch
    logger.info(
        f"[{session_id}] Stitching chunk_{chunk_index} with layout={layout}..."
    )
    publish_event_sync({
        "type": "stitching",
        "session_id": session_id,
        "chunk_index": chunk_index,
        "message": "Stitching videos into combined layout...",
    })
    stitch_chunks(aligned_paths, output_path, layout)

    logger.info(f"[{session_id}] ✅ chunk_{chunk_index} done → {output_path}")
    return output_path