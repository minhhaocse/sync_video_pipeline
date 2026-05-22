import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

import aiofiles

from app.config import get_settings
from app.database import get_db
from app.models import Session
from app.ws.manager import manager
from app.workers.tasks import process_full_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/simulate", tags=["simulate"])
settings = get_settings()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@router.post("/upload")
async def simulate_upload(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload full video files for multiple cameras to simulate a live sync session.
    Supports dynamic number of cameras (cam1, cam2, ..., camN).
    """
    try:
        form_data = await request.form()
        logger.info(f"📥 /simulate/upload Received form keys: {list(form_data.keys())}")

        session_id_str = form_data.get("session_id")
        layout = form_data.get("layout", "hstack")
        sync_strategy = form_data.get("sync_strategy", "auto")
        selected_cameras_str = form_data.get("selected_cameras", "")

        if not session_id_str:
            raise HTTPException(status_code=400, detail="Missing session_id")

        try:
            session_id = uuid.UUID(session_id_str)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid session_id format: {session_id_str}")

        session = await db.get(Session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # ── Discovery Phase ───────────────────────────────────────────────────────
        inputs = []  # list of (cam_id, UploadFile)
        processed_prefixes = set()

        # 1. Primary discovery: structured camN_id / camN_file pairs
        for key in list(form_data.keys()):
            if key.startswith("cam") and key.endswith("_id"):
                prefix = key[:-3]
                cam_id = form_data.get(key)
                file_key = f"{prefix}_file"
                cam_file = form_data.get(file_key)

                if cam_id and cam_file and hasattr(cam_file, "filename"):
                    inputs.append((str(cam_id), cam_file))
                    processed_prefixes.add(prefix)
                    logger.info(f"✅ Discovered: {prefix} -> ID={cam_id}, File={cam_file.filename}")

        # 2. Secondary discovery: files without explicit camN_id in form (fallback)
        for key in list(form_data.keys()):
            if key.startswith("cam") and key.endswith("_file"):
                prefix = key[:-5]
                if prefix not in processed_prefixes:
                    cam_file = form_data.get(key)
                    if cam_file and hasattr(cam_file, "filename"):
                        cam_id = prefix
                        inputs.append((cam_id, cam_file))
                        logger.info(f"⚠️ Discovered (fallback): {prefix} -> ID={cam_id}, File={cam_file.filename}")

        if not inputs:
            msg = f"No valid camera files found. Use keys cam1_id/cam1_file, etc. Received keys: {list(form_data.keys())}"
            logger.error(f"❌ {msg}")
            raise HTTPException(status_code=400, detail=msg)

        # ── Selection Phase ───────────────────────────────────────────────────────
        selected_cam_ids = [c.strip() for c in selected_cameras_str.split(",") if c.strip()]
        if not selected_cam_ids:
            selected_cam_ids = [cam_id for cam_id, _ in inputs]
            logger.info(f"ℹ️ No selected_cameras provided, defaulting to all: {selected_cam_ids}")
        else:
            logger.info(f"🎯 User selected cameras: {selected_cam_ids}")

        session.camera_count = len(selected_cam_ids)
        session.layout = layout
        session.sync_strategy = sync_strategy
        session.status = "processing"
        await db.commit()

        storage_base = Path(settings.storage_base).resolve()
        session_dir = storage_base / "raw" / str(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for cam_id, file in inputs:
            suffix = Path(file.filename).suffix.lower() if file.filename else ".mp4"
            ext = suffix if suffix in ALLOWED_EXTENSIONS else ".mp4"
            dest_path = session_dir / f"{cam_id}{ext}"

            async with aiofiles.open(dest_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    await f.write(chunk)

            saved_files.append(dest_path)
            logger.info(f"✅ Saved simulation source for {cam_id} → {dest_path}")

        if not saved_files:
            raise HTTPException(status_code=400, detail="No camera files were saved for simulation.")

        process_full_session.delay(
            session_id=str(session_id),
            cam_ids=selected_cam_ids,
            layout=layout,
            sync_strategy=sync_strategy,
        )

        await manager.broadcast(str(session_id), {
            "type": "master_started",
            "session_id": str(session_id),
            "message": "Building final synced video…",
        })

        return {
            "status": "success",
            "session_id": str(session_id),
            "uploaded_files": len(saved_files),
            "selected_cameras": selected_cam_ids,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Simulation upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

