import logging
from datetime import datetime, timezone
from pathlib import Path

from app.workers.celery_app import celery_app
from app.services.sync_pipeline import run_sync_pipeline, run_full_sync_pipeline
from app.services.master_pipeline import run_master_pipeline
from app.services.stitching import StitchLayout
from app.ws.redis_bridge import publish_event_sync
from app.diag_logger import log_diag

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_chunk_set(
    self,
    session_id: str,
    chunk_index: int,
    cam_ids: list[str],
    layout: str = "hstack",
    sync_strategy: str = "auto",
) -> dict:
    """
    Phase 1 — Live/Preview pipeline.

    Celery task to process a complete set of camera chunks.
    Triggered automatically when all cameras have uploaded for a given chunk_index.
    On completion, publishes a 'chunk_done' event to Redis so the FastAPI
    WebSocket bridge can forward it to connected browser clients.
    """
    try:
        log_diag(f"👷 [Task] STARTING: session={session_id} chunk={chunk_index} strategy={sync_strategy}")
        logger.info(f"[Task] Processing session={session_id} chunk={chunk_index} cams={cam_ids} strategy={sync_strategy}")

        output_path = run_sync_pipeline(
            session_id=session_id,
            chunk_index=chunk_index,
            cam_ids=cam_ids,
            layout=StitchLayout(layout),
            strategy_name=sync_strategy,
        )

        # Build the public-facing URL for the synced video file.
        # Nginx serves /var/www/synced/ → /static/synced/{session_id}/...
        # FastAPI also mounts /static/synced/ as a static directory.
        relative_url = f"/static/synced/{session_id}/{output_path.name}"

        # Publish the event via Redis so the FastAPI WS bridge can forward it.
        publish_event_sync({
            "type": "chunk_done",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "url": relative_url,
        })

        log_diag(f"✅ [Task] COMPLETED: session={session_id} chunk={chunk_index} -> {output_path}")
        logger.info(f"[Task] ✅ chunk={chunk_index} done → {output_path}")

        return {
            "status": "completed",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "output": str(output_path),
            "strategy_used": sync_strategy,
            "url": relative_url,
        }

    except Exception as exc:
        log_diag(f"❌ [Task] FAILED: session={session_id} chunk={chunk_index}: {exc}")
        logger.error(f"[Task] Failed chunk={chunk_index} session={session_id}: {exc}", exc_info=True)
        # Notify frontend of failure as well
        publish_event_sync({
            "type": "error",
            "session_id": session_id,
            "chunk_index": chunk_index,
            "message": str(exc),
        })
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def produce_master_video(
    self,
    session_id: str,
    cam_ids: list[str],
    layout: str = "hstack",
) -> dict:
    """
    Phase 2 — Final Master pipeline.

    WHY THIS EXISTS:
        The Phase-1 pipeline processes 2-second chunks on the fly to give
        the user instant visual feedback. However, stitching tiny segments
        creates micro-stutters and audio discontinuities at chunk boundaries.

        This task runs AFTER all chunks have been processed. It:
          1. Concatenates the raw uploaded chunks per device into one long file.
          2. Applies the chunk-0 sync offset (already computed by Phase 1) once
             on the entire file — guaranteeing smooth, continuous playback.
          3. Stitches all cameras into a single high-bitrate master export.

        The result is a production-quality "final cut" with no boundary artifacts.

    Args:
        session_id: UUID string of the session.
        cam_ids:    Camera/device IDs to include.
        layout:     Stitch layout (hstack, vstack, grid_2x2).
    """
    from app.database import SyncSessionLocal
    from app.models import MasterVideo
    import uuid

    log_diag(f"🎬 [Master] STARTING: session={session_id} cams={cam_ids}")

    # -- Mark as processing in DB --
    try:
        with SyncSessionLocal() as db:
            sess_uuid = uuid.UUID(session_id)
            mv = db.query(MasterVideo).filter_by(session_id=sess_uuid).first()
            if not mv:
                mv = MasterVideo(session_id=sess_uuid)
                db.add(mv)
            mv.status = "processing"
            mv.started_at = datetime.now(timezone.utc)
            mv.error = None
            db.commit()
    except Exception as db_err:
        logger.warning(f"[Master] Could not update DB status to processing: {db_err}")

    publish_event_sync({
        "type": "master_started",
        "session_id": session_id,
        "message": "Building final master video…",
    })

    try:
        master_path = run_master_pipeline(
            session_id=session_id,
            cam_ids=cam_ids,
            layout=StitchLayout(layout),
        )

        relative_url = f"/static/master/{session_id}/{master_path.name}"

        # -- Mark as completed in DB --
        try:
            with SyncSessionLocal() as db:
                mv = db.query(MasterVideo).filter_by(session_id=uuid.UUID(session_id)).first()
                if mv:
                    mv.status = "completed"
                    mv.file_path = str(master_path)
                    mv.url = relative_url
                    mv.finished_at = datetime.now(timezone.utc)
                    db.commit()
        except Exception as db_err:
            logger.warning(f"[Master] Could not update DB status to completed: {db_err}")

        publish_event_sync({
            "type": "master_done",
            "session_id": session_id,
            "url": relative_url,
            "message": "🎬 Master video is ready!",
        })

        log_diag(f"✅ [Master] DONE: session={session_id} → {master_path}")
        return {
            "status": "completed",
            "session_id": session_id,
            "url": relative_url,
            "output": str(master_path),
        }

    except Exception as exc:
        error_msg = str(exc)
        log_diag(f"❌ [Master] FAILED: session={session_id}: {error_msg}")
        logger.error(f"[Master] Failed session={session_id}: {error_msg}", exc_info=True)

        # -- Mark as failed in DB --
        try:
            with SyncSessionLocal() as db:
                mv = db.query(MasterVideo).filter_by(session_id=uuid.UUID(session_id)).first()
                if mv:
                    mv.status = "failed"
                    mv.error = error_msg
                    mv.finished_at = datetime.now(timezone.utc)
                session = db.query(Session).filter_by(id=uuid.UUID(session_id)).first()
                if session:
                    session.status = "failed"
                db.commit()
        except Exception as db_err:
            logger.warning(f"[Master] Could not update DB status to failed: {db_err}")

        publish_event_sync({
            "type": "master_error",
            "session_id": session_id,
            "message": f"Master render failed: {error_msg}",
        })
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_full_session(
    self,
    session_id: str,
    cam_ids: list[str],
    layout: str = "hstack",
    sync_strategy: str = "auto",
) -> dict:
    """
    Process the full session: concat chunks -> sync full videos.
    This replaces both chunk-by-chunk processing and the manual master render step.
    """
    from app.database import SyncSessionLocal
    from app.models import MasterVideo
    import uuid

    try:
        log_diag(f"👷 [Full Task] STARTING: session={session_id} strategy={sync_strategy}")
        logger.info(f"[Full Task] Processing session={session_id} cams={cam_ids} strategy={sync_strategy}")

        # -- Mark as processing in DB --
        try:
            with SyncSessionLocal() as db:
                sess_uuid = uuid.UUID(session_id)
                mv = db.query(MasterVideo).filter_by(session_id=sess_uuid).first()
                if not mv:
                    mv = MasterVideo(session_id=sess_uuid)
                    db.add(mv)
                mv.status = "processing"
                mv.started_at = datetime.now(timezone.utc)
                mv.error = None
                db.commit()
        except Exception as db_err:
            logger.warning(f"[Full Task] Could not update DB status to processing: {db_err}")

        publish_event_sync({
            "type": "master_started",
            "session_id": session_id,
            "message": "Building final master video (Full Pipeline)…",
        })

        output_path = run_full_sync_pipeline(
            session_id=session_id,
            cam_ids=cam_ids,
            layout=StitchLayout(layout),
            strategy_name=sync_strategy,
        )

        relative_url = f"/static/synced/{session_id}/{output_path.name}"

        # -- Mark as completed in DB --
        try:
            from app.models import MasterVideo, Session
            with SyncSessionLocal() as db:
                mv = db.query(MasterVideo).filter_by(session_id=uuid.UUID(session_id)).first()
                if mv:
                    mv.status = "completed"
                    mv.file_path = str(output_path)
                    mv.url = relative_url
                    mv.finished_at = datetime.now(timezone.utc)
                session = db.query(Session).filter_by(id=uuid.UUID(session_id)).first()
                if session:
                    session.status = "completed"
                db.commit()
        except Exception as db_err:
            logger.warning(f"[Full Task] Could not update DB status to completed: {db_err}")

        publish_event_sync({
            "type": "master_done",
            "session_id": session_id,
            "url": relative_url,
            "message": "🎬 Master video is ready!",
        })

        log_diag(f"✅ [Full Task] COMPLETED: session={session_id} -> {output_path}")
        logger.info(f"[Full Task] ✅ session done → {output_path}")

        return {
            "status": "completed",
            "session_id": session_id,
            "output": str(output_path),
            "strategy_used": sync_strategy,
            "url": relative_url,
        }

    except Exception as exc:
        error_msg = str(exc)
        log_diag(f"❌ [Full Task] FAILED: session={session_id}: {error_msg}")
        logger.error(f"[Full Task] Failed session={session_id}: {error_msg}", exc_info=True)

        # -- Mark as failed in DB --
        try:
            with SyncSessionLocal() as db:
                mv = db.query(MasterVideo).filter_by(session_id=uuid.UUID(session_id)).first()
                if mv:
                    mv.status = "failed"
                    mv.error = error_msg
                    mv.finished_at = datetime.now(timezone.utc)
                    db.commit()
        except Exception as db_err:
            logger.warning(f"[Full Task] Could not update DB status to failed: {db_err}")

        publish_event_sync({
            "type": "master_error",
            "session_id": session_id,
            "message": f"Full sync failed: {error_msg}",
        })
        raise self.retry(exc=exc)
