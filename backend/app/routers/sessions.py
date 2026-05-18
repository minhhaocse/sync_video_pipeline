import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Session, Offset
from app.schemas import SessionCreate, SessionOut, OffsetOut
from app.diag_logger import log_diag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(data: SessionCreate, db: AsyncSession = Depends(get_db)):
    try:
        log_diag(f"📝 Received request to create session: {data.name} ({data.camera_count} cams)")
        logger.info(f"Creating session: name={data.name}, cams={data.camera_count}")
        session = Session(
            name=data.name,
            camera_count=data.camera_count,
            sync_strategy=data.sync_strategy,
            layout=data.layout,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        log_diag(f"✅ Session created successfully in DB: {session.id}")
        logger.info(f"✅ Session created: {session.id}")
        
        # Construct response with master_url (will be None for new sessions)
        response_data = {
            "id": session.id,
            "name": session.name,
            "camera_count": session.camera_count,
            "status": session.status,
            "sync_strategy": session.sync_strategy,
            "layout": session.layout,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "master_url": None,
        }
        return SessionOut(**response_data)
    except Exception as e:
        log_diag(f"❌ ERROR in create_session: {e}")
        logger.error(f"❌ Failed to create session in DB: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    try:
        from sqlalchemy.orm import joinedload
        
        log_diag(f"🔍 Listing sessions: skip={skip}, limit={limit}")
        result = await db.execute(
            select(Session)
            .order_by(Session.created_at.desc())
            .offset(skip)
            .limit(limit)
            .options(joinedload(Session.master_video))
        )
        items = result.scalars().unique().all()
        log_diag(f"✅ Found {len(items)} sessions")
        
        # Construct response with master_url included
        response_items = []
        for session in items:
            response_data = {
                "id": session.id,
                "name": session.name,
                "camera_count": session.camera_count,
                "status": session.status,
                "sync_strategy": session.sync_strategy,
                "layout": session.layout,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "master_url": session.master_video.url if session.master_video else None,
            }
            response_items.append(SessionOut(**response_data))
        
        return response_items
    except Exception as e:
        log_diag(f"❌ ERROR in list_sessions: {e}")
        import traceback
        log_diag(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    from sqlalchemy.orm import joinedload
    result = await db.execute(
        select(Session).where(Session.id == session_id).options(joinedload(Session.master_video))
    )
    session = result.scalars().unique().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Manually construct response to include master_url from the relationship
    response_data = {
        "id": session.id,
        "name": session.name,
        "camera_count": session.camera_count,
        "status": session.status,
        "sync_strategy": session.sync_strategy,
        "layout": session.layout,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "master_url": session.master_video.url if session.master_video else None,
    }
    return SessionOut(**response_data)


@router.get("/{session_id}/offsets", response_model=list[OffsetOut])
async def get_offsets(session_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Offset).where(Offset.session_id == session_id)
    )
    return result.scalars().all()


@router.get("/{session_id}/chunks")
async def get_chunks(session_id: UUID):
    from pathlib import Path
    from app.config import get_settings
    settings = get_settings()
    synced_dir = Path(settings.storage_base) / "synced" / str(session_id)
    if not synced_dir.exists():
        return []
    chunks = []
    for file in synced_dir.glob("synced_chunk_*.mp4"):
        name = file.stem
        parts = name.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            chunks.append(int(parts[2]))
    return sorted(list(set(chunks)))


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()
