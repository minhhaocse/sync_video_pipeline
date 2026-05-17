import asyncio
print("!!! DEBUG: main.py is being executed !!!", flush=True)
import logging
import traceback
import sys
from contextlib import asynccontextmanager
from pathlib import Path

print("!!! DEBUG: Importing FastAPI", flush=True)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

print("!!! DEBUG: Importing Config", flush=True)
from app.config import get_settings

print("!!! DEBUG: Importing Database", flush=True)
from app.database import engine, Base

print("!!! DEBUG: Importing Live Router (Check for directory creation)", flush=True)
from app.routers import live
print("!!! DEBUG: Live Router imported", flush=True)

print("!!! DEBUG: Importing other Routers", flush=True)
from app.routers import sessions, ws, simulate
print("!!! DEBUG: All Routers imported", flush=True)

from app.ws.manager import manager
from app.ws.redis_bridge import start_redis_subscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

from app.diag_logger import log_diag

log_diag("🚀 Backend application starting...")
log_diag(f"📂 Storage base: {settings.storage_base}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_diag("🎬 Entering lifespan startup...")
    # Startup: create tables + ensure storage dirs exist
    try:
        # Ensure directories exist (moved from module level in live.py to avoid import crashes)
        live.ensure_directories()
        log_diag(f"🔗 Attempting to connect to database at: {settings.database_url}")
        async with engine.begin() as conn:
            log_diag("✅ DB Connection established. Verifying tables...")
            await conn.run_sync(Base.metadata.create_all)
            log_diag("✅ Base tables verified. Running migrations...")
            try:
                from sqlalchemy import text
                await conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS camera_count INTEGER DEFAULT 3"))
                await conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'recording'"))
                await conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sync_strategy VARCHAR(50) DEFAULT 'auto'"))
                await conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS layout VARCHAR(50) DEFAULT 'hstack'"))
                log_diag("✅ Migrations completed.")
            except Exception as e:
                log_diag(f"⚠️ Migration warning: {e}")
                logger.warning(f"Could not alter sessions table: {e}")
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        log_diag(f"❌ DATABASE STARTUP CRITICAL ERROR: {e}")
        log_diag(traceback.format_exc())
        logger.warning(
            f"⚠️  Database unavailable at startup ({e.__class__.__name__}: {e}). "
            "Live streaming endpoints will still work. "
            "Session/upload API endpoints require a running PostgreSQL."
        )

    # Start Redis pub/sub → WebSocket bridge
    bridge_task = asyncio.create_task(start_redis_subscriber(manager))
    
    # Recover active session state
    from app.routers.live import recover_active_session
    await recover_active_session()

    logger.info("✅ Redis WS bridge started")
    logger.info("✅ VideoSync API started")
    yield
    # Shutdown
    bridge_task.cancel()
    await engine.dispose()
    logger.info("VideoSync API shut down")


app = FastAPI(
    title="VideoSync Pipeline API",
    description="Multi-camera video ingestion, synchronization, and stitching platform",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log everything
    import traceback
    error_type = exc.__class__.__name__
    error_detail = str(exc)
    stack_trace = traceback.format_exc()
    
    log_diag(f"‼️ UNHANDLED {error_type} on {request.method} {request.url.path}")
    log_diag(f"Detail: {error_detail}")
    log_diag(stack_trace)
    
    # If it's already an HTTPException, preserve its status code
    status_code = 500
    if isinstance(exc, StarletteHTTPException):
        status_code = exc.status_code
        
    return JSONResponse(
        status_code=status_code,
        content={"detail": f"{error_type}: {error_detail}"},
    )

# ── CORS ──────────────────────────────────────────────────────────────────────
cors_raw = settings.allowed_origins
origins = cors_raw if isinstance(cors_raw, list) else cors_raw.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    log_diag(f"📡 INCOMING: {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    log_diag(f"🏁 OUTGOING: {request.method} {request.url.path} -> {response.status_code}")
    return response



# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(sessions.router)
app.include_router(live.router) # MP_web live streaming logic
app.include_router(live.ws_router) # WebSocket endpoints (no prefix)
app.include_router(ws.router)
app.include_router(simulate.router)

# ── Static files for synced video playback ────────────────────────────────────
synced_dir = Path(settings.storage_base) / "synced"

synced_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/synced", StaticFiles(directory=str(synced_dir)), name="synced")

# Phase-2 master videos
master_dir = Path(settings.storage_base) / "master"
master_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/master", StaticFiles(directory=str(master_dir)), name="master")


@app.get("/health")
async def health():
    log_diag("🩺 Health check request received")
    return {"status": "ok", "version": "1.0.0"}
