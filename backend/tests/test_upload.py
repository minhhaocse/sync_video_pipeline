"""
Tests for the /api/upload-chunk endpoint.
"""
import io
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

TEST_DB_URL = "sqlite+aiosqlite:///./test_videosync.db"

from app.main import app
from app.database import Base, get_db
from app.models import Session

engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def session_id(client):
    """Create a test session and return its ID."""
    resp = await client.post("/api/sessions", json={"name": "Test Session", "camera_count": 2})
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_upload_chunk_success(client, session_id, tmp_path):
    """Upload a valid chunk and verify the response."""
    dummy_video = tmp_path / "camA.mp4"
    dummy_video.write_bytes(b"\x00" * 1024)  # 1KB dummy file

    with open(dummy_video, "rb") as f:
        resp = await client.post(
            "/api/upload-chunk",
            data={
                "cam_id": "camA",
                "chunk_index": "0",
                "session_id": session_id,
            },
            files={"file": ("camA.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["cam_id"] == "camA"
    assert body["chunk_index"] == 0
    assert not body["processing_triggered"]  # Only 1 of 2 cams uploaded


@pytest.mark.asyncio
async def test_upload_invalid_file_type(client, session_id, tmp_path):
    """Reject files with unsupported extensions."""
    dummy = tmp_path / "virus.exe"
    dummy.write_bytes(b"\x00" * 100)

    with open(dummy, "rb") as f:
        resp = await client.post(
            "/api/upload-chunk",
            data={"cam_id": "camA", "chunk_index": "0", "session_id": session_id},
            files={"file": ("virus.exe", f, "application/octet-stream")},
        )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_nonexistent_session(client, tmp_path):
    """Return 404 for unknown session IDs."""
    dummy = tmp_path / "camA.mp4"
    dummy.write_bytes(b"\x00" * 100)

    with open(dummy, "rb") as f:
        resp = await client.post(
            "/api/upload-chunk",
            data={"cam_id": "camA", "chunk_index": "0", "session_id": str(uuid.uuid4())},
            files={"file": ("camA.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 404
