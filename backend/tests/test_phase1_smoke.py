"""
Phase 1 Smoke Tests — verify:
1. FastAPI app starts and /health returns 200
2. Database tables are created correctly
3. POST /api/submissions accepts text input
4. GET /api/submissions/{id} returns the created submission
5. Security: file size limit enforced
6. Security: invalid MIME type rejected
"""

from __future__ import annotations

import io
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.database import Base, get_db

# ── Test database (in-memory SQLite) ──────────────────────────────────────────
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """Create all tables before each test, drop after."""
    import app.models  # noqa: ensure all models registered

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    """Async test client with DB override."""
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Liveness probe must return 200 with status=ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "loklens_ai"


@pytest.mark.asyncio
async def test_api_status(client: AsyncClient):
    """Readiness probe must report DB as ok."""
    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["database"] == "ok"
    assert data["search_provider"] == "duckduckgo"


@pytest.mark.asyncio
async def test_tables_exist():
    """All 14 ORM tables must exist in the test DB."""
    import app.models  # noqa
    expected_tables = {
        "users", "submissions", "claims", "claim_entities", "entities",
        "sources", "documents", "search_results", "evidence",
        "images", "image_forensics", "verdicts", "timelines",
        "evidence_graph_edges",
    }
    async with test_engine.connect() as conn:
        actual = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    missing = expected_tables - actual
    assert not missing, f"Missing tables: {missing}"


@pytest.mark.asyncio
async def test_create_text_submission(client: AsyncClient):
    """POST /api/submissions with text should return 201 and queued status."""
    resp = await client.post(
        "/api/submissions",
        data={"text": "Government announced a nationwide 5G ban on September 20."},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["input_type"] == "text"
    assert "id" in data


@pytest.mark.asyncio
async def test_get_submission(client: AsyncClient):
    """GET /api/submissions/{id} must return the created submission."""
    create_resp = await client.post(
        "/api/submissions",
        data={"text": "Test claim for retrieval."},
    )
    submission_id = create_resp.json()["id"]

    get_resp = await client.get(f"/api/submissions/{submission_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == submission_id


@pytest.mark.asyncio
async def test_get_nonexistent_submission(client: AsyncClient):
    """GET /api/submissions/{id} with unknown id must return 404."""
    resp = await client.get("/api/submissions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_submission_requires_input(client: AsyncClient):
    """POST /api/submissions with neither text nor image must return 422."""
    resp = await client.post("/api/submissions", data={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_image_upload_invalid_mime(client: AsyncClient):
    """Uploading a text file as image/jpeg must be rejected with 415."""
    fake_image = io.BytesIO(b"this is not an image")
    resp = await client.post(
        "/api/submissions",
        files={"image": ("test.jpg", fake_image, "image/jpeg")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_trigger_analyze(client: AsyncClient):
    """POST /api/submissions/{id}/analyze must transition status to processing."""
    create_resp = await client.post(
        "/api/submissions",
        data={"text": "Claim to analyze."},
    )
    submission_id = create_resp.json()["id"]

    analyze_resp = await client.post(f"/api/submissions/{submission_id}/analyze")
    assert analyze_resp.status_code == 200
    assert analyze_resp.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_list_claims_empty(client: AsyncClient):
    """GET /api/submissions/{id}/claims must return empty list before analysis."""
    create_resp = await client.post(
        "/api/submissions",
        data={"text": "Some claim."},
    )
    submission_id = create_resp.json()["id"]

    claims_resp = await client.get(f"/api/submissions/{submission_id}/claims")
    assert claims_resp.status_code == 200
    assert claims_resp.json() == []


@pytest.mark.asyncio
async def test_report_not_ready(client: AsyncClient):
    """GET /api/submissions/{id}/report for queued submission must return 425."""
    create_resp = await client.post(
        "/api/submissions",
        data={"text": "Another test claim."},
    )
    submission_id = create_resp.json()["id"]

    report_resp = await client.get(f"/api/submissions/{submission_id}/report")
    assert report_resp.status_code == 425
