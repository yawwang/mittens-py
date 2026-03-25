"""Tests for the FastAPI web application."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from mittens.api import create_app
from mittens.registry import Registry
from mittens.types import MittensConfig

MITTENS_DIR = Path.home() / "mittens"


@pytest.fixture
def config(tmp_path):
    return MittensConfig(
        mittens_dir=str(MITTENS_DIR),
        project_dir=str(tmp_path),
        project_name="test-project",
        default_model="anthropic/claude-sonnet-4-20250514",
        hook_model="anthropic/claude-haiku-4-5-20251001",
    )


@pytest.fixture
def app(config, tmp_path):
    return create_app(config, db_path=str(tmp_path / "test.db"))


@pytest_asyncio.fixture
async def client(app):
    # Manually trigger lifespan since ASGITransport doesn't
    from asgi_lifespan import LifespanManager
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


class TestWorkflowEndpoints:
    @pytest.mark.asyncio
    async def test_list_workflows(self, client):
        resp = await client.get("/api/workflows")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert any(w["id"] == "autonomous-build" for w in data)

    @pytest.mark.asyncio
    async def test_list_talents(self, client):
        resp = await client.get("/api/talents")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert any(t["id"] == "founder" for t in data)


class TestRunEndpoints:
    @pytest.mark.asyncio
    async def test_list_runs_empty(self, client):
        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_run(self, client):
        resp = await client.get("/api/runs/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_run(self, client):
        resp = await client.delete("/api/runs/nonexistent")
        assert resp.status_code == 404


class TestStaticFiles:
    @pytest.mark.asyncio
    async def test_index_page(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "Mittens" in resp.text

    @pytest.mark.asyncio
    async def test_css(self, client):
        resp = await client.get("/static/style.css")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_js(self, client):
        resp = await client.get("/static/app.js")
        assert resp.status_code == 200
