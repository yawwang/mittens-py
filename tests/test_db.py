"""Tests for the SQLite database layer."""

import pytest
import pytest_asyncio

from mittens.db import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.init_db()
    yield database
    await database.close()


class TestRuns:
    @pytest.mark.asyncio
    async def test_create_run(self, db):
        run_id = await db.create_run("autonomous-build", "Build a thing", "LOW")
        assert run_id

    @pytest.mark.asyncio
    async def test_get_run(self, db):
        run_id = await db.create_run("autonomous-build", "Build a thing", "LOW")
        run = await db.get_run(run_id)
        assert run is not None
        assert run["workflow_id"] == "autonomous-build"
        assert run["mission"] == "Build a thing"
        assert run["status"] == "IN_PROGRESS"

    @pytest.mark.asyncio
    async def test_update_run_status(self, db):
        run_id = await db.create_run("test", "Test", "LOW")
        await db.update_run(run_id, status="COMPLETED", total_phases=3)
        run = await db.get_run(run_id)
        assert run["status"] == "COMPLETED"
        assert run["total_phases"] == 3
        assert run["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_list_runs(self, db):
        await db.create_run("wf1", "Mission 1", "LOW")
        await db.create_run("wf2", "Mission 2", "HIGH")
        runs = await db.list_runs()
        assert len(runs) == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_run(self, db):
        run = await db.get_run("nonexistent")
        assert run is None

    @pytest.mark.asyncio
    async def test_custom_run_id(self, db):
        run_id = await db.create_run("wf", "m", "LOW", run_id="custom-id")
        assert run_id == "custom-id"
        run = await db.get_run("custom-id")
        assert run is not None


class TestEvents:
    @pytest.mark.asyncio
    async def test_add_event(self, db):
        run_id = await db.create_run("wf", "m", "LOW")
        event_id = await db.add_event(
            run_id, "PHASE_START", "2026-03-24T00:00:00Z",
            {"Phase": "orient", "Loop": "1"}, "orient",
        )
        assert event_id > 0

    @pytest.mark.asyncio
    async def test_get_events(self, db):
        run_id = await db.create_run("wf", "m", "LOW")
        await db.add_event(run_id, "PHASE_START", "2026-03-24T00:00:00Z", {"Phase": "orient"}, "orient")
        await db.add_event(run_id, "PHASE_COMPLETE", "2026-03-24T00:01:00Z", {"Phase": "orient"}, "orient")

        events = await db.get_events(run_id)
        assert len(events) == 2
        assert events[0]["event_type"] == "PHASE_START"
        assert events[0]["fields"]["Phase"] == "orient"

    @pytest.mark.asyncio
    async def test_filter_events_by_type(self, db):
        run_id = await db.create_run("wf", "m", "LOW")
        await db.add_event(run_id, "PHASE_START", "t1", {}, None)
        await db.add_event(run_id, "TALENT_ACTIVATED", "t2", {}, None)
        await db.add_event(run_id, "PHASE_COMPLETE", "t3", {}, None)

        events = await db.get_events(run_id, event_type="TALENT_ACTIVATED")
        assert len(events) == 1


class TestArtifacts:
    @pytest.mark.asyncio
    async def test_add_artifact(self, db):
        run_id = await db.create_run("wf", "m", "LOW")
        art_id = await db.add_artifact(run_id, "MISSION_BRIEF", "artifacts/mission-brief.md")
        assert art_id > 0

    @pytest.mark.asyncio
    async def test_get_artifacts(self, db):
        run_id = await db.create_run("wf", "m", "LOW")
        await db.add_artifact(run_id, "MISSION_BRIEF", "artifacts/mission-brief.md")
        await db.add_artifact(run_id, "CODE", "src/main.py")

        arts = await db.get_artifacts(run_id)
        assert len(arts) == 2
        assert arts[0]["name"] == "MISSION_BRIEF"
