"""FastAPI web application for the Mittens runtime.

Provides REST endpoints for workflow management, run history, artifact
access, and WebSocket streaming. The web UI is served as static files.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mittens.config import load_config
from mittens.db import Database
from mittens.registry import Registry
from mittens.types import MittensConfig
from mittens.ws import EventBroadcaster, websocket_handler

logger = logging.getLogger(__name__)


# -- Request/Response models --


class RunRequest(BaseModel):
    workflow_id: str = "autonomous-build"
    mission: str
    tier: str | None = None


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


# -- Application factory --


def create_app(
    config: MittensConfig | None = None,
    db_path: str | None = None,
) -> FastAPI:
    """Create the FastAPI application with all routes."""
    if config is None:
        config = load_config()

    registry = Registry(config.mittens_dir)
    broadcaster = EventBroadcaster()

    if db_path is None:
        db_path = str(Path(config.project_dir) / "artifacts" / "mittens.db")

    db = Database(db_path)

    # Active run tasks (run_id -> asyncio.Task)
    active_runs: dict[str, asyncio.Task] = {}
    # Cancellation flags
    cancel_flags: dict[str, bool] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db.init_db()
        yield
        for task in active_runs.values():
            task.cancel()
        await db.close()

    app = FastAPI(
        title="Mittens",
        description="Portable runtime for the Mittens team system",
        version="0.3.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Workflow & talent listing --

    @app.get("/api/workflows")
    async def list_workflows():
        workflows = []
        for wf_id in registry.list_workflows():
            try:
                wf = registry.workflow(wf_id)
                workflows.append({
                    "id": wf.id,
                    "name": wf.name,
                    "description": wf.description,
                    "phases": [{"id": p.id, "name": p.name} for p in wf.phases],
                })
            except Exception:
                workflows.append({"id": wf_id, "name": wf_id, "phases": []})
        return workflows

    @app.get("/api/talents")
    async def list_talents():
        talents = []
        for tid in registry.list_talents():
            try:
                doc = registry.talent(tid)
                talents.append({
                    "id": tid,
                    "name": doc.frontmatter.get("name", tid),
                    "purpose": doc.frontmatter.get("purpose", ""),
                    "skills": doc.frontmatter.get("skills", []),
                })
            except Exception:
                talents.append({"id": tid, "name": tid})
        return talents

    # -- Run management --

    @app.post("/api/runs", response_model=RunResponse)
    async def create_run(req: RunRequest):
        run_id = str(uuid.uuid4())[:8]

        await db.create_run(
            workflow_id=req.workflow_id,
            mission=req.mission,
            tier=req.tier or "AUTO",
            run_id=run_id,
        )

        cancel_flags[run_id] = False

        async def run_workflow():
            from mittens.artifacts import ArtifactTracker
            from mittens.async_orchestrator import AsyncOrchestrator
            from mittens.capabilities import CapabilityResolver
            from mittens.hooks import HookRunner
            from mittens.ledger import Ledger
            from mittens.llm import LLMAdapter
            from mittens.types import ComplexityTier

            llm = LLMAdapter(model=config.default_model)
            hook_llm = LLMAdapter(model=config.hook_model)
            ledger = Ledger(
                config.project_dir, config.project_name, req.mission,
                event_callback=broadcaster.make_callback(run_id),
            )
            artifacts = ArtifactTracker(config.project_dir)
            caps = CapabilityResolver(config.capabilities, registry)
            hooks = HookRunner(registry, hook_llm, config.project_dir, config.mittens_dir)

            orch = AsyncOrchestrator(
                registry=registry, llm=llm, ledger=ledger,
                artifacts=artifacts, capabilities=caps, hooks=hooks,
                project_dir=config.project_dir, stream=False, config=config,
            )

            tier = ComplexityTier[req.tier] if req.tier else None

            try:
                state = await orch.arun_workflow(req.workflow_id, req.mission, tier)

                # Sync events to DB
                for event in ledger.events:
                    phase_id = event.fields.get("Phase")
                    await db.add_event(
                        run_id, event.event_type, event.timestamp,
                        event.fields, phase_id,
                    )

                await db.update_run(
                    run_id,
                    status="COMPLETED",
                    total_phases=state.current_phase_index + 1,
                    total_iterations=state.total_iterations,
                    cost_json=json.dumps(orch.cost.summary()),
                )
            except asyncio.CancelledError:
                await db.update_run(run_id, status="CANCELLED")
            except Exception as e:
                logger.error("Run %s failed: %s", run_id, e)
                await db.update_run(run_id, status="FAILED")
            finally:
                active_runs.pop(run_id, None)
                cancel_flags.pop(run_id, None)

        task = asyncio.create_task(run_workflow())
        active_runs[run_id] = task

        return RunResponse(
            run_id=run_id,
            status="IN_PROGRESS",
            message=f"Started workflow {req.workflow_id}",
        )

    @app.get("/api/runs")
    async def list_runs():
        return await db.list_runs()

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run = await db.get_run(run_id)
        if not run:
            raise HTTPException(404, f"Run {run_id} not found")
        run["is_active"] = run_id in active_runs
        return run

    @app.get("/api/runs/{run_id}/events")
    async def get_events(run_id: str, event_type: str | None = None):
        return await db.get_events(run_id, event_type)

    @app.get("/api/runs/{run_id}/artifacts")
    async def get_artifacts(run_id: str):
        return await db.get_artifacts(run_id)

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    async def get_artifact_content(run_id: str, name: str):
        arts = await db.get_artifacts(run_id)
        for art in arts:
            if art["name"] == name:
                path = Path(art["path"])
                if path.exists():
                    return {"name": name, "content": path.read_text()}
                return {"name": name, "content": "(file not found)"}
        raise HTTPException(404, f"Artifact {name} not found in run {run_id}")

    @app.get("/api/runs/{run_id}/cost")
    async def get_cost(run_id: str):
        run = await db.get_run(run_id)
        if not run:
            raise HTTPException(404)
        if run["cost_json"]:
            return json.loads(run["cost_json"])
        return {"message": "Cost data not yet available"}

    @app.delete("/api/runs/{run_id}")
    async def cancel_run(run_id: str):
        if run_id not in active_runs:
            raise HTTPException(404, "Run not active")
        active_runs[run_id].cancel()
        return {"status": "cancelled", "run_id": run_id}

    # -- WebSocket --

    @app.websocket("/ws/runs/{run_id}")
    async def ws_run_events(websocket: WebSocket, run_id: str):
        await websocket_handler(websocket, run_id, broadcaster)

    # -- Static files (web UI) --

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        @app.get("/", response_class=HTMLResponse)
        async def index():
            return (static_dir / "index.html").read_text()

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
