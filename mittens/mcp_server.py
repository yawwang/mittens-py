"""MCP server — expose Mittens orchestrator as an MCP server.

Allows any MCP-compatible client (Claude Desktop, Cursor, etc.) to
drive Mittens workflows via tools and read artifacts via resources.

Transport: stdio (default) or SSE.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from mittens.config import load_config
from mittens.registry import Registry
from mittens.types import MittensConfig

logger = logging.getLogger(__name__)


def create_server(
    config: MittensConfig | None = None,
    registry: Registry | None = None,
) -> Server:
    """Create and configure the MCP server with all tools and resources."""
    if config is None:
        config = load_config()
    if registry is None:
        registry = Registry(config.mittens_dir)

    server = Server("mittens", version="0.2.0")

    # -- Tools --

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="run_workflow",
                description="Run a Mittens workflow (e.g., autonomous-build)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "Workflow ID (e.g., 'autonomous-build')",
                        },
                        "mission": {
                            "type": "string",
                            "description": "The mission/task description",
                        },
                        "tier": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH"],
                            "description": "Complexity tier (auto-classified if omitted)",
                        },
                    },
                    "required": ["workflow_id", "mission"],
                },
            ),
            types.Tool(
                name="get_status",
                description="Get current project status from the status ledger",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="list_workflows",
                description="List all available Mittens workflows",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="list_talents",
                description="List all available Mittens talents",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="invoke_skill",
                description="Get instructions for a specific skill",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "Skill ID (e.g., 'write-spec')",
                        },
                    },
                    "required": ["skill_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[types.TextContent]:
        if name == "run_workflow":
            return await _handle_run_workflow(config, registry, arguments)
        elif name == "get_status":
            return _handle_get_status(config)
        elif name == "list_workflows":
            return _handle_list_workflows(registry)
        elif name == "list_talents":
            return _handle_list_talents(registry)
        elif name == "invoke_skill":
            return _handle_invoke_skill(registry, arguments)
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    # -- Resources --

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        resources = [
            types.Resource(
                uri="mittens://status-ledger",
                name="Status Ledger",
                description="Current workflow status ledger",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="mittens://config",
                name="Configuration",
                description="Current Mittens configuration",
                mimeType="application/json",
            ),
        ]

        # Add artifact resources
        artifacts_dir = Path(config.project_dir) / "artifacts"
        if artifacts_dir.exists():
            for f in sorted(artifacts_dir.iterdir()):
                if f.is_file() and f.suffix in (".md", ".json", ".txt"):
                    resources.append(types.Resource(
                        uri=f"mittens://artifacts/{f.name}",
                        name=f.stem,
                        description=f"Artifact: {f.name}",
                        mimeType="text/markdown" if f.suffix == ".md" else "application/json",
                    ))

        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> str | bytes:
        uri_str = str(uri)

        if uri_str == "mittens://status-ledger":
            ledger_path = Path(config.project_dir) / "artifacts" / "status-ledger.md"
            if ledger_path.exists():
                return ledger_path.read_text()
            return "No status ledger found."

        if uri_str == "mittens://config":
            return json.dumps({
                "mittens_dir": config.mittens_dir,
                "project_dir": config.project_dir,
                "default_model": config.default_model,
                "hook_model": config.hook_model,
                "capabilities": sorted(config.capabilities),
                "stream": config.stream,
            }, indent=2)

        if uri_str.startswith("mittens://artifacts/"):
            filename = uri_str.split("mittens://artifacts/", 1)[1]
            artifact_path = Path(config.project_dir) / "artifacts" / filename
            if artifact_path.exists():
                return artifact_path.read_text()
            return f"Artifact not found: {filename}"

        return f"Unknown resource: {uri_str}"

    return server


# -- Tool handlers --


async def _handle_run_workflow(
    config: MittensConfig,
    registry: Registry,
    arguments: dict[str, Any],
) -> list[types.TextContent]:
    """Start a workflow run."""
    from mittens.artifacts import ArtifactTracker
    from mittens.async_orchestrator import AsyncOrchestrator
    from mittens.capabilities import CapabilityResolver
    from mittens.hooks import HookRunner
    from mittens.ledger import Ledger
    from mittens.llm import LLMAdapter
    from mittens.types import ComplexityTier

    workflow_id = arguments["workflow_id"]
    mission = arguments["mission"]
    tier_str = arguments.get("tier")

    llm = LLMAdapter(model=config.default_model)
    hook_llm = LLMAdapter(model=config.hook_model)
    ledger = Ledger(config.project_dir, config.project_name, mission)
    artifacts = ArtifactTracker(config.project_dir)
    caps = CapabilityResolver(config.capabilities, registry)
    hooks = HookRunner(registry, hook_llm, config.project_dir, config.mittens_dir)

    orch = AsyncOrchestrator(
        registry=registry, llm=llm, ledger=ledger,
        artifacts=artifacts, capabilities=caps, hooks=hooks,
        project_dir=config.project_dir, stream=False, config=config,
    )

    tier = ComplexityTier[tier_str] if tier_str else None
    state = await orch.arun_workflow(workflow_id, mission, tier)

    summary = (
        f"Completed: {state.workflow_id} ({state.tier.value})\n"
        f"Phases: {state.current_phase_index + 1}\n"
        f"Iterations: {state.total_iterations}\n"
        f"Artifacts: {', '.join(state.artifacts.keys()) or 'none'}\n"
        f"Cost: {orch.cost.summary()}"
    )
    return [types.TextContent(type="text", text=summary)]


def _handle_get_status(config: MittensConfig) -> list[types.TextContent]:
    ledger_path = Path(config.project_dir) / "artifacts" / "status-ledger.md"
    if not ledger_path.exists():
        return [types.TextContent(type="text", text="No status ledger found.")]

    from mittens.ledger import extract_last_phase

    content = ledger_path.read_text()
    last_phase = extract_last_phase(content) or "unknown"

    return [types.TextContent(type="text", text=f"Last phase: {last_phase}\n\n{content[-2000:]}")]


def _handle_list_workflows(registry: Registry) -> list[types.TextContent]:
    workflows = registry.list_workflows()
    lines = []
    for wf_id in workflows:
        try:
            wf = registry.workflow(wf_id)
            phases = " → ".join(p.id for p in wf.phases)
            lines.append(f"- **{wf_id}**: {wf.description}\n  Phases: {phases}")
        except Exception:
            lines.append(f"- **{wf_id}**")
    return [types.TextContent(type="text", text="\n".join(lines) or "No workflows found.")]


def _handle_list_talents(registry: Registry) -> list[types.TextContent]:
    talents = registry.list_talents()
    lines = []
    for tid in talents:
        try:
            doc = registry.talent(tid)
            purpose = doc.frontmatter.get("purpose", "")
            lines.append(f"- **{tid}**: {purpose}")
        except Exception:
            lines.append(f"- **{tid}**")
    return [types.TextContent(type="text", text="\n".join(lines) or "No talents found.")]


def _handle_invoke_skill(
    registry: Registry, arguments: dict[str, Any]
) -> list[types.TextContent]:
    skill_id = arguments["skill_id"]
    try:
        doc = registry.skill(skill_id)
        instructions = registry.skill_instructions(skill_id)
        caps = doc.frontmatter.get("requires_capabilities", ["file_read", "file_write"])
        text = (
            f"# {doc.frontmatter.get('name', skill_id)}\n\n"
            f"**Capabilities:** {', '.join(caps)}\n\n"
            f"## Instructions\n\n{instructions}"
        )
        return [types.TextContent(type="text", text=text)]
    except FileNotFoundError:
        return [types.TextContent(type="text", text=f"Skill not found: {skill_id}")]


# -- Entry points --


async def run_stdio_server(config: MittensConfig | None = None) -> None:
    """Run the MCP server over stdio transport."""
    if config is None:
        config = load_config()
    registry = Registry(config.mittens_dir)
    server = create_server(config, registry)

    init_options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


async def run_sse_server(
    config: MittensConfig | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Run the MCP server over SSE transport."""
    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn
    except ImportError:
        raise ImportError(
            "SSE transport requires additional dependencies: "
            "pip install 'mittens-py[mcp-sse]'"
        )

    if config is None:
        config = load_config()
    registry = Registry(config.mittens_dir)
    server = create_server(config, registry)
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )

    app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages/", endpoint=sse.handle_post_message, methods=["POST"]),
    ])

    config_uvicorn = uvicorn.Config(app, host=host, port=port)
    uv_server = uvicorn.Server(config_uvicorn)
    await uv_server.serve()
