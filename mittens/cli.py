"""Click-based CLI for the Mittens runtime."""

from __future__ import annotations

import logging
import sys

import click

from mittens import __version__
from mittens.artifacts import ArtifactTracker
from mittens.capabilities import CapabilityResolver
from mittens.config import load_config
from mittens.hooks import HookRunner
from mittens.ledger import Ledger
from mittens.llm import LLMAdapter
from mittens.orchestrator import Orchestrator
from mittens.registry import Registry
from mittens.session import load_session
from mittens.types import ComplexityTier


@click.group()
@click.version_option(__version__, prog_name="mittens")
@click.option("--config", "config_path", default=None, help="Config file path")
@click.option("--model", default=None, help="Override default model")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, model: str | None, verbose: bool) -> None:
    """Mittens — portable runtime for the Mittens team system."""
    ctx.ensure_object(dict)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s: %(message)s")
    ctx.obj["config_path"] = config_path
    ctx.obj["model_override"] = model
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("mission")
@click.option(
    "--tier",
    type=click.Choice(["LOW", "MEDIUM", "HIGH"], case_sensitive=False),
    default=None,
    help="Override complexity tier (auto-classified if omitted)",
)
@click.option("--workflow", default="autonomous-build", help="Workflow to run")
@click.option("--async", "use_async", is_flag=True, help="Use async orchestrator")
@click.pass_context
def auto(ctx: click.Context, mission: str, tier: str | None, workflow: str, use_async: bool) -> None:
    """Run a full autonomous build."""
    config = load_config(
        config_path=ctx.obj["config_path"],
        model_override=ctx.obj["model_override"],
    )
    registry = Registry(config.mittens_dir)

    llm = LLMAdapter(
        model=config.default_model,
    )
    hook_llm = LLMAdapter(
        model=config.hook_model,
    )

    # Load plugins if configured
    from mittens.plugins import PluginLoader
    plugin_loader = PluginLoader(
        enabled=config.plugins_enabled or None
    )
    plugin_loader.load_all()
    plugin_reg = plugin_loader.registry if plugin_loader.loaded_plugins else None

    ledger = Ledger(config.project_dir, config.project_name, mission)
    artifacts = ArtifactTracker(config.project_dir)
    caps = CapabilityResolver(config.capabilities, registry)
    hooks = HookRunner(
        registry, hook_llm, config.project_dir, config.mittens_dir,
        plugin_registry=plugin_reg,
    )

    tier_enum = ComplexityTier[tier.upper()] if tier else None

    if use_async:
        import asyncio
        from mittens.async_orchestrator import AsyncOrchestrator

        async_orch = AsyncOrchestrator(
            registry=registry, llm=llm, ledger=ledger,
            artifacts=artifacts, capabilities=caps, hooks=hooks,
            project_dir=config.project_dir, stream=config.stream, config=config,
        )
        async_orch._sync.plugin_registry = plugin_reg
        state = asyncio.run(async_orch.arun_workflow(workflow, mission, tier_enum))
        cost_data = async_orch.cost.summary()
    else:
        orchestrator = Orchestrator(
            registry=registry, llm=llm, ledger=ledger,
            artifacts=artifacts, capabilities=caps, hooks=hooks,
            project_dir=config.project_dir, stream=config.stream, config=config,
        )
        orchestrator.plugin_registry = plugin_reg
        state = orchestrator.run_workflow(workflow, mission, tier_enum)
        cost_data = orchestrator.cost.summary()

    # Print summary
    click.echo(f"\nCompleted: {state.workflow_id} ({state.tier.value})")
    click.echo(f"Phases run: {state.current_phase_index + 1}")
    click.echo(f"Total iterations: {state.total_iterations}")
    click.echo(f"Artifacts: {', '.join(state.artifacts.keys()) or 'none'}")
    click.echo(f"\nCost: {cost_data}")
    click.echo(f"Ledger: {ledger.path}")


@cli.command()
@click.argument("component_type", type=click.Choice(["talent", "skill", "workflow", "hook"]))
@click.argument("name")
@click.pass_context
def invoke(ctx: click.Context, component_type: str, name: str) -> None:
    """Directly invoke a talent, skill, workflow, or hook."""
    config = load_config(
        config_path=ctx.obj["config_path"],
        model_override=ctx.obj["model_override"],
    )
    registry = Registry(config.mittens_dir)

    if component_type == "talent":
        doc = registry.talent(name)
        click.echo(f"Talent: {doc.frontmatter['name']}")
        click.echo(f"Purpose: {doc.frontmatter['purpose']}")
        click.echo(f"Skills: {', '.join(doc.frontmatter.get('skills', []))}")
    elif component_type == "skill":
        doc = registry.skill(name)
        click.echo(f"Skill: {doc.frontmatter['name']}")
        click.echo(f"Description: {doc.frontmatter.get('description', '')}")
        caps = doc.frontmatter.get("requires_capabilities", ["file_read", "file_write"])
        click.echo(f"Capabilities: {', '.join(caps)}")
    elif component_type == "workflow":
        wf = registry.workflow(name)
        click.echo(f"Workflow: {wf.name}")
        click.echo(f"Phases: {' -> '.join(p.id for p in wf.phases)}")
    elif component_type == "hook":
        doc = registry.hook(name)
        click.echo(f"Hook: {doc.frontmatter.get('name', name)}")


@cli.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """Run hook audit (PASS/WARN/BLOCK) without blocking."""
    config = load_config(config_path=ctx.obj["config_path"])
    registry = Registry(config.mittens_dir)

    click.echo(f"Mittens dir: {config.mittens_dir}")
    click.echo(f"Talents: {len(registry.list_talents())}")
    click.echo(f"Skills: {len(registry.list_skills())}")
    click.echo(f"Workflows: {len(registry.list_workflows())}")
    click.echo(f"Hooks: {len(registry.list_hooks())}")

    # Check for skills with undeclared capabilities
    for skill_id in registry.list_skills():
        doc = registry.skill(skill_id)
        instructions = doc.sections.get("Instructions", "")
        has_bash_refs = any(
            kw in instructions.lower()
            for kw in ["run ", "execute", "bash", "shell", "command"]
        )
        caps = doc.frontmatter.get("requires_capabilities")
        if has_bash_refs and not caps:
            click.echo(f"  WARN: {skill_id} may need requires_capabilities")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current project status from the ledger."""
    config = load_config(config_path=ctx.obj["config_path"])
    ledger_path = f"{config.project_dir}/artifacts/status-ledger.md"

    try:
        content = open(ledger_path).read()
    except FileNotFoundError:
        click.echo("No status ledger found. Run 'mittens auto' first.")
        return

    from mittens.ledger import extract_last_phase

    last_phase = extract_last_phase(content)

    click.echo(f"Ledger: {ledger_path}")
    if last_phase:
        click.echo(f"Last phase: {last_phase}")


@cli.command()
@click.pass_context
def cost(ctx: click.Context) -> None:
    """Show token usage and cost breakdown (from current ledger)."""
    config = load_config(config_path=ctx.obj["config_path"])
    ledger_path = f"{config.project_dir}/artifacts/status-ledger.md"

    try:
        content = open(ledger_path).read()
    except FileNotFoundError:
        click.echo("No status ledger found.")
        return

    # Count events
    events = content.count("### ")
    cost_events = content.count("### COST_CHECK")
    click.echo(f"Total events: {events}")
    click.echo(f"Cost checkpoints: {cost_events}")


@cli.command()
@click.pass_context
def save(ctx: click.Context) -> None:
    """Save current session state for later resume."""
    from mittens.session import save_session
    from mittens.types import RunState

    config = load_config(config_path=ctx.obj["config_path"])
    snapshot_path = config.project_dir + "/artifacts/session-snapshot.json"

    try:
        import json
        data = json.loads(open(snapshot_path).read())
        click.echo(f"Session snapshot exists: {snapshot_path}")
        click.echo(f"  Workflow: {data.get('workflow_id')}")
        click.echo(f"  Phase index: {data.get('current_phase_index')}")
        click.echo(f"  Saved at: {data.get('timestamp')}")
    except FileNotFoundError:
        click.echo("No active session to save. Run 'mittens auto' first.")
        click.echo("(Sessions are auto-saved after each phase.)")


@cli.command()
@click.pass_context
def resume(ctx: click.Context) -> None:
    """Resume from the last saved session snapshot."""
    config = load_config(
        config_path=ctx.obj["config_path"],
        model_override=ctx.obj["model_override"],
    )

    try:
        snapshot = load_session(config.project_dir)
    except FileNotFoundError:
        click.echo("No session snapshot found. Run 'mittens auto' first.")
        return

    click.echo(f"Resuming: {snapshot.workflow_id} (tier={snapshot.tier})")
    click.echo(f"  From phase index: {snapshot.current_phase_index + 1}")
    click.echo(f"  Mission: {snapshot.mission}")

    registry = Registry(config.mittens_dir)
    llm = LLMAdapter(model=config.default_model)
    hook_llm = LLMAdapter(model=config.hook_model)

    # Create ledger in append mode (don't overwrite existing)
    ledger = Ledger(config.project_dir, config.project_name, snapshot.mission)
    artifacts = ArtifactTracker(config.project_dir)
    caps = CapabilityResolver(config.capabilities, registry)
    hooks = HookRunner(registry, hook_llm, config.project_dir, config.mittens_dir)

    orchestrator = Orchestrator(
        registry=registry,
        llm=llm,
        ledger=ledger,
        artifacts=artifacts,
        capabilities=caps,
        hooks=hooks,
        project_dir=config.project_dir,
        stream=config.stream,
        config=config,
    )

    state = orchestrator.resume_workflow(snapshot)

    click.echo(f"\nCompleted: {state.workflow_id} ({state.tier.value})")
    click.echo(f"Total iterations: {state.total_iterations}")
    click.echo(f"Artifacts: {', '.join(state.artifacts.keys()) or 'none'}")
    click.echo(f"\nCost: {orchestrator.cost.summary()}")
    click.echo(f"Ledger: {ledger.path}")


@cli.command()
@click.option(
    "--transport", type=click.Choice(["stdio", "sse"]), default="stdio",
    help="MCP transport (stdio or sse)",
)
@click.option("--port", default=8080, help="Port for SSE transport")
@click.option("--host", default="0.0.0.0", help="Host for SSE transport")
@click.pass_context
def serve(ctx: click.Context, transport: str, port: int, host: str) -> None:
    """Start the MCP server for external clients."""
    import asyncio
    from mittens.mcp_server import run_sse_server, run_stdio_server

    config = load_config(
        config_path=ctx.obj["config_path"],
        model_override=ctx.obj["model_override"],
    )

    if transport == "stdio":
        click.echo("Starting MCP server (stdio transport)...", err=True)
        asyncio.run(run_stdio_server(config))
    else:
        click.echo(f"Starting MCP server (SSE transport on {host}:{port})...", err=True)
        asyncio.run(run_sse_server(config, host=host, port=port))


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8000, help="Port to listen on")
@click.pass_context
def web(ctx: click.Context, host: str, port: int) -> None:
    """Start the web UI and API server."""
    import uvicorn
    from mittens.api import create_app

    config = load_config(
        config_path=ctx.obj["config_path"],
        model_override=ctx.obj["model_override"],
    )

    app = create_app(config)
    click.echo(f"Starting Mittens web UI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.group()
def plugins() -> None:
    """Manage Mittens plugins."""
    pass


@plugins.command("list")
@click.pass_context
def plugins_list(ctx: click.Context) -> None:
    """List discovered plugins."""
    from mittens.plugins import PluginLoader

    config = load_config(config_path=ctx.obj["config_path"])
    loader = PluginLoader(enabled=config.plugins_enabled or None)
    loader.load_all()

    if not loader.loaded_plugins:
        click.echo("No plugins found.")
        return

    for info in loader.loaded_plugins:
        click.echo(f"  {info.name} (v{info.version})")
        if info.skills:
            click.echo(f"    Skills: {', '.join(info.skills)}")
        if info.hooks:
            click.echo(f"    Hooks: {', '.join(info.hooks)}")


@plugins.command("info")
@click.argument("name")
@click.pass_context
def plugins_info(ctx: click.Context, name: str) -> None:
    """Show details about a specific plugin."""
    from mittens.plugins import PluginLoader

    config = load_config(config_path=ctx.obj["config_path"])
    loader = PluginLoader(enabled=config.plugins_enabled or None)
    loader.load_all()

    for info in loader.loaded_plugins:
        if info.name == name:
            click.echo(f"Plugin: {info.name}")
            click.echo(f"Version: {info.version}")
            click.echo(f"Skills: {', '.join(info.skills) or 'none'}")
            click.echo(f"Hooks: {', '.join(info.hooks) or 'none'}")
            return

    click.echo(f"Plugin not found: {name}")
