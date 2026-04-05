# mittens-py

A portable Python runtime for the [Mittens team system](https://github.com/tinaw/mittens) — the markdown-defined framework of talents, skills, workflows, and hooks that orchestrates multi-agent software engineering.

## Why this exists

Mittens defines an entire team system in markdown: 11 cognitive roles (talents), 36 reusable capabilities (skills), 8 project playbooks (workflows), and 4 enforcement mechanisms (hooks). Today, Claude Code executes that system by interpreting the markdown files directly at runtime. This works, but it means:

- **No programmatic control.** You can't trigger a workflow from a script, CI pipeline, or web UI — only from an interactive Claude session.
- **No state inspection.** The status ledger is a markdown file. Querying "what phase are we in?" means parsing text.
- **No parallelism.** Sequential phase execution is the only option. Running two talents concurrently on separate worktrees requires manual coordination.
- **No extensibility.** Adding a custom skill or hook means editing the framework's markdown, not installing a plugin.

mittens-py solves these by providing a Python runtime that reads the same markdown files and exposes the full orchestration loop as a library, CLI, API server, and MCP server.

## Architecture

```
Mittens markdown files (source of truth)
        │
        ▼
    ┌─────────┐
    │ Registry │ ── parses talents, skills, workflows, hooks
    └────┬────┘
         │
    ┌────▼─────┐     ┌──────────┐     ┌───────┐
    │Orchestrator│───▶│LLMAdapter│───▶│litellm│───▶ any LLM provider
    └────┬─────┘     └──────────┘     └───────┘
         │
    ┌────▼────┐   ┌──────────┐   ┌────────────┐
    │  Ledger │   │  Hooks   │   │  Artifacts │
    └─────────┘   └──────────┘   └────────────┘
```

The orchestrator drives the phase loop: activate a talent, dispatch its skills as LLM tool calls, enforce hook checks at phase boundaries, log everything to the ledger, and advance to the next phase. The markdown files are never modified — they remain the single source of truth.

## Features

| Version | What it adds |
|---------|-------------|
| **v1** | Registry, parser, orchestrator, LLM adapter, ledger, hooks, CLI |
| **v1.5** | Per-talent model routing, session save/resume, streaming |
| **v2** | Async orchestrator, parallel talent instances, git worktree isolation |
| **v2.5** | MCP server (stdio + SSE transport) |
| **v3** | FastAPI web UI, SQLite history, WebSocket event streaming |
| **v3.5** | Plugin system — decorators, registry, loader |

## Installation

```bash
pip install mittens-py

# With optional features:
pip install mittens-py[web]    # FastAPI web UI + SQLite + WebSocket
pip install mittens-py[mcp]    # MCP server
pip install mittens-py[all]    # Everything
```

Requires Python 3.10+ and a Mittens framework directory (defaults to `~/mittens`).

## Quick start

```bash
# Run a full autonomous build
mittens auto "Build a REST API for user management" --tier HIGH

# Use the async orchestrator with parallel talent instances
mittens auto "Refactor auth module" --async

# Resume an interrupted session
mittens resume

# Inspect a component
mittens invoke talent staff-engineer
mittens invoke workflow autonomous-build

# Start the web UI
mittens web --port 8000

# Start the MCP server for IDE integration
mittens serve --transport stdio
```

## Configuration

Create `mittens.toml` in your project root (or pass `--config`):

```toml
[mittens]
mittens_dir = "~/mittens"
default_model = "anthropic/claude-sonnet-4-20250514"
hook_model = "anthropic/claude-haiku-4-5-20251001"

[model_overrides]
staff-engineer = "anthropic/claude-sonnet-4-20250514"
founder = "anthropic/claude-sonnet-4-20250514"

[plugins]
enabled = ["my-plugin"]
```

All LLM calls go through [litellm](https://github.com/BerriAI/litellm), so any provider it supports (OpenAI, Anthropic, Google, local models) works out of the box.

## Modules

| Module | Purpose |
|--------|---------|
| `registry.py` | Parses and indexes markdown files from the Mittens directory |
| `parser.py` | Frontmatter + section parser for markdown documents |
| `orchestrator.py` | Synchronous phase loop — talent activation, skill dispatch, hook enforcement |
| `async_orchestrator.py` | Async variant with parallel instance support |
| `llm.py` | LLM adapter (sync/async, streaming, tool calls) via litellm |
| `ledger.py` | Append-only status ledger for run events |
| `hooks.py` | Phase-transition checks — AUTO (bash), PROSE (LLM-evaluated), PLUGIN |
| `artifacts.py` | Artifact tracking and file I/O |
| `capabilities.py` | Maps skills to required tool capabilities |
| `session.py` | Save/resume workflow state to JSON snapshots |
| `config.py` | TOML configuration loader with defaults |
| `worktrees.py` | Git worktree management for parallel instances |
| `api.py` | FastAPI REST API and static web UI |
| `ws.py` | WebSocket event broadcaster for real-time streaming |
| `db.py` | SQLite read-cache for run history |
| `mcp_server.py` | MCP server with tools and resources |
| `plugin_api.py` | Plugin decorators and registry |
| `plugins.py` | Plugin discovery and loading via entry points |
| `cli.py` | Click-based CLI |
| `types.py` | Shared dataclasses and enums |

## Testing

```bash
pip install mittens-py[dev]
pytest
```

124 tests covering the registry, parser, orchestrator (sync + async), ledger, hooks, session save/resume, worktrees, plugins, API, WebSocket, MCP server, and database layer. All tests use explicit fixtures with no shared mutable state.

## License

MIT
