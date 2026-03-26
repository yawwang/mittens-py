"""Internal transport types for the Mittens runtime.

These are ephemeral in-memory representations — the markdown files
in MITTENS_DIR are the source of truth, not these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckStatus(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class HookVerdict(Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class ComplexityTier(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ParsedDoc:
    """A parsed markdown file: frontmatter dict + body sections dict."""

    path: str
    frontmatter: dict[str, Any]
    body: str
    sections: dict[str, str]


@dataclass(frozen=True)
class PhaseSpec:
    """A single phase within a workflow."""

    id: str
    name: str
    description: str
    talents: list[str]
    inputs: list[str]
    outputs: list[str]
    exit_criteria: list[str]
    owner_talent: str | None = None
    consulting_talents: list[str] = field(default_factory=list)
    escalates_to: str | None = None
    required_skills: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowSpec:
    """A parsed workflow definition."""

    id: str
    name: str
    version: str
    description: str
    phases: list[PhaseSpec]
    body: str


@dataclass
class LedgerEvent:
    """A single event in the status ledger."""

    event_type: str
    timestamp: str
    fields: dict[str, Any]


@dataclass
class PhaseResult:
    """Outcome of running a single phase."""

    phase_id: str
    verdict: HookVerdict
    artifacts_produced: list[str]
    loop_count: int
    warnings: list[str]


@dataclass
class RunState:
    """Mutable state carried through the orchestration loop."""

    workflow_id: str
    tier: ComplexityTier
    current_phase_index: int = 0
    total_iterations: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    budget_remaining: float | None = None
    active_talent: str | None = None
    current_system_prompt: str = ""


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMToolResponse:
    """Response from an LLM call that may include tool calls."""

    content: str | None
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    model: str


@dataclass
class CheckResult:
    """Result of a single hook check."""

    check_type: str  # AUTO, PROSE, SKILL, PLUGIN
    description: str
    result: CheckStatus
    reasoning: str = ""


def categorize_checks(
    checks: list[CheckResult],
) -> tuple[list[CheckResult], list[CheckResult], list[CheckResult]]:
    """Split checks into (failed, warned, passed) groups."""
    failed = [c for c in checks if c.result == CheckStatus.FAIL]
    warned = [c for c in checks if c.result == CheckStatus.WARN]
    passed = [c for c in checks if c.result == CheckStatus.PASS]
    return failed, warned, passed


@dataclass(frozen=True)
class MittensConfig:
    """Runtime configuration."""

    mittens_dir: str
    project_dir: str
    project_name: str
    default_model: str
    hook_model: str
    model_overrides: dict[str, str] = field(default_factory=dict)
    capabilities: set[str] = field(
        default_factory=lambda: {"file_read", "file_write", "bash", "git_write", "test_exec"}
    )
    max_budget_usd: float | None = None
    stream: bool = True
    verbosity: str = "normal"
    plugins_enabled: list[str] = field(default_factory=list)
    plugin_config: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SessionSnapshot:
    """Serializable snapshot of a workflow run for save/resume."""

    workflow_id: str
    mission: str
    tier: str  # ComplexityTier.value
    current_phase_index: int
    total_iterations: int
    artifacts: dict[str, str]
    flags: dict[str, bool]
    budget_remaining: float | None
    active_talent: str | None
    ledger_event_count: int
    timestamp: str
    mittens_version: str = "0.1.0"


@dataclass
class InstanceSpec:
    """Specification for a parallel talent instance."""

    talent_id: str
    instance_num: int
    task_description: str = ""
    worktree_path: str | None = None


@dataclass
class InstanceResult:
    """Result from a parallel talent instance."""

    instance_id: str  # e.g., "software-engineer#1"
    phase_id: str
    artifacts_produced: list[str] = field(default_factory=list)
    worktree_path: str | None = None
    success: bool = True
    error: str | None = None
