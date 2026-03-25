"""Plugin API — decorators and protocols for extending Mittens.

Provides @mittens_skill and @mittens_hook decorators that plugins use
to register custom Python implementations of skills and hook checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class SkillContext:
    """Context passed to plugin skill executors."""

    phase_id: str
    talent_id: str
    artifacts: dict[str, str]
    project_dir: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result from a plugin skill execution."""

    success: bool
    output: str
    artifacts_produced: dict[str, str] = field(default_factory=dict)


@dataclass
class HookContext:
    """Context passed to plugin hook checks."""

    phase_id: str
    tier: str
    flags: dict[str, bool]
    project_dir: str
    artifacts_dir: str


class SkillExecutor(Protocol):
    """Protocol for plugin skill functions."""

    def __call__(self, context: SkillContext) -> SkillResult: ...


class HookCheck(Protocol):
    """Protocol for plugin hook functions."""

    def __call__(self, context: HookContext) -> tuple[str, str]: ...
    # Returns (result: "PASS"|"WARN"|"FAIL", reasoning: str)


@dataclass
class SkillRegistration:
    """A registered plugin skill."""

    name: str
    capabilities: list[str]
    executor: Callable[[SkillContext], SkillResult]
    description: str = ""


@dataclass
class HookRegistration:
    """A registered plugin hook check."""

    name: str
    check_type: str  # "PLUGIN"
    checker: Callable[[HookContext], tuple[str, str]]
    description: str = ""


class PluginRegistry:
    """Registration target for plugins to register skills and hooks."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillRegistration] = {}
        self._hooks: list[HookRegistration] = []

    def register_skill(self, registration: SkillRegistration) -> None:
        self._skills[registration.name] = registration

    def register_hook(self, registration: HookRegistration) -> None:
        self._hooks.append(registration)

    def get_skill(self, name: str) -> SkillRegistration | None:
        return self._skills.get(name)

    def get_hooks(self) -> list[HookRegistration]:
        return list(self._hooks)

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    @property
    def hook_names(self) -> list[str]:
        return [h.name for h in self._hooks]


# -- Decorators --

# Module-level pending registrations (collected before registry exists)
_pending_skills: list[SkillRegistration] = []
_pending_hooks: list[HookRegistration] = []


def mittens_skill(
    name: str,
    capabilities: list[str] | None = None,
    description: str = "",
):
    """Decorator to register a Python function as a Mittens skill executor.

    Usage:
        @mittens_skill("my-lint", capabilities=["bash"])
        def run_lint(context: SkillContext) -> SkillResult:
            ...
    """
    def decorator(func: Callable[[SkillContext], SkillResult]):
        reg = SkillRegistration(
            name=name,
            capabilities=capabilities or [],
            executor=func,
            description=description or func.__doc__ or "",
        )
        _pending_skills.append(reg)
        func._mittens_skill = reg  # type: ignore
        return func
    return decorator


def mittens_hook(
    name: str,
    check_type: str = "PLUGIN",
    description: str = "",
):
    """Decorator to register a Python function as a Mittens hook check.

    Usage:
        @mittens_hook("my-check")
        def check_something(context: HookContext) -> tuple[str, str]:
            return ("PASS", "All good")
    """
    def decorator(func: Callable[[HookContext], tuple[str, str]]):
        reg = HookRegistration(
            name=name,
            check_type=check_type,
            checker=func,
            description=description or func.__doc__ or "",
        )
        _pending_hooks.append(reg)
        func._mittens_hook = reg  # type: ignore
        return func
    return decorator


def flush_pending(registry: PluginRegistry) -> None:
    """Transfer pending decorator registrations to a PluginRegistry."""
    for reg in _pending_skills:
        registry.register_skill(reg)
    for reg in _pending_hooks:
        registry.register_hook(reg)
    _pending_skills.clear()
    _pending_hooks.clear()
