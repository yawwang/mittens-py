"""Capability resolution and produce/execute split.

Implements the decision matrix from schemas/capability.schema.md:
- Skills declare requires_capabilities
- Runtime has available capabilities
- If required is a subset of available: direct execution
- If bash/test_exec missing but file ops available: produce/execute split
"""

from __future__ import annotations

from mittens.registry import Registry

ALL_CAPABILITIES = frozenset({
    "bash", "file_read", "file_write", "network", "git_write", "test_exec"
})


class CapabilityResolver:
    """Resolve whether skills can run directly or need a produce/execute split."""

    def __init__(self, available: set[str], registry: Registry):
        self.available = frozenset(available)
        self.registry = registry
        self._cache: dict[str, frozenset[str]] = {}

    def _required(self, skill_id: str) -> frozenset[str]:
        """Cached lookup of required capabilities for a skill."""
        if skill_id not in self._cache:
            self._cache[skill_id] = frozenset(self.registry.skill_capabilities(skill_id))
        return self._cache[skill_id]

    def can_execute(self, skill_id: str) -> bool:
        """Check if the current runtime can fully execute this skill."""
        return self._required(skill_id).issubset(self.available)

    def needs_split(self, skill_id: str) -> bool:
        """Check if this skill needs produce/execute split.

        Split is needed when bash or test_exec is missing but file
        operations are available.
        """
        missing = self._required(skill_id) - self.available
        has_file_ops = {"file_read", "file_write"}.issubset(self.available)
        return bool(missing & {"bash", "test_exec"}) and has_file_ops

    def missing_capabilities(self, skill_id: str) -> set[str]:
        """Return capabilities required by the skill but missing at runtime."""
        return set(self._required(skill_id) - self.available)

    def split_plan(self, skill_id: str) -> tuple[list[str], list[str]]:
        """Return (agent_capabilities, orchestrator_capabilities) for a split.

        Agent does: file_read, file_write, network (produce)
        Orchestrator does: bash, test_exec, git_write (execute)
        """
        required = self._required(skill_id)
        agent_caps = sorted(required & self.available)
        orch_caps = sorted(required - self.available)
        return agent_caps, orch_caps
