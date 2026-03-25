"""Artifact tracking and staleness detection.

Maintains the dependency graph from hooks/artifact-update.md and flags
downstream artifacts as stale when an upstream artifact changes.
"""

from __future__ import annotations

from pathlib import Path

# Canonical dependency graph from hooks/artifact-update.md
DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "MISSION_BRIEF": ["PRD", "REQUIREMENTS", "UX_SPEC"],
    "PRD": ["TECH_SPEC", "ADR", "TASK_BREAKDOWN", "UX_SPEC"],
    "REQUIREMENTS": ["TECH_SPEC", "ADR", "TASK_BREAKDOWN"],
    "TECH_SPEC": ["TASK_BREAKDOWN", "CODE", "TESTS"],
    "TASK_BREAKDOWN": ["CODE"],
    "CODE": ["TESTS", "DOCUMENTATION"],
}


class ArtifactTracker:
    """Track produced artifacts and detect staleness via dependency graph."""

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir)
        self.artifacts_dir = self.project_dir / "artifacts"
        self._produced: dict[str, str] = {}  # name -> path
        self._stale: set[str] = set()

    def register(self, artifact_name: str, path: str) -> None:
        """Record that an artifact was produced or updated."""
        self._produced[artifact_name] = path
        self._stale.discard(artifact_name)

    def flag_downstream(self, changed_artifact: str) -> list[str]:
        """Flag downstream dependents as potentially stale.

        Returns list of newly flagged artifact names.
        """
        dependents = DEPENDENCY_GRAPH.get(changed_artifact, [])
        newly_stale = []
        for dep in dependents:
            if dep in self._produced and dep not in self._stale:
                self._stale.add(dep)
                newly_stale.append(dep)
        return newly_stale

    def stale_artifacts(self) -> set[str]:
        """Return the set of currently stale artifact names."""
        return set(self._stale)

    def resolve(self, artifact_name: str) -> None:
        """Mark an artifact as no longer stale."""
        self._stale.discard(artifact_name)

    def exists(self, artifact_name: str) -> bool:
        """Check if an artifact file exists on disk."""
        path = self.artifacts_dir / f"{artifact_name}.md"
        return path.exists()

    def get_path(self, artifact_name: str) -> str | None:
        """Return the path of a produced artifact, or None."""
        return self._produced.get(artifact_name)

    @property
    def produced(self) -> dict[str, str]:
        """All produced artifacts: name -> path."""
        return dict(self._produced)
