"""Append-only status ledger writer.

Produces markdown events matching the format defined in
hooks/status-ledger.md. The ledger file is the single source of truth
for what happened during a workflow execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mittens.types import LedgerEvent


def utc_now() -> str:
    """UTC timestamp in ISO format (shared utility)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_last_phase(content: str) -> str | None:
    """Extract the last phase name from ledger content."""
    for line in reversed(content.splitlines()):
        if line.startswith("**Phase:**"):
            return line.split(":", 1)[1].strip()
    return None


class Ledger:
    """Append-only markdown ledger for workflow events."""

    def __init__(
        self,
        project_dir: str | Path,
        project_name: str,
        mission: str,
        event_callback: Callable[[LedgerEvent], None] | None = None,
    ):
        self.path = Path(project_dir) / "artifacts" / "status-ledger.md"
        self._events: list[LedgerEvent] = []
        self._event_callback = event_callback
        self._ensure_header(project_name, mission)

    def _ensure_header(self, project_name: str, mission: str) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = (
                f"# Status Ledger: {project_name}\n"
                f"**Started:** {self._now()}\n"
                f"**Mission:** {mission}\n"
                f"**Status:** IN_PROGRESS\n\n---\n\n"
                f"## Event Log\n\n"
            )
            self.path.write_text(header)

    # -- Convenience methods for common events --

    def phase_start(
        self, phase_id: str, loop: int, talents: list[str]
    ) -> LedgerEvent:
        return self.log(
            "PHASE_START",
            Phase=phase_id,
            Loop=str(loop),
            Assigned_talents=", ".join(talents),
        )

    def phase_complete(
        self,
        phase_id: str,
        result: str,
        artifacts: list[str],
        warnings: list[str],
    ) -> LedgerEvent:
        return self.log(
            "PHASE_COMPLETE",
            Phase=phase_id,
            Result=result,
            Artifacts_produced=", ".join(artifacts) if artifacts else "none",
            Warnings=", ".join(warnings) if warnings else "none",
        )

    def talent_activated(
        self, talent_id: str, phase_id: str, handoff_from: str | None
    ) -> LedgerEvent:
        return self.log(
            "TALENT_ACTIVATED",
            Talent=talent_id,
            Phase=phase_id,
            Handoff_from=handoff_from or "none",
        )

    def skill_invoked(
        self,
        skill_id: str,
        talent_id: str,
        phase_id: str,
        result: str,
        duration_s: float,
    ) -> LedgerEvent:
        return self.log(
            "SKILL_INVOKED",
            Skill=skill_id,
            Invoked_by=talent_id,
            Phase=phase_id,
            Result=result,
            Duration=f"{duration_s:.1f}s",
        )

    def hook_result(
        self,
        hook_name: str,
        phase_id: str,
        result: str,
        details: str,
        action_required: str | None = None,
    ) -> LedgerEvent:
        return self.log(
            "HOOK_RESULT",
            Hook=hook_name,
            Phase=phase_id,
            Result=result,
            Details=details,
            Action_required=action_required or "none",
        )

    def loop_iteration(
        self,
        phase_id: str,
        iteration_from: int,
        iteration_to: int,
        block_reason: str,
    ) -> LedgerEvent:
        return self.log(
            "LOOP_ITERATION",
            Phase=phase_id,
            Iteration=f"{iteration_from} -> {iteration_to}",
            Block_reason=block_reason,
        )

    def cost_check(
        self, phase_id: str, cost_so_far: float, budget_remaining: float, action: str
    ) -> LedgerEvent:
        return self.log(
            "COST_CHECK",
            Phase=phase_id,
            Estimated_cost=f"${cost_so_far:.2f}",
            Budget_remaining=f"${budget_remaining:.2f}",
            Action=action,
        )

    def project_complete(
        self,
        status: str,
        total_phases: int,
        total_iterations: int,
        artifacts: list[str],
    ) -> LedgerEvent:
        return self.log(
            "PROJECT_COMPLETE",
            Final_status=status,
            Total_phases=str(total_phases),
            Total_iterations=str(total_iterations),
            Artifacts_produced=", ".join(artifacts) if artifacts else "none",
        )

    # -- Core log method --

    def log(self, event_type: str, **fields: Any) -> LedgerEvent:
        """Append a structured event to the ledger file."""
        event = LedgerEvent(
            event_type=event_type,
            timestamp=self._now(),
            fields=fields,
        )
        self._events.append(event)
        self._append_to_file(event)
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception:
                pass  # Don't let callback failures break the ledger
        return event

    @property
    def events(self) -> list[LedgerEvent]:
        """All events logged this session (in-memory)."""
        return list(self._events)

    def _append_to_file(self, event: LedgerEvent) -> None:
        lines = [f"### {event.event_type}\n"]
        lines.append(f"**Timestamp:** {event.timestamp}\n")
        for key, value in event.fields.items():
            label = key.replace("_", " ")
            lines.append(f"**{label}:** {value}\n")
        lines.append("\n")
        with open(self.path, "a") as f:
            f.writelines(lines)

    @staticmethod
    def _now() -> str:
        return utc_now()
