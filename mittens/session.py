"""Session save/resume — serialize RunState for recovery.

Writes a JSON snapshot to artifacts/session-snapshot.json after each
phase. On resume, the orchestrator skips completed phases and continues.
"""

from __future__ import annotations

import json
from pathlib import Path

from mittens.ledger import utc_now

from mittens.types import ComplexityTier, RunState, SessionSnapshot


def save_session(
    state: RunState,
    ledger_event_count: int,
    mission: str,
    project_dir: str | Path,
) -> Path:
    """Serialize current run state to a JSON snapshot."""
    snapshot = SessionSnapshot(
        workflow_id=state.workflow_id,
        mission=mission,
        tier=state.tier.value,
        current_phase_index=state.current_phase_index,
        total_iterations=state.total_iterations,
        artifacts=dict(state.artifacts),
        flags=dict(state.flags),
        budget_remaining=state.budget_remaining,
        active_talent=state.active_talent,
        ledger_event_count=ledger_event_count,
        timestamp=utc_now(),
    )

    path = Path(project_dir) / "artifacts" / "session-snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_snapshot_to_dict(snapshot), indent=2))
    return path


def load_session(project_dir: str | Path) -> SessionSnapshot:
    """Load a session snapshot from disk."""
    path = Path(project_dir) / "artifacts" / "session-snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"No session snapshot found at {path}")

    data = json.loads(path.read_text())
    return _dict_to_snapshot(data)


def restore_run_state(snapshot: SessionSnapshot) -> RunState:
    """Reconstruct a RunState from a snapshot."""
    return RunState(
        workflow_id=snapshot.workflow_id,
        tier=ComplexityTier(snapshot.tier),
        current_phase_index=snapshot.current_phase_index,
        total_iterations=snapshot.total_iterations,
        artifacts=dict(snapshot.artifacts),
        flags=dict(snapshot.flags),
        budget_remaining=snapshot.budget_remaining,
        active_talent=snapshot.active_talent,
    )


def _snapshot_to_dict(snapshot: SessionSnapshot) -> dict:
    return {
        "workflow_id": snapshot.workflow_id,
        "mission": snapshot.mission,
        "tier": snapshot.tier,
        "current_phase_index": snapshot.current_phase_index,
        "total_iterations": snapshot.total_iterations,
        "artifacts": snapshot.artifacts,
        "flags": snapshot.flags,
        "budget_remaining": snapshot.budget_remaining,
        "active_talent": snapshot.active_talent,
        "ledger_event_count": snapshot.ledger_event_count,
        "timestamp": snapshot.timestamp,
        "mittens_version": snapshot.mittens_version,
    }


def _dict_to_snapshot(data: dict) -> SessionSnapshot:
    return SessionSnapshot(
        workflow_id=data["workflow_id"],
        mission=data["mission"],
        tier=data["tier"],
        current_phase_index=data["current_phase_index"],
        total_iterations=data["total_iterations"],
        artifacts=data.get("artifacts", {}),
        flags=data.get("flags", {}),
        budget_remaining=data.get("budget_remaining"),
        active_talent=data.get("active_talent"),
        ledger_event_count=data["ledger_event_count"],
        timestamp=data["timestamp"],
        mittens_version=data.get("mittens_version", "0.1.0"),
    )
