"""Tests for session save/resume."""

import json
from pathlib import Path

import pytest

from mittens.session import load_session, restore_run_state, save_session
from mittens.types import ComplexityTier, RunState, SessionSnapshot


@pytest.fixture
def run_state():
    return RunState(
        workflow_id="autonomous-build",
        tier=ComplexityTier.LOW,
        current_phase_index=1,
        total_iterations=3,
        artifacts={"MISSION_BRIEF": "artifacts/mission-brief.md"},
        flags={"output_is_human_facing": True},
        budget_remaining=8.50,
        active_talent="software-engineer",
    )


class TestSaveSession:
    def test_creates_snapshot_file(self, tmp_path, run_state):
        path = save_session(run_state, 10, "Build a thing", tmp_path)
        assert path.exists()
        assert path.name == "session-snapshot.json"

    def test_snapshot_contains_all_fields(self, tmp_path, run_state):
        save_session(run_state, 10, "Build a thing", tmp_path)
        data = json.loads(
            (tmp_path / "artifacts" / "session-snapshot.json").read_text()
        )
        assert data["workflow_id"] == "autonomous-build"
        assert data["mission"] == "Build a thing"
        assert data["tier"] == "LOW"
        assert data["current_phase_index"] == 1
        assert data["total_iterations"] == 3
        assert data["artifacts"] == {"MISSION_BRIEF": "artifacts/mission-brief.md"}
        assert data["flags"] == {"output_is_human_facing": True}
        assert data["budget_remaining"] == 8.50
        assert data["active_talent"] == "software-engineer"
        assert data["ledger_event_count"] == 10
        assert "timestamp" in data

    def test_creates_artifacts_directory(self, tmp_path, run_state):
        save_session(run_state, 5, "Test", tmp_path)
        assert (tmp_path / "artifacts").is_dir()

    def test_overwrites_existing_snapshot(self, tmp_path, run_state):
        save_session(run_state, 5, "First", tmp_path)
        run_state.current_phase_index = 2
        save_session(run_state, 8, "First", tmp_path)
        data = json.loads(
            (tmp_path / "artifacts" / "session-snapshot.json").read_text()
        )
        assert data["current_phase_index"] == 2
        assert data["ledger_event_count"] == 8


class TestLoadSession:
    def test_round_trip(self, tmp_path, run_state):
        save_session(run_state, 10, "Build a thing", tmp_path)
        snapshot = load_session(tmp_path)
        assert snapshot.workflow_id == "autonomous-build"
        assert snapshot.tier == "LOW"
        assert snapshot.current_phase_index == 1
        assert snapshot.total_iterations == 3
        assert snapshot.ledger_event_count == 10
        assert snapshot.mission == "Build a thing"

    def test_missing_snapshot_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path)

    def test_preserves_artifacts_and_flags(self, tmp_path, run_state):
        save_session(run_state, 5, "Test", tmp_path)
        snapshot = load_session(tmp_path)
        assert snapshot.artifacts == {"MISSION_BRIEF": "artifacts/mission-brief.md"}
        assert snapshot.flags == {"output_is_human_facing": True}


class TestRestoreRunState:
    def test_restores_state(self):
        snapshot = SessionSnapshot(
            workflow_id="autonomous-build",
            mission="Build it",
            tier="MEDIUM",
            current_phase_index=2,
            total_iterations=5,
            artifacts={"CODE": "src/main.py"},
            flags={},
            budget_remaining=5.0,
            active_talent="tech-lead",
            ledger_event_count=20,
            timestamp="2026-03-24T00:00:00+00:00",
        )
        state = restore_run_state(snapshot)
        assert state.workflow_id == "autonomous-build"
        assert state.tier == ComplexityTier.MEDIUM
        assert state.current_phase_index == 2
        assert state.total_iterations == 5
        assert state.artifacts == {"CODE": "src/main.py"}
        assert state.budget_remaining == 5.0
        assert state.active_talent == "tech-lead"

    def test_handles_none_budget(self):
        snapshot = SessionSnapshot(
            workflow_id="test",
            mission="Test",
            tier="LOW",
            current_phase_index=0,
            total_iterations=1,
            artifacts={},
            flags={},
            budget_remaining=None,
            active_talent=None,
            ledger_event_count=0,
            timestamp="2026-03-24T00:00:00+00:00",
        )
        state = restore_run_state(snapshot)
        assert state.budget_remaining is None
        assert state.active_talent is None
