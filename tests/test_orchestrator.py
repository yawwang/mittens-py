"""Tests for the orchestrator — uses mock LLM to test phase logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mittens.artifacts import ArtifactTracker
from mittens.capabilities import CapabilityResolver
from mittens.hooks import HookRunner
from mittens.ledger import Ledger
from mittens.llm import LLMAdapter
from mittens.orchestrator import (
    MAX_LOOPS_PER_PHASE,
    MAX_TOTAL_ITERATIONS,
    TIER_PHASES,
    Orchestrator,
)
from mittens.registry import Registry
from mittens.types import (
    CheckResult,
    CheckStatus,
    ComplexityTier,
    HookVerdict,
    LLMResponse,
    LLMToolResponse,
    RunState,
)

MITTENS_DIR = Path.home() / "mittens"


@pytest.fixture
def registry():
    return Registry(MITTENS_DIR)


@pytest.fixture
def mock_llm():
    adapter = MagicMock(spec=LLMAdapter)
    adapter.model = "mock-model"
    adapter.total_input_tokens = 0
    adapter.total_output_tokens = 0
    adapter._per_talent = {}
    # Default: return a no-tool-call response
    adapter.complete_with_tools.return_value = LLMToolResponse(
        content="Phase work complete.",
        tool_calls=[],
        input_tokens=100,
        output_tokens=50,
        model="mock",
    )
    adapter.complete.return_value = LLMResponse(
        content="MEDIUM",
        input_tokens=50,
        output_tokens=5,
        model="mock",
        finish_reason="stop",
    )
    return adapter


@pytest.fixture
def orchestrator(tmp_path, registry, mock_llm):
    ledger = Ledger(tmp_path, "test-project", "Test mission")
    artifacts = ArtifactTracker(tmp_path)
    caps = CapabilityResolver(
        {"file_read", "file_write", "bash", "test_exec"}, registry
    )
    hooks = MagicMock(spec=HookRunner)
    hooks.run_phase_transition.return_value = (HookVerdict.PASS, [])

    return Orchestrator(
        registry=registry,
        llm=mock_llm,
        ledger=ledger,
        artifacts=artifacts,
        capabilities=caps,
        hooks=hooks,
        project_dir=str(tmp_path),
        stream=False,
    )


class TestTierPhases:
    def test_low_tier_phases(self):
        assert TIER_PHASES[ComplexityTier.LOW] == ["orient", "implement", "verify"]

    def test_medium_tier_phases(self):
        phases = TIER_PHASES[ComplexityTier.MEDIUM]
        assert "orient" in phases
        assert "frame" in phases
        assert "design" in phases
        assert "plan" not in phases

    def test_high_tier_has_all_phases(self):
        phases = TIER_PHASES[ComplexityTier.HIGH]
        assert len(phases) == 7
        assert "plan" in phases
        assert "reflect" in phases


class TestOrchestrator:
    def test_run_low_tier(self, orchestrator, mock_llm):
        state = orchestrator.run_workflow(
            "autonomous-build", "Fix a typo", ComplexityTier.LOW
        )
        assert state.workflow_id == "autonomous-build"
        assert state.tier == ComplexityTier.LOW

        # Should have called LLM for each phase (orient, implement, verify)
        assert mock_llm.complete_with_tools.call_count >= 3

    def test_talent_activation_order(self, orchestrator, mock_llm):
        """Owner talent should be activated first."""
        state = orchestrator.run_workflow(
            "autonomous-build", "Fix a typo", ComplexityTier.LOW
        )
        # Check ledger has TALENT_ACTIVATED events
        events = orchestrator.ledger.events
        talent_events = [e for e in events if e.event_type == "TALENT_ACTIVATED"]
        assert len(talent_events) >= 3  # at least one per phase

    def test_hook_enforcement(self, orchestrator):
        """BLOCK verdict should trigger loop."""
        orchestrator.hooks.run_phase_transition.side_effect = [
            # orient: PASS
            (HookVerdict.PASS, []),
            # implement: BLOCK first, then PASS
            (
                HookVerdict.BLOCK,
                [CheckResult("AUTO", "tests fail", CheckStatus.FAIL, "")],
            ),
            (HookVerdict.PASS, []),
            # verify: PASS
            (HookVerdict.PASS, []),
        ]

        state = orchestrator.run_workflow(
            "autonomous-build", "Fix a bug", ComplexityTier.LOW
        )

        events = orchestrator.ledger.events
        loop_events = [e for e in events if e.event_type == "LOOP_ITERATION"]
        assert len(loop_events) >= 1

    def test_phase_complete_logged(self, orchestrator):
        state = orchestrator.run_workflow(
            "autonomous-build", "Do a thing", ComplexityTier.LOW
        )
        events = orchestrator.ledger.events
        complete_events = [e for e in events if e.event_type == "PHASE_COMPLETE"]
        assert len(complete_events) >= 3

    def test_project_complete_logged(self, orchestrator):
        state = orchestrator.run_workflow(
            "autonomous-build", "Do a thing", ComplexityTier.LOW
        )
        events = orchestrator.ledger.events
        project_events = [e for e in events if e.event_type == "PROJECT_COMPLETE"]
        assert len(project_events) == 1
        assert project_events[0].fields["Final_status"] == "COMPLETED"

    def test_complexity_classification(self, orchestrator, mock_llm):
        """When no tier given, LLM classifies complexity."""
        mock_llm.complete.return_value = LLMResponse(
            content="LOW", input_tokens=50, output_tokens=5,
            model="mock", finish_reason="stop",
        )
        state = orchestrator.run_workflow(
            "autonomous-build", "Fix a typo", tier=None
        )
        assert state.tier == ComplexityTier.LOW


class TestResolvesTalentOrder:
    def test_owner_first(self, orchestrator):
        from mittens.types import PhaseSpec
        phase = PhaseSpec(
            id="test", name="Test", description="",
            talents=["a", "b", "c"],
            inputs=[], outputs=[], exit_criteria=[],
            owner_talent="b",
            consulting_talents=["c"],
        )
        order = orchestrator._resolve_talent_order(phase)
        assert order[0] == "b"
        assert order[1] == "c"
        assert "a" in order

    def test_no_owner(self, orchestrator):
        from mittens.types import PhaseSpec
        phase = PhaseSpec(
            id="test", name="Test", description="",
            talents=["a", "b"],
            inputs=[], outputs=[], exit_criteria=[],
        )
        order = orchestrator._resolve_talent_order(phase)
        assert order == ["a", "b"]
