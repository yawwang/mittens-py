"""Tests for the async orchestrator — mirrors key sync tests."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mittens.artifacts import ArtifactTracker
from mittens.async_orchestrator import AsyncOrchestrator
from mittens.capabilities import CapabilityResolver
from mittens.hooks import HookRunner
from mittens.ledger import Ledger
from mittens.llm import LLMAdapter
from mittens.registry import Registry
from mittens.types import (
    CheckResult,
    CheckStatus,
    ComplexityTier,
    HookVerdict,
    InstanceSpec,
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

    # Sync methods (used by _sync orchestrator internally)
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

    # Async methods
    adapter.acomplete_with_tools = AsyncMock(return_value=LLMToolResponse(
        content="Async phase work complete.",
        tool_calls=[],
        input_tokens=100,
        output_tokens=50,
        model="mock",
    ))
    adapter.acomplete = AsyncMock(return_value=LLMResponse(
        content="LOW",
        input_tokens=50,
        output_tokens=5,
        model="mock",
        finish_reason="stop",
    ))
    adapter.astream_with_tools = AsyncMock(return_value=LLMToolResponse(
        content="Streamed work.",
        tool_calls=[],
        input_tokens=80,
        output_tokens=40,
        model="mock",
    ))
    return adapter


@pytest.fixture
def async_orch(tmp_path, registry, mock_llm):
    ledger = Ledger(tmp_path, "test-project", "Async test mission")
    artifacts = ArtifactTracker(tmp_path)
    caps = CapabilityResolver(
        {"file_read", "file_write", "bash", "test_exec"}, registry
    )
    hooks = MagicMock(spec=HookRunner)
    hooks.run_phase_transition.return_value = (HookVerdict.PASS, [])

    return AsyncOrchestrator(
        registry=registry,
        llm=mock_llm,
        ledger=ledger,
        artifacts=artifacts,
        capabilities=caps,
        hooks=hooks,
        project_dir=str(tmp_path),
        stream=False,
        config=None,
    )


class TestAsyncOrchestrator:
    @pytest.mark.asyncio
    async def test_arun_low_tier(self, async_orch, mock_llm):
        state = await async_orch.arun_workflow(
            "autonomous-build", "Fix a typo", ComplexityTier.LOW
        )
        assert state.workflow_id == "autonomous-build"
        assert state.tier == ComplexityTier.LOW
        assert mock_llm.acomplete_with_tools.call_count >= 3

    @pytest.mark.asyncio
    async def test_talent_activation_logged(self, async_orch):
        state = await async_orch.arun_workflow(
            "autonomous-build", "Fix a typo", ComplexityTier.LOW
        )
        events = async_orch.ledger.events
        talent_events = [e for e in events if e.event_type == "TALENT_ACTIVATED"]
        assert len(talent_events) >= 3

    @pytest.mark.asyncio
    async def test_hook_enforcement(self, async_orch):
        async_orch.hooks.run_phase_transition.side_effect = [
            (HookVerdict.PASS, []),
            (HookVerdict.BLOCK, [CheckResult("AUTO", "tests fail", CheckStatus.FAIL, "")]),
            (HookVerdict.PASS, []),
            (HookVerdict.PASS, []),
        ]
        state = await async_orch.arun_workflow(
            "autonomous-build", "Fix a bug", ComplexityTier.LOW
        )
        events = async_orch.ledger.events
        loop_events = [e for e in events if e.event_type == "LOOP_ITERATION"]
        assert len(loop_events) >= 1

    @pytest.mark.asyncio
    async def test_project_complete_logged(self, async_orch):
        state = await async_orch.arun_workflow(
            "autonomous-build", "Do a thing", ComplexityTier.LOW
        )
        events = async_orch.ledger.events
        complete = [e for e in events if e.event_type == "PROJECT_COMPLETE"]
        assert len(complete) == 1

    @pytest.mark.asyncio
    async def test_auto_save_after_phase(self, async_orch, tmp_path):
        state = await async_orch.arun_workflow(
            "autonomous-build", "Test save", ComplexityTier.LOW
        )
        snapshot_path = tmp_path / "artifacts" / "session-snapshot.json"
        assert snapshot_path.exists()


class TestParseInstances:
    def test_no_instances(self, async_orch):
        from mittens.types import PhaseSpec
        phase = PhaseSpec(
            id="test", name="Test", description="",
            talents=["software-engineer", "tech-lead"],
            inputs=[], outputs=[], exit_criteria=[],
        )
        result = async_orch._parse_instances(phase)
        assert result is None

    def test_parses_instance_syntax(self, async_orch):
        from mittens.types import PhaseSpec
        phase = PhaseSpec(
            id="test", name="Test", description="",
            talents=["software-engineer#1", "software-engineer#2", "tech-lead"],
            inputs=[], outputs=[], exit_criteria=[],
        )
        result = async_orch._parse_instances(phase)
        assert result is not None
        assert len(result) == 2
        assert result[0].talent_id == "software-engineer"
        assert result[0].instance_num == 1
        assert result[1].instance_num == 2


class TestAsyncBash:
    @pytest.mark.asyncio
    async def test_arun_bash_simple(self, async_orch):
        result = await async_orch._arun_bash("echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_arun_bash_error(self, async_orch):
        result = await async_orch._arun_bash("exit 1")
        assert "Exit code: 1" in result
