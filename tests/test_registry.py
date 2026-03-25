"""Tests for the framework registry."""

from pathlib import Path

import pytest

from mittens.registry import Registry

MITTENS_DIR = Path.home() / "mittens"


@pytest.fixture
def registry():
    return Registry(MITTENS_DIR)


class TestRegistry:
    def test_list_talents(self, registry):
        talents = registry.list_talents()
        assert len(talents) >= 10
        assert "founder" in talents
        assert "staff-engineer" in talents
        assert "qa-engineer" in talents

    def test_list_skills(self, registry):
        skills = registry.list_skills()
        assert len(skills) >= 30
        assert "write-spec" in skills
        assert "code-review" in skills

    def test_list_workflows(self, registry):
        workflows = registry.list_workflows()
        assert len(workflows) >= 4
        assert "autonomous-build" in workflows

    def test_list_hooks(self, registry):
        hooks = registry.list_hooks()
        assert len(hooks) >= 4
        assert "phase-transition" in hooks
        assert "status-ledger" in hooks

    def test_talent_lookup(self, registry):
        doc = registry.talent("founder")
        assert doc.frontmatter["id"] == "founder"
        assert "System Prompt" in doc.sections

    def test_talent_system_prompt(self, registry):
        prompt = registry.talent_system_prompt("founder")
        assert len(prompt) > 50
        assert "You" in prompt  # second-person perspective

    def test_skill_lookup(self, registry):
        doc = registry.skill("write-spec")
        assert doc.frontmatter["id"] == "write-spec"

    def test_skill_instructions(self, registry):
        instructions = registry.skill_instructions("write-spec")
        assert len(instructions) > 50

    def test_skill_capabilities_default(self, registry):
        caps = registry.skill_capabilities("write-spec")
        assert caps == ["file_read", "file_write"]

    def test_skill_capabilities_declared(self, registry):
        caps = registry.skill_capabilities("verification-loop")
        assert "bash" in caps
        assert "test_exec" in caps

    def test_workflow_spec(self, registry):
        wf = registry.workflow("autonomous-build")
        assert wf.id == "autonomous-build"
        assert len(wf.phases) >= 3
        phase_ids = [p.id for p in wf.phases]
        assert "orient" in phase_ids
        assert "implement" in phase_ids
        assert "verify" in phase_ids

    def test_workflow_phase_has_exit_criteria(self, registry):
        wf = registry.workflow("autonomous-build")
        for phase in wf.phases:
            assert len(phase.exit_criteria) > 0, (
                f"Phase {phase.id} has no exit criteria"
            )

    def test_missing_talent_raises(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.talent("nonexistent-talent")

    def test_caching(self, registry):
        doc1 = registry.talent("founder")
        doc2 = registry.talent("founder")
        assert doc1 is doc2  # same object from cache
