"""Tests for the status ledger writer."""

import re
from pathlib import Path

import pytest

from mittens.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path, "test-project", "Build a thing")


class TestLedger:
    def test_creates_ledger_file(self, ledger):
        assert ledger.path.exists()
        content = ledger.path.read_text()
        assert "# Status Ledger: test-project" in content
        assert "**Mission:** Build a thing" in content
        assert "**Status:** IN_PROGRESS" in content

    def test_phase_start(self, ledger):
        ledger.phase_start("orient", 1, ["founder", "research-agent"])
        content = ledger.path.read_text()
        assert "### PHASE_START" in content
        assert "**Phase:** orient" in content
        assert "founder, research-agent" in content

    def test_phase_complete(self, ledger):
        ledger.phase_complete("orient", "PASS", ["MISSION_BRIEF"], [])
        content = ledger.path.read_text()
        assert "### PHASE_COMPLETE" in content
        assert "**Result:** PASS" in content
        assert "MISSION_BRIEF" in content

    def test_talent_activated(self, ledger):
        ledger.talent_activated("founder", "orient", None)
        content = ledger.path.read_text()
        assert "### TALENT_ACTIVATED" in content
        assert "**Talent:** founder" in content
        assert "**Handoff from:** none" in content

    def test_skill_invoked(self, ledger):
        ledger.skill_invoked("write-spec", "staff-engineer", "design", "SUCCESS", 12.5)
        content = ledger.path.read_text()
        assert "### SKILL_INVOKED" in content
        assert "**Duration:** 12.5s" in content

    def test_hook_result(self, ledger):
        ledger.hook_result("phase-transition", "orient", "PASS", "3 checks run")
        content = ledger.path.read_text()
        assert "### HOOK_RESULT" in content
        assert "**Result:** PASS" in content

    def test_loop_iteration(self, ledger):
        ledger.loop_iteration("implement", 1, 2, "Tests failed")
        content = ledger.path.read_text()
        assert "### LOOP_ITERATION" in content
        assert "1 -> 2" in content
        assert "Tests failed" in content

    def test_project_complete(self, ledger):
        ledger.project_complete("COMPLETED", 3, 4, ["CODE", "TESTS"])
        content = ledger.path.read_text()
        assert "### PROJECT_COMPLETE" in content
        assert "COMPLETED" in content

    def test_events_in_memory(self, ledger):
        ledger.phase_start("orient", 1, ["founder"])
        ledger.phase_complete("orient", "PASS", [], [])
        assert len(ledger.events) == 2
        assert ledger.events[0].event_type == "PHASE_START"
        assert ledger.events[1].event_type == "PHASE_COMPLETE"

    def test_timestamps_are_iso8601(self, ledger):
        ledger.phase_start("orient", 1, ["founder"])
        content = ledger.path.read_text()
        # ISO 8601 pattern
        assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", content)

    def test_append_only(self, ledger):
        ledger.phase_start("orient", 1, ["founder"])
        ledger.phase_complete("orient", "PASS", [], [])
        ledger.phase_start("implement", 1, ["software-engineer"])
        content = ledger.path.read_text()
        # All three events present in order
        events = re.findall(r"### (\w+)", content)
        assert events == ["PHASE_START", "PHASE_COMPLETE", "PHASE_START"]

    def test_no_duplicate_header(self, ledger):
        # Creating a second Ledger for the same path should not re-write header
        ledger.phase_start("orient", 1, ["founder"])
        ledger2 = Ledger(ledger.path.parent.parent, "test-project", "Build a thing")
        content = ledger2.path.read_text()
        assert content.count("# Status Ledger") == 1
