"""Tests for the markdown+YAML frontmatter parser."""

from pathlib import Path

import pytest

from mittens.parser import parse_doc, _extract_sections

MITTENS_DIR = Path.home() / "mittens"


class TestParseDoc:
    def test_parse_talent(self):
        doc = parse_doc(MITTENS_DIR / "talents" / "founder.md")
        assert doc.frontmatter["id"] == "founder"
        assert doc.frontmatter["name"] == "Founder"
        assert doc.frontmatter["version"] == "1.0.0"
        assert isinstance(doc.frontmatter["strengths"], list)
        assert len(doc.frontmatter["strengths"]) >= 3
        assert isinstance(doc.frontmatter["skills"], list)
        assert isinstance(doc.frontmatter["phases"], list)
        assert "reasoning" in doc.frontmatter
        assert doc.frontmatter["reasoning"]["style"] in {
            "analytical", "creative", "systematic", "investigative", "pragmatic"
        }
        assert "System Prompt" in doc.sections
        assert len(doc.sections["System Prompt"]) > 0

    def test_parse_skill(self):
        doc = parse_doc(MITTENS_DIR / "skills" / "write-spec.md")
        assert doc.frontmatter["id"] == "write-spec"
        assert isinstance(doc.frontmatter["inputs"], list)
        assert isinstance(doc.frontmatter["outputs"], list)
        assert "Instructions" in doc.sections
        assert len(doc.sections["Instructions"]) > 0

    def test_parse_skill_with_capabilities(self):
        doc = parse_doc(MITTENS_DIR / "skills" / "verification-loop.md")
        caps = doc.frontmatter.get("requires_capabilities", [])
        assert "bash" in caps
        assert "test_exec" in caps

    def test_parse_workflow(self):
        doc = parse_doc(MITTENS_DIR / "workflows" / "autonomous-build.md")
        assert doc.frontmatter["id"] == "autonomous-build"
        phases = doc.frontmatter["phases"]
        assert isinstance(phases, list)
        assert len(phases) >= 3
        # Check first phase has required fields
        first = phases[0]
        assert "id" in first
        assert "talents" in first
        assert "inputs" in first
        assert "outputs" in first
        assert "exit_criteria" in first

    def test_parse_hook(self):
        doc = parse_doc(MITTENS_DIR / "hooks" / "phase-transition.md")
        assert doc.frontmatter is not None
        assert len(doc.body) > 0


class TestExtractSections:
    def test_basic_sections(self):
        body = """## Section One

Content of section one.

## Section Two

Content of section two.
More content.
"""
        sections = _extract_sections(body)
        assert "Section One" in sections
        assert "Section Two" in sections
        assert "Content of section one." in sections["Section One"]
        assert "More content." in sections["Section Two"]

    def test_nested_sections(self):
        body = """## Parent

Parent content.

### Child

Child content.

## Next Parent

Next content.
"""
        sections = _extract_sections(body)
        assert "Parent" in sections
        assert "Child" in sections
        assert "Next Parent" in sections
        # Parent content should include child since child is deeper
        assert "Parent content." in sections["Parent"]

    def test_empty_body(self):
        sections = _extract_sections("")
        assert sections == {}
