"""Framework index — scans MITTENS_DIR and provides lookup by type and ID.

Caches parsed results to avoid re-reading during a single run.
Never imports llm.py — keeps indexing testable without mocking LLM calls.
"""

from __future__ import annotations

from pathlib import Path

from mittens.parser import parse_doc
from mittens.types import ParsedDoc, PhaseSpec, WorkflowSpec


class Registry:
    """Read-only index of all Mittens definition files."""

    def __init__(self, mittens_dir: str | Path):
        self._dir = Path(mittens_dir)
        self._cache: dict[str, ParsedDoc] = {}

    # -- Single-item lookups --

    def talent(self, talent_id: str) -> ParsedDoc:
        return self._load("talents", talent_id)

    def skill(self, skill_id: str) -> ParsedDoc:
        return self._load("skills", skill_id)

    def workflow(self, workflow_id: str) -> WorkflowSpec:
        doc = self._load("workflows", workflow_id)
        return self._build_workflow_spec(doc)

    def hook(self, hook_id: str) -> ParsedDoc:
        return self._load("hooks", hook_id)

    # -- Listing --

    def list_talents(self) -> list[str]:
        return self._list_ids("talents")

    def list_skills(self) -> list[str]:
        return self._list_ids("skills")

    def list_workflows(self) -> list[str]:
        return self._list_ids("workflows")

    def list_hooks(self) -> list[str]:
        return self._list_ids("hooks")

    # -- Capability queries --

    def skill_capabilities(self, skill_id: str) -> list[str]:
        """Return requires_capabilities for a skill.

        Defaults to [file_read, file_write] per capability.schema.md.
        """
        doc = self.skill(skill_id)
        return doc.frontmatter.get(
            "requires_capabilities", ["file_read", "file_write"]
        )

    def talent_system_prompt(self, talent_id: str) -> str:
        """Extract the System Prompt section from a talent card."""
        doc = self.talent(talent_id)
        return doc.sections.get("System Prompt", "")

    def skill_instructions(self, skill_id: str) -> str:
        """Extract the Instructions section from a skill."""
        doc = self.skill(skill_id)
        return doc.sections.get("Instructions", "")

    # -- Internal --

    def _load(self, category: str, item_id: str) -> ParsedDoc:
        key = f"{category}/{item_id}"
        if key not in self._cache:
            path = self._dir / category / f"{item_id}.md"
            if not path.exists():
                raise FileNotFoundError(
                    f"No {category[:-1]} found: {path}"
                )
            self._cache[key] = parse_doc(path)
        return self._cache[key]

    def _list_ids(self, category: str) -> list[str]:
        folder = self._dir / category
        if not folder.is_dir():
            return []
        return sorted(
            p.stem
            for p in folder.glob("*.md")
            if p.stem not in ("INDEX", "README")
        )

    def _build_workflow_spec(self, doc: ParsedDoc) -> WorkflowSpec:
        fm = doc.frontmatter
        phases = []
        for p in fm["phases"]:
            phases.append(
                PhaseSpec(
                    id=p["id"],
                    name=p["name"],
                    description=p.get("description", ""),
                    talents=p["talents"],
                    inputs=p.get("inputs", []),
                    outputs=p.get("outputs", []),
                    exit_criteria=p.get("exit_criteria", []),
                    owner_talent=p.get("owner_talent"),
                    consulting_talents=p.get("consulting_talents", []),
                    escalates_to=p.get("escalates_to"),
                    required_skills=p.get("required_skills", []),
                )
            )
        return WorkflowSpec(
            id=fm["id"],
            name=fm["name"],
            version=fm.get("version", "0.0.0"),
            description=fm.get("description", ""),
            phases=phases,
            body=doc.body,
        )
