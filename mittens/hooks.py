"""Hook runner — evaluates phase-transition checks.

Three check types:
- AUTO: bash commands run via subprocess (bin/check-phase.sh)
- PROSE: LLM-evaluated semantic checks against artifacts
- SKILL: invokes a named skill and uses its result

The hook runner is orchestrator-owned — it uses the hook_model,
not the active talent's model.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from mittens.llm import LLMAdapter
from mittens.plugin_api import HookContext, PluginRegistry
from mittens.registry import Registry
from mittens.types import CheckResult, CheckStatus, HookVerdict

logger = logging.getLogger(__name__)


class HookRunner:
    """Run phase-transition hooks and return verdicts."""

    def __init__(
        self,
        registry: Registry,
        hook_llm: LLMAdapter,
        project_dir: str,
        mittens_dir: str,
        plugin_registry: PluginRegistry | None = None,
    ):
        self.registry = registry
        self.llm = hook_llm
        self.project_dir = project_dir
        self.mittens_dir = mittens_dir
        self.plugin_registry = plugin_registry

    def run_phase_transition(
        self,
        phase_id: str,
        tier: str,
        flags: dict[str, bool],
    ) -> tuple[HookVerdict, list[CheckResult]]:
        """Run post-{phase} checks. Returns (verdict, check_details)."""
        all_checks: list[CheckResult] = []

        # 1. AUTO checks via bash script (if available)
        auto_checks = self._run_auto_checks(phase_id, tier, flags)
        all_checks.extend(auto_checks)

        # 2. PROSE checks via LLM evaluation
        prose_checks = self._run_prose_checks(phase_id, tier, flags)
        all_checks.extend(prose_checks)

        # 3. PLUGIN checks from registered hook checks
        if self.plugin_registry:
            plugin_checks = self._run_plugin_checks(phase_id, tier, flags)
            all_checks.extend(plugin_checks)

        verdict = self._aggregate(all_checks)
        return verdict, all_checks

    def _run_plugin_checks(
        self, phase_id: str, tier: str, flags: dict[str, bool]
    ) -> list[CheckResult]:
        """Run registered plugin hook checks."""
        results = []
        context = HookContext(
            phase_id=phase_id,
            tier=tier,
            flags=flags,
            project_dir=self.project_dir,
            artifacts_dir=str(Path(self.project_dir) / "artifacts"),
        )
        for hook in self.plugin_registry.get_hooks():
            try:
                result_str, reasoning = hook.checker(context)
                results.append(CheckResult(
                    check_type="PLUGIN",
                    description=hook.name,
                    result=CheckStatus(result_str),
                    reasoning=reasoning,
                ))
            except Exception as e:
                results.append(CheckResult(
                    check_type="PLUGIN",
                    description=hook.name,
                    result=CheckStatus.FAIL,
                    reasoning=f"Plugin hook error: {e}",
                ))
        return results

    def _run_auto_checks(
        self, phase_id: str, tier: str, flags: dict[str, bool]
    ) -> list[CheckResult]:
        """Run bin/check-phase.sh if it exists."""
        script = Path(self.mittens_dir) / "bin" / "check-phase.sh"
        if not script.exists():
            return []

        cmd = [
            "bash",
            str(script),
            f"post-{phase_id}",
            self.project_dir,
            f"tier={tier}",
        ]
        for key, value in flags.items():
            cmd.append(f"{key}={str(value).lower()}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            return self._parse_auto_output(result.stdout, result.returncode)
        except subprocess.TimeoutExpired:
            return [
                CheckResult(
                    check_type="AUTO",
                    description="check-phase.sh timed out",
                    result=CheckStatus.FAIL,
                    reasoning="Script exceeded 60s timeout",
                )
            ]
        except FileNotFoundError:
            logger.debug("check-phase.sh not found, skipping AUTO checks")
            return []

    def _parse_auto_output(
        self, stdout: str, returncode: int
    ) -> list[CheckResult]:
        """Parse structured output from check-phase.sh.

        Expected format: lines of "PASS|WARN|FAIL: description"
        Falls back to treating the whole output as a single check.
        """
        results = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            for prefix in ("PASS:", "WARN:", "FAIL:"):
                if line.upper().startswith(prefix):
                    status = CheckStatus(prefix[:-1])
                    desc = line[len(prefix) :].strip()
                    results.append(
                        CheckResult(
                            check_type="AUTO",
                            description=desc,
                            result=status,
                        )
                    )
                    break
            else:
                results.append(
                    CheckResult(
                        check_type="AUTO",
                        description=line,
                        result=CheckStatus.PASS if returncode == 0 else CheckStatus.FAIL,
                    )
                )

        if not results and returncode != 0:
            results.append(
                CheckResult(
                    check_type="AUTO",
                    description="check-phase.sh failed",
                    result=CheckStatus.FAIL,
                    reasoning=stdout[:500] if stdout else f"exit code {returncode}",
                )
            )
        return results

    def _run_prose_checks(
        self, phase_id: str, tier: str, flags: dict[str, bool]
    ) -> list[CheckResult]:
        """Evaluate prose-based exit criteria via LLM.

        Reads the phase-transition hook to find exit criteria for this
        phase, then asks the LLM to evaluate each one.
        """
        # Build artifact context
        artifacts_dir = Path(self.project_dir) / "artifacts"
        artifact_context = self._gather_artifact_context(artifacts_dir)
        if not artifact_context:
            return []

        # Get exit criteria from the hook definition
        hook_doc = self.registry.hook("phase-transition")
        exit_criteria = self._extract_exit_criteria(hook_doc.body, phase_id)

        results = []
        for criterion in exit_criteria:
            check_result = self._evaluate_criterion(
                criterion, artifact_context, phase_id
            )
            results.append(check_result)

        return results

    def _gather_artifact_context(self, artifacts_dir: Path) -> str:
        """Read artifact files for LLM evaluation context."""
        if not artifacts_dir.exists():
            return ""
        max_chars = 2000
        parts = []
        for path in sorted(artifacts_dir.glob("*.md")):
            if path.name == "status-ledger.md":
                continue  # Too large, not needed for checks
            with open(path) as f:
                content = f.read(max_chars + 1)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... (truncated)"
            parts.append(f"### {path.stem}\n{content}")
        return "\n\n".join(parts)

    def _extract_exit_criteria(self, hook_body: str, phase_id: str) -> list[str]:
        """Extract PROSE exit criteria for a phase from the hook body.

        This is a best-effort extraction — the hook body format varies.
        Falls back to returning an empty list if parsing fails.
        """
        # Look for a section matching "post-{phase_id}" and extract criteria
        criteria = []
        in_phase = False
        for line in hook_body.splitlines():
            lower = line.lower().strip()
            if f"post-{phase_id}" in lower:
                in_phase = True
                continue
            if in_phase:
                if line.startswith("## ") or line.startswith("# "):
                    break
                if "`[PROSE]`" in line:
                    # Extract the check description after the tag
                    desc = line.split("`[PROSE]`", 1)[-1].strip().lstrip("- ")
                    if desc:
                        criteria.append(desc)
        return criteria

    def _evaluate_criterion(
        self, criterion: str, artifact_context: str, phase_id: str
    ) -> CheckResult:
        """Ask the LLM to evaluate a single prose criterion."""
        try:
            response = self.llm.complete(
                system=(
                    "You are a quality gate evaluator. Your job is to check "
                    "whether a specific criterion is met by examining project "
                    "artifacts. Respond with exactly PASS or FAIL on the first "
                    "line, then explain your reasoning briefly."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Phase: {phase_id}\n"
                            f"Criterion: {criterion}\n\n"
                            f"Artifacts:\n{artifact_context}"
                        ),
                    }
                ],
                max_tokens=500,
            )
            first_line = response.content.strip().split("\n")[0].upper()
            passed = "PASS" in first_line
            return CheckResult(
                check_type="PROSE",
                description=criterion,
                result=CheckStatus.PASS if passed else CheckStatus.FAIL,
                reasoning=response.content,
            )
        except Exception as e:
            logger.warning("PROSE check failed for '%s': %s", criterion, e)
            return CheckResult(
                check_type="PROSE",
                description=criterion,
                result=CheckStatus.WARN,
                reasoning=f"Evaluation error: {e}",
            )

    @staticmethod
    def _aggregate(checks: list[CheckResult]) -> HookVerdict:
        """Aggregate individual check results into a single verdict."""
        if not checks:
            return HookVerdict.PASS
        has_fail = any(c.result == CheckStatus.FAIL for c in checks)
        has_warn = any(c.result == CheckStatus.WARN for c in checks)
        if has_fail:
            return HookVerdict.BLOCK
        if has_warn:
            return HookVerdict.WARN
        return HookVerdict.PASS
