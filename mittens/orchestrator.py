"""Core orchestration engine — the heart of the Mittens runtime.

Implements the phase loop: talent activation, skill dispatch, hook
enforcement, loop handling, and safety rails. Faithfully reproduces
the execution flow from commands/auto/SKILL.md.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from mittens.artifacts import ArtifactTracker
from mittens.capabilities import CapabilityResolver
from mittens.hooks import HookRunner
from mittens.ledger import Ledger
from mittens.llm import CostAggregator, LLMAdapter, tools_for_capabilities
from mittens.plugin_api import PluginRegistry, SkillContext
from mittens.registry import Registry
from mittens.session import save_session
from mittens.types import (
    CheckStatus,
    ComplexityTier,
    HookVerdict,
    LLMToolResponse,
    MittensConfig,
    PhaseSpec,
    RunState,
    SessionSnapshot,
    ToolCall,
    WorkflowSpec,
)

logger = logging.getLogger(__name__)

# Phase skip rules by tier (from autonomous-build.md complexity routing)
TIER_PHASES: dict[ComplexityTier, list[str]] = {
    ComplexityTier.LOW: ["orient", "implement", "verify"],
    ComplexityTier.MEDIUM: ["orient", "frame", "design", "implement", "verify"],
    ComplexityTier.HIGH: [
        "orient", "frame", "design", "plan", "implement", "verify", "reflect"
    ],
}

MAX_LOOPS_PER_PHASE = 2
MAX_TOTAL_ITERATIONS = 15
MAX_AGENT_TURNS = 20  # Safety limit on tool-use turns within a phase


def _build_tool_call_message(response: LLMToolResponse) -> dict[str, Any]:
    """Build the assistant message containing tool calls (required by API)."""
    msg: dict[str, Any] = {"role": "assistant"}
    if response.content:
        msg["content"] = response.content
    msg["tool_calls"] = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
        }
        for tc in response.tool_calls
    ]
    return msg


class Orchestrator:
    """Run Mittens workflows by orchestrating LLM calls, tools, and hooks."""

    def __init__(
        self,
        registry: Registry,
        llm: LLMAdapter,
        ledger: Ledger,
        artifacts: ArtifactTracker,
        capabilities: CapabilityResolver,
        hooks: HookRunner,
        project_dir: str,
        stream: bool = True,
        config: MittensConfig | None = None,
    ):
        self.registry = registry
        self._default_llm = llm
        self._active_llm = llm
        self._llm_cache: dict[str, LLMAdapter] = {llm.model: llm}
        self.cost = CostAggregator()
        self.cost.register(llm)
        self.ledger = ledger
        self.artifacts = artifacts
        self.caps = capabilities
        self.hooks = hooks
        self.project_dir = project_dir
        self.stream = stream
        self.config = config
        self.plugin_registry: PluginRegistry | None = None
        self._mission: str | None = None

    @property
    def llm(self) -> LLMAdapter:
        """Active LLM adapter (may differ per talent via model_overrides)."""
        return self._active_llm

    @llm.setter
    def llm(self, adapter: LLMAdapter) -> None:
        self._default_llm = adapter
        self._active_llm = adapter

    def run_workflow(
        self,
        workflow_id: str,
        mission: str,
        tier: ComplexityTier | None = None,
    ) -> RunState:
        """Main entry point: run a complete workflow."""
        self._mission = mission
        workflow = self.registry.workflow(workflow_id)

        # Classify complexity if not provided
        if tier is None:
            tier = self._classify_complexity(mission)

        active_phases = self._filter_phases(workflow, tier)
        logger.info(
            "Running %s (tier=%s, phases=%s)",
            workflow_id,
            tier.value,
            [p.id for p in active_phases],
        )

        state = RunState(
            workflow_id=workflow_id,
            tier=tier,
        )

        self._run_phases(active_phases, state, start_index=0)
        return state

    def resume_workflow(
        self,
        snapshot: SessionSnapshot,
    ) -> RunState:
        """Resume a workflow from a saved session snapshot."""
        from mittens.session import restore_run_state

        self._mission = snapshot.mission
        state = restore_run_state(snapshot)
        workflow = self.registry.workflow(state.workflow_id)
        active_phases = self._filter_phases(workflow, state.tier)

        # Resume from the next phase after the last completed one
        start = snapshot.current_phase_index + 1
        logger.info(
            "Resuming %s from phase index %d (tier=%s)",
            state.workflow_id, start, state.tier.value,
        )

        self._run_phases(active_phases, state, start_index=start)
        return state

    def _run_phases(
        self,
        phases: list[PhaseSpec],
        state: RunState,
        start_index: int = 0,
    ) -> None:
        """Run phases from start_index through end, with auto-save."""
        for i in range(start_index, len(phases)):
            phase = phases[i]
            state.current_phase_index = i
            self._run_phase(phase, state)

            # Auto-save after each phase
            if self._mission:
                try:
                    save_session(
                        state,
                        len(self.ledger.events),
                        self._mission,
                        self.project_dir,
                    )
                except Exception as e:
                    logger.warning("Auto-save failed: %s", e)

            if state.total_iterations >= MAX_TOTAL_ITERATIONS:
                self.ledger.log(
                    "HARD_STOP",
                    Reason="Total iteration ceiling exceeded",
                    Total_iterations=str(state.total_iterations),
                )
                logger.error("Hard stop: %d iterations exceeded ceiling", state.total_iterations)
                break

        self.ledger.project_complete(
            status="COMPLETED",
            total_phases=len(phases),
            total_iterations=state.total_iterations,
            artifacts=list(state.artifacts.keys()),
        )

    def _classify_complexity(self, mission: str) -> ComplexityTier:
        """Ask the LLM to classify mission complexity."""
        system_prompt = self.registry.talent_system_prompt("founder")
        response = self.llm.complete(
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Classify this mission's complexity as LOW, MEDIUM, or HIGH.\n\n"
                        "LOW: single concern, <=2 components, binary correctness\n"
                        "MEDIUM: 2-4 concerns, multiple acceptance criteria, non-obvious edge cases\n"
                        "HIGH: multiple subsystems, algorithms, multi-state UI, SDK\n\n"
                        f"Mission: {mission}\n\n"
                        "Respond with exactly one word: LOW, MEDIUM, or HIGH"
                    ),
                }
            ],
            max_tokens=10,
        )
        text = response.content.strip().upper()
        for tier in ComplexityTier:
            if tier.value in text:
                return tier
        return ComplexityTier.MEDIUM  # safe default

    def _filter_phases(
        self, workflow: WorkflowSpec, tier: ComplexityTier
    ) -> list[PhaseSpec]:
        """Return only the phases active for this complexity tier."""
        allowed_ids = set(TIER_PHASES.get(tier, TIER_PHASES[ComplexityTier.HIGH]))
        return [p for p in workflow.phases if p.id in allowed_ids]

    def _run_phase(self, phase: PhaseSpec, state: RunState) -> None:
        """Execute a single phase with loop support."""
        loop_count = 0

        while loop_count <= MAX_LOOPS_PER_PHASE:
            state.total_iterations += 1
            loop_count += 1

            self.ledger.phase_start(phase.id, loop_count, phase.talents)
            logger.info("Phase %s (loop %d)", phase.id, loop_count)

            # 1. Resolve talent order (owner first, then consulting, then rest)
            talent_order = self._resolve_talent_order(phase)

            # 2. Activate primary talent and execute phase work
            primary_talent = talent_order[0]
            self._activate_talent(primary_talent, phase.id, state)

            produced = self._execute_phase_work(phase, state)

            # 3. Track produced artifacts
            for name, path in produced.items():
                self.artifacts.register(name, path)
                state.artifacts[name] = path
                stale = self.artifacts.flag_downstream(name)
                if stale:
                    self.ledger.log(
                        "ARTIFACT_STALE",
                        Trigger=name,
                        Flagged_stale=", ".join(stale),
                    )

            # 4. Run phase-transition hook
            verdict, checks = self.hooks.run_phase_transition(
                phase.id, state.tier.value, state.flags
            )

            failed = [c for c in checks if c.result == CheckStatus.FAIL]
            warned = [c for c in checks if c.result == CheckStatus.WARN]
            passed = [c for c in checks if c.result == CheckStatus.PASS]

            check_summary = f"{len(checks)} checks: {len(passed)} pass, {len(failed)} fail"
            self.ledger.hook_result(
                "phase-transition",
                phase.id,
                verdict.value,
                check_summary,
                None if verdict == HookVerdict.PASS else "Fix failing checks",
            )

            if verdict in (HookVerdict.PASS, HookVerdict.WARN):
                self.ledger.phase_complete(
                    phase.id,
                    verdict.value,
                    list(produced.keys()),
                    [c.description for c in warned],
                )
                return

            if loop_count > MAX_LOOPS_PER_PHASE:
                logger.warning(
                    "Phase %s blocked after %d loops, invoking reassess",
                    phase.id,
                    loop_count,
                )
                self._invoke_reassess(phase, state, checks)
                self.ledger.phase_complete(
                    phase.id, "BLOCK", list(produced.keys()),
                    [c.description for c in failed],
                )
                return

            self.ledger.loop_iteration(
                phase.id,
                loop_count,
                loop_count + 1,
                "; ".join(c.description for c in failed),
            )

    def _resolve_talent_order(self, phase: PhaseSpec) -> list[str]:
        """Owner talent first, then consulting talents, then remaining."""
        order: list[str] = []
        if phase.owner_talent:
            order.append(phase.owner_talent)
        for t in phase.consulting_talents:
            if t not in order:
                order.append(t)
        for t in phase.talents:
            if t not in order:
                order.append(t)
        return order

    def _activate_talent(
        self, talent_id: str, phase_id: str, state: RunState
    ) -> None:
        """Load talent system prompt, swap model if overridden, set as active."""
        system_prompt = self.registry.talent_system_prompt(talent_id)
        self.ledger.talent_activated(talent_id, phase_id, state.active_talent)
        state.active_talent = talent_id
        state.current_system_prompt = system_prompt

        # Per-talent model routing
        overrides = self.config.model_overrides if self.config else {}
        target_model = overrides.get(talent_id, self._default_llm.model)

        if target_model != self._active_llm.model:
            if target_model in self._llm_cache:
                self._active_llm = self._llm_cache[target_model]
            else:
                adapter = LLMAdapter(model=target_model)
                self._llm_cache[target_model] = adapter
                self.cost.register(adapter)
                self._active_llm = adapter
            logger.info("Switched model to %s for talent %s", target_model, talent_id)
        elif target_model == self._default_llm.model:
            self._active_llm = self._default_llm

    def _execute_phase_work(
        self, phase: PhaseSpec, state: RunState
    ) -> dict[str, str]:
        """Send phase context to LLM with tools, return produced artifacts."""
        messages: list[dict[str, Any]] = []
        messages.append({
            "role": "user",
            "content": self._build_phase_prompt(phase, state),
        })

        plugin_produced: dict[str, str] = {}

        for skill_id in phase.required_skills:
            # Check if a plugin executor is registered for this skill
            if self.plugin_registry:
                plugin_skill = self.plugin_registry.get_skill(skill_id)
                if plugin_skill:
                    start_time = time.time()
                    ctx = SkillContext(
                        phase_id=phase.id,
                        talent_id=state.active_talent or "",
                        artifacts=dict(state.artifacts),
                        project_dir=self.project_dir,
                    )
                    try:
                        result = plugin_skill.executor(ctx)
                        for name, path in result.artifacts_produced.items():
                            plugin_produced[name] = path
                        duration = time.time() - start_time
                        self.ledger.skill_invoked(
                            skill_id, state.active_talent or "",
                            phase.id,
                            "SUCCESS" if result.success else "FAILED",
                            duration,
                        )
                        if result.output:
                            messages.append({
                                "role": "user",
                                "content": f"PLUGIN SKILL RESULT ({skill_id}):\n{result.output}",
                            })
                    except Exception as e:
                        self.ledger.skill_invoked(
                            skill_id, state.active_talent or "",
                            phase.id, "FAILED", time.time() - start_time,
                        )
                        logger.warning("Plugin skill %s failed: %s", skill_id, e)
                    continue

            instructions = self.registry.skill_instructions(skill_id)
            if not instructions:
                continue

            if self.caps.can_execute(skill_id):
                messages.append({
                    "role": "user",
                    "content": f"REQUIRED SKILL: {skill_id}\n\n{instructions}",
                })
            elif self.caps.needs_split(skill_id):
                agent_caps, orch_caps = self.caps.split_plan(skill_id)
                messages.append({
                    "role": "user",
                    "content": (
                        f"SKILL: {skill_id} (PRODUCE/EXECUTE SPLIT)\n\n"
                        f"You can: {', '.join(agent_caps)}\n"
                        f"The orchestrator will handle: {', '.join(orch_caps)}\n\n"
                        f"Produce all files needed. The orchestrator will run "
                        f"bash commands and feed results back.\n\n{instructions}"
                    ),
                })

        tools = tools_for_capabilities(self.caps.available)
        produced: dict[str, str] = dict(plugin_produced)

        for turn in range(MAX_AGENT_TURNS):
            if self.stream:
                response = self._active_llm.stream_with_tools(
                    system=state.current_system_prompt,
                    messages=messages,
                    tools=tools,
                )
            else:
                response = self._active_llm.complete_with_tools(
                    system=state.current_system_prompt,
                    messages=messages,
                    tools=tools,
                )

            # Track cost for talent
            self._active_llm.track_for_talent(
                state.active_talent,
                (response.input_tokens, response.output_tokens),
            )

            if not response.tool_calls:
                # LLM is done — append final response and break
                if response.content:
                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                    })
                break

            messages.append(_build_tool_call_message(response))

            for tc in response.tool_calls:
                result = self._execute_tool(tc, state, produced)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            logger.warning("Agent turn limit reached for phase %s", phase.id)

        return produced

    def _execute_tool(
        self, tool_call: ToolCall, state: RunState, produced: dict[str, str]
    ) -> str:
        """Execute a tool call and return the result string."""
        name = tool_call.name
        args = tool_call.arguments

        if name == "read_file":
            return self._tool_read_file(args["path"])
        elif name == "write_file":
            path = args["path"]
            content = args["content"]
            result = self._tool_write_file(path, content)
            # Track as produced artifact if in artifacts/ directory
            artifact_name = self._path_to_artifact_name(path)
            if artifact_name:
                produced[artifact_name] = path
            return result
        elif name == "run_bash":
            return self._tool_run_bash(args["command"])
        else:
            return f"Unknown tool: {name}"

    def _tool_read_file(self, path: str) -> str:
        """Read a file and return its contents."""
        resolved = self._resolve_path(path)
        try:
            content = Path(resolved).read_text()
            if len(content) > 10000:
                return content[:10000] + "\n... (truncated at 10000 chars)"
            return content
        except FileNotFoundError:
            return f"Error: file not found: {resolved}"
        except Exception as e:
            return f"Error reading {resolved}: {e}"

    def _tool_write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        resolved = self._resolve_path(path)
        try:
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            Path(resolved).write_text(content)
            return f"Written: {resolved} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing {resolved}: {e}"

    def _tool_run_bash(self, command: str) -> str:
        """Execute a bash command and return output."""
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_dir,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output[:5000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120s"
        except Exception as e:
            return f"Error: {e}"

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the project directory."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(self.project_dir) / p)

    def _path_to_artifact_name(self, path: str) -> str | None:
        """Extract artifact name from a path if it's in artifacts/."""
        resolved = Path(self._resolve_path(path))
        artifacts_dir = Path(self.project_dir) / "artifacts"
        try:
            relative = resolved.relative_to(artifacts_dir)
            return relative.stem.upper().replace("-", "_")
        except ValueError:
            return None

    def _build_phase_prompt(self, phase: PhaseSpec, state: RunState) -> str:
        """Construct the phase context message for the LLM."""
        lines = [
            f"## Phase: {phase.name}",
            f"",
            f"**Phase ID:** {phase.id}",
            f"**Description:** {phase.description}",
            f"**Required outputs:** {', '.join(phase.outputs)}",
            f"",
            f"**Exit criteria:**",
        ]
        for ec in phase.exit_criteria:
            lines.append(f"- {ec}")

        if state.artifacts:
            lines.append("")
            lines.append("**Available artifacts from previous phases:**")
            for name, path in state.artifacts.items():
                lines.append(f"- {name}: {path}")

        if state.flags:
            lines.append("")
            lines.append("**Flags:**")
            for key, value in state.flags.items():
                lines.append(f"- {key}: {value}")

        lines.append("")
        lines.append(
            "Produce the required outputs by using the available tools "
            "(read_file, write_file, run_bash). When you have produced all "
            "required outputs, summarize what you did."
        )

        return "\n".join(lines)

    def _invoke_reassess(
        self, phase: PhaseSpec, state: RunState, checks: list
    ) -> None:
        """Invoke reassess when stuck after max loops."""
        failed = [c for c in checks if c.result == CheckStatus.FAIL]
        response = self.llm.complete(
            system="You are reassessing a stuck workflow phase.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Phase '{phase.id}' is blocked after {MAX_LOOPS_PER_PHASE} loops.\n\n"
                        f"Failed checks:\n"
                        + "\n".join(f"- {c.description}: {c.reasoning}" for c in failed)
                        + "\n\nShould we: (a) retry with a different approach, "
                        "(b) reduce scope, or (c) abort? Explain your recommendation."
                    ),
                }
            ],
        )
        self.ledger.log(
            "REASSESS",
            Phase=phase.id,
            Recommendation=response.content[:500],
        )
