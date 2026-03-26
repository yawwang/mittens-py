"""Async orchestration engine — wraps the sync Orchestrator for parallel work.

Uses composition: delegates shared logic (talent order, phase prompts, path
resolution) to a sync Orchestrator instance. Adds async LLM calls, parallel
talent instances, and worktree isolation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from mittens.artifacts import ArtifactTracker
from mittens.capabilities import CapabilityResolver
from mittens.hooks import HookRunner
from mittens.ledger import Ledger
from mittens.llm import CostAggregator, LLMAdapter, tools_for_capabilities
from mittens.orchestrator import (
    MAX_AGENT_TURNS,
    MAX_LOOPS_PER_PHASE,
    MAX_TOTAL_ITERATIONS,
    TIER_PHASES,
    Orchestrator,
    _build_tool_call_message,
)
from mittens.registry import Registry
from mittens.session import save_session
from mittens.plugin_api import PluginRegistry, SkillContext
from mittens.types import (
    ComplexityTier,
    HookVerdict,
    InstanceResult,
    InstanceSpec,
    MittensConfig,
    PhaseSpec,
    RunState,
    SessionSnapshot,
    ToolCall,
    WorkflowSpec,
    categorize_checks,
)
from mittens.worktrees import (
    acreate_worktree,
    amerge_worktree,
    aremove_worktree,
)

logger = logging.getLogger(__name__)


class AsyncOrchestrator:
    """Async workflow orchestrator with parallel talent support.

    Wraps a sync Orchestrator for shared logic (talent order, prompts,
    path resolution) and adds async LLM calls + parallel instances.
    """

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
        self._sync = Orchestrator(
            registry=registry,
            llm=llm,
            ledger=ledger,
            artifacts=artifacts,
            capabilities=capabilities,
            hooks=hooks,
            project_dir=project_dir,
            stream=stream,
            config=config,
        )

    @property
    def registry(self) -> Registry:
        return self._sync.registry

    @property
    def ledger(self) -> Ledger:
        return self._sync.ledger

    @property
    def artifacts(self) -> ArtifactTracker:
        return self._sync.artifacts

    @property
    def caps(self) -> CapabilityResolver:
        return self._sync.caps

    @property
    def hooks(self) -> HookRunner:
        return self._sync.hooks

    @property
    def project_dir(self) -> str:
        return self._sync.project_dir

    @property
    def stream(self) -> bool:
        return self._sync.stream

    @property
    def config(self) -> MittensConfig | None:
        return self._sync.config

    @property
    def cost(self) -> CostAggregator:
        return self._sync.cost

    @property
    def _active_llm(self) -> LLMAdapter:
        return self._sync._active_llm

    async def arun_workflow(
        self,
        workflow_id: str,
        mission: str,
        tier: ComplexityTier | None = None,
    ) -> RunState:
        """Async main entry point: run a complete workflow."""
        self._sync._mission = mission
        workflow = self.registry.workflow(workflow_id)

        if tier is None:
            tier = await self._aclassify_complexity(mission)

        active_phases = self._sync._filter_phases(workflow, tier)
        logger.info(
            "Async running %s (tier=%s, phases=%s)",
            workflow_id,
            tier.value,
            [p.id for p in active_phases],
        )

        state = RunState(workflow_id=workflow_id, tier=tier)
        await self._arun_phases(active_phases, state, start_index=0)
        return state

    async def aresume_workflow(self, snapshot: SessionSnapshot) -> RunState:
        """Async resume from a saved snapshot."""
        from mittens.session import restore_run_state

        self._sync._mission = snapshot.mission
        state = restore_run_state(snapshot)
        workflow = self.registry.workflow(state.workflow_id)
        active_phases = self._sync._filter_phases(workflow, state.tier)

        start = snapshot.current_phase_index + 1
        logger.info(
            "Async resuming %s from phase index %d", state.workflow_id, start
        )
        await self._arun_phases(active_phases, state, start_index=start)
        return state

    async def _arun_phases(
        self,
        phases: list[PhaseSpec],
        state: RunState,
        start_index: int = 0,
    ) -> None:
        for i in range(start_index, len(phases)):
            phase = phases[i]
            state.current_phase_index = i

            # Check for parallel instance specs (talent#N syntax)
            instances = self._parse_instances(phase)
            if instances:
                await self._arun_parallel_instances(phase, state, instances)
            else:
                await self._arun_phase(phase, state)

            # Auto-save
            if self._sync._mission:
                try:
                    save_session(
                        state,
                        len(self.ledger.events),
                        self._sync._mission,
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
                break

        self.ledger.project_complete(
            status="COMPLETED",
            total_phases=len(phases),
            total_iterations=state.total_iterations,
            artifacts=list(state.artifacts.keys()),
        )

    async def _arun_phase(self, phase: PhaseSpec, state: RunState) -> None:
        """Execute a single phase asynchronously with loop support."""
        loop_count = 0

        while loop_count <= MAX_LOOPS_PER_PHASE:
            state.total_iterations += 1
            loop_count += 1

            self.ledger.phase_start(phase.id, loop_count, phase.talents)

            talent_order = self._sync._resolve_talent_order(phase)
            primary_talent = talent_order[0]
            self._sync._activate_talent(primary_talent, phase.id, state)

            produced = await self._aexecute_phase_work(phase, state)

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

            # Hooks run synchronously (they may involve subprocess)
            verdict, checks = self.hooks.run_phase_transition(
                phase.id, state.tier.value, state.flags
            )

            failed, warned, passed = categorize_checks(checks)

            check_summary = f"{len(checks)} checks: {len(passed)} pass, {len(failed)} fail"
            self.ledger.hook_result(
                "phase-transition", phase.id, verdict.value,
                check_summary,
                None if verdict == HookVerdict.PASS else "Fix failing checks",
            )

            if verdict in (HookVerdict.PASS, HookVerdict.WARN):
                self.ledger.phase_complete(
                    phase.id, verdict.value, list(produced.keys()),
                    [c.description for c in warned],
                )
                return

            if loop_count > MAX_LOOPS_PER_PHASE:
                self._sync._invoke_reassess(phase, state, checks)
                self.ledger.phase_complete(
                    phase.id, HookVerdict.BLOCK.value, list(produced.keys()),
                    [c.description for c in failed],
                )
                return

            self.ledger.loop_iteration(
                phase.id, loop_count, loop_count + 1,
                "; ".join(c.description for c in failed),
            )

    async def _aexecute_phase_work(
        self, phase: PhaseSpec, state: RunState
    ) -> dict[str, str]:
        """Async agent loop: LLM + tools."""
        messages: list[dict[str, Any]] = []
        messages.append({
            "role": "user",
            "content": self._sync._build_phase_prompt(phase, state),
        })

        plugin_produced: dict[str, str] = {}

        for skill_id in phase.required_skills:
            # Check if a plugin executor is registered for this skill
            if self._sync.plugin_registry:
                plugin_skill = self._sync.plugin_registry.get_skill(skill_id)
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
                        f"Produce all files needed.\n\n{instructions}"
                    ),
                })

        tools = tools_for_capabilities(self.caps.available)
        produced: dict[str, str] = dict(plugin_produced)

        for turn in range(MAX_AGENT_TURNS):
            if self.stream:
                response = await self._active_llm.astream_with_tools(
                    system=state.current_system_prompt,
                    messages=messages,
                    tools=tools,
                )
            else:
                response = await self._active_llm.acomplete_with_tools(
                    system=state.current_system_prompt,
                    messages=messages,
                    tools=tools,
                )

            self._active_llm.track_for_talent(
                state.active_talent,
                (response.input_tokens, response.output_tokens),
            )

            if not response.tool_calls:
                if response.content:
                    messages.append({"role": "assistant", "content": response.content})
                break

            messages.append(_build_tool_call_message(response))

            for tc in response.tool_calls:
                result = await self._aexecute_tool(tc, state, produced)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            logger.warning("Agent turn limit reached for phase %s", phase.id)

        return produced

    async def _aexecute_tool(
        self,
        tool_call: ToolCall,
        state: RunState,
        produced: dict[str, str],
        project_dir: str | None = None,
    ) -> str:
        """Execute a tool call asynchronously."""
        name = tool_call.name
        args = tool_call.arguments
        base_dir = project_dir or self.project_dir

        if name == "read_file":
            return self._sync._tool_read_file(args["path"])
        elif name == "write_file":
            path = args["path"]
            content = args["content"]
            result = self._sync._tool_write_file(path, content)
            artifact_name = self._sync._path_to_artifact_name(path)
            if artifact_name:
                produced[artifact_name] = path
            return result
        elif name == "run_bash":
            return await self._arun_bash(args["command"], cwd=base_dir)
        else:
            return f"Unknown tool: {name}"

    async def _arun_bash(self, command: str, cwd: str | None = None) -> str:
        """Execute a bash command asynchronously."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or self.project_dir,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=120
                )
            except asyncio.TimeoutError:
                proc.kill()
                return "Error: command timed out after 120s"

            output = ""
            if stdout:
                output += stdout.decode()
            if stderr:
                output += f"\nSTDERR:\n{stderr.decode()}"
            if proc.returncode != 0:
                output += f"\nExit code: {proc.returncode}"
            return output[:5000] if output else "(no output)"
        except Exception as e:
            return f"Error: {e}"

    async def _aclassify_complexity(self, mission: str) -> ComplexityTier:
        """Async complexity classification."""
        system_prompt = self.registry.talent_system_prompt("founder")
        response = await self._active_llm.acomplete(
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": (
                    "Classify this mission's complexity as LOW, MEDIUM, or HIGH.\n\n"
                    "LOW: single concern, <=2 components\n"
                    "MEDIUM: 2-4 concerns, multiple acceptance criteria\n"
                    "HIGH: multiple subsystems, algorithms, multi-state UI\n\n"
                    f"Mission: {mission}\n\n"
                    "Respond with exactly one word: LOW, MEDIUM, or HIGH"
                ),
            }],
            max_tokens=10,
        )
        text = response.content.strip().upper()
        for tier in ComplexityTier:
            if tier.value in text:
                return tier
        return ComplexityTier.MEDIUM

    # -- Parallel instance support --

    def _parse_instances(self, phase: PhaseSpec) -> list[InstanceSpec] | None:
        """Detect talent#N syntax in talent list for parallel instances.

        Returns None if no parallel instances are specified.
        """
        instances = []
        for talent in phase.talents:
            if "#" in talent:
                parts = talent.split("#", 1)
                talent_id = parts[0]
                try:
                    num = int(parts[1])
                except ValueError:
                    continue
                instances.append(InstanceSpec(talent_id=talent_id, instance_num=num))

        return instances if instances else None

    async def _arun_parallel_instances(
        self,
        phase: PhaseSpec,
        state: RunState,
        instances: list[InstanceSpec],
    ) -> None:
        """Run multiple talent instances in parallel with worktree isolation."""
        state.total_iterations += 1
        self.ledger.phase_start(phase.id, 1, [f"{i.talent_id}#{i.instance_num}" for i in instances])

        use_worktrees = self.config is not None and "git_write" in (
            self.config.capabilities if self.config else set()
        )

        def _branch_name(inst: InstanceSpec) -> str:
            return f"task/{phase.id}-{inst.talent_id}-{inst.instance_num}"

        if use_worktrees:
            for inst in instances:
                branch = _branch_name(inst)
                try:
                    inst.worktree_path = await acreate_worktree(self.project_dir, branch)
                except Exception as e:
                    logger.warning("Worktree creation failed for %s: %s", branch, e)

        # Spawn parallel tasks
        async def run_instance(inst: InstanceSpec) -> InstanceResult:
            instance_id = f"{inst.talent_id}#{inst.instance_num}"
            self.ledger.log(
                "INSTANCE_START",
                Instance=instance_id,
                Phase=phase.id,
                Task_assigned=inst.task_description or phase.description,
                Worktree=inst.worktree_path or "shared",
            )

            try:
                self._sync._activate_talent(inst.talent_id, phase.id, state)
                produced = await self._aexecute_phase_work(phase, state)
                artifact_names = list(produced.keys())

                self.ledger.log(
                    "INSTANCE_COMPLETE",
                    Instance=instance_id,
                    Phase=phase.id,
                    Result="SUCCESS",
                    Artifacts_produced=", ".join(artifact_names) or "none",
                )

                return InstanceResult(
                    instance_id=instance_id,
                    phase_id=phase.id,
                    artifacts_produced=artifact_names,
                    worktree_path=inst.worktree_path,
                )
            except Exception as e:
                self.ledger.log(
                    "INSTANCE_COMPLETE",
                    Instance=instance_id,
                    Phase=phase.id,
                    Result="FAILED",
                    Errors=str(e),
                )
                return InstanceResult(
                    instance_id=instance_id,
                    phase_id=phase.id,
                    success=False,
                    error=str(e),
                    worktree_path=inst.worktree_path,
                )

        results = await asyncio.gather(
            *(run_instance(inst) for inst in instances),
            return_exceptions=True,
        )

        # Merge worktrees back
        if use_worktrees:
            for inst in instances:
                if inst.worktree_path:
                    branch = _branch_name(inst)
                    try:
                        await amerge_worktree(self.project_dir, branch)
                        await aremove_worktree(self.project_dir, inst.worktree_path)
                    except Exception as e:
                        logger.warning("Worktree merge/cleanup failed: %s", e)

        # Merge produced artifacts into state
        for result in results:
            if isinstance(result, InstanceResult) and result.success:
                for name in result.artifacts_produced:
                    if name in state.artifacts:
                        continue  # First writer wins
                    state.artifacts[name] = name

        self.ledger.phase_complete(
            phase.id, HookVerdict.PASS.value,
            list(state.artifacts.keys()),
            [],
        )
