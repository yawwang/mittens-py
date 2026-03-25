"""Tests for the plugin system — registration, execution, and loading."""

from unittest.mock import MagicMock, patch

import pytest

from mittens.plugin_api import (
    HookContext,
    HookRegistration,
    PluginRegistry,
    SkillContext,
    SkillRegistration,
    SkillResult,
    flush_pending,
    mittens_hook,
    mittens_skill,
    _pending_hooks,
    _pending_skills,
)
from mittens.plugins import PluginLoader


class TestPluginRegistry:
    def test_register_skill(self):
        reg = PluginRegistry()
        skill = SkillRegistration(
            name="test-skill",
            capabilities=["bash"],
            executor=lambda ctx: SkillResult(success=True, output="ok"),
        )
        reg.register_skill(skill)
        assert "test-skill" in reg.skill_names
        assert reg.get_skill("test-skill") is skill

    def test_register_hook(self):
        reg = PluginRegistry()
        hook = HookRegistration(
            name="test-hook",
            check_type="PLUGIN",
            checker=lambda ctx: ("PASS", "all good"),
        )
        reg.register_hook(hook)
        assert "test-hook" in reg.hook_names
        assert len(reg.get_hooks()) == 1

    def test_get_nonexistent_skill(self):
        reg = PluginRegistry()
        assert reg.get_skill("nope") is None

    def test_multiple_skills(self):
        reg = PluginRegistry()
        for i in range(3):
            reg.register_skill(SkillRegistration(
                name=f"skill-{i}",
                capabilities=[],
                executor=lambda ctx: SkillResult(success=True, output=""),
            ))
        assert len(reg.skill_names) == 3


class TestDecorators:
    def setup_method(self):
        _pending_skills.clear()
        _pending_hooks.clear()

    def test_mittens_skill_decorator(self):
        @mittens_skill("my-lint", capabilities=["bash"])
        def run_lint(context):
            return SkillResult(success=True, output="clean")

        assert len(_pending_skills) == 1
        assert _pending_skills[0].name == "my-lint"
        assert hasattr(run_lint, "_mittens_skill")

    def test_mittens_hook_decorator(self):
        @mittens_hook("my-check")
        def check_something(context):
            return ("PASS", "all good")

        assert len(_pending_hooks) == 1
        assert _pending_hooks[0].name == "my-check"

    def test_flush_pending(self):
        @mittens_skill("flush-test")
        def skill_fn(ctx):
            return SkillResult(success=True, output="")

        @mittens_hook("flush-hook")
        def hook_fn(ctx):
            return ("PASS", "")

        reg = PluginRegistry()
        flush_pending(reg)

        assert "flush-test" in reg.skill_names
        assert "flush-hook" in reg.hook_names
        assert len(_pending_skills) == 0
        assert len(_pending_hooks) == 0


class TestSkillExecution:
    def test_skill_executor_called(self):
        called = []

        def my_executor(ctx: SkillContext) -> SkillResult:
            called.append(ctx)
            return SkillResult(
                success=True,
                output="done",
                artifacts_produced={"CODE": "src/main.py"},
            )

        reg = PluginRegistry()
        reg.register_skill(SkillRegistration(
            name="test-exec",
            capabilities=[],
            executor=my_executor,
        ))

        skill = reg.get_skill("test-exec")
        ctx = SkillContext(
            phase_id="implement",
            talent_id="software-engineer",
            artifacts={},
            project_dir="/tmp/test",
        )
        result = skill.executor(ctx)

        assert len(called) == 1
        assert result.success
        assert result.artifacts_produced == {"CODE": "src/main.py"}

    def test_skill_executor_failure(self):
        def failing_executor(ctx):
            return SkillResult(success=False, output="lint errors found")

        reg = PluginRegistry()
        reg.register_skill(SkillRegistration(
            name="failing",
            capabilities=[],
            executor=failing_executor,
        ))

        result = reg.get_skill("failing").executor(
            SkillContext("p", "t", {}, "/tmp")
        )
        assert not result.success


class TestHookExecution:
    def test_hook_checker_pass(self):
        def my_check(ctx: HookContext) -> tuple[str, str]:
            return ("PASS", "everything looks good")

        reg = PluginRegistry()
        reg.register_hook(HookRegistration(
            name="my-check", check_type="PLUGIN", checker=my_check
        ))

        hooks = reg.get_hooks()
        ctx = HookContext(
            phase_id="verify", tier="LOW", flags={},
            project_dir="/tmp", artifacts_dir="/tmp/artifacts",
        )
        result, reasoning = hooks[0].checker(ctx)
        assert result == "PASS"

    def test_hook_checker_fail(self):
        def failing_check(ctx):
            return ("FAIL", "tests not passing")

        reg = PluginRegistry()
        reg.register_hook(HookRegistration(
            name="strict-check", check_type="PLUGIN", checker=failing_check
        ))

        result, _ = reg.get_hooks()[0].checker(
            HookContext("v", "LOW", {}, "/tmp", "/tmp/a")
        )
        assert result == "FAIL"


class TestPluginLoader:
    def test_empty_loader(self):
        loader = PluginLoader()
        loader.load_all()
        assert loader.loaded_plugins == []

    def test_load_module_with_register(self):
        # Create a mock module with a register function
        mock_module = MagicMock()
        mock_module.__version__ = "1.0.0"
        mock_module.register = MagicMock()

        loader = PluginLoader()
        with patch("mittens.plugins.importlib.import_module", return_value=mock_module):
            info = loader.load_module("mittens_test")

        assert info is not None
        assert info.name == "mittens_test"
        assert info.version == "1.0.0"
        mock_module.register.assert_called_once()

    def test_load_module_without_register(self):
        mock_module = MagicMock(spec=[])  # No register attribute

        loader = PluginLoader()
        with patch("mittens.plugins.importlib.import_module", return_value=mock_module):
            info = loader.load_module("bad_plugin")

        assert info is None

    def test_load_module_import_error(self):
        loader = PluginLoader()
        with patch("mittens.plugins.importlib.import_module", side_effect=ImportError("nope")):
            info = loader.load_module("nonexistent")

        assert info is None

    def test_enabled_filter(self):
        loader = PluginLoader(enabled=["mittens_allowed"])
        # The _enabled set should filter during discovery
        assert loader._enabled == {"mittens_allowed"}
