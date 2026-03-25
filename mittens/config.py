"""Configuration loading with layered overrides.

Loading order: defaults -> global config -> project config -> env vars -> CLI flags.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mittens.types import MittensConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


DEFAULTS: dict[str, Any] = {
    "mittens": {
        "mittens_dir": str(Path.home() / "mittens"),
        "project_name": "",
    },
    "model": {
        "default": "anthropic/claude-sonnet-4-20250514",
        "hook_model": "anthropic/claude-haiku-4-5-20251001",
        "override": {},
    },
    "capabilities": {
        "available": ["file_read", "file_write", "bash", "git_write", "test_exec"],
    },
    "budget": {
        "max_total_usd": None,
    },
    "output": {
        "stream": True,
        "verbosity": "normal",
    },
    "plugins": {
        "enabled": [],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dicts."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _apply_env(config: dict) -> dict:
    """Override config values from MITTENS_* environment variables."""
    env_map = {
        "MITTENS_DIR": ("mittens", "mittens_dir"),
        "MITTENS_MODEL": ("model", "default"),
        "MITTENS_HOOK_MODEL": ("model", "hook_model"),
        "MITTENS_CAPABILITIES": ("capabilities", "available"),
        "MITTENS_BUDGET_MAX": ("budget", "max_total_usd"),
        "MITTENS_VERBOSITY": ("output", "verbosity"),
    }
    for env_key, (section, field) in env_map.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        if env_key == "MITTENS_CAPABILITIES":
            value = [v.strip() for v in value.split(",")]
        elif env_key == "MITTENS_BUDGET_MAX":
            value = float(value)
        config.setdefault(section, {})[field] = value
    return config


def load_config(
    project_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    model_override: str | None = None,
) -> MittensConfig:
    """Load configuration with full override chain.

    1. Defaults
    2. Global config (~/.config/mittens/config.toml)
    3. Project config (./mittens.toml or config_path)
    4. Environment variables (MITTENS_*)
    5. CLI flags (model_override)
    """
    project_dir = Path(project_dir) if project_dir else Path.cwd()

    # Layer 1: defaults
    config = dict(DEFAULTS)

    # Layer 2: global config
    global_path = Path.home() / ".config" / "mittens" / "config.toml"
    config = _deep_merge(config, _load_toml(global_path))

    # Layer 3: project config
    if config_path:
        config = _deep_merge(config, _load_toml(Path(config_path)))
    else:
        config = _deep_merge(config, _load_toml(project_dir / "mittens.toml"))

    # Layer 4: environment variables
    config = _apply_env(config)

    # Layer 5: CLI flags
    if model_override:
        config.setdefault("model", {})["default"] = model_override

    # Build config object
    project_name = config["mittens"].get("project_name") or project_dir.name
    mittens_dir = os.path.expanduser(config["mittens"]["mittens_dir"])

    # Parse plugin config
    plugins_section = config.get("plugins", {})
    plugins_enabled = plugins_section.get("enabled", [])
    plugin_config = {
        k: v for k, v in plugins_section.items()
        if k != "enabled" and isinstance(v, dict)
    }

    return MittensConfig(
        mittens_dir=mittens_dir,
        project_dir=str(project_dir),
        project_name=project_name,
        default_model=config["model"]["default"],
        hook_model=config["model"]["hook_model"],
        model_overrides=config["model"].get("override", {}),
        capabilities=set(config["capabilities"]["available"]),
        max_budget_usd=config["budget"].get("max_total_usd"),
        stream=config["output"]["stream"],
        verbosity=config["output"]["verbosity"],
        plugins_enabled=plugins_enabled,
        plugin_config=plugin_config,
    )
