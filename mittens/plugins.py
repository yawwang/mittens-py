"""Plugin loader — discovers and loads Mittens plugins.

Plugins are discovered via two mechanisms:
1. Python packages with entry_points group "mittens.plugins"
2. Python packages with the "mittens_" name prefix

Each plugin must expose a register(registry: PluginRegistry) function.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from mittens.plugin_api import PluginRegistry, flush_pending

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata about a loaded plugin."""

    name: str
    version: str = "unknown"
    skills: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)


class PluginLoader:
    """Discovers and loads Mittens plugins."""

    def __init__(self, enabled: list[str] | None = None):
        self._enabled = set(enabled) if enabled else None
        self._registry = PluginRegistry()
        self._loaded: list[PluginInfo] = []

    @property
    def registry(self) -> PluginRegistry:
        return self._registry

    @property
    def loaded_plugins(self) -> list[PluginInfo]:
        return list(self._loaded)

    def load_all(self) -> None:
        """Discover and load all plugins."""
        self._load_from_entry_points()
        self._load_from_prefix()
        # Flush any pending decorator registrations
        flush_pending(self._registry)

    def load_module(self, module_name: str) -> PluginInfo | None:
        """Load a single plugin by module name."""
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning("Failed to import plugin %s: %s", module_name, e)
            return None

        return self._register_module(mod, module_name)

    def _load_from_entry_points(self) -> None:
        """Load plugins registered via entry_points."""
        try:
            eps = entry_points(group="mittens.plugins")
        except TypeError:
            # Python < 3.12 compatibility
            eps = entry_points().get("mittens.plugins", [])

        for ep in eps:
            if self._enabled is not None and ep.name not in self._enabled:
                logger.debug("Skipping disabled plugin: %s", ep.name)
                continue

            try:
                plugin_obj = ep.load()
                if callable(plugin_obj):
                    # It's the register function directly
                    plugin_obj(self._registry)
                    self._loaded.append(PluginInfo(
                        name=ep.name,
                        skills=self._registry.skill_names,
                        hooks=self._registry.hook_names,
                    ))
                elif hasattr(plugin_obj, "register"):
                    plugin_obj.register(self._registry)
                    self._loaded.append(PluginInfo(
                        name=ep.name,
                        skills=self._registry.skill_names,
                        hooks=self._registry.hook_names,
                    ))
                logger.info("Loaded plugin: %s", ep.name)
            except Exception as e:
                logger.warning("Failed to load plugin %s: %s", ep.name, e)

    def _load_from_prefix(self) -> None:
        """Load plugins matching the mittens_ package prefix."""
        import pkgutil

        for importer, modname, ispkg in pkgutil.iter_modules():
            if not modname.startswith("mittens_"):
                continue
            if self._enabled is not None and modname not in self._enabled:
                continue

            try:
                mod = importlib.import_module(modname)
                self._register_module(mod, modname)
            except ImportError as e:
                logger.warning("Failed to import %s: %s", modname, e)

    def _register_module(self, mod: Any, name: str) -> PluginInfo | None:
        """Register a module as a plugin."""
        if not hasattr(mod, "register"):
            logger.warning("Plugin %s has no register() function", name)
            return None

        try:
            skills_before = set(self._registry.skill_names)
            hooks_before = set(self._registry.hook_names)

            mod.register(self._registry)

            # Also flush any decorator registrations
            flush_pending(self._registry)

            new_skills = set(self._registry.skill_names) - skills_before
            new_hooks = set(self._registry.hook_names) - hooks_before

            info = PluginInfo(
                name=name,
                version=getattr(mod, "__version__", "unknown"),
                skills=list(new_skills),
                hooks=list(new_hooks),
            )
            self._loaded.append(info)
            logger.info("Loaded plugin: %s (skills=%s, hooks=%s)", name, new_skills, new_hooks)
            return info
        except Exception as e:
            logger.warning("Failed to register plugin %s: %s", name, e)
            return None
