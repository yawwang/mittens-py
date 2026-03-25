"""Git worktree management for parallel talent instances.

Provides create/remove/merge operations for isolated working copies.
Each parallel instance gets its own worktree so file writes don't conflict.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def create_worktree(project_dir: str, branch_name: str) -> str:
    """Create a git worktree for isolated parallel work.

    Returns the absolute path to the new worktree directory.
    """
    project = Path(project_dir).resolve()
    worktree_path = project.parent / f"{project.name}-{branch_name}"

    subprocess.run(
        ["git", "worktree", "add", str(worktree_path), "-b", branch_name],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Created worktree at %s (branch: %s)", worktree_path, branch_name)
    return str(worktree_path)


def remove_worktree(project_dir: str, worktree_path: str) -> None:
    """Remove a git worktree and prune."""
    subprocess.run(
        ["git", "worktree", "remove", worktree_path, "--force"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Removed worktree: %s", worktree_path)


def merge_worktree(project_dir: str, branch_name: str) -> str:
    """Merge a worktree branch back into the current branch.

    Returns the merge output.
    """
    result = subprocess.run(
        ["git", "merge", branch_name, "--no-edit"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Merged branch %s", branch_name)
    return result.stdout


def delete_branch(project_dir: str, branch_name: str) -> None:
    """Delete a local branch after merge."""
    subprocess.run(
        ["git", "branch", "-d", branch_name],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    )


def list_worktrees(project_dir: str) -> list[str]:
    """List all active worktree paths."""
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            paths.append(line.split(" ", 1)[1])
    return paths


# Async variants for use in AsyncOrchestrator


async def acreate_worktree(project_dir: str, branch_name: str) -> str:
    """Async version of create_worktree."""
    project = Path(project_dir).resolve()
    worktree_path = project.parent / f"{project.name}-{branch_name}"

    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "add", str(worktree_path), "-b", branch_name,
        cwd=project_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, "git worktree add",
            output=stdout.decode(), stderr=stderr.decode(),
        )
    logger.info("Created worktree at %s (branch: %s)", worktree_path, branch_name)
    return str(worktree_path)


async def aremove_worktree(project_dir: str, worktree_path: str) -> None:
    """Async version of remove_worktree."""
    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "remove", worktree_path, "--force",
        cwd=project_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


async def amerge_worktree(project_dir: str, branch_name: str) -> str:
    """Async version of merge_worktree."""
    proc = await asyncio.create_subprocess_exec(
        "git", "merge", branch_name, "--no-edit",
        cwd=project_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, "git merge",
            output=stdout.decode(), stderr=stderr.decode(),
        )
    return stdout.decode()
