"""Tests for git worktree management (mocked subprocess)."""

from unittest.mock import MagicMock, patch

import pytest

from mittens.worktrees import (
    create_worktree,
    delete_branch,
    list_worktrees,
    merge_worktree,
    remove_worktree,
)


class TestCreateWorktree:
    @patch("mittens.worktrees.subprocess.run")
    def test_creates_worktree(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        path = create_worktree("/home/user/project", "task-a")
        assert "task-a" in path
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert "git" in args[0][0]
        assert "worktree" in args[0][0]
        assert "add" in args[0][0]

    @patch("mittens.worktrees.subprocess.run")
    def test_returns_sibling_path(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        path = create_worktree("/home/user/project", "feature-x")
        assert "project-feature-x" in path


class TestRemoveWorktree:
    @patch("mittens.worktrees.subprocess.run")
    def test_removes_worktree(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        remove_worktree("/home/user/project", "/home/user/project-task-a")
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert "remove" in args[0][0]


class TestMergeWorktree:
    @patch("mittens.worktrees.subprocess.run")
    def test_merges_branch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Merge complete")
        result = merge_worktree("/home/user/project", "task-a")
        assert result == "Merge complete"
        args = mock_run.call_args
        assert "merge" in args[0][0]
        assert "task-a" in args[0][0]


class TestListWorktrees:
    @patch("mittens.worktrees.subprocess.run")
    def test_lists_worktrees(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="worktree /home/user/project\nworktree /home/user/project-task-a\n"
        )
        paths = list_worktrees("/home/user/project")
        assert len(paths) == 2
        assert "/home/user/project" in paths
        assert "/home/user/project-task-a" in paths


class TestDeleteBranch:
    @patch("mittens.worktrees.subprocess.run")
    def test_deletes_branch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        delete_branch("/home/user/project", "task-a")
        args = mock_run.call_args
        assert "branch" in args[0][0]
        assert "-d" in args[0][0]
        assert "task-a" in args[0][0]
