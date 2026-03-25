"""Tests for WebSocket event broadcasting."""

import asyncio

import pytest

from mittens.types import LedgerEvent
from mittens.ws import EventBroadcaster


class TestEventBroadcaster:
    def test_subscribe(self):
        broadcaster = EventBroadcaster()
        queue = broadcaster.subscribe("run-1")
        assert queue is not None
        assert broadcaster.active_subscriptions == 1

    def test_unsubscribe(self):
        broadcaster = EventBroadcaster()
        queue = broadcaster.subscribe("run-1")
        broadcaster.unsubscribe("run-1", queue)
        assert broadcaster.active_subscriptions == 0

    def test_broadcast_to_subscriber(self):
        broadcaster = EventBroadcaster()
        queue = broadcaster.subscribe("run-1")
        event = LedgerEvent(
            event_type="PHASE_START",
            timestamp="2026-03-24T00:00:00Z",
            fields={"Phase": "orient"},
        )
        broadcaster.broadcast("run-1", event)
        assert not queue.empty()
        msg = queue.get_nowait()
        assert msg["event_type"] == "PHASE_START"
        assert msg["fields"]["Phase"] == "orient"

    def test_broadcast_to_multiple_subscribers(self):
        broadcaster = EventBroadcaster()
        q1 = broadcaster.subscribe("run-1")
        q2 = broadcaster.subscribe("run-1")
        event = LedgerEvent(
            event_type="TEST", timestamp="t", fields={}
        )
        broadcaster.broadcast("run-1", event)
        assert not q1.empty()
        assert not q2.empty()

    def test_broadcast_different_runs_isolated(self):
        broadcaster = EventBroadcaster()
        q1 = broadcaster.subscribe("run-1")
        q2 = broadcaster.subscribe("run-2")
        event = LedgerEvent(
            event_type="TEST", timestamp="t", fields={}
        )
        broadcaster.broadcast("run-1", event)
        assert not q1.empty()
        assert q2.empty()

    def test_unsubscribe_nonexistent(self):
        broadcaster = EventBroadcaster()
        queue = asyncio.Queue()
        broadcaster.unsubscribe("nonexistent", queue)  # Should not raise

    def test_make_callback(self):
        broadcaster = EventBroadcaster()
        queue = broadcaster.subscribe("run-1")
        callback = broadcaster.make_callback("run-1")
        event = LedgerEvent(
            event_type="PHASE_COMPLETE", timestamp="t",
            fields={"Phase": "orient", "Result": "PASS"},
        )
        callback(event)
        msg = queue.get_nowait()
        assert msg["event_type"] == "PHASE_COMPLETE"

    def test_queue_full_does_not_raise(self):
        broadcaster = EventBroadcaster()
        # Create a queue with maxsize=1
        queue = broadcaster.subscribe("run-1")
        # Override with tiny queue
        broadcaster._run_queues["run-1"] = [asyncio.Queue(maxsize=1)]
        tiny_q = broadcaster._run_queues["run-1"][0]

        event = LedgerEvent(event_type="E1", timestamp="t", fields={})
        broadcaster.broadcast("run-1", event)  # Fills queue
        broadcaster.broadcast("run-1", event)  # Should not raise (drops silently)
        assert tiny_q.qsize() == 1
