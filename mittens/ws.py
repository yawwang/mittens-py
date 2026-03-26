"""WebSocket event streaming for real-time workflow monitoring.

Each connected client receives LedgerEvent objects as JSON as they
are produced by the orchestrator. Uses asyncio.Queue per client for
backpressure handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from mittens.types import LedgerEvent

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Fan-out event broadcaster for WebSocket clients.

    Register as a ledger event_callback. Pushes events to all
    connected clients via per-client asyncio.Queue instances.
    """

    def __init__(self) -> None:
        self._run_queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Create a new subscription queue for a run."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        if run_id not in self._run_queues:
            self._run_queues[run_id] = []
        self._run_queues[run_id].append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscription queue."""
        if run_id in self._run_queues:
            try:
                self._run_queues[run_id].remove(queue)
            except ValueError:
                pass
            if not self._run_queues[run_id]:
                del self._run_queues[run_id]

    def broadcast(self, run_id: str, event: LedgerEvent) -> None:
        """Push an event to all subscribers for a run.

        This is called synchronously from the ledger callback.
        Uses put_nowait to avoid blocking the orchestrator.
        """
        message = _event_to_dict(event)
        queues = self._run_queues.get(run_id, [])
        for queue in queues:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("WebSocket queue full, dropping event")

    def make_callback(self, run_id: str):
        """Create a ledger event_callback bound to a specific run_id."""
        def callback(event: LedgerEvent) -> None:
            self.broadcast(run_id, event)
        return callback

    @property
    def active_subscriptions(self) -> int:
        return sum(len(qs) for qs in self._run_queues.values())


async def websocket_handler(
    websocket: WebSocket,
    run_id: str,
    broadcaster: EventBroadcaster,
) -> None:
    """Handle a WebSocket connection for streaming run events."""
    await websocket.accept()
    queue = broadcaster.subscribe(run_id)

    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        broadcaster.unsubscribe(run_id, queue)


def _event_to_dict(event: LedgerEvent) -> dict[str, Any]:
    """Convert a LedgerEvent to a JSON-serializable dict."""
    return {
        "type": "event",
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "fields": {k: str(v) for k, v in event.fields.items()},
    }
