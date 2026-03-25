"""LLM adapter — the only module that imports litellm.

Provides a model-agnostic interface for completions, tool use,
and streaming. Tracks token usage per-talent for cost reporting.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, AsyncIterator, Iterator

import litellm

from mittens.types import LLMResponse, LLMToolResponse, ToolCall

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True

# OpenAI function-calling format tool definitions for file I/O and bash
TOOL_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
}

TOOL_WRITE_FILE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file (creates parent directories if needed)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
}

TOOL_RUN_BASH = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Execute a bash command and return stdout/stderr",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"},
            },
            "required": ["command"],
        },
    },
}


def tools_for_capabilities(capabilities: set[str]) -> list[dict]:
    """Build tool definitions based on available capabilities."""
    tools = []
    if "file_read" in capabilities:
        tools.append(TOOL_READ_FILE)
    if "file_write" in capabilities:
        tools.append(TOOL_WRITE_FILE)
    if "bash" in capabilities:
        tools.append(TOOL_RUN_BASH)
    return tools


def _parse_tool_args(args: Any) -> dict:
    """Parse tool call arguments from string or dict."""
    if isinstance(args, str):
        if not args:
            return {}
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
    return args if args else {}


def _build_tool_calls_from_fragments(
    fragments: dict[int, dict],
) -> list[ToolCall]:
    """Build ToolCall list from accumulated streaming fragments."""
    tool_calls = []
    for idx in sorted(fragments):
        frag = fragments[idx]
        args = _parse_tool_args(frag["arguments"])
        tool_calls.append(
            ToolCall(id=frag["id"], name=frag["name"], arguments=args)
        )
    return tool_calls


def _accumulate_fragment(
    fragments: dict[int, dict], tc_delta: Any
) -> None:
    """Accumulate a single streaming tool call delta into fragments."""
    idx = tc_delta.index
    if idx not in fragments:
        fragments[idx] = {"id": "", "name": "", "arguments": ""}
    frag = fragments[idx]
    if tc_delta.id:
        frag["id"] = tc_delta.id
    if tc_delta.function:
        if tc_delta.function.name:
            frag["name"] = tc_delta.function.name
        if tc_delta.function.arguments:
            frag["arguments"] += tc_delta.function.arguments


class LLMAdapter:
    """Model-agnostic LLM client wrapping litellm."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._per_talent: dict[str, tuple[int, int]] = {}

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Single or multi-turn completion without tool use."""
        all_messages = [{"role": "system", "content": system}] + messages

        response = litellm.completion(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        result = LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
        )
        self._track(result)
        return result

    def complete_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMToolResponse:
        """Completion with tool use (function calling)."""
        all_messages = [{"role": "system", "content": system}] + messages

        response = litellm.completion(
            model=self.model,
            messages=all_messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        message = response.choices[0].message
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=_parse_tool_args(tc.function.arguments),
                    )
                )

        result = LLMToolResponse(
            content=message.content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
        )
        self._track(result)
        return result

    def stream(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """Streaming completion for interactive terminal output."""
        all_messages = [{"role": "system", "content": system}] + messages

        response = litellm.completion(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def stream_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        output_stream=None,
    ) -> LLMToolResponse:
        """Streaming completion that also captures tool calls.

        Streams content chunks to output_stream (default: stderr) as they
        arrive, while accumulating tool call fragments. Returns the final
        LLMToolResponse once the stream ends.
        """
        if output_stream is None:
            output_stream = sys.stderr

        all_messages = [{"role": "system", "content": system}] + messages
        response = litellm.completion(
            model=self.model,
            messages=all_messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
            stream_options={"include_usage": True},
        )

        content_parts: list[str] = []
        tool_call_fragments: dict[int, dict] = {}
        input_tokens = 0
        output_tokens = 0

        for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta:
                delta = choice.delta
                if delta.content:
                    content_parts.append(delta.content)
                    output_stream.write(delta.content)
                    output_stream.flush()
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        _accumulate_fragment(tool_call_fragments, tc_delta)

            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        if content_parts:
            output_stream.write("\n")
            output_stream.flush()

        content = "".join(content_parts) or None
        result = LLMToolResponse(
            content=content,
            tool_calls=_build_tool_calls_from_fragments(tool_call_fragments),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )
        self._track(result)
        return result

    # -- Async variants (for AsyncOrchestrator / v2+) --

    async def acomplete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Async single/multi-turn completion without tool use."""
        all_messages = [{"role": "system", "content": system}] + messages

        response = await litellm.acompletion(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        result = LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
        )
        self._track(result)
        return result

    async def acomplete_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMToolResponse:
        """Async completion with tool use (function calling)."""
        all_messages = [{"role": "system", "content": system}] + messages

        response = await litellm.acompletion(
            model=self.model,
            messages=all_messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        message = response.choices[0].message
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=_parse_tool_args(tc.function.arguments),
                    )
                )

        result = LLMToolResponse(
            content=message.content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
        )
        self._track(result)
        return result

    async def astream_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        output_stream=None,
    ) -> LLMToolResponse:
        """Async streaming completion that captures tool calls."""
        if output_stream is None:
            output_stream = sys.stderr

        all_messages = [{"role": "system", "content": system}] + messages
        response = await litellm.acompletion(
            model=self.model,
            messages=all_messages,
            tools=tools if tools else None,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
            stream_options={"include_usage": True},
        )

        content_parts: list[str] = []
        tool_call_fragments: dict[int, dict] = {}
        input_tokens = 0
        output_tokens = 0

        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta:
                delta = choice.delta
                if delta.content:
                    content_parts.append(delta.content)
                    output_stream.write(delta.content)
                    output_stream.flush()
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        _accumulate_fragment(tool_call_fragments, tc_delta)
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        if content_parts:
            output_stream.write("\n")
            output_stream.flush()

        content = "".join(content_parts) or None
        result = LLMToolResponse(
            content=content,
            tool_calls=_build_tool_calls_from_fragments(tool_call_fragments),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )
        self._track(result)
        return result

    def track_for_talent(self, talent_id: str, tokens: tuple[int, int]) -> None:
        """Accumulate token counts for a specific talent."""
        prev_in, prev_out = self._per_talent.get(talent_id, (0, 0))
        self._per_talent[talent_id] = (
            prev_in + tokens[0],
            prev_out + tokens[1],
        )

    def cost_summary(self) -> dict[str, Any]:
        """Return token usage breakdown."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "per_talent": {
                tid: {"input": inp, "output": out}
                for tid, (inp, out) in self._per_talent.items()
            },
        }

    def _track(self, response: LLMResponse | LLMToolResponse) -> None:
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens


class CostAggregator:
    """Collects cost data from multiple LLMAdapter instances."""

    def __init__(self) -> None:
        self._adapters: list[LLMAdapter] = []

    def register(self, adapter: LLMAdapter) -> None:
        if adapter not in self._adapters:
            self._adapters.append(adapter)

    def summary(self) -> dict[str, Any]:
        total_in = sum(a.total_input_tokens for a in self._adapters)
        total_out = sum(a.total_output_tokens for a in self._adapters)
        per_talent: dict[str, dict[str, int]] = {}
        for adapter in self._adapters:
            for tid, (inp, out) in adapter._per_talent.items():
                if tid in per_talent:
                    per_talent[tid]["input"] += inp
                    per_talent[tid]["output"] += out
                else:
                    per_talent[tid] = {"input": inp, "output": out}
        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "models": [a.model for a in self._adapters],
            "per_talent": per_talent,
        }
