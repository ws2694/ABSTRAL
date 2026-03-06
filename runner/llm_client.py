"""Unified multi-provider LLM client for agent execution.

Routes API calls by model prefix:
  - claude-* → Anthropic Messages API
  - gpt-*    → OpenAI Chat Completions API
  - gemini-* → Google Generative AI API

All responses are normalized to LLMResponse for uniform handling
by the topology runner.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TextBlock:
    """Normalized text content block."""
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    """Normalized tool use content block."""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: list                    # list of TextBlock / ToolUseBlock
    stop_reason: str = "end_turn"    # "end_turn" | "tool_use"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0


def get_provider(model: str) -> str:
    """Determine provider from model name."""
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    elif m.startswith("gemini-"):
        return "google"
    else:
        return "anthropic"


# ── Anthropic ────────────────────────────────────────────────────────────────

async def _call_anthropic(
    model: str, system: Any, messages: list, tools: Optional[list],
    max_tokens: int,
) -> LLMResponse:
    """Call Anthropic Messages API."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    kwargs = dict(model=model, max_tokens=max_tokens,
                  system=system, messages=messages)
    if tools:
        kwargs["tools"] = tools

    response = await client.messages.create(**kwargs)

    usage = response.usage
    return LLMResponse(
        content=response.content,  # native Anthropic blocks (TextBlock, ToolUseBlock)
        stop_reason=response.stop_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_create_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


# ── OpenAI ───────────────────────────────────────────────────────────────────

def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schema to OpenAI function-calling format."""
    oai_tools = []
    for t in tools:
        schema = dict(t.get("input_schema", {}))
        schema.pop("cache_control", None)
        tool_def = {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": schema,
            }
        }
        oai_tools.append(tool_def)
    return oai_tools


def _anthropic_messages_to_openai(system: Any, messages: list) -> list[dict]:
    """Convert Anthropic message format to OpenAI format."""
    oai_messages = []

    # System prompt
    sys_text = ""
    if isinstance(system, str):
        sys_text = system
    elif isinstance(system, list):
        sys_text = " ".join(
            b["text"] for b in system if isinstance(b, dict) and "text" in b
        )
    if sys_text:
        oai_messages.append({"role": "system", "content": sys_text})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                oai_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Could be tool_result blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        oai_messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                    elif isinstance(block, str):
                        oai_messages.append({"role": "user", "content": block})

        elif role == "assistant":
            if isinstance(content, str):
                oai_messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # Anthropic assistant content blocks → OpenAI format
                text_parts = []
                tool_calls = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                }
                            })
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                }
                            })
                assistant_msg = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                if not text_parts and not tool_calls:
                    assistant_msg["content"] = ""
                oai_messages.append(assistant_msg)

    return oai_messages


async def _call_openai(
    model: str, system: Any, messages: list, tools: Optional[list],
    max_tokens: int,
) -> LLMResponse:
    """Call OpenAI Chat Completions API."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "openai package required for GPT models. Install: pip install openai"
        )

    client = AsyncOpenAI()
    oai_messages = _anthropic_messages_to_openai(system, messages)

    kwargs = dict(model=model, messages=oai_messages, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = _anthropic_tools_to_openai(tools)

    response = await client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    msg = choice.message

    # Normalize to Anthropic-like content blocks
    content = []
    if msg.content:
        content.append(TextBlock(text=msg.content))
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            content.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=args,
            ))

    stop_reason = "end_turn"
    if choice.finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif msg.tool_calls:
        stop_reason = "tool_use"

    usage = response.usage
    return LLMResponse(
        content=content,
        stop_reason=stop_reason,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        cache_read_tokens=0,
        cache_create_tokens=0,
    )


# ── Google (Gemini) ──────────────────────────────────────────────────────────

async def _call_google(
    model: str, system: Any, messages: list, tools: Optional[list],
    max_tokens: int,
) -> LLMResponse:
    """Call Google Generative AI API.

    Uses google-genai SDK. Converts Anthropic tool/message format.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package required for Gemini models. "
            "Install: pip install google-genai"
        )

    client = genai.Client()

    # System instruction
    sys_text = ""
    if isinstance(system, str):
        sys_text = system
    elif isinstance(system, list):
        sys_text = " ".join(
            b["text"] for b in system if isinstance(b, dict) and "text" in b
        )

    # Convert tools to Gemini format
    gemini_tools = None
    if tools:
        function_declarations = []
        for t in tools:
            schema = dict(t.get("input_schema", {}))
            schema.pop("cache_control", None)
            function_declarations.append(types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=schema,
            ))
        gemini_tools = [types.Tool(function_declarations=function_declarations)]

    # Convert messages to Gemini content format
    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        content = msg["content"]

        if isinstance(content, str):
            gemini_contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=content)]
            ))
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        parts.append(types.Part.from_function_response(
                            name=block.get("tool_name", "tool"),
                            response={"result": block.get("content", "")},
                        ))
                    elif block.get("type") == "text":
                        parts.append(types.Part.from_text(text=block.get("text", "")))
                elif hasattr(block, "type"):
                    if block.type == "text":
                        parts.append(types.Part.from_text(text=block.text))
                    elif block.type == "tool_use":
                        parts.append(types.Part.from_function_response(
                            name=block.name,
                            response=block.input,
                        ))
            if parts:
                gemini_contents.append(types.Content(role=role, parts=parts))

    config = types.GenerateContentConfig(
        system_instruction=sys_text if sys_text else None,
        max_output_tokens=max_tokens,
        tools=gemini_tools,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=gemini_contents,
        config=config,
    )

    # Normalize response
    content_blocks = []
    has_function_call = False
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if part.text:
                content_blocks.append(TextBlock(text=part.text))
            elif part.function_call:
                has_function_call = True
                content_blocks.append(ToolUseBlock(
                    id=f"gemini_{part.function_call.name}",
                    name=part.function_call.name,
                    input=dict(part.function_call.args) if part.function_call.args else {},
                ))

    usage_meta = response.usage_metadata
    return LLMResponse(
        content=content_blocks,
        stop_reason="tool_use" if has_function_call else "end_turn",
        input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        cache_read_tokens=0,
        cache_create_tokens=0,
    )


# ── Unified Entry Point ─────────────────────────────────────────────────────

async def llm_call(
    model: str,
    system: Any,
    messages: list,
    tools: Optional[list],
    max_tokens: int,
    max_retries: int = 8,
    initial_delay: float = 5.0,
    on_event=None,
    agent_id: str = "",
) -> LLMResponse:
    """Unified LLM call with retry logic. Routes to appropriate provider.

    Args:
        model: Model identifier (e.g. "claude-sonnet-4-20250514", "gpt-4o-mini")
        system: System prompt (string or Anthropic-style list)
        messages: Conversation messages in Anthropic format
        tools: Tool definitions in Anthropic format (auto-converted for other providers)
        max_tokens: Maximum output tokens
        max_retries: Max retry attempts on rate limit / overload
        initial_delay: Initial retry delay in seconds
        on_event: Optional event callback for logging
        agent_id: Agent identifier for event logging
    """
    def emit(etype, data=None):
        if on_event:
            on_event(etype, data or {})

    provider = get_provider(model)
    call_fn = {
        "anthropic": _call_anthropic,
        "openai": _call_openai,
        "google": _call_google,
    }[provider]

    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await call_fn(model, system, messages, tools, max_tokens)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                "rate" in err_str and "limit" in err_str
                or "429" in err_str
                or "too many requests" in err_str
            )
            is_overloaded = "overloaded" in err_str or "529" in err_str

            if is_rate_limit and attempt < max_retries - 1:
                wait = delay + (attempt * 2)
                print(f"    [{provider} rate-limit] Waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                emit("rate_limit_wait", {
                    "agent_id": agent_id, "wait_s": round(wait),
                    "attempt": attempt + 1, "provider": provider,
                })
                await asyncio.sleep(wait)
                delay = min(delay * 1.5, 60)
            elif is_overloaded and attempt < max_retries - 1:
                wait = delay + (attempt * 3)
                print(f"    [{provider} overloaded] Waiting {wait:.0f}s...")
                emit("api_overloaded", {
                    "agent_id": agent_id, "wait_s": round(wait),
                    "provider": provider,
                })
                await asyncio.sleep(wait)
                delay = min(delay * 1.5, 60)
            else:
                raise
