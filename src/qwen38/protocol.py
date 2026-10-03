"""OpenAI Chat Completions protocol layer for the Qwen3.8 server.

Everything here is pure Python: request validation and normalization into the
structured messages the Qwen3.8 chat template renders, the template's native
tool-call syntax (``qwen3_coder`` XML), and incremental assembly of streamed
assistant output.  No MLX import, so it is testable without a Metal device.

Two views of one assistant turn exist and must not be confused:

- the generated token IDs, which are what the model cache contains and are
  the only input to cache-reuse decisions;
- the parsed API message (``content`` + ``tool_calls``), a semantic view
  derived deterministically from the decoded text.

The API view is never re-serialized to recover tokens.  ``messages_equal``
only decides whether a replayed message is the one this server produced, so
that the session can substitute the authoritative token IDs for it.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
CONTEXT_OVERFLOW_CODE = "context_length_exceeded"

_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_ROLE_ALIASES = {"developer": "system"}
_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


class ProtocolError(ValueError):
    """A malformed or unsupported request; maps to HTTP 400."""

    def __init__(self, message: str, code: str | None = None, param: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.param = param


# ---------------------------------------------------------------------------
# Request normalization
# ---------------------------------------------------------------------------


def content_text(content: Any, where: str) -> str:
    """Return the text of a string or text-part content value.

    Non-text parts (images, audio, files) are rejected explicitly: this is a
    text-only model and silently dropping or stringifying them would corrupt
    the prompt.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for index, item in enumerate(content):
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                raise ProtocolError(f"{where}.content[{index}] must be an object")
            kind = item.get("type")
            if kind != "text":
                raise ProtocolError(
                    f"{where}.content[{index}] has unsupported type {kind!r}; this model is text-only",
                    param=f"{where}.content",
                )
            text = item.get("text")
            if not isinstance(text, str):
                raise ProtocolError(f"{where}.content[{index}].text must be a string")
            parts.append(text)
        return "".join(parts)
    raise ProtocolError(f"{where}.content must be a string, a list of text parts, or null")


def _arguments_object(value: Any, where: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"{where} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ProtocolError(f"{where} must encode a JSON object")
        return parsed
    raise ProtocolError(f"{where} must be a JSON object string")


def _normalize_tool_calls(raw: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ProtocolError(f"{where}.tool_calls must be a list")
    calls: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        here = f"{where}.tool_calls[{index}]"
        if not isinstance(item, dict):
            raise ProtocolError(f"{here} must be an object")
        kind = item.get("type", "function")
        if kind != "function":
            raise ProtocolError(f"{here}.type {kind!r} is not supported")
        function = item.get("function")
        if not isinstance(function, dict):
            raise ProtocolError(f"{here}.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError(f"{here}.function.name must be a non-empty string")
        call_id = item.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise ProtocolError(f"{here}.id must be a string")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _arguments_object(
                        function.get("arguments"), f"{here}.function.arguments"
                    ),
                },
            }
        )
    return calls


def normalize_messages(raw_messages: Any) -> list[dict[str, Any]]:
    """Validate OpenAI messages and convert them to template-ready dicts.

    The result keeps the structure the Qwen3.8 template renders: assistant
    ``tool_calls`` with ``arguments`` as a mapping, and ``tool_call_id`` on
    tool results.  Text content parts are joined; other parts are rejected.
    """

    if not isinstance(raw_messages, list) or not raw_messages:
        raise ProtocolError("messages must be a non-empty list", param="messages")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        where = f"messages[{index}]"
        if not isinstance(raw, dict):
            raise ProtocolError(f"{where} must be an object")
        role = _ROLE_ALIASES.get(raw.get("role"), raw.get("role"))
        if role not in _ALLOWED_ROLES:
            raise ProtocolError(
                f"{where}.role must be one of {sorted(_ALLOWED_ROLES | set(_ROLE_ALIASES))}"
            )
        text = content_text(raw.get("content"), where)
        if role == "system" and index != 0:
            raise ProtocolError(f"{where}: a system message is only supported first")
        if role == "user" and not text.strip():
            raise ProtocolError(f"{where} has empty user content")
        message: dict[str, Any] = {"role": role, "content": text}
        if role == "assistant" and raw.get("tool_calls"):
            message["tool_calls"] = _normalize_tool_calls(raw["tool_calls"], where)
        if role == "tool":
            call_id = raw.get("tool_call_id")
            if call_id is not None and not isinstance(call_id, str):
                raise ProtocolError(f"{where}.tool_call_id must be a string")
            message["tool_call_id"] = call_id
        normalized.append(message)
    return normalized


def normalize_tools(raw_tools: Any) -> list[dict[str, Any]] | None:
    """Validate function tools; return them unchanged (order and keys kept).

    The template renders each tool with ``tojson`` in the given order and key
    order, so the exact objects are the prompt identity.  An empty list means
    no tools.
    """

    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list):
        raise ProtocolError("tools must be a list", param="tools")
    if not raw_tools:
        return None
    names: set[str] = set()
    for index, tool in enumerate(raw_tools):
        where = f"tools[{index}]"
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ProtocolError(f"{where} must be a function tool", param="tools")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ProtocolError(f"{where}.function must be an object", param="tools")
        name = function.get("name")
        if not isinstance(name, str) or not _FUNCTION_NAME_RE.match(name):
            raise ProtocolError(f"{where}.function.name is missing or invalid", param="tools")
        if name in names:
            raise ProtocolError(f"duplicate tool name {name!r}", param="tools")
        names.add(name)
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise ProtocolError(f"{where}.function.description must be a string", param="tools")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ProtocolError(f"{where}.function.parameters must be a JSON Schema object", param="tools")
    return raw_tools


def tools_identity(tools: Sequence[dict[str, Any]] | None) -> str | None:
    """Exact rendered-order identity of an effective tool set."""

    if not tools:
        return None
    return json.dumps(list(tools), ensure_ascii=False)


def resolve_tool_choice(value: Any, tools: list[dict[str, Any]] | None) -> str:
    """Return ``auto`` or ``none``; reject forms the prompt cannot enforce."""

    if value is None or value == "auto":
        return "auto"
    if value == "none":
        return "none"
    if value == "required" or isinstance(value, dict):
        raise ProtocolError(
            "tool_choice 'required' and named-function forcing are not supported: "
            "this server cannot constrain decoding; use 'auto' or 'none'",
            param="tool_choice",
        )
    raise ProtocolError("tool_choice must be 'auto' or 'none'", param="tool_choice")


@dataclass
class ChatRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    tool_choice: str
    stream: bool
    include_usage: bool
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    mode: str | None
    session_id: str | None

    @property
    def effective_tools(self) -> list[dict[str, Any]] | None:
        """Tools rendered into the prompt (none when tool_choice is 'none')."""

        return self.tools if self.tool_choice == "auto" else None


def _optional_number(body: dict[str, Any], key: str) -> float | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{key} must be a number", param=key)
    return float(value)


def parse_chat_request(body: Any) -> ChatRequest:
    """Validate a ``/v1/chat/completions`` body.

    Unknown harmless fields (``store``, ``metadata``, ``user``, ``seed``,
    ``prompt_cache_key`` ...) are ignored, as OpenAI-compatible servers do.
    Fields whose semantics this server cannot honor are rejected.
    """

    if not isinstance(body, dict):
        raise ProtocolError("request body must be a JSON object")
    messages = normalize_messages(body.get("messages"))
    tools = normalize_tools(body.get("tools"))
    tool_choice = resolve_tool_choice(body.get("tool_choice"), tools)
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise ProtocolError("stream must be a boolean", param="stream")
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise ProtocolError("stream_options must be an object", param="stream_options")
    n = body.get("n", 1)
    if n not in (None, 1):
        raise ProtocolError("only n=1 is supported", param="n")
    stop = body.get("stop")
    if stop not in (None, [], ""):
        raise ProtocolError("stop sequences are not supported", param="stop")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise ProtocolError("only response_format type 'text' is supported", param="response_format")
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ProtocolError("max_completion_tokens/max_tokens must be a positive integer")
    mode = body.get("mode")
    if mode is not None and mode not in {"greedy", "sample"}:
        raise ProtocolError("mode must be greedy or sample", param="mode")
    session_id = body.get("session_id")
    return ChatRequest(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream,
        include_usage=bool(stream_options.get("include_usage")),
        max_tokens=max_tokens,
        temperature=_optional_number(body, "temperature"),
        top_p=_optional_number(body, "top_p"),
        mode=mode,
        session_id=None if session_id is None else str(session_id),
    )


# ---------------------------------------------------------------------------
# Transcript identity
# ---------------------------------------------------------------------------


def _message_key(message: dict[str, Any]) -> tuple[Any, ...]:
    role = message["role"]
    if role == "assistant":
        # The template renders assistant content trimmed, and a tool-only turn
        # may be replayed with null or "" content.  Tool calls compare by id,
        # name and parsed arguments, because clients re-serialize arguments.
        calls = tuple(
            (
                call.get("id"),
                call["function"]["name"],
                json.dumps(call["function"]["arguments"], sort_keys=True, ensure_ascii=False),
            )
            for call in message.get("tool_calls") or ()
        )
        return (role, (message.get("content") or "").strip(), calls)
    if role == "tool":
        return (role, message.get("content") or "", message.get("tool_call_id"))
    return (role, message.get("content") or "")


def messages_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _message_key(left) == _message_key(right)


def extends_transcript(
    previous: Sequence[dict[str, Any]],
    incoming: Sequence[dict[str, Any]],
) -> bool:
    """True when ``incoming`` begins with exactly the stored transcript."""

    if len(incoming) <= len(previous):
        return False
    return all(messages_equal(a, b) for a, b in zip(previous, incoming))


# ---------------------------------------------------------------------------
# Qwen3.8 tool-call parsing (qwen3_coder XML, as the chat template defines it)
# ---------------------------------------------------------------------------


@dataclass
class ParsedAssistant:
    """API view of one generated assistant turn."""

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None
    unknown_tools: list[str] = field(default_factory=list)
    argument_type_mismatches: list[str] = field(default_factory=list)

    def message(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": json.dumps(
                            call["function"]["arguments"], ensure_ascii=False
                        ),
                    },
                }
                for call in self.tool_calls
            ]
        return out

    def stored_message(self) -> dict[str, Any]:
        """Normalized form kept in the session transcript."""

        out: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            out["tool_calls"] = [dict(call) for call in self.tool_calls]
        return out


class _Malformed(Exception):
    pass


def _schema_properties(tools: Sequence[dict[str, Any]] | None, name: str) -> dict[str, Any] | None:
    for tool in tools or ():
        function = tool.get("function") or {}
        if function.get("name") == name:
            parameters = function.get("parameters") or {}
            properties = parameters.get("properties") if isinstance(parameters, dict) else None
            return properties if isinstance(properties, dict) else {}
    return None


def _schema_types(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    if isinstance(kind, list):
        return [item for item in kind if isinstance(item, str)]
    if "enum" in schema and all(isinstance(item, str) for item in schema["enum"]):
        return ["string"]
    return []


def convert_parameter(raw: str, schema: Any) -> tuple[Any, bool]:
    """Invert the template's rendering of one parameter value.

    The template writes strings verbatim and everything else with ``tojson``.
    Returns ``(value, ok)``; ``ok`` is False when the text does not fit the
    declared type, in which case the raw string is kept so the client's own
    schema validation reports the problem to the model.
    """

    types = _schema_types(schema)
    if not types or "string" in types:
        if raw == "null" and "null" in types:
            return None, True
        return raw, True
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.strip().lower()
        if "boolean" in types and lowered in {"true", "false"}:
            return lowered == "true", True
        return raw, False
    checks: dict[str, Callable[[Any], bool]] = {
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
    }
    if any(checks.get(kind, lambda _v: False)(value) for kind in types):
        return value, True
    if "integer" in types and isinstance(value, float) and value.is_integer():
        return int(value), True
    return raw, False


def _parse_parameter_value(text: str, start: int) -> tuple[str, int]:
    """Return (value, index after ``</parameter>``) for a value at ``start``.

    A value may itself contain ``</parameter>`` (for example when writing
    this parser), so the closing tag is the first one followed, after
    optional whitespace, by another parameter or the function close.
    """

    if text.startswith("\n", start):
        start += 1
    search = start
    first_close: int | None = None
    while True:
        close = text.find("</parameter>", search)
        if close < 0:
            if first_close is None:
                raise _Malformed("unterminated <parameter>")
            close = first_close
            break
        if first_close is None:
            first_close = close
        after = close + len("</parameter>")
        rest = text[after:].lstrip()
        if rest.startswith("<parameter=") or rest.startswith("</function>"):
            break
        search = after
    value = text[start:close]
    if value.endswith("\n"):
        value = value[:-1]
    return value, close + len("</parameter>")


def _expect(text: str, pos: int, token: str) -> int:
    if not text.startswith(token, pos):
        raise _Malformed(f"expected {token!r} at offset {pos}")
    return pos + len(token)


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos


def parse_tool_call_block(
    text: str,
    tools: Sequence[dict[str, Any]] | None,
    new_id: Callable[[], str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Parse one or more ``<tool_call>`` blocks; nothing but whitespace may follow."""

    calls: list[dict[str, Any]] = []
    unknown: list[str] = []
    mismatches: list[str] = []
    pos = _skip_ws(text, 0)
    while pos < len(text):
        pos = _expect(text, pos, TOOL_CALL_START)
        pos = _skip_ws(text, pos)
        pos = _expect(text, pos, "<function=")
        close = text.find(">", pos)
        if close < 0:
            raise _Malformed("unterminated <function=")
        name = text[pos:close].strip()
        if not _FUNCTION_NAME_RE.match(name):
            raise _Malformed(f"invalid function name {name!r}")
        pos = close + 1
        properties = _schema_properties(tools, name)
        if properties is None:
            unknown.append(name)
        arguments: dict[str, Any] = {}
        while True:
            pos = _skip_ws(text, pos)
            if text.startswith("</function>", pos):
                pos += len("</function>")
                break
            pos = _expect(text, pos, "<parameter=")
            close = text.find(">", pos)
            if close < 0:
                raise _Malformed("unterminated <parameter=")
            parameter = text[pos:close].strip()
            if not parameter:
                raise _Malformed("empty parameter name")
            if parameter in arguments:
                raise _Malformed(f"duplicate parameter {parameter!r}")
            raw, pos = _parse_parameter_value(text, close + 1)
            value, ok = convert_parameter(raw, (properties or {}).get(parameter))
            if not ok:
                mismatches.append(f"{name}.{parameter}")
            arguments[parameter] = value
        pos = _skip_ws(text, pos)
        pos = _expect(text, pos, TOOL_CALL_END)
        calls.append(
            {"id": new_id(), "type": "function", "function": {"name": name, "arguments": arguments}}
        )
        pos = _skip_ws(text, pos)
    if not calls:
        raise _Malformed("no tool call")
    return calls, unknown, mismatches


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def parse_assistant_text(
    text: str,
    tools: Sequence[dict[str, Any]] | None,
    *,
    new_id: Callable[[], str] = new_call_id,
) -> ParsedAssistant:
    """Split decoded assistant text into content and native tool calls.

    Tool calls are only recognized when tools are in effect.  Any malformed
    or truncated tool syntax leaves the whole text as content and records
    ``parse_error``; it never becomes a partial successful call.
    """

    marker = text.find(TOOL_CALL_START) if tools else -1
    if marker < 0:
        return ParsedAssistant(content=text)
    try:
        calls, unknown, mismatches = parse_tool_call_block(text[marker:], tools, new_id)
    except _Malformed as exc:
        return ParsedAssistant(content=text, parse_error=str(exc))
    content = text[:marker].rstrip()
    return ParsedAssistant(
        content=content or None,
        tool_calls=calls,
        unknown_tools=unknown,
        argument_type_mismatches=mismatches,
    )


# ---------------------------------------------------------------------------
# Streaming assembly
# ---------------------------------------------------------------------------


class StreamAssembler:
    """Turn a growing decoded text into safe ``content`` deltas.

    ``feed(text)`` receives the full decoded text so far and returns the new
    content that is final.  It holds back:

    - a trailing U+FFFD, which marks an incomplete UTF-8 sequence that the
      next token may complete;
    - trailing whitespace, which is dropped before a tool call;
    - any suffix that could be the start of ``<tool_call>``;
    - everything from ``<tool_call>`` on, which is parsed at the end.

    ``finish(parsed)`` returns the remaining content so that the concatenated
    deltas equal ``parsed.content`` exactly (the non-streaming result).
    """

    def __init__(self, tools_active: bool) -> None:
        self.tools_active = tools_active
        self.emitted = ""
        self.in_tool_call = False

    def _safe_prefix(self, text: str) -> str:
        if self.tools_active:
            marker = text.find(TOOL_CALL_START)
            if marker >= 0:
                self.in_tool_call = True
                text = text[:marker]
            else:
                for size in range(min(len(TOOL_CALL_START) - 1, len(text)), 0, -1):
                    if TOOL_CALL_START.startswith(text[-size:]):
                        text = text[:-size]
                        break
        text = text.rstrip("�")
        return text.rstrip()

    def feed(self, text: str) -> str:
        if self.in_tool_call:
            return ""
        safe = self._safe_prefix(text)
        if len(safe) <= len(self.emitted) or not safe.startswith(self.emitted):
            return ""
        delta = safe[len(self.emitted) :]
        self.emitted = safe
        return delta

    def finish(self, parsed: ParsedAssistant) -> str:
        final = parsed.content or ""
        if not final.startswith(self.emitted):
            # Only possible if decoding was not prefix-stable; the streamed
            # text would no longer match the transcript the client stores.
            raise RuntimeError("streamed content is not a prefix of the final content")
        delta = final[len(self.emitted) :]
        self.emitted = final
        return delta


# ---------------------------------------------------------------------------
# Response objects
# ---------------------------------------------------------------------------


def usage_object(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def chunk_object(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any] | None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    choices = [] if delta is None else [
        {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
    ]
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }


def tool_call_deltas(
    calls: Sequence[dict[str, Any]],
    fragment_chars: int = 256,
) -> list[dict[str, Any]]:
    """Deltas announcing each call (index, id, name) then its argument fragments."""

    deltas: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": ""},
                    }
                ]
            }
        )
        arguments = json.dumps(call["function"]["arguments"], ensure_ascii=False)
        for start in range(0, len(arguments), fragment_chars):
            deltas.append(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "function": {"arguments": arguments[start : start + fragment_chars]},
                        }
                    ]
                }
            )
    return deltas


def sse_event(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode("utf-8")


def context_overflow_message(limit: int, requested: int) -> str:
    # OpenAI wording; clients (including Pi) classify it as context overflow.
    return (
        f"This model's maximum context length is {limit} tokens. "
        f"However, your messages resulted in {requested} tokens."
    )
