"""Prompt-emulated tool calling for DeepSeek's web chat.

chat.deepseek.com has no native function-calling channel: the completion body
accepts only text plus the thinking/search flags, and a `tools` field is
silently ignored (verified against the live backend). So tools are serialised
into the prompt and the model's reply is parsed back into OpenAI-style
`tool_calls`.

The protocol is deliberately strict and marker-based: it survives streaming,
multi-tool replies, and the model's habit of adding a short preamble.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"

_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S | re.I)
_PIPE_TAG_RE = re.compile(r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:tool_call|tool|json)?[ \t]*\r?\n(.*?)```", re.S | re.I)
_JSON_START_RE = re.compile(r"^\s*\{")

_STREAM_MARKERS = (
    "<tool_call",
    "```tool_call",
    "```json",
    "```tool",
    "<|tool_call|>",
    '{"tool_calls"',
    "<|dsml",
    "<\uff5c\uff5cdsml",
    "<\uff5cdsml",
)
_HOLD_CHARS = 32

_ARG_KEYS = ("arguments", "parameters", "args", "params", "input")
_NAME_KEYS = ("name", "tool", "tool_name", "function_name")


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None
    raw: str = ""

    def to_openai(self, call_id: str) -> Dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


def tools_fingerprint(tools: Optional[Iterable[Any]]) -> str:
    """Stable hash of the tool set, used to key conversations."""
    if not tools:
        return ""
    canonical = []
    for tool in tools:
        data = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else tool
        fn = data.get("function", data) if isinstance(data, dict) else {}
        canonical.append(
            {
                "name": fn.get("name"),
                "parameters": fn.get("parameters"),
            }
        )
    canonical.sort(key=lambda item: str(item.get("name")))
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def tool_choice_mode(tool_choice: Any) -> Tuple[str, Optional[str]]:
    """Normalise tool_choice to (mode, forced_name)."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        mode = tool_choice.lower()
        if mode in ("auto", "none", "required"):
            return mode, None
        return "auto", None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") or tool_choice.get("name")
        if name:
            return "forced", str(name)
        if tool_choice.get("type") == "required":
            return "required", None
    return "auto", None


def render_tools_block(tools: Optional[Iterable[Any]], tool_choice: Any = None) -> str:
    """Serialise the tool definitions + protocol rules for the prompt."""
    if not tools:
        return ""
    mode, forced = tool_choice_mode(tool_choice)
    if mode == "none":
        return ""

    lines: List[str] = [
        "# TOOLS",
        "You can call the tools listed below. To call one, output ONLY tool blocks in this exact form:",
        f'{OPEN_TAG}{{"name":"<tool_name>","arguments":{{...}}}}{CLOSE_TAG}',
        "Rules:",
        "- `arguments` must be a valid JSON object matching the tool schema.",
        "- Never wrap the block in markdown fences and never explain the call.",
        "- To call several tools, emit several blocks back to back.",
        "- After calling tools, stop and wait: the results arrive in the next message.",
        "- When you can answer without a tool, reply in plain text with no tool block.",
    ]
    if mode == "required":
        lines.append("- You MUST call at least one tool in this reply.")
    elif mode == "forced" and forced:
        lines.append(f'- You MUST call the tool named "{forced}" in this reply.')

    lines.append("")
    lines.append("## Available tools")
    for tool in tools:
        data = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else dict(tool)
        fn = data.get("function", data)
        name = fn.get("name")
        if not name:
            continue
        description = (fn.get("description") or "").strip().replace("\n", " ")
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        lines.append(f"- {name}: {description}" if description else f"- {name}")
        lines.append("  schema: " + json.dumps(params, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def has_tool_syntax(text: str) -> bool:
    return bool(
        _BLOCK_RE.search(text)
        or _PIPE_TAG_RE.search(text)
        or _FENCE_RE.search(text)
        or _DSML_START.search(text)
    )


def parse_tool_calls(text: str) -> Tuple[str, List[ToolCall]]:
    """Split model output into (visible_text, tool_calls).

    Handles the canonical tag form, OpenAI-style JSON, `{"tool_calls": [...]}`
    wrappers, fenced JSON, bare JSON objects, and common JSON malformations.
    """
    if not text:
        return "", []

    calls: List[ToolCall] = []
    spans: List[Tuple[int, int]] = []

    for regex in (_BLOCK_RE, _PIPE_TAG_RE):
        for match in regex.finditer(text):
            parsed = _parse_payload(match.group(1))
            if parsed:
                calls.extend(parsed)
                spans.append(match.span())

    if not calls:
        for match in _FENCE_RE.finditer(text):
            body = match.group(1)
            parsed = _parse_payload(body)
            if parsed:
                calls.extend(parsed)
                spans.append(match.span())

    if not calls:
        parsed = _parse_payload(text)
        if parsed:
            calls.extend(parsed)
            spans.append((0, len(text)))

    if not calls:
        dsml_text, dsml_calls = parse_dsml_tool_calls(text)
        if dsml_calls:
            return dsml_text, dsml_calls
        return text, []

    visible = _strip_spans(text, spans).strip()
    return visible, calls


def _strip_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    out: List[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            out.append(text[cursor:start])
        cursor = max(cursor, end)
    out.append(text[cursor:])
    return "".join(out)


def _parse_payload(payload: str) -> List[ToolCall]:
    data = _loads_json(payload)
    if data is None:
        return []
    return _calls_from_data(data, payload)


def _calls_from_data(data: Any, raw: str) -> List[ToolCall]:
    if isinstance(data, list):
        calls: List[ToolCall] = []
        for item in data:
            calls.extend(_calls_from_data(item, raw))
        return calls
    if not isinstance(data, dict):
        return []

    if isinstance(data.get("tool_calls"), list):
        return _calls_from_data(data["tool_calls"], raw)

    function = data.get("function")
    if isinstance(function, dict) and function.get("name"):
        return [
            ToolCall(
                name=str(function["name"]),
                arguments=_coerce_arguments(function),
                call_id=data.get("id"),
                raw=raw,
            )
        ]

    name = _first_present(data, _NAME_KEYS)
    if isinstance(name, str) and name.strip():
        return [
            ToolCall(
                name=name.strip(),
                arguments=_coerce_arguments(data),
                call_id=data.get("id"),
                raw=raw,
            )
        ]
    return []


def _coerce_arguments(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in _ARG_KEYS:
        if key in data:
            value = data[key]
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                parsed = _loads_json(value)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": value}
            if value is None:
                return {}
            return {"value": value}
    extras = {
        k: v
        for k, v in data.items()
        if k not in _NAME_KEYS and k not in ("id", "type", "function", "tool_calls")
    }
    return extras


def _first_present(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _loads_json(text: str) -> Any:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate:
        return None
    candidate = re.sub(r"^```[a-zA-Z_]*\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)

    for attempt in (candidate, _repair_json(candidate), _extract_json_object(candidate)):
        if not attempt:
            continue
        try:
            return json.loads(attempt)
        except (ValueError, TypeError):
            continue
    return None


def _repair_json(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if '"' not in repaired and "'" in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


class ToolCallStream:
    """Separates visible text from tool calls while the answer streams.

    Emits ``("text", str)`` items as text is confirmed to be prose, and a single
    ``("tool_calls", [ToolCall])`` item when the reply turns out to be a call.
    Text that was already emitted is never retracted; only the region at or
    after a marker (or a leading JSON object) is held back.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.buffer = ""
        self.emitted = 0
        self.holding = False
        self.saw_text = False

    def feed(self, text: str) -> List[Tuple[str, Any]]:
        if not text:
            return []
        self.buffer += text
        if not self.enabled:
            self.emitted = len(self.buffer)
            return [("text", text)]
        if self.holding:
            return []

        pending = self.buffer[self.emitted:]
        if not pending:
            return []

        if not self.saw_text and _JSON_START_RE.match(pending):
            self.holding = True
            return []

        marker_at = _find_marker(pending)
        if marker_at is not None:
            head = pending[:marker_at]
            self.emitted += marker_at
            self.holding = True
            if head:
                self.saw_text = True
                return [("text", head)]
            return []

        safe = len(pending) - _partial_marker_tail(pending)
        if safe <= 0:
            return []
        chunk = pending[:safe]
        self.emitted += safe
        if chunk:
            self.saw_text = True
        return [("text", chunk)]

    def flush(self) -> List[Tuple[str, Any]]:
        remainder = self.buffer[self.emitted:]
        if not remainder.strip():
            return []
        if not self.enabled:
            return []

        visible, calls = parse_tool_calls(remainder)
        if calls:
            if visible.strip():
                self.saw_text = True
                return [("text", visible), ("tool_calls", calls)]
            return [("tool_calls", calls)]
        return [("text", remainder)]


def _find_marker(text: str) -> Optional[int]:
    lowered = text.lower()
    best: Optional[int] = None
    for marker in _STREAM_MARKERS:
        index = lowered.find(marker)
        if index >= 0 and (best is None or index < best):
            best = index
    return best


def _partial_marker_tail(text: str) -> int:
    lowered = text.lower()
    longest = 0
    window = min(len(lowered), _HOLD_CHARS)
    for size in range(1, window + 1):
        tail = lowered[-size:]
        if any(marker.startswith(tail) for marker in _STREAM_MARKERS):
            longest = size
    return longest

_FW_BAR = "\uff5c"
_DSML_START = re.compile(r"<\s*/?\s*[|\uff5c]\s*[|\uff5c]?\s*DSML", re.I)
_DSML_BLOCK = re.compile(
    r"<\|DSML\|\s*(?:tool_)?calls\s*>(.*?)</\|DSML\|\s*(?:tool_)?calls\s*>",
    re.S | re.I,
)
_DSML_INVOKE = re.compile(
    r"<\|DSML\|\s*invoke\s+name=\"([^\"]+)\"\s*>(.*?)</\|DSML\|\s*invoke\s*>",
    re.S | re.I,
)
_DSML_PARAM = re.compile(
    r"<\|DSML\|\s*parameter\s+name=\"([^\"]+)\"(?:\s+string=\"(true|false)\")?\s*>"
    r"(.*?)</\|DSML\|\s*parameter\s*>",
    re.S | re.I,
)


def _normalize_dsml(text: str) -> str:
    text = text.replace(_FW_BAR, "|")
    text = re.sub(r"<\s*/\s*\|\s*\|?\s*DSML\s*\|\s*\|?\s*", "</|DSML|", text, flags=re.I)
    text = re.sub(r"<\s*\|\s*\|?\s*DSML\s*\|\s*\|?\s*", "<|DSML|", text, flags=re.I)
    return text


def parse_dsml_tool_calls(text: str) -> Tuple[str, List[ToolCall]]:
    """Parse DeepSeek's DSML tool-call dialect.

    Observed shape (fullwidth bars, sometimes with a missing `tool_` prefix):

        <||DSML|| calls>
        <||DSML|| invoke name="read">
        <||DSML|| parameter name="filePath" string="true">C:\\a.txt</||DSML|| parameter>
        </||DSML|| invoke>
        </||DSML|| calls>
    """
    if not text or not _DSML_START.search(text):
        return text, []

    normalized = _normalize_dsml(text)
    blocks = _DSML_BLOCK.findall(normalized)
    scope = "\n".join(blocks) if blocks else normalized

    calls: List[ToolCall] = []
    for invoke in _DSML_INVOKE.finditer(scope):
        name = invoke.group(1).strip()
        if not name:
            continue
        arguments: Dict[str, Any] = {}
        for param in _DSML_PARAM.finditer(invoke.group(2)):
            key = param.group(1).strip()
            is_string = (param.group(2) or "").lower() == "true"
            raw = html.unescape(param.group(3)).strip()
            arguments[key] = raw if is_string else _maybe_json(raw)
        calls.append(ToolCall(name=name, arguments=arguments, raw=invoke.group(0)))

    if not calls:
        return text, []

    start = _DSML_START.search(text).start()
    return text[:start].strip(), calls


def _maybe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


