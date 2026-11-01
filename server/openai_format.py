"""Translate between OpenAI's chat-completions shapes and the DeepSeek bridge.

DeepSeek's web protocol has no roles, no tool channel and no usage counters we
can trust for the prompt side: it takes one `prompt` string and streams
fragments. So messages are rendered into a labelled transcript, and the stream
is re-emitted as OpenAI frames (content, reasoning, tool_calls, citations).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import APPEND_SOURCES, MAX_TOOL_RESULT_CHARS
from .content import text_of
from .toolbridge import ToolCall

_ROLE_LABELS = {
    "system": "System",
    "developer": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool result",
}

_CITATION_RE = re.compile(r"\[citation:(\d+)\]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 3)


def _truncate(text: str) -> str:
    from .digest import digest_tool_output

    return digest_tool_output(text, MAX_TOOL_RESULT_CHARS)


def render_message(message: Dict[str, Any]) -> str:
    role = message.get("role") or "user"
    body = text_of(message.get("content"))

    if role == "tool":
        name = message.get("name") or "tool"
        call_id = message.get("tool_call_id")
        header = f"{name} ({call_id})" if call_id else str(name)
        return f"Tool result [{header}]:\n{_truncate(body)}"

    label = _ROLE_LABELS.get(role, role.capitalize())
    if role in ("system", "developer"):
        return f"{label}:\n{body}" if body else ""

    calls = message.get("tool_calls") or []
    rendered_calls = []
    for call in calls:
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"raw": arguments}
        rendered_calls.append(
            f"<tool_call>{json.dumps({'name': name, 'arguments': arguments or {}}, ensure_ascii=False)}</tool_call>"
        )
    if rendered_calls:
        joined = "\n".join(rendered_calls)
        return f"{label}:\n{body}\n{joined}".strip()
    return f"{label}:\n{body}"


def messages_to_prompt(messages: List[Dict[str, Any]], *, append_cue: bool = True) -> str:
    """Render a full conversation into a single DeepSeek prompt."""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return text_of(messages[0].get("content"))

    blocks = [block for block in (render_message(m) for m in messages) if block.strip()]
    prompt = "\n\n".join(blocks)
    if append_cue and len(blocks) > 1:
        prompt += "\n\nAssistant:"
    return prompt


def messages_to_prompt_tail(messages: List[Dict[str, Any]]) -> str:
    """Render only the part of the history the DeepSeek thread has not seen."""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return text_of(messages[0].get("content"))
    return messages_to_prompt(messages, append_cue=False)


def extract_system(messages: List[Dict[str, Any]]) -> str:
    parts = [
        text_of(m.get("content"))
        for m in messages
        if m.get("role") in ("system", "developer")
    ]
    return "\n\n".join(p for p in parts if p.strip())


def with_tools_block(
    messages: List[Dict[str, Any]], tools: Optional[List[Any]], tool_choice: Any
) -> List[Dict[str, Any]]:
    """Return a copy of the history with the tool protocol folded into the system turn."""
    from .toolbridge import render_tools_block

    block = render_tools_block(tools, tool_choice)
    if not block:
        return list(messages)

    out = [dict(m) for m in messages]
    for message in out:
        if message.get("role") in ("system", "developer"):
            message["content"] = (text_of(message.get("content")) + "\n\n" + block).strip()
            return out
    out.insert(0, {"role": "system", "content": block})
    return out


def tail_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The part of the history after the last assistant turn (usually: new input)."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            tail = messages[index + 1:]
            return tail or messages[-1:]
    return messages[-1:]


def build_annotations(results: List[Dict]) -> List[Dict]:
    annotations = []
    for index, item in enumerate(results):
        url = item.get("url")
        if not url:
            continue
        annotations.append(
            {
                "type": "url_citation",
                "url_citation": {
                    "url": url,
                    "title": item.get("title") or url,
                    "start_index": 0,
                    "end_index": 0,
                    "cite_index": item.get("cite_index", index + 1),
                },
            }
        )
    return annotations


def format_sources(results: List[Dict], answer: str) -> str:
    """Append a compact source list so `[citation:N]` markers are resolvable."""
    if not results:
        return ""
    cited = {int(n) for n in _CITATION_RE.findall(answer or "")}
    chosen = [r for r in results if r.get("cite_index") in cited] or list(results)
    lines = ["", "Fuentes:"]
    for index, item in enumerate(chosen[:12], start=1):
        number = item.get("cite_index") or index
        title = (item.get("title") or "").strip() or item.get("url", "")
        site = item.get("site_name") or ""
        suffix = f" ({site})" if site else ""
        lines.append(f"[{number}] {title} — {item.get('url', '')}{suffix}")
    return "\n".join(lines)


def rewrite_citations(text: str, results: List[Dict]) -> str:
    by_index = {r.get("cite_index"): r for r in results if r.get("cite_index")}
    if not by_index:
        return text

    def replace(match: re.Match) -> str:
        item = by_index.get(int(match.group(1)))
        if not item or not item.get("url"):
            return match.group(0)
        title = (item.get("title") or "source").replace("]", ")")
        return f"[{title}]({item['url']})"

    return _CITATION_RE.sub(replace, text)


def usage(prompt_text: str, completion_text: str, upstream_tokens: Optional[int]) -> Dict[str, int]:
    estimated_completion = estimate_tokens(completion_text)
    if upstream_tokens and upstream_tokens > 0:
        total = max(upstream_tokens, estimated_completion)
        completion_tokens = min(estimated_completion, total)
        prompt_tokens = max(total - completion_tokens, 1)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max(completion_tokens, 1),
            "total_tokens": prompt_tokens + max(completion_tokens, 1),
        }
    prompt_tokens = estimate_tokens(prompt_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": estimated_completion,
        "total_tokens": prompt_tokens + estimated_completion,
    }


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _now() -> int:
    return int(time.time())


def completion_response(
    *,
    model: str,
    content: str,
    prompt: str,
    conversation_id: Optional[str],
    reasoning: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    search_results: Optional[List[Dict]] = None,
    token_usage: Optional[int] = None,
) -> Dict[str, Any]:
    calls = [
        call.to_openai(call.call_id or f"call_{uuid.uuid4().hex[:24]}")
        for call in (tool_calls or [])
    ]
    message: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    annotations = build_annotations(search_results or [])
    if annotations:
        message["annotations"] = annotations

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": usage(prompt, content, token_usage),
        "search_results": search_results or [],
    }


def stream_chunks(
    *,
    model: str,
    stream,
    prompt: str = "",
    include_usage: bool = False,
    tools_enabled: bool = False,
    sources: bool = APPEND_SOURCES,
    result: Optional[Dict[str, Any]] = None,
    tool_names: Optional[Iterable[str]] = None,
) -> Iterable[str]:
    """Yield OpenAI SSE frames for one completion."""
    from .toolbridge import ToolCallStream

    chunk_id = _id()
    created = _now()

    def frame(delta: Dict, finish: Optional[str] = None, extra: Optional[Dict] = None) -> str:
        obj: Dict[str, Any] = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    yield frame({"role": "assistant", "content": ""})

    splitter = ToolCallStream(enabled=tools_enabled, allowed_names=tool_names)
    collected_calls: List[ToolCall] = []
    answer_parts: List[str] = []
    reasoning_parts: List[str] = []

    for event in stream:
        name = type(event).__name__
        if name == "TextDelta":
            for kind, payload in splitter.feed(event.text):
                if kind == "text":
                    answer_parts.append(payload)
                    yield frame({"content": payload})
                elif kind == "tool_calls":
                    collected_calls.extend(payload)
        elif name == "ReasoningDelta":
            reasoning_parts.append(event.text)
            yield frame({"reasoning_content": event.text, "reasoning": event.text})

    for kind, payload in splitter.flush():
        if kind == "text":
            if payload.strip():
                answer_parts.append(payload)
                yield frame({"content": payload})
        elif kind == "tool_calls":
            collected_calls.extend(payload)

    state = getattr(stream, "state", None)
    results = state.sorted_results() if state is not None else []
    answer = "".join(answer_parts)

    if sources and results:
        suffix = format_sources(results, answer)
        if suffix:
            answer_parts.append(suffix)
            yield frame({"content": suffix})

    if collected_calls:
        delta_calls = []
        for index, call in enumerate(collected_calls):
            call_id = call.call_id or f"call_{uuid.uuid4().hex[:24]}"
            delta_calls.append(
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
            )
        yield frame({"tool_calls": delta_calls})

    finish_reason = "tool_calls" if collected_calls else "stop"
    final_extra: Dict[str, Any] = {"conversation_id": getattr(stream, "conversation_id", None)}
    if results:
        final_extra["search_results"] = [
            {
                "url": r.get("url"),
                "title": r.get("title"),
                "cite_index": r.get("cite_index"),
                "site_name": r.get("site_name"),
            }
            for r in results[:20]
        ]
    yield frame({}, finish=finish_reason, extra=final_extra)

    if result is not None:
        result["text"] = "".join(answer_parts)
        result["reasoning"] = "".join(reasoning_parts)
        result["calls"] = [
            {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}
            for call in collected_calls
        ]
        result["finish_reason"] = finish_reason

    if include_usage:
        usage_obj = usage(prompt, "".join(answer_parts), getattr(state, "token_usage", None))
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [], 'usage': usage_obj, 'conversation_id': final_extra['conversation_id']}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"

@dataclass
class Answer:
    text: str
    reasoning: str
    calls: List[ToolCall]
    state: Any


def collect_answer(
    stream,
    *,
    tools_enabled: bool = False,
    sources: bool = APPEND_SOURCES,
    tool_names: Optional[Iterable[str]] = None,
) -> Answer:
    """Consume a completion stream into a single answer (non-streaming path)."""
    from .toolbridge import ToolCallStream
    from deepseek.events import ReasoningDelta, TextDelta

    splitter = ToolCallStream(enabled=tools_enabled, allowed_names=tool_names)
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    calls: List[ToolCall] = []

    def absorb(items):
        for kind, payload in items:
            if kind == "text":
                text_parts.append(payload)
            elif kind == "tool_calls":
                calls.extend(payload)

    for event in stream:
        if isinstance(event, TextDelta):
            absorb(splitter.feed(event.text))
        elif isinstance(event, ReasoningDelta):
            reasoning_parts.append(event.text)
    absorb(splitter.flush())

    state = getattr(stream, "state", None)
    results = state.sorted_results() if state is not None else []
    if sources and results:
        suffix = format_sources(results, "".join(text_parts))
        if suffix:
            text_parts.append(suffix)

    return Answer("".join(text_parts), "".join(reasoning_parts), calls, state)




