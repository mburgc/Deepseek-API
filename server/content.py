"""Helpers for reading and fingerprinting chat messages.

Kept dependency-free so both the thread store and the OpenAI translator can use
it without import cycles.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def text_of(content: Any) -> str:
    """Flatten OpenAI message content (string or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    parts: List[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            kind = part.get("type")
            if kind in (None, "text", "input_text"):
                parts.append(str(part.get("text") or ""))
            elif kind == "tool_result":
                parts.append(str(part.get("content") or part.get("text") or ""))
    return "\n".join(p for p in parts if p)


def image_parts(content: Any) -> List[Dict[str, Any]]:
    """Extract OpenAI image_url parts from a message's content."""
    if not isinstance(content, list):
        return []
    out: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
            out.append(part)
    return out


def file_parts(content: Any) -> List[Dict[str, Any]]:
    if not isinstance(content, list):
        return []
    out: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("file", "input_file"):
            out.append(part)
    return out


def canonical_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of a message that must match when resuming a conversation.

    Deliberately excludes ids, reasoning traces and provider metadata: clients
    legitimately drop or rewrite those when replaying history.
    """
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append([function.get("name"), function.get("arguments")])
    return {
        "role": message.get("role"),
        "name": message.get("name"),
        "text": text_of(message.get("content")),
        "tool_calls": calls,
        "tool_call_id": message.get("tool_call_id"),
    }


def canonical_hash(messages: List[Dict[str, Any]]) -> str:
    blob = json.dumps(
        [canonical_message(m) for m in messages],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def as_message_dicts(messages: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            out.append(message.model_dump(exclude_none=True))
        elif isinstance(message, dict):
            out.append(message)
    return out


def last_user_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = text_of(message.get("content"))
            if text.strip():
                return text
    return None
