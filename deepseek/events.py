"""Typed events emitted by the DeepSeek completion stream.

DeepSeek's web completion endpoint does not stream OpenAI-style deltas. It
streams a JSON-patch-like protocol over SSE:

    event: ready
    data: {"request_message_id":1,"response_message_id":2,...}

    data: {"v":{"response":{"message_id":2,"fragments":[{"type":"THINK",...}]}}}

    data: {"p":"response/fragments/-1/content","o":"APPEND","v":"Hello"}
    data: {"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE",...}]}
    data: {"p":"response","o":"BATCH","v":[{"p":"status","o":"SET","v":"FINISHED"}]}

`deepseek.sse` turns that into the small event set below, which is what the
rest of the bridge (and the OpenAI shim) consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Ready:
    """Stream accepted; carries the assistant message id up front."""

    request_message_id: Optional[int] = None
    response_message_id: Optional[int] = None
    model_type: Optional[str] = None


@dataclass(frozen=True)
class TextDelta:
    """A chunk of the visible answer (fragment type RESPONSE/TEXT)."""

    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    """A chunk of chain-of-thought (fragment type THINK)."""

    text: str


@dataclass(frozen=True)
class SearchUpdate:
    """Web-search progress: queries issued and/or sources found."""

    queries: Tuple[str, ...] = ()
    results: Tuple[Dict, ...] = ()
    status: Optional[str] = None
    finished: bool = False


@dataclass(frozen=True)
class StatusEvent:
    """Generation status (WIP/FINISHED) and real token usage when known."""

    status: Optional[str] = None
    token_usage: Optional[int] = None


@dataclass(frozen=True)
class TitleEvent:
    """Auto-generated conversation title."""

    title: str


@dataclass(frozen=True)
class DoneEvent:
    """Stream closed."""

    auto_resume: bool = False


@dataclass
class StreamState:
    """Mutable bookkeeping accumulated while a stream is consumed.

    The client reads this after exhausting the event iterator to build the
    resumable conversation id, usage numbers and citation list.
    """

    message_id: Optional[int] = None
    request_message_id: Optional[int] = None
    model_type: Optional[str] = None
    status: Optional[str] = None
    token_usage: Optional[int] = None
    conversation_mode: Optional[str] = None
    thinking_enabled: Optional[bool] = None
    search_enabled: Optional[bool] = None
    search_triggered: bool = False
    title: Optional[str] = None
    auto_resume: bool = False
    search_queries: List[str] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    emitted_answer: str = ""
    emitted_reasoning: str = ""

    def sorted_results(self) -> List[Dict]:
        """Search results ordered by their cite_index (the number in `[citation:N]`)."""
        return sorted(
            self.search_results,
            key=lambda r: (r.get("cite_index") is None, r.get("cite_index") or 0),
        )
