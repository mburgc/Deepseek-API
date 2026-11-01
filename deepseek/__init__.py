"""Unofficial OpenAI-compatible client for chat.deepseek.com."""

from .auth import Session, get_session, login
from .client import (
    CompletionStream,
    DeepSeekAPIError,
    DeepSeekClient,
    Reply,
)
from .events import (
    DoneEvent,
    Ready,
    ReasoningDelta,
    SearchUpdate,
    StatusEvent,
    TextDelta,
    TitleEvent,
)
from .pow import DeepSeekPow

__all__ = [
    "Session",
    "get_session",
    "login",
    "DeepSeekClient",
    "CompletionStream",
    "DeepSeekAPIError",
    "Reply",
    "DeepSeekPow",
    "TextDelta",
    "ReasoningDelta",
    "SearchUpdate",
    "StatusEvent",
    "TitleEvent",
    "DoneEvent",
    "Ready",
]
