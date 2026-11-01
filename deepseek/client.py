"""Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into typed events (deepseek.sse)

Two levels of API are exposed:

* ``client.request(...)`` returns a `CompletionStream` that yields typed events
  (text, reasoning, search sources, status). Use it from the server bridge.
* ``client.chat(...)`` / ``client.stream(...)`` are the simple helpers that only
  return the answer text (kept for backwards compatibility).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import httpx

from .auth import Session, get_session
from .events import ReasoningDelta, StreamState, TextDelta
from .pow import DeepSeekPow
from .sse import iter_events

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"
UPLOAD_PATH = "/api/v0/file/upload_file"

DEFAULT_MODEL_TYPE = "default"

_CID_SEP = ":"


class DeepSeekAPIError(RuntimeError):
    """An API-level failure returned inside an HTTP 200 envelope.

    DeepSeek answers 200 with ``{"code":...,"data":{"biz_code":...,"biz_msg":...}}``
    even for errors, so HTTP status alone is not enough to detect failure.
    """

    def __init__(self, message: str, *, code=None, biz_code=None, status=None):
        super().__init__(message)
        self.code = code
        self.biz_code = biz_code
        self.status = status

    @property
    def is_rate_limit(self) -> bool:
        return self.code == 429 or self.biz_code in (7, 429)


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation."""

    text: str
    conversation_id: str
    reasoning: str = ""
    search_results: List[Dict] = field(default_factory=list)
    token_usage: Optional[int] = None

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _unwrap(data: dict, status: Optional[int] = None) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if not isinstance(data, dict):
        raise DeepSeekAPIError(f"Unexpected response: {data!r}", status=status)
    if data.get("code") != 0:
        raise DeepSeekAPIError(
            f"DeepSeek API error: {data.get('msg') or data}",
            code=data.get("code"),
            status=status,
        )
    payload = data.get("data") or {}
    if payload.get("biz_code") not in (0, None):
        raise DeepSeekAPIError(
            f"DeepSeek error {payload.get('biz_code')}: {payload.get('biz_msg')}",
            code=data.get("code"),
            biz_code=payload.get("biz_code"),
            status=status,
        )
    biz = payload.get("biz_data")
    if biz is None:
        raise DeepSeekAPIError(f"Unexpected response shape: {data}", status=status)
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
        http: Optional[httpx.Client] = None,
    ):
        self.session = session or get_session(allow_interactive=allow_interactive)
        self._pow = DeepSeekPow()
        # The wasmtime Store behind the PoW solver is not reentrant; serialise
        # access so concurrent requests don't corrupt it.
        self._pow_lock = threading.Lock()
        self._owns_http = http is None
        self._http = http or httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(120.0, read=300.0),
        )

    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        r = self._http.post("/api/v0/chat_session/create", json={})
        r.raise_for_status()
        return _unwrap(r.json(), r.status_code)["chat_session"]["id"]

    def pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        """Fetch a challenge and solve it. Retries once if the challenge expired."""
        last_error: Optional[Exception] = None
        for _ in range(2):
            r = self._http.post(
                "/api/v0/chat/create_pow_challenge",
                json={"target_path": target_path},
            )
            r.raise_for_status()
            challenge = _unwrap(r.json(), r.status_code)["challenge"]
            with self._pow_lock:
                try:
                    return self._pow.make_header(challenge)
                except RuntimeError as e:
                    last_error = e
        raise DeepSeekAPIError(f"Could not solve proof-of-work: {last_error}")

    def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        return self.pow_header(target_path)

    # --- account helpers ----------------------------------------------------

    def current_user(self) -> Dict:
        r = self._http.get("/api/v0/users/current")
        r.raise_for_status()
        return _unwrap(r.json(), r.status_code)

    def client_settings(self, device_id: str = "bridge", os_name: str = "web") -> Dict:
        r = self._http.get(
            "/api/v0/client/settings", params={"did": device_id, "os": os_name}
        )
        r.raise_for_status()
        return _unwrap(r.json(), r.status_code).get("settings") or {}

    def list_sessions(self, count: int = 20) -> List[Dict]:
        r = self._http.get("/api/v0/chat_session/fetch_page", params={"count": count})
        r.raise_for_status()
        return _unwrap(r.json(), r.status_code).get("chat_sessions") or []

    def history(self, session_id: str, count: int = 100) -> Dict:
        r = self._http.get(
            "/api/v0/chat/history_messages",
            params={"chat_session_id": session_id, "count": count},
        )
        r.raise_for_status()
        return _unwrap(r.json(), r.status_code)

    # --- public API ---------------------------------------------------------

    def request(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        parent_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[List[str]] = None,
        action=None,
        preempt: bool = False,
    ) -> "CompletionStream":
        """Start a completion and return a stream of typed events.

        A new thread is created when neither `session_id` nor `conversation_id`
        is given; the selected `model` only applies to new threads (a thread's
        model is fixed at creation).
        """
        if conversation_id and (session_id or model is not None):
            raise ValueError(
                "pass either conversation_id or session_id+model, not both"
            )
        if conversation_id:
            session_id, parent_id = _decode_cid(conversation_id)

        if session_id is None:
            session_id = self.create_chat_session()
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            model_type = None if parent_id is not None else (model or None)

        return CompletionStream(
            client=self,
            prompt=prompt,
            session_id=session_id,
            parent_id=parent_id,
            model_type=model_type,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids or [],
            action=action,
            preempt=preempt,
        )

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[List[str]] = None,
    ) -> "CompletionStream":
        """Stream a reply, yielding only the answer text chunks.

        Kept for backwards compatibility; use `request()` for typed events.
        """
        return self.request(
            prompt,
            conversation_id=conversation_id,
            model=model,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
        )

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[List[str]] = None,
    ) -> Reply:
        """Return the complete reply (`.text`) plus its `.conversation_id`."""
        s = self.request(
            prompt,
            conversation_id=conversation_id,
            model=model,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
        )
        parts: List[str] = []
        reasoning: List[str] = []
        for event in s:
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, ReasoningDelta):
                reasoning.append(event.text)
        return Reply(
            text="".join(parts),
            conversation_id=s.conversation_id,
            reasoning="".join(reasoning),
            search_results=s.state.sorted_results(),
            token_usage=s.state.token_usage,
        )

    def close(self) -> None:
        if self._owns_http:
            self._http.close()


class CompletionStream:
    """Iterable of typed events for one completion.

    After consumption, `state` holds the assistant message id, real token usage
    and search results, and `conversation_id` is the resumable token.
    """

    def __init__(
        self,
        *,
        client: DeepSeekClient,
        prompt: str,
        session_id: str,
        parent_id: Optional[int],
        model_type: Optional[str],
        thinking: bool,
        search: bool,
        ref_file_ids: List[str],
        action,
        preempt: bool,
    ):
        self._client = client
        self.prompt = prompt
        self.session_id = session_id
        self.parent_id = parent_id
        self.model_type = model_type
        self.thinking = thinking
        self.search = search
        self.ref_file_ids = ref_file_ids
        self.action = action
        self.preempt = preempt
        self.state = StreamState()
        self._consumed = False

    def body(self) -> Dict:
        payload = {
            "chat_session_id": self.session_id,
            "parent_message_id": self.parent_id,
            "prompt": self.prompt,
            "ref_file_ids": list(self.ref_file_ids),
            "thinking_enabled": self.thinking,
            "search_enabled": self.search,
            "action": self.action,
            "preempt": self.preempt,
        }
        if self.model_type is not None:
            payload["model_type"] = self.model_type
        return payload

    def __iter__(self) -> Iterator[object]:
        last_error: Optional[DeepSeekAPIError] = None
        for attempt in range(3):
            emitted = False
            try:
                for event in self._attempt():
                    emitted = True
                    yield event
                return
            except DeepSeekAPIError as exc:
                if not exc.is_rate_limit or emitted or attempt == 2:
                    raise
                last_error = exc
                time.sleep(3.0 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def _attempt(self) -> Iterator[object]:
        headers = {"x-ds-pow-response": self._client.pow_header(COMPLETION_PATH)}
        with self._client._http.stream(
            "POST", COMPLETION_PATH, json=self.body(), headers=headers
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise DeepSeekAPIError(
                    f"DeepSeek HTTP {resp.status_code}: {resp.text[:300]}",
                    status=resp.status_code,
                )
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                resp.read()
                raise _error_from_body(resp.text, resp.status_code)

            lines = _guarded_lines(resp.iter_lines(), resp.status_code)
            self._consumed = True
            for event in iter_events(lines, self.state):
                yield event

    @property
    def message_id(self) -> Optional[int]:
        return self.state.message_id

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self.session_id, self.state.message_id)


def _guarded_lines(lines: Iterator[str], status: Optional[int]) -> Iterator[str]:
    """Yield stream lines, raising if the body is really a JSON error envelope.

    DeepSeek sometimes answers a completion with a plain JSON error (HTTP 200,
    content-type not set to SSE), e.g. ``biz_code 9 invalid ref file id``.
    """
    first_payload_seen = False
    for line in lines:
        if not first_payload_seen:
            stripped = line.strip("\r")
            if not stripped:
                continue
            first_payload_seen = True
            if not (stripped.startswith("data:") or stripped.startswith("event:")):
                if stripped.startswith("{"):
                    raise _error_from_body(stripped, status)
        yield line


def _error_from_body(raw: str, status: Optional[int]) -> DeepSeekAPIError:
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return DeepSeekAPIError(f"DeepSeek returned: {raw[:300]}", status=status)
    try:
        _unwrap(payload, status)
    except DeepSeekAPIError as e:
        return e
    return DeepSeekAPIError(f"DeepSeek returned: {raw[:300]}", status=status)




