"""Parser for DeepSeek's completion SSE stream.

The wire format is a fragment-based JSON patch protocol, not plain text deltas.
Key facts (verified against real captures in tests/fixtures):

* Frames are either a full ``{"v": {"response": {...}}}`` snapshot, a patch
  ``{"p": <path>, "o": <SET|APPEND|BATCH>, "v": <value>}``, or a bare
  ``{"v": "text"}`` that appends to the currently active fragment.
* ``p`` paths are relative to the response object; ``-1`` means "last fragment".
* A ``BATCH`` value is a list of *relative* patches that must be joined onto the
  batch's own path before being applied (this is what the app's own state
  engine does). Batches can be nested.
* ``RESPONSE`` and ``THINK`` fragments share the same ``.../content`` path.
  Only the fragment's *type* tells them apart, so the parser tracks it.
* A new fragment often arrives as a list: ``{"p":"response/fragments",
  "o":"APPEND","v":[{...}]}`` — the first chunk of the answer is inside it.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List, Optional

from .events import (
    DoneEvent,
    Ready,
    ReasoningDelta,
    SearchUpdate,
    StatusEvent,
    StreamState,
    TextDelta,
    TitleEvent,
)

_TEXT_PREFIX = "data:"
_EVENT_PREFIX = "event:"

_THINK_TYPES = {"THINK", "THINKING", "REASONING"}
_TEXT_TYPES = {"RESPONSE", "TEXT", "ANSWER", None}


def iter_events(lines: Iterable[str], state: Optional[StreamState] = None) -> Iterator[object]:
    """Turn SSE lines into typed events.

    `lines` is anything iterable of strings (e.g. ``httpx.Response.iter_lines``).
    Pass a `StreamState` to collect message ids, usage and citations; a fresh
    one is created if omitted (and is then only reachable via the events).
    """
    st = state if state is not None else StreamState()
    fragments: List[Dict] = []
    event_name: Optional[str] = None

    for raw in lines:
        if raw is None:
            continue
        line = raw.strip("\r")
        if not line:
            event_name = None
            continue
        if line.startswith(_EVENT_PREFIX):
            event_name = line[len(_EVENT_PREFIX):].strip()
            continue
        if not line.startswith(_TEXT_PREFIX):
            continue

        payload = line[len(_TEXT_PREFIX):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue

        try:
            results = list(_handle_frame(obj, fragments, st, event_name))
        finally:
            event_name = None
        for ev in results:
            yield ev


# --------------------------------------------------------------------------
# frame dispatch
# --------------------------------------------------------------------------


def _handle_frame(obj: Dict, fragments: List[Dict], st: StreamState,
                  event_name: Optional[str]) -> Iterator[object]:
    if event_name == "ready":
        st.request_message_id = _as_int(
            obj.get("request_message_id", obj.get("requestMessageId"))
        )
        rid = _as_int(obj.get("response_message_id", obj.get("responseMessageId")))
        if rid is not None:
            st.message_id = rid
        st.model_type = obj.get("model_type") or obj.get("modelType")
        yield Ready(st.request_message_id, st.message_id, st.model_type)
        return

    if event_name == "title":
        title = obj.get("content") or obj.get("title")
        if isinstance(title, str) and title:
            st.title = title
            yield TitleEvent(title)
        return

    if event_name == "close":
        st.auto_resume = bool(obj.get("auto_resume"))
        yield DoneEvent(st.auto_resume)
        return

    if event_name == "update_session":
        return

    value = obj.get("v")

    # Snapshot: the whole response object.
    if isinstance(value, dict) and "response" in value:
        yield from _apply_snapshot(value["response"], fragments, st)
        return

    # Patch frame.
    path = obj.get("p")
    if isinstance(path, str):
        op = obj.get("o")
        if op == "BATCH" and isinstance(value, list):
            for sub in value:
                if not isinstance(sub, dict):
                    continue
                sub = dict(sub)
                sub["p"] = _join(path, sub.get("p"))
                yield from _apply_patch(sub, fragments, st)
        else:
            yield from _apply_patch(obj, fragments, st)
        return

    # Bare append to the active (last) fragment.
    if isinstance(value, str) and fragments:
        yield from _emit(fragments[-1], value)


# --------------------------------------------------------------------------
# snapshots and patches
# --------------------------------------------------------------------------


def _apply_snapshot(resp: Dict, fragments: List[Dict], st: StreamState) -> Iterator[object]:
    mid = _as_int(resp.get("message_id", resp.get("id")))
    had_ready = st.request_message_id is not None or st.message_id is not None
    if mid is not None:
        st.message_id = mid

    if "status" in resp and isinstance(resp["status"], str):
        st.status = resp["status"]
    if "conversation_mode" in resp:
        st.conversation_mode = resp.get("conversation_mode")
    if "search_triggered" in resp:
        st.search_triggered = bool(resp.get("search_triggered"))
    if "thinking_enabled" in resp:
        st.thinking_enabled = bool(resp.get("thinking_enabled"))
    if "search_enabled" in resp:
        st.search_enabled = bool(resp.get("search_enabled"))
    usage = _as_int(resp.get("accumulated_token_usage"))
    if usage is not None:
        st.token_usage = usage

    fragments.clear()
    for frag in resp.get("fragments") or []:
        if not isinstance(frag, dict):
            continue
        entry = {
            "type": frag.get("type"),
            "content": frag.get("content") or "",
            "status": frag.get("status"),
        }
        fragments.append(entry)

        if entry["type"] == "SEARCH":
            queries = _extract_queries(frag)
            if queries:
                st.search_queries = queries
                yield SearchUpdate(queries=tuple(queries), status=frag.get("status"))
            results = frag.get("results") or []
            if results:
                yield from _merge_results(results, st)
        else:
            yield from _emit_full(entry, entry["content"])

    if mid is not None and not had_ready:
        # No `event: ready` frame: announce the id now so callers can resume.
        yield Ready(st.request_message_id, st.message_id, st.model_type)


def _apply_patch(patch: Dict, fragments: List[Dict], st: StreamState) -> Iterator[object]:
    path = patch.get("p")
    if not isinstance(path, str):
        return
    op = patch.get("o") or "SET"
    value = patch.get("v")

    if path == "response/fragments":
        if isinstance(value, list):
            for frag in value:
                if not isinstance(frag, dict):
                    continue
                entry = {
                    "type": frag.get("type"),
                    "content": frag.get("content") or "",
                    "status": frag.get("status"),
                }
                fragments.append(entry)
                if entry["type"] == "SEARCH":
                    queries = _extract_queries(frag)
                    if queries:
                        st.search_queries = queries
                        yield SearchUpdate(queries=tuple(queries), status=frag.get("status"))
                    if frag.get("results"):
                        yield from _merge_results(frag["results"], st)
                else:
                    yield from _emit_full(entry, entry["content"])
        return

    if path.startswith("response/fragments/"):
        rest = path[len("response/fragments/"):]
        index_part, _, field_path = rest.partition("/")
        frag = _resolve_fragment(index_part, fragments)
        if frag is None:
            return
        if not field_path:
            if isinstance(value, dict):
                frag.update(value)
            return
        yield from _apply_fragment_field(frag, field_path, op, value, st)
        return

    if path == "response/status":
        if isinstance(value, str):
            st.status = value
            yield StatusEvent(value, st.token_usage)
        return

    if path.endswith("accumulated_token_usage"):
        usage = _as_int(value)
        if usage is not None:
            st.token_usage = usage
            yield StatusEvent(st.status, usage)
        return

    if path == "response/conversation_mode":
        if isinstance(value, str):
            st.conversation_mode = value
        return

    if path == "response/search_triggered":
        st.search_triggered = bool(value)
        return

    # Everything else (elapsed_secs, has_pending_fragment, quasi_status, ...) is
    # UI bookkeeping we don't need.


def _apply_fragment_field(frag: Dict, field_path: str, op: str, value,
                          st: StreamState) -> Iterator[object]:
    field = field_path.split("/")[0]

    if field in ("content", "thinking_content", "reasoning_content"):
        if not isinstance(value, str):
            return
        if field != "content":
            # An explicit reasoning path inside a fragment.
            frag["content"] = (frag.get("content") or "") + value
            yield ReasoningDelta(value)
            return
        old = frag.get("content") or ""
        if op == "APPEND":
            frag["content"] = old + value
            yield from _emit(frag, value)
        else:
            frag["content"] = value
            if value.startswith(old):
                yield from _emit(frag, value[len(old):])
            else:
                yield from _emit_full(frag, value, replace=True)
        return

    if field == "results":
        if isinstance(value, list):
            yield from _merge_results(value, st)
        return

    if field == "status":
        if isinstance(value, str):
            frag["status"] = value
            if frag.get("type") == "SEARCH":
                yield SearchUpdate(status=value, finished=value == "FINISHED")
        return

    if field == "queries":
        if isinstance(value, list):
            queries = _extract_queries({"queries": value})
            if queries:
                st.search_queries = queries
                yield SearchUpdate(queries=tuple(queries))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _emit(frag: Dict, text: str) -> Iterator[object]:
    if not text:
        return
    ftype = frag.get("type")
    if ftype in _THINK_TYPES:
        yield ReasoningDelta(text)
    elif ftype in _TEXT_TYPES:
        yield TextDelta(text)


def _emit_full(frag: Dict, text: str, replace: bool = False) -> Iterator[object]:
    if not text:
        return
    ftype = frag.get("type")
    if ftype in _THINK_TYPES:
        yield ReasoningDelta(text)
    elif ftype in _TEXT_TYPES:
        yield TextDelta(text)


def _resolve_fragment(index_part: str, fragments: List[Dict]) -> Optional[Dict]:
    if not fragments:
        return None
    if index_part == "-1":
        return fragments[-1]
    if index_part.isdigit():
        idx = int(index_part)
        if 0 <= idx < len(fragments):
            return fragments[idx]
    return None


def _extract_queries(frag: Dict) -> List[str]:
    out: List[str] = []
    for q in frag.get("queries") or []:
        if isinstance(q, dict):
            text = q.get("query") or q.get("q")
        else:
            text = q
        if isinstance(text, str) and text.strip():
            out.append(text.strip())
    return out


def _merge_results(results: Iterable, st: StreamState) -> Iterator[object]:
    added: List[Dict] = []
    seen = {
        (r.get("cite_index"), r.get("url"))
        for r in st.search_results
        if isinstance(r, dict)
    }
    for item in results:
        if not isinstance(item, dict):
            continue
        key = (item.get("cite_index"), item.get("url"))
        if key in seen:
            continue
        seen.add(key)
        st.search_results.append(item)
        added.append(item)
    if added:
        yield SearchUpdate(results=tuple(added), queries=tuple(st.search_queries))


def _join(parent: str, child) -> str:
    if not child:
        return parent
    child = str(child)
    if parent in ("", "response"):
        return "response/" + child if parent == "response" else child
    return parent + "/" + child


def _as_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
