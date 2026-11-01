"""SSE parser tests driven by real captured streams plus synthetic edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepseek.events import (
    DoneEvent,
    Ready,
    ReasoningDelta,
    SearchUpdate,
    StatusEvent,
    StreamState,
    TextDelta,
    TitleEvent,
)
from deepseek.sse import iter_events

FIXTURES = Path(__file__).parent / "fixtures"


def parse(name: str):
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    state = StreamState()
    events = list(iter_events(lines, state))
    return events, state


def text_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def reasoning_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, ReasoningDelta))


def test_thinking_stream_keeps_reasoning_out_of_answer():
    """THINK deltas and the RESPONSE list-append must be separated correctly.

    Regression: the old parser emitted THINK text as the answer and dropped the
    RESPONSE fragment entirely (it arrived as a list).
    """
    events, state = parse("thinking_list_append.sse")

    assert reasoning_of(events).startswith("We need answer only OK")
    assert text_of(events) == "OK"
    assert state.message_id == 2
    assert isinstance(events[0], Ready)
    assert events[0].response_message_id == 2


def test_search_stream_surfaces_queries_results_and_status():
    events, state = parse("search_news.sse")

    texts = text_of(events)
    assert texts.startswith("A PHP security vulnerability")
    assert "[citation:2]" in texts

    search_events = [e for e in events if isinstance(e, SearchUpdate)]
    assert search_events, "search updates must be emitted"
    assert state.search_queries, "queries must be captured"
    assert len(state.search_results) >= 10
    assert all("url" in r for r in state.search_results)
    # cite_index 1..N with 6 absent: sorted_results must stay ordered.
    indexes = [r["cite_index"] for r in state.sorted_results() if r.get("cite_index")]
    assert indexes == sorted(indexes)

    assert state.status == "FINISHED"
    assert state.token_usage == 74
    assert state.search_triggered is True
    assert any(isinstance(e, StatusEvent) and e.status == "FINISHED" for e in events)


def test_plain_answer_stream_and_title_and_close():
    events, state = parse("plain_answer.sse")

    assert text_of(events).startswith("I can't call functions")
    assert reasoning_of(events) == ""
    assert state.message_id == 2
    assert any(isinstance(e, TitleEvent) for e in events)
    assert any(isinstance(e, DoneEvent) for e in events)


def test_bare_appends_without_path_frames():
    lines = [
        'data: {"v":{"response":{"message_id":7,"fragments":'
        '[{"id":1,"type":"RESPONSE","content":"Hel"}]}}}',
        'data: {"v":"lo"}',
        'data: {"v":" world"}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
    ]
    state = StreamState()
    events = list(iter_events(lines, state))

    assert text_of(events) == "Hello world"
    assert state.message_id == 7


def test_set_content_replaces_without_duplicating():
    lines = [
        'data: {"v":{"response":{"message_id":3,"fragments":'
        '[{"id":1,"type":"RESPONSE","content":"abc"}]}}}',
        'data: {"p":"response/fragments/-1/content","o":"SET","v":"abcdef"}',
    ]
    events = list(iter_events(lines, StreamState()))
    assert text_of(events) == "abcdef"


def test_nested_batch_relative_paths_are_joined():
    lines = [
        'data: {"v":{"response":{"message_id":4,"fragments":[]}}}',
        'data: {"p":"response","o":"BATCH","v":['
        '{"p":"fragments","o":"APPEND","v":[{"id":2,"type":"RESPONSE","content":"Hi"}]},'
        '{"p":"has_pending_fragment","o":"SET","v":false}]}',
        'data: {"p":"response","o":"BATCH","v":['
        '{"p":"accumulated_token_usage","v":12},'
        '{"p":"quasi_status","v":"FINISHED"}]}',
        'data: {"p":"response/fragments/-1","o":"BATCH","v":['
        '{"p":"status","v":"FINISHED"},{"p":"content","v":"Hi there"}]}',
    ]
    state = StreamState()
    events = list(iter_events(lines, state))

    # The nested SET carries the full content, so only the suffix is new.
    assert text_of(events) == "Hi there"
    assert state.token_usage == 12


def test_thinking_content_path_is_reasoning():
    lines = [
        'data: {"v":{"response":{"message_id":5,"fragments":'
        '[{"id":1,"type":"RESPONSE","content":"42"}]}}}',
        'data: {"p":"response/fragments/-1/thinking_content","o":"APPEND","v":"let me think"}',
    ]
    events = list(iter_events(lines, StreamState()))
    assert text_of(events) == "42"
    assert reasoning_of(events) == "let me think"


def test_malformed_and_non_data_lines_are_ignored():
    lines = [
        "event: ready",
        'data: {"request_message_id":1,"response_message_id":9,"model_type":"default"}',
        "",
        "event: update_session",
        'data: {"updated_at":123}',
        "data: not-json",
        "id: 42",
        ": comment",
        'data: {"v":{"response":{"message_id":9,"fragments":'
        '[{"type":"RESPONSE","content":"ok"}]}}}',
    ]
    state = StreamState()
    events = list(iter_events(lines, state))
    assert text_of(events) == "ok"
    assert state.message_id == 9
