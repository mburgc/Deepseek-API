"""OpenAI translation tests: prompts, responses, streaming frames."""

from __future__ import annotations

import json

from deepseek.events import ReasoningDelta, StreamState, TextDelta
from server.openai_format import (
    collect_answer,
    completion_response,
    format_sources,
    messages_to_prompt,
    messages_to_prompt_tail,
    rewrite_citations,
    stream_chunks,
    tail_messages,
    usage,
    with_tools_block,
)
from server.toolbridge import ToolCall


def test_single_user_message_is_sent_verbatim():
    messages = [{"role": "user", "content": "hola"}]
    assert messages_to_prompt(messages) == "hola"


def test_multi_turn_renders_roles_tool_calls_and_results():
    messages = [
        {"role": "system", "content": "eres un agente"},
        {"role": "user", "content": "lee a.txt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "contenido"},
    ]
    prompt = messages_to_prompt(messages)

    assert "System:" in prompt
    assert "<tool_call>" in prompt and "read_file" in prompt
    assert "Tool result" in prompt and "contenido" in prompt
    assert prompt.endswith("Assistant:")


def test_with_tools_block_appends_to_existing_system():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "read_file", "parameters": {"type": "object"}},
        }
    ]
    out = with_tools_block(messages, tools, "auto")

    assert out is not messages
    assert out[0]["content"].startswith("sys")
    assert "# TOOLS" in out[0]["content"]
    assert messages[0]["content"] == "sys"


def test_with_tools_block_creates_system_when_missing():
    out = with_tools_block(
        [{"role": "user", "content": "u"}],
        [{"type": "function", "function": {"name": "t"}}],
        "auto",
    )
    assert out[0]["role"] == "system"


def test_tail_messages_returns_input_after_last_assistant():
    messages = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "tool", "tool_call_id": "c", "content": "3"},
        {"role": "user", "content": "4"},
    ]
    assert tail_messages(messages) == [
        {"role": "tool", "tool_call_id": "c", "content": "3"},
        {"role": "user", "content": "4"},
    ]
    assert messages_to_prompt_tail(tail_messages(messages)).endswith("4")
    assert messages_to_prompt_tail([{"role": "user", "content": "solo"}]) == "solo"


def test_completion_response_maps_reasoning_tools_and_citations():
    results = [
        {"url": "https://a", "title": "A", "cite_index": 1, "site_name": "a.com"},
    ]
    response = completion_response(
        model="deepseek-chat",
        content="hola [citation:1]",
        prompt="p",
        conversation_id="s:2",
        reasoning="pensé",
        tool_calls=[ToolCall(name="read_file", arguments={"path": "a"})],
        search_results=results,
        token_usage=42,
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "pensé"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert choice["message"]["annotations"][0]["url_citation"]["url"] == "https://a"
    assert response["usage"]["total_tokens"] == 42
    assert response["conversation_id"] == "s:2"


def test_sources_and_citation_rewrite():
    results = [{"url": "https://a", "title": "Titulo", "cite_index": 2}]
    assert "[2] Titulo — https://a" in format_sources(results, "x [citation:2]")
    assert rewrite_citations("ver [citation:2]", results) == "ver [Titulo](https://a)"


class FakeStream:
    def __init__(self, events, *, message_id=2, session_id="sess-1", results=None):
        self._events = events
        self.session_id = session_id
        self.state = StreamState()
        self.state.message_id = message_id
        self.state.token_usage = 55
        self.state.search_results = results or []

    def __iter__(self):
        yield from self._events

    @property
    def message_id(self):
        return self.state.message_id

    @property
    def conversation_id(self):
        return f"{self.session_id}:{self.state.message_id}"


def frames(text: str):
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                out.append("[DONE]")
            else:
                out.append(json.loads(payload))
    return out


def test_stream_chunks_emits_role_content_reasoning_and_usage():
    stream = FakeStream(
        [TextDelta("Hola "), ReasoningDelta("pensando"), TextDelta("mundo")]
    )
    result = {}
    text = "".join(
        stream_chunks(
            model="deepseek-chat",
            stream=stream,
            prompt="hi",
            include_usage=True,
            result=result,
        )
    )

    parsed = frames(text)
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        f["choices"][0]["delta"].get("content", "")
        for f in parsed
        if isinstance(f, dict) and f.get("choices")
    )
    assert content == "Hola mundo"
    assert any(
        f["choices"][0]["delta"].get("reasoning_content") == "pensando"
        for f in parsed
        if isinstance(f, dict) and f.get("choices")
    )
    usage_chunk = [f for f in parsed if isinstance(f, dict) and not f["choices"]]
    assert usage_chunk and usage_chunk[0]["usage"]["total_tokens"] > 0
    assert parsed[-1] == "[DONE]"
    assert result["text"] == "Hola mundo"
    assert result["finish_reason"] == "stop"


def test_stream_chunks_maps_tool_calls():
    stream = FakeStream(
        [TextDelta('<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>')]
    )
    result = {}
    text = "".join(
        stream_chunks(
            model="deepseek-chat", stream=stream, tools_enabled=True, result=result
        )
    )

    parsed = frames(text)
    tool_frames = [
        f["choices"][0]["delta"]["tool_calls"]
        for f in parsed
        if isinstance(f, dict) and f.get("choices") and "tool_calls" in f["choices"][0]["delta"]
    ]
    assert tool_frames
    assert tool_frames[0][0]["function"]["name"] == "read_file"
    finish = [f["choices"][0]["finish_reason"] for f in parsed if isinstance(f, dict) and f["choices"]]
    assert "tool_calls" in finish
    assert result["calls"][0]["name"] == "read_file"


def test_collect_answer_none_streaming_path():
    stream = FakeStream(
        [
            TextDelta("Te "),
            TextDelta('<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'),
        ]
    )
    answer = collect_answer(stream, tools_enabled=True)

    assert answer.text.strip() == "Te"
    assert answer.calls[0].name == "read_file"
    assert answer.calls[0].arguments == {"path": "a"}



def test_usage_never_double_counts_the_prompt():
    prompt = "filler " * 10000
    result = usage(prompt, "short answer", 190000)

    assert result["total_tokens"] == 190000
    assert result["prompt_tokens"] + result["completion_tokens"] == 190000
    assert result["prompt_tokens"] > 100000


def test_usage_falls_back_to_estimates_without_upstream_count():
    result = usage("hola", "una respuesta", None)

    assert result["prompt_tokens"] >= 1
    assert result["completion_tokens"] >= 1
    assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
