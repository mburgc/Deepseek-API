"""End-to-end API tests with a fake upstream client (no network, no account)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import server.api as api
from deepseek.events import ReasoningDelta, StreamState, TextDelta
from deepseek.files import FileRecord
from server.threads import ThreadStore


class FakeStream:
    def __init__(self, events, *, session_id="sess-1", message_id=2, usage=77, results=None):
        self._events = events
        self.session_id = session_id
        self.state = StreamState()
        self.state.message_id = message_id
        self.state.token_usage = usage
        self.state.search_results = results or []

    def __iter__(self):
        yield from self._events

    @property
    def message_id(self):
        return self.state.message_id

    @property
    def conversation_id(self):
        return f"{self.session_id}:{self.state.message_id}"


class FakeClient:
    def __init__(self, events, **stream_kwargs):
        self.events = events
        self.stream_kwargs = stream_kwargs
        self.calls = []

    def request(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return FakeStream(list(self.events), **self.stream_kwargs)

    def current_user(self):
        return {"id": "u1", "email": "test@example.com"}

    def close(self):
        pass


class FakeFiles:
    def __init__(self):
        self.records = {}

    def upload(self, data, filename, mime=None):
        record = FileRecord(
            id="file-test",
            file_name=filename,
            status="SUCCESS",
            file_size=len(data),
            mime=mime,
        )
        self.records[record.id] = record
        return record

    def get(self, file_id):
        return self.records[file_id]

    def download(self, file_id):
        return b"contenido", "text/plain", "nota.txt"

    def forget(self, file_id):
        self.records.pop(file_id, None)

    def close(self):
        pass


class FakeServices:
    def __init__(self, client, tmp_path):
        self.client = client
        self.threads = ThreadStore(tmp_path / "bridge.db")
        self.files = FakeFiles()


@pytest.fixture
def wired(tmp_path, monkeypatch):
    def build(events, **stream_kwargs):
        client = FakeClient(events, **stream_kwargs)
        services = FakeServices(client, tmp_path)
        monkeypatch.setattr(api, "get_services", lambda: services)
        monkeypatch.setattr(api, "SESSION_PREWARM", False)
        return TestClient(api.app), client

    return build


def parse_frames(text: str):
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


def test_non_stream_chat(wired):
    client_http, fake = wired([TextDelta("Hola "), ReasoningDelta("pienso"), TextDelta("mundo")])

    response = client_http.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hola mundo"
    assert body["choices"][0]["message"]["reasoning_content"] == "pienso"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["conversation_id"] == "sess-1:2"
    assert body["usage"]["total_tokens"] == 77
    assert fake.calls[0]["prompt"] == "hi"


def test_second_turn_resumes_thread_and_sends_only_the_tail(wired):
    client_http, fake = wired([TextDelta("Hola mundo")])

    client_http.post(
        "/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
    )
    response = client_http.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hola mundo"},
                {"role": "user", "content": "again"},
            ],
        },
    )

    assert response.status_code == 200
    assert len(fake.calls) == 2
    second = fake.calls[1]
    assert second["session_id"] == "sess-1"
    assert second["parent_id"] == 2
    assert second["prompt"] == "again"


def test_stream_chat_emits_openai_frames(wired):
    client_http, _ = wired([TextDelta("uno "), TextDelta("dos")])

    response = client_http.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    frames = parse_frames(response.text)
    content = "".join(
        f["choices"][0]["delta"].get("content", "")
        for f in frames
        if isinstance(f, dict) and f.get("choices")
    )
    assert content == "uno dos"
    assert frames[-1] == "[DONE]"
    assert any(isinstance(f, dict) and not f["choices"] for f in frames)


def test_tools_produce_tool_calls(wired):
    client_http, _ = wired(
        [TextDelta('<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>')]
    )

    response = client_http.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "lee a.txt"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "read",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
    )

    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt"}


def test_tool_choice_none_disables_tools(wired):
    client_http, fake = wired([TextDelta("sin herramientas")])

    client_http.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "tool_choice": "none",
        },
    )

    assert "TOOLS" not in fake.calls[0]["prompt"]


def test_unknown_model_returns_404(wired):
    client_http, _ = wired([TextDelta("x")])
    response = client_http.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_empty_messages_returns_400(wired):
    client_http, _ = wired([TextDelta("x")])
    response = client_http.post("/v1/chat/completions", json={"model": "deepseek-chat", "messages": []})
    assert response.status_code == 400


def test_models_and_health(wired):
    client_http, _ = wired([TextDelta("x")])

    models = client_http.get("/v1/models").json()["data"]
    ids = {m["id"] for m in models}
    assert "deepseek-chat" in ids
    assert "deepseek-expert-think-search" in ids
    assert client_http.get("/healthz").json() == {"status": "ok"}


def test_file_upload_and_download(wired):
    client_http, _ = wired([TextDelta("x")])

    upload = client_http.post(
        "/v1/files", files={"file": ("nota.txt", b"hola", "text/plain")}
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["id"] == "file-test"
    assert body["filename"] == "nota.txt"

    content = client_http.get("/v1/files/file-test/content")
    assert content.status_code == 200
    assert content.content == b"contenido"
    assert "nota.txt" in content.headers["content-disposition"]


def test_readyz_reports_session(wired):
    client_http, _ = wired([TextDelta("x")])
    response = client_http.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_health_aliases_and_root(wired):
    client_http, _ = wired([TextDelta("x")])

    for path in ("/health", "/v1/health", "/healthz"):
        assert client_http.get(path).status_code == 200, path
    assert client_http.get("/").status_code == 200
    assert client_http.get("/ready").status_code == 200


def test_http_logging_records_the_404_path(wired):
    client_http, _ = wired([TextDelta("x")])
    capture = _LogCapture()
    api._log.addHandler(capture)
    try:
        response = client_http.get("/nope")
    finally:
        api._log.removeHandler(capture)

    assert response.status_code == 404
    assert any("http GET /nope -> 404" in line for line in capture.messages)


def test_chat_logs_model_and_thread(wired):
    client_http, _ = wired([TextDelta("ok")])
    capture = _LogCapture()
    api._log.addHandler(capture)
    try:
        response = client_http.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        api._log.removeHandler(capture)

    assert response.status_code == 200
    assert any("chat model=deepseek-chat" in line for line in capture.messages)
    assert any("mode=new" in line for line in capture.messages)
    assert any("thread saved" in line for line in capture.messages)


def test_serialized_endpoint_holds_the_lock_until_the_stream_finishes():
    active = 0
    peak = 0
    guard = threading.Lock()

    @api._serialized
    async def endpoint():
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)

        async def body():
            nonlocal active
            try:
                yield b"uno"
                await asyncio.sleep(0.05)
            finally:
                with guard:
                    active -= 1

        return StreamingResponse(body())

    async def call():
        response = await endpoint()
        async for _ in response.body_iterator:
            pass

    async def run():
        await asyncio.gather(call(), call())

    asyncio.run(run())

    assert peak == 1
