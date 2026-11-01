"""Client tests against a mocked HTTP transport (no network, no real account)."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from deepseek.auth import Session
from deepseek.client import DeepSeekAPIError, DeepSeekClient
from deepseek.events import ReasoningDelta, TextDelta

SSE_OK = "\n".join(
    [
        "event: ready",
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"default"}',
        "",
        'data: {"v":{"response":{"message_id":2,"fragments":'
        '[{"id":2,"type":"THINK","content":"hmm"}]}}}',
        'data: {"v":" more"}',
        'data: {"p":"response/fragments","o":"APPEND","v":'
        '[{"id":3,"type":"RESPONSE","content":"Hello"}]}',
        'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" world"}',
        'data: {"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":33}]}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
        "event: title",
        'data: {"content":"Greeting"}',
        "event: close",
        'data: {"click_behavior":"none","auto_resume":false}',
        "",
    ]
)


class FakePow:
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0

    def make_header(self, challenge: dict) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("challenge expired")
        return "pow-token"


def make_client(handler, pow_fail_times: int = 0) -> DeepSeekClient:
    session = Session(token="token", cookies={}, user_agent="ua", captured_at=time.time())
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x")
    client = DeepSeekClient(session=session, http=http)
    client._pow = FakePow(fail_times=pow_fail_times)
    return client


def biz(payload, code=0, biz_code=0, biz_msg=""):
    return {
        "code": code,
        "msg": "",
        "data": {"biz_code": biz_code, "biz_msg": biz_msg, "biz_data": payload},
    }


def base_handler(stream_body: str, stream_headers=None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v0/chat_session/create":
            return httpx.Response(
                200, json=biz({"chat_session": {"id": "sess-1"}})
            )
        if path == "/api/v0/chat/create_pow_challenge":
            return httpx.Response(
                200,
                json=biz(
                    {
                        "challenge": {
                            "challenge": "c",
                            "salt": "s",
                            "signature": "sig",
                            "difficulty": 1,
                            "expire_at": 1,
                            "target_path": "/api/v0/chat/completion",
                        }
                    }
                ),
            )
        if path == "/api/v0/chat/completion":
            headers = stream_headers or {"content-type": "text/event-stream"}
            return httpx.Response(200, content=stream_body.encode(), headers=headers)
        return httpx.Response(404, json={"detail": "not found"})

    return handler


def test_request_yields_typed_events_and_resumable_id():
    client = make_client(base_handler(SSE_OK))
    stream = client.request("hi")

    events = list(stream)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))

    assert text == "Hello world"
    assert reasoning == "hmm more"
    assert stream.state.token_usage == 33
    assert stream.conversation_id == "sess-1:2"


def test_resume_uses_session_and_parent_and_omits_model():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/chat/completion":
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=SSE_OK.encode(),
                                  headers={"content-type": "text/event-stream"})
        return base_handler(SSE_OK)(request)

    client = make_client(handler)
    list(client.request("again", conversation_id="sess-1:2"))

    assert seen["chat_session_id"] == "sess-1"
    assert seen["parent_message_id"] == 2
    assert "model_type" not in seen


def test_json_error_envelope_raises_api_error():
    body = json.dumps(
        {
            "code": 0,
            "msg": "",
            "data": {"biz_code": 9, "biz_msg": "invalid ref file id", "biz_data": None},
        }
    )
    client = make_client(base_handler(body, {"content-type": "application/json"}))

    with pytest.raises(DeepSeekAPIError) as excinfo:
        list(client.request("hi", ref_file_ids=["nope"]))

    assert excinfo.value.biz_code == 9
    assert "invalid ref file id" in str(excinfo.value)


def test_json_error_without_json_content_type_is_detected():
    body = json.dumps(
        {"code": 0, "msg": "", "data": {"biz_code": 3, "biz_msg": "boom", "biz_data": None}}
    )
    client = make_client(base_handler(body))

    with pytest.raises(DeepSeekAPIError, match="boom"):
        list(client.request("hi"))


def test_http_error_is_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/chat/completion":
            return httpx.Response(503, text="upstream down")
        return base_handler(SSE_OK)(request)

    client = make_client(handler)
    with pytest.raises(DeepSeekAPIError) as excinfo:
        list(client.request("hi"))
    assert excinfo.value.status == 503


def test_pow_is_retried_once_when_challenge_expires():
    client = make_client(base_handler(SSE_OK), pow_fail_times=1)

    events = list(client.request("hi"))
    assert any(isinstance(e, TextDelta) for e in events)
    assert client._pow.calls == 2


def test_chat_helper_returns_reply_and_reasoning():
    client = make_client(base_handler(SSE_OK))
    reply = client.chat("hi")

    assert reply.text == "Hello world"
    assert reply.reasoning == "hmm more"
    assert reply.conversation_id == "sess-1:2"
    assert reply.token_usage == 33


