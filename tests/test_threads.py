"""Thread store tests: prefix-extension matching, invalidation, TTL."""

from __future__ import annotations

from server.threads import ThreadStore


def make(tmp_path, ttl=3600.0) -> ThreadStore:
    return ThreadStore(tmp_path / "bridge.db", ttl_seconds=ttl)


def test_resume_only_when_history_is_an_extension(tmp_path):
    store = make(tmp_path)
    first_turn = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "buenas"},
    ]
    store.save(
        key="k",
        session_id="sess-1",
        last_message_id=2,
        messages_seen=first_turn,
        model_id="deepseek-chat",
    )

    extended = first_turn + [{"role": "user", "content": "otra"}]
    thread = store.lookup("k", extended)

    assert thread is not None
    assert thread.session_id == "sess-1"
    assert thread.last_message_id == 2
    assert thread.seen_count == 2
    assert thread.conversation_id == "sess-1:2"


def test_no_resume_when_history_diverges(tmp_path):
    store = make(tmp_path)
    store.save(
        key="k",
        session_id="s",
        last_message_id=2,
        messages_seen=[
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "buenas"},
        ],
        model_id="m",
    )

    edited = [
        {"role": "user", "content": "hola EDITADO"},
        {"role": "assistant", "content": "buenas"},
        {"role": "user", "content": "otra"},
    ]
    assert store.lookup("k", edited) is None


def test_no_resume_without_new_messages(tmp_path):
    store = make(tmp_path)
    seen = [{"role": "user", "content": "hola"}]
    store.save(key="k", session_id="s", last_message_id=1, messages_seen=seen, model_id="m")
    assert store.lookup("k", seen) is None


def test_tool_arguments_and_results_are_part_of_the_identity(tmp_path):
    store = make(tmp_path)
    turn = [
        {"role": "user", "content": "lee"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path":"a"}'}}
            ],
        },
    ]
    store.save(key="k", session_id="s", last_message_id=9, messages_seen=turn, model_id="m")

    same = turn + [{"role": "tool", "content": "x"}]
    different = [
        turn[0],
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path":"b"}'}}
            ],
        },
        {"role": "tool", "content": "x"},
    ]

    assert store.lookup("k", same) is not None
    assert store.lookup("k", different) is None


def test_expired_thread_is_dropped(tmp_path):
    store = make(tmp_path, ttl=0.0)
    msgs = [{"role": "user", "content": "hola"}]
    store.save(key="k", session_id="s", last_message_id=1, messages_seen=msgs, model_id="m")

    assert store.lookup("k", msgs + [{"role": "user", "content": "nuevo"}]) is None
    assert store.get("k") is None


def test_prune_removes_old_rows(tmp_path):
    store = make(tmp_path, ttl=0.0)
    store.save(
        key="k",
        session_id="s",
        last_message_id=1,
        messages_seen=[{"role": "user", "content": "a"}],
        model_id="m",
    )
    assert store.prune() == 1
