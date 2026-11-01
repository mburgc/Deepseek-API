"""Planner tests: append, branch (edit), anchor fallback and the @edit marker."""

from __future__ import annotations

from server.threads import ThreadStore


def make(tmp_path, ttl=3600.0) -> ThreadStore:
    return ThreadStore(tmp_path / "bridge.db", ttl_seconds=ttl)


def turn(user: str, assistant: str):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def save_turn(store, key, messages, ids, anchor="a1", session="sess"):
    store.save(
        key=key,
        session_id=session,
        last_message_id=max(v for v in ids.values() if v is not None),
        messages_seen=messages,
        model_id="m",
        anchor=anchor,
        message_ids=ids,
    )


def test_append_when_history_extends(tmp_path):
    store = make(tmp_path)
    first = turn("hola", "buenas")
    save_turn(store, "k", first, {0: 1, 1: 2})

    plan = store.plan("k", first + [{"role": "user", "content": "otra"}], "a1")

    assert plan.mode == "append"
    assert plan.send_from == 2
    assert plan.parent_id == 2
    assert plan.thread.session_id == "sess"


def test_branch_when_a_message_is_edited(tmp_path):
    store = make(tmp_path)
    history = turn("hola", "buenas") + turn("que tal", "bien")
    save_turn(store, "k", history, {0: 1, 1: 2, 2: 3, 3: 4})

    edited = turn("hola", "buenas") + [{"role": "user", "content": "que tal EDITADO"}]
    plan = store.plan("k", edited, "a1")

    assert plan.mode == "branch"
    assert plan.send_from == 2          # solo se reenvía desde el mensaje editado
    assert plan.parent_id == 2          # el asistente anterior al mensaje editado
    assert plan.fork_index == 2


def test_retry_forks_from_the_assistant_never_from_a_user(tmp_path):
    """Regresión: un prompt crea un turno de usuario, así que el padre no puede
    ser otro usuario. DeepSeek responde `invalid message role` si eso pasa."""
    store = make(tmp_path)
    history = turn("hola", "buenas") + turn("que tal", "bien")
    save_turn(store, "k", history, {0: 1, 1: 2, 2: 3, 3: 4})

    retry = turn("hola", "buenas") + [{"role": "user", "content": "que tal"}]
    plan = store.plan("k", retry, "a1")

    assert plan.mode == "branch"
    assert plan.parent_id == 2          # a1, NO u2 (id 3)
    assert plan.send_from == 2          # se reenvía u2


def test_branch_walks_back_when_the_fork_has_no_id(tmp_path):
    store = make(tmp_path)
    history = turn("hola", "buenas") + turn("que tal", "bien")
    # solo el primer turno tiene ids
    save_turn(store, "k", history, {0: 1, 1: 2, 2: None, 3: 4})

    edited = turn("hola", "buenas") + [{"role": "user", "content": "otro"}]
    plan = store.plan("k", edited, "a1")

    assert plan.mode == "branch"
    assert plan.parent_id == 2          # cae al ancestro con id conocido


def test_new_when_no_ids_were_recorded(tmp_path):
    store = make(tmp_path)
    history = turn("hola", "buenas")
    store.save(
        key="k", session_id="s", last_message_id=2,
        messages_seen=history, model_id="m", anchor="a1",
    )

    plan = store.plan("k", turn("hola", "buenas") + [{"role": "user", "content": "x"}], "a1")

    assert plan.mode == "append"        # append no necesita ids
    assert plan.parent_id == 2

    diverged = [{"role": "user", "content": "otro"}]
    assert store.plan("k", diverged, "a1").mode == "new"


def test_anchor_finds_the_thread_when_the_key_changes(tmp_path):
    store = make(tmp_path)
    first = turn("hola", "buenas")
    save_turn(store, "key-vieja", first, {0: 1, 1: 2}, anchor="anchor-x")

    plan = store.plan("key-nueva", first + [{"role": "user", "content": "sigo"}], "anchor-x")

    assert plan.mode == "append"
    assert plan.thread.key == "key-vieja"


def test_edit_marker_truncates_and_appends_the_instruction():
    from server.api import _apply_edit_marker

    messages = [
        {"role": "system", "content": "eres un asistente"},
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "buenas"},
        {"role": "user", "content": "otra cosa"},
        {"role": "assistant", "content": "respuesta"},
        {"role": "user", "content": "@edit más corto por favor"},
    ]

    out = _apply_edit_marker(messages)

    assert out[-1] == {"role": "user", "content": "más corto por favor"}
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "user"]
    assert out[-2]["content"] == "otra cosa"   # se conserva el turno que produjo la respuesta
    assert out[2]["content"] == "buenas"       # y todo el historial anterior


def test_edit_marker_only_affects_directives():
    from server.api import _apply_edit_marker

    messages = [{"role": "user", "content": "hola @edit"}]
    assert _apply_edit_marker(messages) == messages


def test_edit_marker_keeps_history_when_target_is_out_of_range():
    from server.api import _apply_edit_marker

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hola"},
        {"role": "user", "content": "@edit 9 no existe"},
    ]
    assert _apply_edit_marker(messages) == messages

# --- @think / @search -------------------------------------------------------

def test_think_directive_sets_the_flag_and_strips_the_prefix():
    from server.api import _apply_directives

    msgs = [{"role": "user", "content": "@think explicame los agujeros negros"}]
    out, think, search = _apply_directives(msgs)

    assert out[-1]["content"] == "explicame los agujeros negros"
    assert think is True
    assert search is None


def test_search_and_think_can_be_combined():
    from server.api import _apply_directives

    msgs = [{"role": "user", "content": "@search @think que paso hoy en chile"}]
    out, think, search = _apply_directives(msgs)

    assert out[-1]["content"] == "que paso hoy en chile"
    assert think is True and search is True


def test_spanish_aliases_and_other_sigils():
    from server.api import _apply_directives

    for text, key in (("!busca noticias", "search"), ("#piensa esto", "think"), ("/web algo", "search")):
        out, think, search = _apply_directives([{"role": "user", "content": text}])
        value = think if key == "think" else search
        assert value is True, text


def test_plain_message_is_untouched():
    from server.api import _apply_directives

    msgs = [{"role": "user", "content": "hola que tal"}]
    out, think, search = _apply_directives(msgs)
    assert out == msgs and think is None and search is None


def test_directive_only_does_not_leave_an_empty_turn():
    from server.api import _apply_directives

    out, think, _ = _apply_directives([{"role": "user", "content": "@think"}])
    assert think is True
    assert out[-1]["content"].strip()
