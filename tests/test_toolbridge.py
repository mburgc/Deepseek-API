"""Tool-protocol tests: prompt rendering, parsing variants, stream splitter."""

from __future__ import annotations

import json

from server.toolbridge import (
    ToolCallStream,
    parse_tool_calls,
    render_tools_block,
    tool_choice_mode,
    tools_fingerprint,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def test_render_tools_block_includes_schemas_and_protocol():
    block = render_tools_block(TOOLS, "auto")

    assert "# TOOLS" in block
    assert "<tool_call>" in block
    assert "read_file" in block and "run_shell" in block
    assert '"required":["path"]' in block
    assert "MUST call" not in block


def test_render_tools_block_forced_and_none():
    assert "MUST call the tool named \"read_file\"" in render_tools_block(
        TOOLS, {"type": "function", "function": {"name": "read_file"}}
    )
    assert "MUST call at least one tool" in render_tools_block(TOOLS, "required")
    assert render_tools_block(TOOLS, "none") == ""
    assert render_tools_block(None, "auto") == ""


def test_tool_choice_mode_normalisation():
    assert tool_choice_mode(None) == ("auto", None)
    assert tool_choice_mode("none") == ("none", None)
    assert tool_choice_mode({"type": "function", "function": {"name": "x"}}) == ("forced", "x")


def test_tools_fingerprint_is_stable_and_order_independent():
    a = tools_fingerprint(TOOLS)
    b = tools_fingerprint(list(reversed(TOOLS)))
    assert a == b
    assert a != tools_fingerprint(TOOLS[:1])


def test_parse_canonical_tag():
    text = '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'
    visible, calls = parse_tool_calls(text)

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.txt"}
    assert json.loads(calls[0].to_openai("call_1")["function"]["arguments"]) == {"path": "a.txt"}


def test_parse_multiple_calls_with_preamble():
    text = (
        "I'll inspect both files.\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>\n'
        '<tool_call>{"name":"read_file","arguments":{"path":"b"}}</tool_call>'
    )
    visible, calls = parse_tool_calls(text)

    assert visible == "I'll inspect both files."
    assert [c.arguments["path"] for c in calls] == ["a", "b"]


def test_parse_fenced_json_and_openai_wrapper():
    fenced = '```json\n{"name":"run_shell","arguments":{"command":"ls"}}\n```'
    _, calls = parse_tool_calls(fenced)
    assert calls[0].name == "run_shell"

    wrapped = '{"tool_calls":[{"function":{"name":"read_file","arguments":"{\\"path\\":\\"x\\"}"}}]}'
    _, calls = parse_tool_calls(wrapped)
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "x"}


def test_parse_bare_json_and_trailing_comma_repair():
    _, calls = parse_tool_calls('{"name":"read_file","arguments":{"path":"x",}}')
    assert calls[0].arguments == {"path": "x"}


def test_parse_plain_text_is_untouched():
    text = "Just a normal answer with no tools."
    visible, calls = parse_tool_calls(text)
    assert visible == text
    assert calls == []


def test_stream_passes_prose_through():
    stream = ToolCallStream(enabled=True)
    parts = []
    for chunk in ["Hello", " there", ", friend."]:
        parts.extend(stream.feed(chunk))
    parts.extend(stream.flush())

    assert "".join(payload for kind, payload in parts if kind == "text") == "Hello there, friend."
    assert all(kind == "text" for kind, _ in parts)


def test_stream_splits_text_from_call():
    stream = ToolCallStream(enabled=True)
    parts = []
    parts.extend(stream.feed("Let me check. "))
    parts.extend(stream.feed('<tool_call>{"name":"read_file",'))
    parts.extend(stream.feed('"arguments":{"path":"a"}}</tool_call>'))
    parts.extend(stream.flush())

    text = "".join(payload for kind, payload in parts if kind == "text")
    calls = [payload for kind, payload in parts if kind == "tool_calls"]

    assert text.strip() == "Let me check."
    assert len(calls) == 1 and calls[0][0].name == "read_file"


def test_stream_detects_leading_json_without_marker():
    stream = ToolCallStream(enabled=True)
    parts = []
    parts.extend(stream.feed('{"name":"run_shell",'))
    parts.extend(stream.feed('"arguments":{"command":"ls"}}'))
    parts.extend(stream.flush())

    assert [kind for kind, _ in parts] == ["tool_calls"]
    assert parts[0][1][0].arguments == {"command": "ls"}


def test_stream_does_not_leak_partial_marker():
    stream = ToolCallStream(enabled=True)
    first = stream.feed("Answer <tool")
    assert "".join(payload for _, payload in first).strip() == "Answer"
    rest = stream.feed('_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>')
    flushed = stream.flush()

    assert not any("tool_call" in payload for kind, payload in first if kind == "text")
    assert [kind for kind, _ in rest + flushed] == ["tool_calls"]


def test_stream_disabled_returns_raw_text():
    stream = ToolCallStream(enabled=False)
    parts = stream.feed('<tool_call>{"name":"x","arguments":{}}</tool_call>')
    assert parts == [("text", '<tool_call>{"name":"x","arguments":{}}</tool_call>')]


BAR = chr(0xFF5C)
DSML_CAPTURED = (
    f"<{BAR}{BAR}DSML{BAR}{BAR} calls>\n"
    f'<{BAR}{BAR}DSML{BAR}{BAR} invoke name="read">\n'
    f'<{BAR}{BAR}DSML{BAR}{BAR} parameter name="filePath" string="true">'
    f"C:\\a.txt</{BAR}{BAR}DSML{BAR}{BAR} parameter>\n"
    f"</{BAR}{BAR}DSML{BAR}{BAR} invoke>\n"
    f"</{BAR}{BAR}DSML{BAR}{BAR} calls>"
)


def test_parse_captured_dsml_variant():
    visible, calls = parse_tool_calls(DSML_CAPTURED)

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "read"
    assert calls[0].arguments == {"filePath": "C:\\a.txt"}


def test_parse_dsml_typed_params_and_preamble():
    text = (
        "Voy a leerlo.\n"
        "<|DSML|tool_calls>\n"
        '<|DSML|invoke name="run">\n'
        '<|DSML|parameter name="n" string="false">3</|DSML|parameter>\n'
        '<|DSML|parameter name="flag" string="false">true</|DSML|parameter>\n'
        '<|DSML|parameter name="obj" string="false">{"a":1}</|DSML|parameter>\n'
        "</|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    visible, calls = parse_tool_calls(text)

    assert visible == "Voy a leerlo."
    assert calls[0].arguments == {"n": 3, "flag": True, "obj": {"a": 1}}


def test_stream_holds_back_dsml():
    stream = ToolCallStream(enabled=True)
    parts = []
    parts += stream.feed(f"<{BAR}{BAR}DSML{BAR}{BAR} calls>\n<{BAR}{BAR}DSML{BAR}{BAR} invoke")
    parts += stream.feed(f' name="read">\n<{BAR}{BAR}DSML{BAR}{BAR} parameter name="p"')
    parts += stream.feed(f' string="true">x</{BAR}{BAR}DSML{BAR}{BAR} parameter>')
    parts += stream.feed(f"\n</{BAR}{BAR}DSML{BAR}{BAR} invoke>\n</{BAR}{BAR}DSML{BAR}{BAR} calls>")
    parts += stream.flush()

    assert [kind for kind, _ in parts] == ["tool_calls"]
    assert parts[0][1][0].name == "read"
    assert parts[0][1][0].arguments == {"p": "x"}


# --- Windows paths, escapes and prose JSON ----------------------------------


def test_windows_path_with_single_backslashes_is_recovered():
    text = (
        r'<tool_call>{"name":"read","arguments":'
        r'{"filePath":"D:\ProjectoRTS\Assets\Scripts\WaveManager.cs"}}</tool_call>'
    )
    _, calls = parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].arguments["filePath"] == "D:/ProjectoRTS/Assets/Scripts/WaveManager.cs"


def test_windows_path_with_control_escapes_is_recovered():
    text = (
        r'<tool_call>{"name":"read","arguments":'
        r'{"filePath":"D:\tools\new\build\foo.cs"}}</tool_call>'
    )
    _, calls = parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].arguments["filePath"] == "D:/tools/new/build/foo.cs"


def test_report_windows_paths_are_all_recovered():
    paths = [
        r"D:\Deepseek\DeepSeek-Bridge-v1.0",
        r"D:\Deepseek\DeepSeek-Bridge-v1.0\src\bridge\plugin.ts",
        r"C:\Users\PANTRUCAS0\AppData\Local\Temp\opencode\robust1.txt",
        r"D:\ProjectoRTS\Assets\Scripts\WaveManager.cs",
    ]
    for path in paths:
        text = (
            '<tool_call>{"name":"read","arguments":'
            '{"filePath":"' + path + '"}}</tool_call>'
        )
        _, calls = parse_tool_calls(text)
        assert len(calls) == 1, path
        assert calls[0].arguments["filePath"] == path.replace("\\", "/")


def test_edit_arguments_accept_raw_newlines():
    text = (
        '<tool_call>{"name":"edit","arguments":'
        '{"filePath":"a.cs","oldString":"line1\nline2","newString":"line3\nline4"}}'
        "</tool_call>"
    )
    _, calls = parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].arguments["oldString"] == "line1\nline2"
    assert calls[0].arguments["newString"] == "line3\nline4"


def test_regex_pattern_keeps_backslash_escapes():
    text = r'<tool_call>{"name":"grep","arguments":{"pattern":"WaveCleared\b"}}</tool_call>'
    _, calls = parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].arguments["pattern"] == "WaveCleared\\b"


def test_prose_json_is_not_a_tool_call():
    text = (
        "El componente se ve asi:\n"
        '{"name":"WaveManager","speed":5}\n'
        "Y se usa en la escena 3."
    )
    visible, calls = parse_tool_calls(text, allowed_names={"read"})

    assert calls == []
    assert visible == text


def test_bare_json_only_counts_when_the_name_is_known():
    text = '{"name":"WaveManager","speed":5}'
    visible, calls = parse_tool_calls(text, allowed_names={"read"})

    assert calls == []
    assert visible == text


def test_bare_json_with_known_name_still_works():
    text = '{"name":"read","arguments":{"path":"a.txt"}}'
    _, calls = parse_tool_calls(text, allowed_names={"read"})

    assert len(calls) == 1
    assert calls[0].name == "read"


def test_fenced_json_inside_prose_is_not_a_tool_call():
    text = 'Ejemplo:\n```json\n{"name":"WaveManager","speed":5}\n```\nFin.'
    visible, calls = parse_tool_calls(text, allowed_names={"read"})

    assert calls == []
    assert visible == text


def test_mixed_json_and_dsml_are_both_parsed():
    bar = chr(0xFF5C)
    json_call = (
        '<tool_call>{"name":"read","arguments":'
        '{"filePath":"D:/ProjectoRTS/A.cs"}}</tool_call>'
    )
    dsml_call = (
        f"<{bar}{bar}DSML{bar}{bar} invoke name=\"read\">\n"
        f"<{bar}{bar}DSML{bar}{bar} parameter name=\"filePath\" string=\"true\">"
        f"D:/ProjectoRTS/B.cs</{bar}{bar}DSML{bar}{bar} parameter>\n"
        f"</{bar}{bar}DSML{bar}{bar} invoke>"
    )
    visible, calls = parse_tool_calls(json_call + "\n" + dsml_call)

    assert [c.arguments["filePath"] for c in calls] == [
        "D:/ProjectoRTS/A.cs",
        "D:/ProjectoRTS/B.cs",
    ]
    assert "<tool_call>" not in visible
    assert "DSML" not in visible
