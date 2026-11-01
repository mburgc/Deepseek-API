"""Digest of oversized tool outputs: structure preserved, budget respected."""

from __future__ import annotations

from server.digest import digest_tool_output


def test_short_text_is_untouched():
    text = "hola\nmundo"
    assert digest_tool_output(text, 1000) == text


def test_long_text_keeps_head_tail_and_index():
    lines = ["import os"]
    for i in range(2000):
        lines.append(f"def funcion_{i}(x):")
        lines.append(f"    return x + {i}")
    lines.append("ERROR FINAL: 0xDEADBEEF")
    text = "\n".join(lines)

    out = digest_tool_output(text, 4000)

    assert len(out) <= 4200, "el digest debe respetar el presupuesto"
    assert "[truncado:" in out
    assert "### indice" in out and "def funcion_0" in out
    assert "### inicio" in out and "import os" in out
    assert "### final" in out
    assert "ERROR FINAL: 0xDEADBEEF" in out, "el final es lo que mas se pierde"
    assert "offset/limit" in out


def test_no_index_when_nothing_structural():
    text = "\n".join(f"linea {i} de relleno" for i in range(3000))
    out = digest_tool_output(text, 2000)

    assert "### indice" not in out
    assert "### inicio" in out and "### final" in out
    assert len(out) <= 2200


def test_small_budget_still_produces_something_useful():
    text = "A" * 50000
    out = digest_tool_output(text, 800)

    assert "[truncado:" in out
    assert len(out) <= 900


def test_zero_limit_disables_digest():
    text = "x" * 1000
    assert digest_tool_output(text, 0) == text
