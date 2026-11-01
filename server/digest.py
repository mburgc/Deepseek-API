"""Digest of oversized tool outputs.

A coding agent can return a 200 KB file read in a single tool message. Sending
it whole burns the context window; cutting it blindly to the first N chars loses
exactly the part that usually matters (the end of a stack trace, the last
function of a file, the tail of a log).

This module keeps a *structured* digest instead:

* a header with the real size,
* an index of structural lines (`def`, `class`, headings, `key: value` blocks)
  with their line numbers, so the model knows what it is missing,
* the beginning and the end of the content,
* a hint telling the model how to request the rest.

It is deterministic, costs no extra tokens and never calls a model.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Lines that make a useful table of contents.
_INDEX_RE = re.compile(
    r"^\s*(?:"
    r"export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:def|class|function|func|fn|pub\s+fn|public|private|protected|internal|"
    r"interface|struct|enum|impl|trait|type|module|namespace|package|sub|proc)\b"
    r"|^\s*#{1,6}\s"
    r"|^\s*\[?[A-Za-z0-9_.\-]+\]?\s*[:=]\s*$"
    r"|^\s*(?:CREATE|ALTER|SELECT|INSERT|UPDATE|DELETE)\s",
    re.IGNORECASE,
)

MAX_INDEX_ENTRIES = 40


def _index_block(lines: List[str]) -> str:
    entries: List[Tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        if len(entries) >= MAX_INDEX_ENTRIES:
            break
        if _INDEX_RE.match(line):
            text = line.strip()
            if len(text) > 110:
                text = text[:107] + "..."
            entries.append((number, text))
    if not entries:
        return ""
    body = "\n".join(f"{number}: {text}" for number, text in entries)
    return "### indice\n" + body + "\n"


def _cut_at_newline(text: str, limit: int, from_end: bool) -> str:
    if limit <= 0:
        return ""
    if from_end:
        chunk = text[-limit:]
        index = chunk.find("\n")
        return chunk[index + 1:] if 0 <= index < 200 else chunk
    chunk = text[:limit]
    index = chunk.rfind("\n")
    return chunk[:index] if index > limit - 200 else chunk


def digest_tool_output(text: str, limit: int) -> str:
    """Return ``text`` unchanged when it fits, or a structured digest."""
    if limit <= 0 or len(text) <= limit:
        return text

    lines = text.splitlines()
    total_lines = len(lines)
    header = (
        f"[truncado: {total_lines} lineas / {len(text)} chars]\n"
        "[RESULTADO INCOMPLETO: NO inventes lineas, conteos ni hashes. "
        "Si necesitas el contenido omitido, volve a leerlo con offset/limit.]\n"
    )

    index = _index_block(lines) if limit >= 1500 else ""
    hint = (
        "### nota\n"
        "Contenido recortado. No supongas ni inventes lo que falta. "
        "Para ver una parte concreta, volve a leerlo con offset/limit "
        "(o pedi un rango de lineas).\n"
    )

    overhead = len(header) + len(index) + len(hint) + len("### inicio\n") + len("\n...\n### final\n")
    budget = limit - overhead
    if budget < 200:
        # Extremely small budget: just keep head and tail, no index.
        index = ""
        overhead = len(header) + len(hint) + len("### inicio\n") + len("\n...\n### final\n")
        budget = max(200, limit - overhead)

    head_chars = int(budget * 0.65)
    tail_chars = budget - head_chars
    head = _cut_at_newline(text, head_chars, from_end=False)
    tail = _cut_at_newline(text, tail_chars, from_end=True)

    return f"{header}{index}### inicio\n{head}\n...\n### final\n{tail}\n{hint}"
