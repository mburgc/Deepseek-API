"""Conversation threading.

A DeepSeek chat session keeps its own server-side history and is resumed with
``parent_message_id``. Reusing it avoids re-sending the whole OpenAI transcript
(and the system prompt) on every turn, which is what makes agent loops usable.

Two things matter here:

* **Identity.** A thread is keyed by ``key`` (system + model + toggles + tools)
  and also carries an ``anchor`` (hash of the first non-system message), so a
  conversation can still be found when the system prompt or the tool set changes
  mid-flight.

* **Branching.** DeepSeek's completion appends a child to ``parent_message_id``.
  Passing an *ancestor* therefore creates a sibling branch instead of a new
  session — that is how an edited message is handled: the bridge replays only
  the messages after the fork point and keeps everything before it.

The store keeps, per thread, the id DeepSeek assigned to each message it has
seen. ``plan()`` compares the incoming OpenAI history against that chain and
decides between ``append``, ``branch`` and ``new``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .content import canonical_message, canonical_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    key             TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    last_message_id INTEGER,
    seen_count      INTEGER NOT NULL,
    history_hash    TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at);

CREATE TABLE IF NOT EXISTS thread_messages (
    thread_key  TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    message_id  INTEGER,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (thread_key, idx)
);
"""

# Columns added after the first release; applied with ALTER when missing.
_EXTRA_COLUMNS = (
    ("anchor", "TEXT"),
    ("profile_hash", "TEXT"),
    ("last_response", "TEXT"),
)


def message_hash(message: Dict[str, Any]) -> str:
    """Stable hash of a single message, used for prefix comparison."""
    import hashlib
    import json

    blob = json.dumps(canonical_message(message), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Thread:
    key: str
    session_id: str
    last_message_id: Optional[int]
    seen_count: int
    history_hash: str
    model_id: str
    updated_at: float
    anchor: Optional[str] = None
    profile_hash: Optional[str] = None
    last_response: Optional[str] = None

    @property
    def conversation_id(self) -> str:
        if self.last_message_id is None:
            return self.session_id
        return f"{self.session_id}:{self.last_message_id}"


@dataclass
class Plan:
    """How the incoming history relates to what the thread already saw."""

    mode: str  # "new" | "append" | "branch"
    thread: Optional[Thread] = None
    send_from: int = 0
    parent_id: Optional[int] = None
    fork_index: int = -1
    profile_changed: bool = False

    @property
    def is_new(self) -> bool:
        return self.mode == "new"


@dataclass
class _Seen:
    idx: int
    message_id: Optional[int]
    content_hash: str
    role: str = ""


class ThreadStore:
    def __init__(self, path: Path, ttl_seconds: float = 12 * 3600.0):
        self.path = Path(path)
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        have = {row["name"] for row in conn.execute("PRAGMA table_info(threads)")}
        for column, ctype in _EXTRA_COLUMNS:
            if column not in have:
                conn.execute(f"ALTER TABLE threads ADD COLUMN {column} {ctype}")
        conn.commit()

    # --- lookups ---------------------------------------------------------

    def _row(self, conn: sqlite3.Connection, key: str) -> Optional[sqlite3.Row]:
        return conn.execute("SELECT * FROM threads WHERE key = ?", (key,)).fetchone()

    def _row_by_anchor(self, conn: sqlite3.Connection, anchor: str) -> Optional[sqlite3.Row]:
        if not anchor:
            return None
        return conn.execute(
            "SELECT * FROM threads WHERE anchor = ? ORDER BY updated_at DESC LIMIT 1",
            (anchor,),
        ).fetchone()

    def _seen(self, conn: sqlite3.Connection, key: str) -> List[_Seen]:
        rows = conn.execute(
            "SELECT idx, message_id, content_hash, role FROM thread_messages "
            "WHERE thread_key = ? ORDER BY idx",
            (key,),
        ).fetchall()
        return [
            _Seen(r["idx"], r["message_id"], r["content_hash"], r["role"] or "")
            for r in rows
        ]

    def _dead(self, conn: sqlite3.Connection, thread: Thread) -> bool:
        return time.time() - thread.updated_at > self.ttl

    # --- planning --------------------------------------------------------

    def plan(
        self,
        key: str,
        messages: List[Dict[str, Any]],
        anchor: Optional[str] = None,
        profile_hash: Optional[str] = None,
    ) -> Plan:
        """Decide between appending, branching and starting fresh."""
        with self._lock:
            conn = self._connect()
            try:
                row = self._row(conn, key)
                found_by_anchor = False
                if row is None:
                    row = self._row_by_anchor(conn, anchor or "")
                    found_by_anchor = row is not None
                if row is None:
                    return Plan("new")
                thread = _row_to_thread(row)
                if self._dead(conn, thread):
                    conn.execute("DELETE FROM threads WHERE key = ?", (thread.key,))
                    conn.execute("DELETE FROM thread_messages WHERE thread_key = ?", (thread.key,))
                    conn.commit()
                    return Plan("new")
                if (
                    found_by_anchor
                    and thread.profile_hash
                    and profile_hash
                    and thread.profile_hash != profile_hash
                ):
                    # Same opening, different system prompt: almost certainly
                    # another project/session, not a profile rewrite.
                    return Plan("new")

                seen = self._seen(conn, thread.key)
                if not seen:
                    return Plan("new")

                # The system/developer turns are not part of the conversational
                # chain: a rewritten system prompt must not force a new session.
                common = 0
                for index, message in enumerate(messages):
                    if (message.get("role") or "") in ("system", "developer"):
                        continue
                    if index >= thread.seen_count or index >= len(seen):
                        break
                    if seen[index].content_hash != message_hash(message):
                        break
                    common = index + 1

                if common == 0:
                    return Plan("new")

                profile_changed = bool(
                    profile_hash and thread.profile_hash and thread.profile_hash != profile_hash
                )

                if common == thread.seen_count and len(messages) > thread.seen_count:
                    return Plan(
                        "append", thread, thread.seen_count, thread.last_message_id, -1,
                        profile_changed=profile_changed,
                    )

                # Diverges (edit) or the client dropped the last answer (retry).
                # The fork point must be an ASSISTANT message: a prompt always
                # creates a user turn, and DeepSeek rejects user -> user.
                for back in range(common - 1, -1, -1):
                    if back >= len(seen):
                        continue
                    entry = seen[back]
                    if entry.message_id is None:
                        continue
                    if (entry.role or "").lower() != "assistant":
                        continue
                    return Plan(
                        "branch", thread, back + 1, entry.message_id, back + 1,
                        profile_changed=profile_changed,
                    )
                return Plan("new")
            finally:
                conn.close()

    def lookup(self, key: str, messages: List[Dict[str, Any]]) -> Optional[Thread]:
        """Backwards-compatible: a thread only when the history is an extension."""
        plan = self.plan(key, messages)
        if plan.mode != "append":
            return None
        return plan.thread

    def get(self, key: str) -> Optional[Thread]:
        with self._lock:
            conn = self._connect()
            try:
                row = self._row(conn, key)
                return _row_to_thread(row) if row else None
            finally:
                conn.close()

    # --- writes ----------------------------------------------------------

    def save(
        self,
        *,
        key: str,
        session_id: str,
        last_message_id: Optional[int],
        messages_seen: List[Dict[str, Any]],
        model_id: str,
        anchor: Optional[str] = None,
        profile_hash: Optional[str] = None,
        message_ids: Optional[Dict[int, Optional[int]]] = None,
        last_response: Optional[str] = None,
        keep_ids_before: int = 0,
    ) -> Thread:
        """Persist the thread.

        ``message_ids`` maps an index in ``messages_seen`` to the DeepSeek
        message id, when known. ``keep_ids_before`` preserves the ids already
        stored for indices below that value (they are not re-sent, so the caller
        cannot know them).
        """
        now = time.time()
        history_hash = canonical_hash(messages_seen)
        thread = Thread(
            key=key,
            session_id=session_id,
            last_message_id=last_message_id,
            seen_count=len(messages_seen),
            history_hash=history_hash,
            model_id=model_id,
            updated_at=now,
            anchor=anchor,
            profile_hash=profile_hash,
            last_response=last_response,
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO threads (key, session_id, last_message_id, seen_count,
                                         history_hash, model_id, created_at, updated_at,
                                         anchor, profile_hash, last_response)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        session_id = excluded.session_id,
                        last_message_id = excluded.last_message_id,
                        seen_count = excluded.seen_count,
                        history_hash = excluded.history_hash,
                        model_id = excluded.model_id,
                        updated_at = excluded.updated_at,
                        anchor = COALESCE(excluded.anchor, threads.anchor),
                        profile_hash = excluded.profile_hash,
                        last_response = COALESCE(excluded.last_response, threads.last_response)
                    """,
                    (
                        key,
                        session_id,
                        last_message_id,
                        thread.seen_count,
                        history_hash,
                        model_id,
                        now,
                        now,
                        anchor,
                        profile_hash,
                        last_response,
                    ),
                )

                if keep_ids_before:
                    conn.execute(
                        "DELETE FROM thread_messages WHERE thread_key = ? AND idx >= ?",
                        (key, keep_ids_before),
                    )
                else:
                    conn.execute("DELETE FROM thread_messages WHERE thread_key = ?", (key,))

                rows = []
                for index, message in enumerate(messages_seen):
                    if keep_ids_before and index < keep_ids_before:
                        continue  # already stored; the caller never saw these ids
                    mid = (message_ids or {}).get(index)
                    rows.append((key, index, message.get("role") or "user", mid, message_hash(message)))
                conn.executemany(
                    "INSERT OR REPLACE INTO thread_messages "
                    "(thread_key, idx, role, message_id, content_hash) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()
        return thread

    def delete(self, key: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM threads WHERE key = ?", (key,))
                conn.execute("DELETE FROM thread_messages WHERE thread_key = ?", (key,))
                conn.commit()
            finally:
                conn.close()

    def prune(self) -> int:
        cutoff = time.time() - self.ttl
        with self._lock:
            conn = self._connect()
            try:
                stale = [
                    row["key"]
                    for row in conn.execute(
                        "SELECT key FROM threads WHERE updated_at < ?", (cutoff,)
                    )
                ]
                for key in stale:
                    conn.execute("DELETE FROM thread_messages WHERE thread_key = ?", (key,))
                cursor = conn.execute("DELETE FROM threads WHERE updated_at < ?", (cutoff,))
                conn.commit()
                return cursor.rowcount or 0
            finally:
                conn.close()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            conn = self._connect()
            try:
                threads = conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
                reusable = conn.execute("SELECT COUNT(*) FROM threads WHERE seen_count > 2").fetchone()[0]
                return {"threads": threads, "reusable": reusable}
            finally:
                conn.close()


def _row_to_thread(row: sqlite3.Row) -> Thread:
    keys = row.keys()
    return Thread(
        key=row["key"],
        session_id=row["session_id"],
        last_message_id=row["last_message_id"],
        seen_count=row["seen_count"],
        history_hash=row["history_hash"],
        model_id=row["model_id"],
        updated_at=row["updated_at"],
        anchor=row["anchor"] if "anchor" in keys else None,
        profile_hash=row["profile_hash"] if "profile_hash" in keys else None,
        last_response=row["last_response"] if "last_response" in keys else None,
    )
