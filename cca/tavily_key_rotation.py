"""Persistent per-key budgets for Tavily HTTP attempts.

Reserve immediately before sending each request, including retries. A reservation
is intentionally not refunded after a timeout or transport error: the server may
already have received that request. All workers must use the same database path.
Only SHA-256 key fingerprints and integer counters are stored in the database.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import re
import sqlite3
from typing import Iterator, Sequence


class TavilyKeyPoolError(RuntimeError):
    """Invalid key-pool configuration or unusable quota storage."""

    fatal_tavily_error = True


class TavilyKeyPoolExhausted(TavilyKeyPoolError):
    """Every configured key has reached its local request-attempt budget."""


_META_SQL = """CREATE TABLE tavily_quota_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL CHECK (version = 1)
)"""
_USAGE_SQL = """CREATE TABLE tavily_key_usage (
    fingerprint TEXT PRIMARY KEY NOT NULL,
    attempts INTEGER NOT NULL CHECK (typeof(attempts) = 'integer' AND attempts >= 0)
)"""
_EXPECTED_SCHEMA = {"tavily_quota_meta": _META_SQL, "tavily_key_usage": _USAGE_SQL}
_ROTATION_SQL = """CREATE TABLE tavily_rotation_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_fingerprint TEXT
)"""
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


def _normalize_sql(value: str) -> str:
    return " ".join(value.split()).lower()


class TavilyKeyPool:
    """Use configured keys in order, with atomic, restart-safe reservations.

    ``reserve()`` returns the selected key but does not send a request. The
    caller must invoke it once for each HTTP attempt. Key identities, rather
    than list positions, retain usage when configuration order changes.
    """

    def __init__(
        self,
        keys: Sequence[str],
        usage_path: str | Path,
        per_key_limit: int = 950,
    ) -> None:
        if isinstance(keys, (str, bytes)):
            raise TavilyKeyPoolError("Tavily keys must be a non-empty sequence of strings.")
        normalized = []
        try:
            for key in keys:
                if not isinstance(key, str) or not key.strip():
                    raise TavilyKeyPoolError("Tavily keys cannot contain blank or non-string entries.")
                normalized.append(key.strip())
        except TypeError:
            raise TavilyKeyPoolError("Tavily keys must be a non-empty sequence of strings.") from None
        if not normalized:
            raise TavilyKeyPoolError("At least one Tavily key is required.")
        if len(set(normalized)) != len(normalized):
            raise TavilyKeyPoolError("Duplicate Tavily keys are not allowed.")
        if (
            isinstance(per_key_limit, bool)
            or not isinstance(per_key_limit, int)
            or not 0 < per_key_limit <= 2**63 - 1
        ):
            raise TavilyKeyPoolError("Tavily per-key limit must be a positive integer.")
        if not usage_path or str(usage_path) == ":memory:":
            raise TavilyKeyPoolError("A persistent Tavily quota database path is required.")
        self._keys = tuple(normalized)
        self._fingerprints = tuple(sha256(key.encode("utf-8")).hexdigest() for key in self._keys)
        self.per_key_limit = per_key_limit
        try:
            self.usage_path = Path(usage_path).expanduser().resolve()
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, TypeError, ValueError):
            raise TavilyKeyPoolError("Tavily quota database path is unusable.") from None
        with self._transaction() as conn:
            # Initialization is serialized with reservations in other processes.
            entries = conn.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if not entries:
                conn.execute(_META_SQL)
                conn.execute(_USAGE_SQL)
                conn.execute("INSERT INTO tavily_quota_meta (id, version) VALUES (1, 1)")
            self._validate_storage(conn)
            if not any(name == "tavily_rotation_state" for (name,) in entries):
                conn.execute(_ROTATION_SQL)
                conn.execute("INSERT INTO tavily_rotation_state (id, last_fingerprint) VALUES (1, NULL)")
            if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise TavilyKeyPoolError("Tavily quota database failed validation; no request was reserved.")
            for fingerprint in self._fingerprints:
                conn.execute(
                    "INSERT OR IGNORE INTO tavily_key_usage (fingerprint, attempts) VALUES (?, 0)",
                    (fingerprint,),
                )

    def __repr__(self) -> str:
        return f"TavilyKeyPool(key_count={len(self._keys)}, per_key_limit={self.per_key_limit})"

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = sqlite3.connect(str(self.usage_path), timeout=30, isolation_level=None)
            conn.execute("PRAGMA trusted_schema = OFF")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except TavilyKeyPoolError:
            if conn is not None:
                conn.rollback()
            raise
        except (OSError, sqlite3.Error, ValueError):
            if conn is not None:
                conn.rollback()
            # Do not include SQL, key material, or lower-level diagnostic payloads.
            raise TavilyKeyPoolError("Tavily quota storage is unavailable; no request was reserved.") from None
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _validate_storage(conn: sqlite3.Connection) -> None:
        schema = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        expected = dict(_EXPECTED_SCHEMA)
        if any(name == "tavily_rotation_state" for _, name, _ in schema):
            expected["tavily_rotation_state"] = _ROTATION_SQL
        if len(schema) != len(expected) or any(
            kind != "table"
            or name not in expected
            or not isinstance(sql, str)
            or _normalize_sql(sql) != _normalize_sql(expected[name])
            for kind, name, sql in schema
        ):
            raise TavilyKeyPoolError("Tavily quota database schema is invalid; no request was reserved.")
        if conn.execute("SELECT id, version FROM tavily_quota_meta").fetchall() != [(1, 1)]:
            raise TavilyKeyPoolError("Tavily quota database version is invalid; no request was reserved.")
        for fingerprint, attempts in conn.execute("SELECT fingerprint, attempts FROM tavily_key_usage"):
            if (
                not isinstance(fingerprint, str)
                or not _FINGERPRINT.fullmatch(fingerprint)
                or not isinstance(attempts, int)
                or attempts < 0
            ):
                raise TavilyKeyPoolError("Tavily quota counters are invalid; no request was reserved.")
        if "tavily_rotation_state" in expected:
            rotation = conn.execute("SELECT id, last_fingerprint FROM tavily_rotation_state").fetchall()
            if len(rotation) != 1 or rotation[0][0] != 1:
                raise TavilyKeyPoolError("Tavily rotation state is invalid; no request was reserved.")
            fingerprint = rotation[0][1]
            if fingerprint is not None and (
                not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint)
                or conn.execute("SELECT 1 FROM tavily_key_usage WHERE fingerprint = ?", (fingerprint,)).fetchone() is None
            ):
                raise TavilyKeyPoolError("Tavily rotation state is invalid; no request was reserved.")

    def reserve(self) -> str:
        """Persist one attempt for the first key below its configured limit."""
        with self._transaction() as conn:
            self._validate_storage(conn)
            for key, fingerprint in zip(self._keys, self._fingerprints):
                row = conn.execute(
                    "SELECT attempts FROM tavily_key_usage WHERE fingerprint = ?", (fingerprint,)
                ).fetchone()
                if row is None:
                    raise TavilyKeyPoolError("A Tavily quota counter is missing; no request was reserved.")
                if row[0] < self.per_key_limit:
                    conn.execute(
                        "UPDATE tavily_key_usage SET attempts = attempts + 1 WHERE fingerprint = ?",
                        (fingerprint,),
                    )
                    return key
            raise TavilyKeyPoolExhausted(
                "All configured Tavily keys reached their per-key request-attempt limit; no search was sent."
            )

    def _next_round_robin(self, conn: sqlite3.Connection, excluded_keys: Sequence[str]) -> tuple[str, str] | None:
        excluded = set(excluded_keys)
        last = conn.execute("SELECT last_fingerprint FROM tavily_rotation_state WHERE id = 1").fetchone()[0]
        start = (self._fingerprints.index(last) + 1) % len(self._keys) if last in self._fingerprints else 0
        has_budget = False
        for offset in range(len(self._keys)):
            position = (start + offset) % len(self._keys)
            key, fingerprint = self._keys[position], self._fingerprints[position]
            row = conn.execute("SELECT attempts FROM tavily_key_usage WHERE fingerprint = ?", (fingerprint,)).fetchone()
            if row is None:
                raise TavilyKeyPoolError("A Tavily quota counter is missing; no request was reserved.")
            if row[0] < self.per_key_limit:
                has_budget = True
                if key not in excluded:
                    return key, fingerprint
        if has_budget:
            return None
        raise TavilyKeyPoolExhausted(
            "All configured Tavily keys reached their per-key request-attempt limit; no search was sent."
        )

    def peek_next_round_robin(self, excluded_keys: Sequence[str] = ()) -> str | None:
        """Select the next uncapped key without reserving quota or moving the cursor."""
        with self._transaction() as conn:
            self._validate_storage(conn)
            selected = self._next_round_robin(conn, excluded_keys)
            return selected[0] if selected is not None else None

    def reserve_round_robin(self, preferred_key: str | None = None,
                            excluded_keys: Sequence[str] = ()) -> str | None:
        """Atomically count a rotating key and persist its cursor across restarts.

        A stale preferred key returns None without changing the database. This
        lets transports wait for that account's rate gate before reserving it.
        """
        if preferred_key is not None and preferred_key not in self._keys:
            raise TavilyKeyPoolError("The requested Tavily key is not configured.")
        with self._transaction() as conn:
            self._validate_storage(conn)
            selected = self._next_round_robin(conn, excluded_keys)
            if selected is None or (preferred_key is not None and selected[0] != preferred_key):
                return None
            key, fingerprint = selected
            conn.execute("UPDATE tavily_key_usage SET attempts = attempts + 1 WHERE fingerprint = ?", (fingerprint,))
            conn.execute("UPDATE tavily_rotation_state SET last_fingerprint = ? WHERE id = 1", (fingerprint,))
            return key

    def usage(self) -> list[dict[str, int | str]]:
        """Return configured order and masked counters, never plaintext keys."""
        with self._transaction() as conn:
            self._validate_storage(conn)
            summary = []
            for slot, fingerprint in enumerate(self._fingerprints, start=1):
                row = conn.execute(
                    "SELECT attempts FROM tavily_key_usage WHERE fingerprint = ?", (fingerprint,)
                ).fetchone()
                if row is None:
                    raise TavilyKeyPoolError("A Tavily quota counter is missing; usage is unavailable.")
                used = row[0]
                summary.append({
                    "slot": slot,
                    "used": used,
                    "remaining": max(self.per_key_limit - used, 0),
                    "fingerprint": fingerprint[:12],
                })
            return summary
