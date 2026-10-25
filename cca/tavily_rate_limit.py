"""Shared pacing and bounded retries for official pooled Tavily searches.

The gate is process-wide; the existing SQLite pool remains the cross-process
attempt counter. Waiting never reserves quota, and every actual HTTP attempt
retains its reservation, including rate-limit responses and transport failures.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable, Iterator

import requests

from .tavily_key_rotation import TavilyKeyPool, TavilyKeyPoolError, TavilyKeyPoolExhausted

SEARCH_URL = "https://api.tavily.com/search"
MIN_START_INTERVAL = 0.75  # 80 RPM, below the documented development-key 100 RPM.
MAX_ATTEMPTS = 3
DEFAULT_COOLDOWN = 60.0


class TavilySearchRequestError(RuntimeError):
    """A counted request failed; provider bodies and credentials are excluded."""

    fatal_tavily_error = True

    def __init__(self, message: str, status_code: int | None = None, *, halt_batch: bool = False):
        super().__init__(message)
        self.status_code = status_code
        # A missing search invalidates this task, but usually not every task.
        self.halt_batch = halt_batch


def _status_error(status: int, *, circuit: bool = False, halt_batch: bool = False) -> RuntimeError:
    suffix = "; pooled searches stopped" if circuit else ""
    message = f"Tavily search returned HTTP {status}{suffix}."
    if status in (432, 433):
        error = TavilyKeyPoolError(message)
        error.status_code = status
        error.halt_batch = halt_batch
        return error
    return TavilySearchRequestError(message, status_code=status, halt_batch=halt_batch)


def retry_after_seconds(value: object, *, now: datetime | None = None) -> float:
    """Accept Retry-After seconds or an HTTP date, otherwise use 60 seconds."""
    if not isinstance(value, str):
        return DEFAULT_COOLDOWN
    try:
        seconds = float(value.strip())
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    except (TypeError, ValueError):
        pass
    try:
        timestamp = parsedate_to_datetime(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (timestamp - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_COOLDOWN


class TavilyRateGate:
    """Paced starts, at most four requests in flight, and a shared circuit."""

    def __init__(self, min_start_interval: float = MIN_START_INTERVAL,
                 *, max_in_flight: int = 4, clock: Callable[[], float] | None = None,
                 wait: Callable[[float], None] | None = None):
        self._interval = min_start_interval
        self._max_in_flight = max_in_flight
        self._clock = clock or time.monotonic
        self._wait = wait  # Injectable only for offline tests; production uses Condition.wait.
        self._condition = threading.Condition()
        self._in_flight = 0
        self._next_start = 0.0
        self._cooldown_until = 0.0
        self._consecutive_429 = 0
        self._abort_status: int | None = None

    @contextmanager
    def request_slot(self, reserve: Callable[[], object] | None = None,
                     *, wait_for_slot: bool = True) -> Iterator[object]:
        acquired = False
        reservation = None
        with self._condition:
            while True:
                if self._abort_status is not None:
                    raise _status_error(self._abort_status, circuit=True)
                if self._in_flight >= self._max_in_flight:
                    if not wait_for_slot:
                        break
                    self._condition.wait()
                    continue
                now = self._clock()
                delay = max(self._next_start, self._cooldown_until) - now
                if delay > 0:
                    if not wait_for_slot:
                        break
                    if self._wait is not None:
                        self._wait(delay)
                    else:
                        self._condition.wait(timeout=min(delay, 60.0))
                    continue
                reservation = reserve() if reserve is not None else True
                if reservation is None:
                    break  # A stale round-robin selection consumes neither quota nor pacing.
                self._in_flight += 1
                self._next_start = now + self._interval
                acquired = True
                break
        try:
            yield reservation
        finally:
            if acquired:
                with self._condition:
                    self._in_flight -= 1
                    self._condition.notify_all()
                with _GATES_CHANGED:
                    _GATES_CHANGED.notify_all()

    def record_response(self, status: int, retry_after: object = None) -> None:
        with self._condition:
            if status == 429:
                self._consecutive_429 += 1
                self._cooldown_until = max(
                    self._cooldown_until, self._clock() + retry_after_seconds(retry_after)
                )
                if self._consecutive_429 >= MAX_ATTEMPTS:
                    self._abort_status = status
            else:
                self._consecutive_429 = 0
                if status in (401, 403, 432, 433):
                    self._abort_status = status
            self._condition.notify_all()
        with _GATES_CHANGED:
            _GATES_CHANGED.notify_all()

    def raise_if_aborted(self) -> None:
        with self._condition:
            if self._abort_status is not None:
                raise _status_error(self._abort_status, circuit=True)

    @property
    def abort_status(self) -> int | None:
        with self._condition:
            return self._abort_status

    def ready_delay(self) -> float | None:
        with self._condition:
            if self._abort_status is not None:
                return math.inf
            if self._in_flight >= self._max_in_flight:
                return None  # A response or request-slot release will notify waiters.
            return max(0.0, max(self._next_start, self._cooldown_until) - self._clock())

    def wait_for_change(self, delay: float) -> None:
        if self._wait is not None:
            self._wait(delay)
        else:
            with _GATES_CHANGED:
                _GATES_CHANGED.wait(timeout=min(delay, 1.0))


class SharedTavilyGuard:
    """Adapt to a limit affecting several accounts without changing normal RR.

    Two distinct keys returning 429 within a minute activate a shared cooldown
    and 80-RPM pacing for five minutes. This is a recovery policy, not a claim
    about whether the provider's undocumented limit is by IP or by account.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None,
                 wait: Callable[[float], None] | None = None):
        self._clock = clock or time.monotonic
        self._wait = wait
        self._lock = threading.Lock()
        self._recent_429: dict[str, float] = {}
        self._fallback_until = 0.0
        self._cooldown_until = 0.0
        self._next_start = 0.0
        self._in_flight = 0

    def _ready_delay(self, now: float) -> float | None:
        if now < self._fallback_until and self._in_flight >= 4:
            return None
        deadline = self._cooldown_until
        if now < self._fallback_until:
            deadline = max(deadline, self._next_start)
        return max(0.0, deadline - now)

    def ready_delay(self) -> float | None:
        with self._lock:
            return self._ready_delay(self._clock())

    def try_reserve(self, reserve: Callable[[], object]) -> object:
        with self._lock:
            now = self._clock()
            delay = self._ready_delay(now)
            if delay is None or delay > 0:
                return None
            key = reserve()
            if key is not None:
                self._in_flight += 1
                self._next_start = now + MIN_START_INTERVAL
            return key

    def release(self) -> None:
        with self._lock:
            self._in_flight -= 1
        with _GATES_CHANGED:
            _GATES_CHANGED.notify_all()

    def record_response(self, key: str, status: int, retry_after: object = None) -> None:
        if status != 429:
            return
        with self._lock:
            now = self._clock()
            self._recent_429 = {identity: timestamp for identity, timestamp in self._recent_429.items()
                                if timestamp >= now - 60.0}
            self._recent_429[sha256(key.encode("utf-8")).hexdigest()] = now
            if len(self._recent_429) >= 2 or now < self._fallback_until:
                self._fallback_until = now + 300.0
                self._cooldown_until = max(self._cooldown_until,
                                           now + max(DEFAULT_COOLDOWN, retry_after_seconds(retry_after)))
        with _GATES_CHANGED:
            _GATES_CHANGED.notify_all()

    def wait_for_change(self, delay: float) -> None:
        if self._wait is not None:
            self._wait(delay)
        else:
            with _GATES_CHANGED:
                _GATES_CHANGED.wait(timeout=min(delay, 1.0))


_GATES_LOCK = threading.Lock()
_GATES_CHANGED = threading.Condition()
_GATES: dict[tuple[str, str], TavilyRateGate] = {}
_SHARED_GUARDS: dict[str, SharedTavilyGuard] = {}


def _pool_identity(pool: TavilyKeyPool) -> str:
    usage_path = getattr(pool, "usage_path", None)
    return (os.path.normcase(str(Path(usage_path).expanduser().resolve()))
            if usage_path is not None else f"pool-object:{id(pool)}")


def get_shared_guard(pool: TavilyKeyPool) -> SharedTavilyGuard:
    identity = _pool_identity(pool)
    with _GATES_LOCK:
        if identity not in _SHARED_GUARDS:
            _SHARED_GUARDS[identity] = SharedTavilyGuard()
        return _SHARED_GUARDS[identity]


def get_rate_gate(pool: TavilyKeyPool, key: str | None = None) -> TavilyRateGate:
    """Each account key has one gate shared by transports using the same database."""
    database = _pool_identity(pool)
    identity = sha256(key.encode("utf-8")).hexdigest() if key is not None else "legacy"
    cache_key = (database, identity)
    with _GATES_LOCK:
        if cache_key not in _GATES:
            _GATES[cache_key] = TavilyRateGate()
        return _GATES[cache_key]


def _all_accounts_unusable(pool: TavilyKeyPool) -> bool:
    if not hasattr(pool, "peek_next_round_robin"):
        return get_rate_gate(pool).abort_status is not None
    blocked = tuple(key for key in pool._keys if get_rate_gate(pool, key).abort_status is not None)
    try:
        return pool.peek_next_round_robin(excluded_keys=blocked) is None
    except TavilyKeyPoolExhausted:
        return True


@contextmanager
def _round_robin_slot(pool: TavilyKeyPool) -> Iterator[tuple[TavilyRateGate, str, SharedTavilyGuard]]:
    shared = get_shared_guard(pool)
    if not hasattr(pool, "peek_next_round_robin"):
        # Small legacy adapters may supply only reserve(); real pools use RR.
        gate = get_rate_gate(pool)
        while True:
            delay = shared.ready_delay()
            if delay is None or delay > 0:
                shared.wait_for_change(1.0 if delay is None else delay)
                continue
            with gate.request_slot(lambda: shared.try_reserve(pool.reserve)) as key:
                if key is None:
                    continue
                try:
                    yield gate, key, shared
                finally:
                    shared.release()
            break
        return
    while True:
        shared_delay = shared.ready_delay()
        if shared_delay is None or shared_delay > 0:
            shared.wait_for_change(1.0 if shared_delay is None else shared_delay)
            continue
        keys = tuple(pool._keys)
        blocked = tuple(key for key in keys if get_rate_gate(pool, key).abort_status is not None)
        if pool.peek_next_round_robin(excluded_keys=blocked) is None:
            status = next((get_rate_gate(pool, key).abort_status for key in blocked), 429)
            raise _status_error(status, circuit=True, halt_batch=True)
        excluded = set(blocked)
        delayed_gates = []
        while True:
            key = pool.peek_next_round_robin(excluded_keys=tuple(excluded))
            if key is None:
                break
            gate = get_rate_gate(pool, key)
            slot = gate.request_slot(
                lambda: shared.try_reserve(
                    lambda: pool.reserve_round_robin(preferred_key=key, excluded_keys=tuple(excluded))
                ),
                wait_for_slot=False,
            )
            try:
                reserved = slot.__enter__()
            except (TavilySearchRequestError, TavilyKeyPoolError):
                if gate.abort_status is None:
                    raise
            else:
                try:
                    if reserved is not None:
                        yield gate, reserved, shared
                        return
                finally:
                    if reserved is not None:
                        shared.release()
                    slot.__exit__(None, None, None)
            excluded.add(key)
            delayed_gates.append(gate)
        delays = [(gate.ready_delay(), gate) for gate in delayed_gates]
        positive = [(delay, gate) for delay, gate in delays if delay is not None and math.isfinite(delay)]
        if positive:
            delay, gate = min(positive, key=lambda item: item[0])
            gate.wait_for_change(max(delay, 0.001))
        elif delayed_gates:
            delayed_gates[0].wait_for_change(1.0)


def pooled_search_post(pool: TavilyKeyPool, payload: dict, *, timeout: int):
    last_status = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with _round_robin_slot(pool) as (gate, key, shared):
                try:
                    response = requests.post(
                        SEARCH_URL, json=payload,
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=timeout, allow_redirects=False,
                    )
                except requests.RequestException as exc:
                    raise TavilySearchRequestError(
                        f"Tavily request failed ({type(exc).__name__}); its attempt remains counted."
                    ) from None
                status = response.status_code
                last_status = status
                headers = getattr(response, "headers", {})
                retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
                gate.record_response(status, retry_after)
                shared.record_response(key, status, retry_after)
                if status in (401, 403, 429, 432, 433) and attempt < MAX_ATTEMPTS - 1:
                    continue
                if status != 200:
                    raise _status_error(status, halt_batch=_all_accounts_unusable(pool))
                return response
        except TavilyKeyPoolExhausted:
            if last_status is not None:
                raise _status_error(last_status, circuit=True, halt_batch=True) from None
            raise
    raise TavilySearchRequestError("Tavily retry attempts exhausted.", status_code=429)
