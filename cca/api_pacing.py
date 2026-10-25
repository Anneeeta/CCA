"""Process-wide request pacing for the MiMo account/model route."""
from threading import Lock
import time


class RequestPacer:
    def __init__(self, interval=1.0, *, clock=None, sleep=None):
        self.interval = interval
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.lock = Lock()
        self.next_start = 0.0

    def wait(self):
        with self.lock:
            delay = self.next_start - self.clock()
            if delay > 0:
                self.sleep(delay)
            self.next_start = self.clock() + self.interval


_LOCK = Lock()
_MIMO_PACERS = {}


def pace_mimo_request(base_url, model):
    # Share across clients and keys: MiMo's published limit aggregates keys for
    # one account and model. A 1s interval leaves room under its 100-RPM limit.
    identity = (base_url.rstrip('/').lower(), model.lower())
    with _LOCK:
        pacer = _MIMO_PACERS.setdefault(identity, RequestPacer())
    pacer.wait()
