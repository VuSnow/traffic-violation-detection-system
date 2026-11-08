"""The thread that runs the queued jobs, one at a time: the GPU is shared, so jobs never run side by side."""

from __future__ import annotations

import logging
import threading

from tvd.service.core import ReviewService
from tvd.service.logs import log_event


class Worker:
    """Runs ``service.run_next`` until stopped, and waits ``idle_seconds`` whenever the queue is empty.

    After each job, finished jobs older than the retention are deleted. ``stop`` lets the running job end first.
    An unexpected error is kept in ``errors``, and the loop goes on: a dead loop would leave every later job waiting.
    """

    def __init__(self, service: ReviewService, idle_seconds: float = 2.0) -> None:
        self.service = service
        self.idle_seconds = idle_seconds
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="tvd-jobs", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.service.run_next()
                if job is not None:
                    self.service.remove_old()
            except Exception as error:  # A job's own failure is kept by run_next: this is anything else, such as a locked database.
                self.errors.append(f"{type(error).__name__}: {error}")
                log_event("worker_error", logging.ERROR, exc_info=True, error=self.errors[-1])
                job = None
            if job is None:
                self._stop.wait(self.idle_seconds)
