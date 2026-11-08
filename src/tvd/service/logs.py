"""The service's log: one JSON object per line, in a file that is rotated, and on standard output for ``docker logs``.

Each line has a time, a level and an event, with that event's fields, such as a job id. No line ever holds a plate number:
events carry ids, states, counts and reasons, never a record.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from tvd.service.settings import LoggingSettings

LOGGER = "tvd.service"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname, "event": record.getMessage(), **getattr(record, "fields", {}),
        }
        if record.exc_info:
            line["traceback"] = self.formatException(record.exc_info)  # Its own key: an event may have an "error" field.
        return json.dumps(line, default=str)


def log_event(event: str, level: int = logging.INFO, exc_info: bool = False, **fields: Any) -> None:
    """``log_event("job_done", job_id=...)``: one line of the service's log."""
    logging.getLogger(LOGGER).log(level, event, exc_info=exc_info, extra={"fields": fields})


def setup_logging(settings: LoggingSettings) -> Path:
    """Send the service's events to ``<folder>/service.log`` and to standard output, and return the file's path.

    Called when the service starts. Called again, it replaces its handlers, so no line is written twice.
    """
    settings.folder.mkdir(parents=True, exist_ok=True)
    path = settings.folder / "service.log"
    logger = logging.getLogger(LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = JsonFormatter()
    to_file = RotatingFileHandler(path, maxBytes=int(settings.max_mb * 2**20), backupCount=settings.keep_files, encoding="utf-8")
    to_output = logging.StreamHandler(sys.stdout)
    for handler in (to_file, to_output):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(settings.level)
    logger.propagate = False  # The lines are complete here; the root logger would print them a second time.
    return path
