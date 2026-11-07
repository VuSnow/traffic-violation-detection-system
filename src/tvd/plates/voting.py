"""One plate per vehicle, from the reads of many frames, as Notebook 06 does.

One frame's confidence is not enough: the OCR can be sure of a wrong text, and of an empty one. So the
frames vote, and each vehicle ends in one explicit status:

- ``read``: enough frames agree on one text;
- ``uncertain``: there is a text, but too few frames agree, so it is only a suggestion;
- ``unreadable``: a plate was found, but no frame gave a usable text;
- ``no_plate_found``: no plate of this vehicle was found.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tvd.plates.reading import PlateRead
from tvd.plates.settings import ReadingSettings

READ = "read"
UNCERTAIN = "uncertain"
UNREADABLE = "unreadable"
NO_PLATE = "no_plate_found"


@dataclass(frozen=True)
class PlateResult:
    status: str
    text: str | None
    votes: int
    usable_reads: int
    plates_found: int
    agreement: float
    mean_prob: float
    other_texts: dict[str, int]
    reads: tuple[PlateRead, ...] = field(default=(), repr=False)

    def best_read(self) -> PlateRead | None:
        """The read to keep as evidence: of the chosen text if there is one, and with the largest plate."""
        pool = [r for r in self.reads if r.text == self.text and r.rejected is None] or list(self.reads)
        return max(pool, key=lambda r: r.plate_width * r.plate_height, default=None)

    def as_json(self) -> dict[str, Any]:
        return {
            "status": self.status, "text": self.text, "votes": self.votes, "usable_reads": self.usable_reads, "plates_found": self.plates_found,
            "agreement": self.agreement, "mean_prob": self.mean_prob, "other_texts": dict(self.other_texts), "reads": [r.as_json() for r in self.reads],
        }


def vote(reads: list[PlateRead], settings: ReadingSettings) -> PlateResult:
    """Combine one vehicle's reads. On a tie, the text with more total confidence wins."""
    usable = [r for r in reads if r.rejected is None]
    if not reads:
        return PlateResult(NO_PLATE, None, 0, 0, 0, 0.0, 0.0, {}, ())
    if not usable:
        return PlateResult(UNREADABLE, None, 0, 0, len(reads), 0.0, 0.0, {}, tuple(reads))
    votes = Counter(r.text for r in usable)
    text = max(votes, key=lambda t: (votes[t], sum(r.mean_prob for r in usable if r.text == t)))
    agreement = votes[text] / len(usable)
    agreed = votes[text] >= settings.min_votes and agreement >= settings.min_agreement
    return PlateResult(
        status=READ if agreed else UNCERTAIN,
        text=text,
        votes=votes[text],
        usable_reads=len(usable),
        plates_found=len(reads),
        agreement=round(agreement, 3),
        mean_prob=round(float(np.mean([r.mean_prob for r in usable if r.text == text])), 4),
        other_texts={t: n for t, n in votes.items() if t != text},
        reads=tuple(reads),
    )
