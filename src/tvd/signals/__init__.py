"""Signal readers. Importing this package registers every reader in it."""

from tvd.signals import glow, lamps  # noqa: F401  Registers "glow" and "lamps".
from tvd.signals.base import CONFLICT, GREEN, RED, SIGNAL_READERS, UNKNOWN, YELLOW, BaseSignalReader, combine_states, smooth_states

__all__ = ["CONFLICT", "GREEN", "RED", "SIGNAL_READERS", "UNKNOWN", "YELLOW", "BaseSignalReader", "combine_states", "smooth_states"]
