"""The evidence of each candidate: its clip, and images from it."""

from tvd.evidence.clips import Clip, clip_range, cut_clips
from tvd.evidence.settings import ClipSettings, EvidenceSettings, ImageSettings, RecordSettings, load_evidence_settings

__all__ = ["Clip", "ClipSettings", "EvidenceSettings", "ImageSettings", "RecordSettings", "clip_range", "cut_clips", "load_evidence_settings"]
