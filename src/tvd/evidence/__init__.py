"""The evidence of each candidate: its clip, and images from it."""

from tvd.evidence.clips import Clip, clip_range, cut_clips
from tvd.evidence.images import EvidenceImage, evidence_frames, write_evidence_images
from tvd.evidence.settings import ClipSettings, EvidenceSettings, ImageSettings, RecordSettings, load_evidence_settings

__all__ = [
    "Clip", "ClipSettings", "EvidenceImage", "EvidenceSettings", "ImageSettings", "RecordSettings",
    "clip_range", "cut_clips", "evidence_frames", "load_evidence_settings", "write_evidence_images",
]
