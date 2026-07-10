"""Reusable interfaces for circuit-discovery studies (TO_FIX.md §17.1).

A study instantiates the five-pillar protocol on one target computation. These lightweight
Protocols name the four moving parts every study shares; they are intentionally minimal — do not
refactor existing code onto them until a real study (SpliceBERT) shows which abstractions are
actually needed.
"""

from gfmcd.studies.protocols import BehaviorAssay, CueConflict, InterventionAssay, ShortcutBaseline

__all__ = ["BehaviorAssay", "CueConflict", "ShortcutBaseline", "InterventionAssay"]
