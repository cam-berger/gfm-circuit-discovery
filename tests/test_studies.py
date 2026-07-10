"""The study interfaces import and are structurally checkable (TO_FIX.md §17.1)."""

from __future__ import annotations

from gfmcd.studies import BehaviorAssay, CueConflict, InterventionAssay, ShortcutBaseline


def test_protocols_are_runtime_checkable():
    class Baseline:
        def fit(self, train):
            return self

        def score(self, test):
            return {"acc": 1.0}

    assert isinstance(Baseline(), ShortcutBaseline)
    # A class missing a required method is not an instance.
    class NotAnAssay:
        pass

    assert not isinstance(NotAnAssay(), BehaviorAssay)
    assert CueConflict is not None and InterventionAssay is not None
