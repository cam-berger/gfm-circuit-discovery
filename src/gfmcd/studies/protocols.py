"""Study interfaces (TO_FIX.md §17.1). Small on purpose.

Each study provides concrete implementations; the harness (behavioral assay -> shortcut ceiling ->
cue conflict -> intervention) is the same across studies. Keeping these as `Protocol`s means a study
can satisfy them structurally without importing a base class.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BehaviorAssay(Protocol):
    """Does the model exhibit the target competence? Returns named scalar metrics."""

    def score(self, model: Any, batch: Any) -> dict[str, float]: ...


@runtime_checkable
class CueConflict(Protocol):
    """Build a conflict batch: matched local cues, conflicting relation. Returns clean/corrupt + targets."""

    def build(self, batch: Any, seed: int) -> Any: ...


@runtime_checkable
class ShortcutBaseline(Protocol):
    """A registered pooled shortcut. Its ceiling is what the circuit must beat (shortcut-gap)."""

    def fit(self, train: Any) -> Any: ...

    def score(self, test: Any) -> dict[str, float]: ...


@runtime_checkable
class InterventionAssay(Protocol):
    """Causal tests. Necessity: ablate components. Sufficiency/mediation: patch clean->corrupt."""

    def necessity(self, model: Any, batch: Any, components: Any) -> dict[str, float]: ...

    def sufficiency(self, model: Any, clean: Any, corrupt: Any, components: Any) -> dict[str, float]: ...
