"""Frozen preflight gates for any atlas result-producing run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

_HEX = frozenset("0123456789abcdef")


class GateState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ANOMALOUS = "ANOMALOUS"
    NOT_RUN = "NOT_RUN"


MANDATORY_GATES = frozenset(
    {
        "positive_control",
        "negative_control",
        "sensitivity",
        "resolution",
        "invariants",
        "matching_balance",
        "token_geometry",
        "model_health",
    }
)


@dataclass(frozen=True)
class GateOutcome:
    name: str
    state: GateState
    threshold: Mapping[str, Any]
    observed: Mapping[str, Any]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name cannot be blank")
        object.__setattr__(self, "threshold", dict(self.threshold))
        object.__setattr__(self, "observed", dict(self.observed))
        object.__setattr__(self, "notes", tuple(self.notes))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateOutcome:
        return cls(
            name=str(value["name"]),
            state=GateState(value["state"]),
            threshold=dict(value.get("threshold", {})),
            observed=dict(value.get("observed", {})),
            notes=tuple(str(x) for x in value.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "threshold": dict(self.threshold),
            "observed": dict(self.observed),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PreflightReport:
    report_id: str
    config_sha256: str
    checkpoint_revision: str
    dataset_release: str
    input_sha256: Mapping[str, str]
    split: str
    outcomes: tuple[GateOutcome, ...]
    frozen: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.split == "confirmation":
            raise ValueError("preflight must not consume the sealed confirmation split")
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "input_sha256", dict(self.input_sha256))
        if not self.input_sha256 or any(
            not name or len(digest) != 64 or set(digest.lower()) - _HEX
            for name, digest in self.input_sha256.items()
        ):
            raise ValueError("preflight must bind named input files by SHA-256")
        names = [outcome.name for outcome in self.outcomes]
        if len(names) != len(set(names)):
            raise ValueError("preflight report contains duplicate gate names")

    @property
    def decision(self) -> str:
        by_name = {outcome.name: outcome for outcome in self.outcomes}
        if not self.frozen or not MANDATORY_GATES <= set(by_name):
            return "INSTRUMENT_UNRESOLVED"
        if any(by_name[name].state is not GateState.PASS for name in MANDATORY_GATES):
            return "INSTRUMENT_UNRESOLVED"
        return "QUALIFIED"

    def assert_qualified(
        self,
        *,
        expected_config_sha256: str,
        expected_checkpoint_revision: str,
        expected_input_sha256: Mapping[str, str],
    ) -> None:
        if self.config_sha256 != expected_config_sha256:
            raise RuntimeError("preflight config hash does not match the target run")
        if self.checkpoint_revision != expected_checkpoint_revision:
            raise RuntimeError("preflight checkpoint does not match the target run")
        if self.input_sha256 != dict(expected_input_sha256):
            raise RuntimeError("preflight data inputs do not match the target run")
        if self.decision != "QUALIFIED":
            states = {outcome.name: outcome.state.value for outcome in self.outcomes}
            raise RuntimeError(f"atlas instrument is not qualified: {states}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "config_sha256": self.config_sha256,
            "checkpoint_revision": self.checkpoint_revision,
            "dataset_release": self.dataset_release,
            "input_sha256": dict(self.input_sha256),
            "split": self.split,
            "frozen": self.frozen,
            "decision": self.decision,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PreflightReport:
        report = cls(
            report_id=str(value["report_id"]),
            config_sha256=str(value["config_sha256"]),
            checkpoint_revision=str(value["checkpoint_revision"]),
            dataset_release=str(value["dataset_release"]),
            input_sha256={
                str(name): str(digest)
                for name, digest in dict(value["input_sha256"]).items()
            },
            split=str(value["split"]),
            outcomes=tuple(GateOutcome.from_dict(x) for x in value["outcomes"]),
            frozen=bool(value["frozen"]),
            notes=tuple(str(x) for x in value.get("notes", ())),
        )
        declared = value.get("decision")
        if declared is not None and declared != report.decision:
            raise ValueError("declared preflight decision disagrees with gate outcomes")
        return report


def load_preflight_report(path: str | Path) -> PreflightReport:
    return PreflightReport.from_dict(json.loads(Path(path).read_text()))


def config_sha256(path: str | Path) -> str:
    """Hash canonical parsed JSON, independent of whitespace."""

    value = json.loads(Path(path).read_text())
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def frozen_config_sha256(path: str | Path) -> str:
    """Refuse a draft or threshold-incomplete target configuration."""

    value = json.loads(Path(path).read_text())
    if value.get("status") != "FROZEN":
        raise RuntimeError("atlas preflight configuration is not FROZEN")
    missing = _null_paths(value.get("mandatory_gates", {}), "mandatory_gates")
    missing.extend(_null_paths(value.get("behavior", {}), "behavior"))
    if missing:
        raise RuntimeError(
            "frozen atlas config contains unresolved thresholds: " + ", ".join(missing)
        )
    return config_sha256(path)


def _null_paths(value: Any, prefix: str) -> list[str]:
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        return [
            path
            for key, nested in value.items()
            for path in _null_paths(nested, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        return [
            path
            for index, nested in enumerate(value)
            for path in _null_paths(nested, f"{prefix}[{index}]")
        ]
    return []
