"""Paired multi-view sparse transcoders for checkpoint families.

The module is deliberately model-agnostic.  A *view* is one checkpoint's
representation of the same ordered examples.  Each view gets its own detector
and writer because independently trained checkpoints need not share native
coordinates, while a consistency loss can give the fine-tuned views a common
feature index.

The intended use is to approximate a frozen MLP map

``MLP input -> sparse features -> MLP write``

without using labels or a downstream classifier during fitting.  Causal
faithfulness remains a separate evaluation on the frozen native model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


LABEL_BEARING_FIELDS = frozenset(
    {
        "label",
        "labels",
        "margin",
        "margins",
        "logit",
        "logits",
        "classifier_weight",
        "classifier_bias",
        "outcome",
        "outcomes",
        "psi",
        "delta_psi",
    }
)
LABEL_FREE_SEQUENCE_PACKAGE_SCHEMA = "gfmcd.label_free_sequence_states.v1"
LABEL_FREE_PACKAGE_FIELDS = frozenset(
    {
        "schema",
        "cohort_namespace",
        "partition",
        "label_free",
        "sparse_training_authorized",
        "created_at_utc",
        "protocol",
        "implementation_addendum",
        "authorization",
        "sources",
        "counts",
        "states",
    }
)
LABEL_FREE_STATE_FIELDS = frozenset(
    {
        "state_id",
        "example_id",
        "gene_id",
        "endpoint",
        "variant_id",
        "variant_kind",
        "sequence",
        "sequence_sha256",
        "target_position",
        "reference_state_id",
        "changed_positions",
        "hamming",
        "codon_start",
        "original_codon",
        "alternate_codon",
    }
)
LABEL_FREE_SOURCE_FIELDS = frozenset({"path", "sha256", "size_bytes", "role"})
_SHA256_HEX = frozenset("0123456789abcdef")


def assert_label_free_payload(payload: Any, *, path: str = "$") -> None:
    """Reject outcome-bearing fields before sparse-dictionary training."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in LABEL_BEARING_FIELDS:
                raise ValueError(f"label-bearing field {path}.{key} is forbidden")
            assert_label_free_payload(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_label_free_payload(value, path=f"{path}[{index}]")


def label_free_package_binding(
    authorization: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> str:
    """Bind the strict authorization declaration to all source records."""

    declaration = {
        key: value for key, value in authorization.items() if key != "binding_sha256"
    }
    payload = json.dumps(
        {"authorization": declaration, "sources": sources},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_label_free_state_rows(
    payload: Mapping[str, Any],
    *,
    namespace: str,
    partition: str = "discovery",
    verify_source_files: bool = False,
) -> list[dict[str, Any]]:
    """Validate the flat sequence-state contract consumed by activation capture."""

    assert_label_free_payload(payload)
    unknown_package_fields = set(payload) - LABEL_FREE_PACKAGE_FIELDS
    if unknown_package_fields:
        raise ValueError(
            "sequence package contains non-allowlisted fields: "
            f"{sorted(unknown_package_fields)}"
        )
    if payload.get("schema") != LABEL_FREE_SEQUENCE_PACKAGE_SCHEMA:
        raise ValueError("sequence package schema mismatch")
    if payload.get("cohort_namespace") != namespace:
        raise ValueError("sequence package cohort namespace mismatch")
    if payload.get("partition") != partition:
        raise ValueError("sequence package partition mismatch")
    if payload.get("label_free") is not True:
        raise ValueError("sequence package must explicitly declare label_free=true")
    if payload.get("sparse_training_authorized") is not True:
        raise ValueError("sequence package is not authorized for sparse training")
    authorization = payload.get("authorization")
    sources = payload.get("sources")
    if not isinstance(authorization, Mapping) or not isinstance(sources, Mapping):
        raise ValueError("sequence package requires authorization and source bindings")
    if (
        authorization.get("status") != "PRECONDITIONS_VERIFIED"
        or authorization.get("preflight_status") != "PASS"
        or authorization.get("discovery_status") != "COMPLETE"
        or authorization.get("producer")
        != "scripts/splicebert/export_gtex_sparse_transcoder_sequence_package.py"
    ):
        raise ValueError("sequence package authorization declaration is invalid")
    binding = authorization.get("binding_sha256")
    if not _valid_sha256(binding) or binding != label_free_package_binding(
        authorization,
        sources,
    ):
        raise ValueError("sequence package authorization binding mismatch")
    required_sources = {
        "preflight",
        "discovery",
        "protocol",
        "addendum",
        "basic_gtf",
        "comprehensive_gtf",
        "fasta",
        "producer",
    }
    if set(sources) != required_sources:
        raise ValueError("sequence package source family mismatch")
    for name, raw_source in sources.items():
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"sequence package source {name} is not a record")
        unknown_source_fields = set(raw_source) - LABEL_FREE_SOURCE_FIELDS
        if unknown_source_fields:
            raise ValueError(
                f"sequence package source {name} has non-allowlisted fields: "
                f"{sorted(unknown_source_fields)}"
            )
        if (
            not isinstance(raw_source.get("path"), str)
            or not _valid_sha256(raw_source.get("sha256"))
            or not isinstance(raw_source.get("size_bytes"), int)
            or raw_source.get("size_bytes", -1) < 0
            or raw_source.get("role") != name
        ):
            raise ValueError(f"sequence package source {name} record is invalid")
        if verify_source_files:
            path = Path(raw_source["path"]).expanduser().resolve()
            if (
                not path.is_file()
                or path.stat().st_size != raw_source["size_bytes"]
                or _sha256_path(path) != raw_source["sha256"]
            ):
                raise RuntimeError(f"sequence package source {name} changed")
    rows = payload.get("states")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sequence package requires a non-empty states list")
    required = {
        "state_id",
        "example_id",
        "gene_id",
        "endpoint",
        "variant_id",
        "variant_kind",
        "sequence",
        "target_position",
    }
    validated: list[dict[str, Any]] = []
    state_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise ValueError(f"state row {index} is missing required fields")
        unknown_state_fields = set(raw) - LABEL_FREE_STATE_FIELDS
        if unknown_state_fields:
            raise ValueError(
                f"state row {index} contains non-allowlisted fields: "
                f"{sorted(unknown_state_fields)}"
            )
        row = dict(raw)
        state_id = str(row["state_id"])
        if not state_id or state_id in state_ids:
            raise ValueError("state_id values must be non-empty and unique")
        state_ids.add(state_id)
        sequence = str(row["sequence"]).upper().replace("U", "T")
        if not sequence or set(sequence) - set("ACGTN"):
            raise ValueError(f"state {state_id} has an invalid DNA sequence")
        target = int(row["target_position"])
        if not 0 <= target < len(sequence):
            raise ValueError(f"state {state_id} target is outside its sequence")
        endpoint = str(row["endpoint"])
        if endpoint not in {"donor", "acceptor", "background"}:
            raise ValueError(f"state {state_id} has an invalid endpoint")
        row["state_id"] = state_id
        row["example_id"] = str(row["example_id"])
        row["gene_id"] = str(row["gene_id"])
        row["endpoint"] = endpoint
        row["variant_id"] = str(row["variant_id"])
        row["variant_kind"] = str(row["variant_kind"])
        row["sequence"] = sequence
        observed_sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if (
            "sequence_sha256" not in row
            or row["sequence_sha256"] != observed_sequence_sha256
        ):
            raise ValueError(f"state {state_id} sequence hash mismatch")
        row["target_position"] = target
        changed_positions = row.get("changed_positions")
        if (
            not isinstance(changed_positions, list)
            or any(not isinstance(position, int) for position in changed_positions)
            or changed_positions != sorted(set(changed_positions))
        ):
            raise ValueError(f"state {state_id} has invalid changed_positions")
        row["changed_positions"] = changed_positions
        if (
            not isinstance(row.get("hamming"), int)
            or row["hamming"] != len(changed_positions)
        ):
            raise ValueError(f"state {state_id} hamming dose mismatch")
        reference = row.get("reference_state_id")
        row["reference_state_id"] = None if reference is None else str(reference)
        validated.append(row)
    by_state_id = {row["state_id"]: row for row in validated}
    for row in validated:
        reference = row["reference_state_id"]
        if reference is None:
            if (
                row["variant_kind"] != "clean"
                or row["variant_id"] != "clean"
                or row["changed_positions"]
                or row["hamming"] != 0
            ):
                raise ValueError("only unchanged clean states may omit a reference")
            continue
        if reference not in state_ids:
            raise ValueError(
                f"state {row['state_id']} references unknown state {reference}"
            )
        if row["variant_kind"] == "clean":
            raise ValueError("a clean state may not reference another state")
        reference_row = by_state_id[reference]
        if reference_row["reference_state_id"] is not None:
            raise ValueError("changed states must reference a clean root state")
        for field in ("example_id", "gene_id", "endpoint", "target_position"):
            if row[field] != reference_row[field]:
                raise ValueError(
                    f"state {row['state_id']} crosses reference identity field {field}"
                )
        if len(row["sequence"]) != len(reference_row["sequence"]):
            raise ValueError("changed/reference sequence lengths differ")
        differences = [
            index
            for index, (clean, changed) in enumerate(
                zip(reference_row["sequence"], row["sequence"])
            )
            if clean != changed
        ]
        if differences != row["changed_positions"] or len(differences) != row["hamming"]:
            raise ValueError(
                f"state {row['state_id']} changed-position audit mismatch"
            )
        if not differences:
            raise ValueError("changed state is identical to its clean reference")
    return validated


@dataclass(frozen=True)
class MultiViewTranscoderConfig:
    """Architecture shared by every checkpoint view."""

    input_dim: int
    output_dim: int
    num_features: int
    views: tuple[str, ...]
    aligned_views: tuple[str, ...]
    threshold_init: float = 0.1
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if min(self.input_dim, self.output_dim, self.num_features) < 1:
            raise ValueError("input, output, and feature dimensions must be positive")
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("views must be a non-empty unique sequence")
        if not self.aligned_views or len(set(self.aligned_views)) != len(
            self.aligned_views
        ):
            raise ValueError("aligned_views must be a non-empty unique sequence")
        if not set(self.aligned_views).issubset(self.views):
            raise ValueError("aligned_views must be drawn from views")
        if self.threshold_init <= 0.0 or self.eps <= 0.0:
            raise ValueError("threshold_init and eps must be positive")


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(tensor)))


class MultiViewSparseTranscoder(nn.Module):
    """Checkpoint-specific detectors/writers with a shared sparse feature index.

    Decoder columns are normalized during every forward pass.  This removes the
    scale degeneracy in which encoder activations shrink while decoder weights
    grow, making activation sparsity and cross-view consistency comparable.
    """

    def __init__(self, config: MultiViewTranscoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict(
            {
                view: nn.Linear(config.input_dim, config.num_features)
                for view in config.views
            }
        )
        self.decoder_raw = nn.ParameterDict(
            {
                view: nn.Parameter(
                    torch.empty(config.output_dim, config.num_features)
                )
                for view in config.views
            }
        )
        self.output_bias = nn.ParameterDict(
            {
                view: nn.Parameter(torch.zeros(config.output_dim))
                for view in config.views
            }
        )
        initial_log_threshold = _inverse_softplus(config.threshold_init)
        self.log_threshold = nn.ParameterDict(
            {
                view: nn.Parameter(
                    torch.full((config.num_features,), initial_log_threshold)
                )
                for view in config.views
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use ordinary linear initialization and unit-scale writer seeds."""

        initial_log_threshold = _inverse_softplus(self.config.threshold_init)
        for view in self.config.views:
            self.encoders[view].reset_parameters()
            nn.init.normal_(
                self.decoder_raw[view],
                mean=0.0,
                std=self.config.output_dim**-0.5,
            )
            nn.init.zeros_(self.output_bias[view])
            nn.init.constant_(self.log_threshold[view], initial_log_threshold)

    def _validate_view(self, view: str) -> None:
        if view not in self.config.views:
            raise KeyError(f"unknown view {view!r}")

    def decoder(self, view: str) -> torch.Tensor:
        """Return the ``[output_dim, features]`` unit-column-norm writer."""

        self._validate_view(view)
        return F.normalize(
            self.decoder_raw[view],
            p=2.0,
            dim=0,
            eps=self.config.eps,
        )

    def thresholds(self, view: str) -> torch.Tensor:
        """Return positive learned activation thresholds for one view."""

        self._validate_view(view)
        return F.softplus(self.log_threshold[view])

    def encode(self, view: str, inputs: torch.Tensor) -> torch.Tensor:
        """Detect non-negative sparse features from one checkpoint's MLP input."""

        self._validate_view(view)
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_dim:
            raise ValueError("inputs must be [batch, input_dim]")
        preactivation = self.encoders[view](inputs)
        return F.relu(preactivation - self.thresholds(view))

    def decode(
        self,
        view: str,
        features: torch.Tensor,
        *,
        include_bias: bool = True,
    ) -> torch.Tensor:
        """Decode sparse features into one checkpoint's MLP-write coordinates."""

        self._validate_view(view)
        if features.ndim != 2 or features.shape[1] != self.config.num_features:
            raise ValueError("features must be [batch, num_features]")
        output = features @ self.decoder(view).T
        if include_bias:
            output = output + self.output_bias[view]
        return output

    def forward(
        self,
        view: str,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(features, reconstructed_write)`` for one view."""

        features = self.encode(view, inputs)
        return features, self.decode(view, features)

    def selected_feature_write(
        self,
        view: str,
        features: torch.Tensor,
        selected: Sequence[int],
    ) -> torch.Tensor:
        """Return only the additive write from selected feature coordinates."""

        if features.ndim != 2 or features.shape[1] != self.config.num_features:
            raise ValueError("features must be [batch, num_features]")
        indices = torch.as_tensor(
            tuple(int(index) for index in selected),
            dtype=torch.long,
            device=features.device,
        )
        if indices.ndim != 1 or indices.numel() != torch.unique(indices).numel():
            raise ValueError("selected feature indices must be unique")
        if indices.numel() == 0:
            return torch.zeros(
                (features.shape[0], self.config.output_dim),
                dtype=features.dtype,
                device=features.device,
            )
        if torch.any(indices < 0) or torch.any(indices >= self.config.num_features):
            raise ValueError("selected feature index outside dictionary")
        return features[:, indices] @ self.decoder(view)[:, indices].T


@dataclass(frozen=True)
class StructuredMultiViewTranscoderConfig:
    """Frozen shared/tuned/private BatchTopK allocation.

    Half of the dictionary is shared, one quarter is available only to tuned
    views, and one quarter is private reconstruction capacity.  Fine-tuned
    views spend half of their per-example activity budget on shared features
    and one quarter on each remaining block.  The base view cannot activate
    tuned-only features and reallocates that activity to its private block.
    """

    input_dim: int
    output_dim: int
    num_features: int
    active_features: int
    views: tuple[str, ...]
    tuned_views: tuple[str, ...]
    threshold_momentum: float = 0.95
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.output_dim,
            self.num_features,
            self.active_features,
        ) < 1:
            raise ValueError("dimensions and active_features must be positive")
        if self.num_features % 4 or self.active_features % 4:
            raise ValueError("feature width and active count must be divisible by four")
        if self.active_features > self.num_features // 2:
            raise ValueError("active feature budget exceeds block capacity")
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("views must be a non-empty unique sequence")
        if not self.tuned_views or len(set(self.tuned_views)) != len(
            self.tuned_views
        ):
            raise ValueError("tuned_views must be a non-empty unique sequence")
        if not set(self.tuned_views).issubset(self.views):
            raise ValueError("tuned_views must be drawn from views")
        if set(self.views) == set(self.tuned_views):
            raise ValueError("at least one non-tuned contrast view is required")
        if not 0.0 <= self.threshold_momentum < 1.0:
            raise ValueError("threshold_momentum must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    @property
    def shared_slice(self) -> slice:
        return slice(0, self.num_features // 2)

    @property
    def tuned_slice(self) -> slice:
        return slice(self.num_features // 2, 3 * self.num_features // 4)

    @property
    def private_slice(self) -> slice:
        return slice(3 * self.num_features // 4, self.num_features)

    @property
    def consistent_slice(self) -> slice:
        return slice(0, 3 * self.num_features // 4)


def batch_topk(
    values: torch.Tensor,
    active_per_example: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep exactly ``batch * active_per_example`` positive entries if available."""

    if values.ndim != 2 or active_per_example < 1:
        raise ValueError("values must be 2D and active_per_example positive")
    positive = F.relu(values)
    count = min(positive.numel(), positive.shape[0] * active_per_example)
    flat = positive.flatten()
    selected_values, selected_indices = torch.topk(
        flat,
        k=count,
        sorted=False,
    )
    keep = selected_values > 0.0
    output = torch.zeros_like(flat)
    output[selected_indices[keep]] = selected_values[keep]
    threshold = (
        selected_values[keep].min()
        if torch.any(keep)
        else values.new_tensor(float("inf"))
    )
    return output.view_as(values), threshold


def row_topk(values: torch.Tensor, active: int) -> torch.Tensor:
    """Deployment-time fixed-L0 counterpart to BatchTopK training."""

    if values.ndim != 2 or not 1 <= active <= values.shape[1]:
        raise ValueError("active must fit the 2D feature block")
    positive = F.relu(values)
    selected_values, selected_indices = torch.topk(
        positive,
        k=active,
        dim=1,
        sorted=False,
    )
    selected_values = torch.where(
        selected_values > 0.0,
        selected_values,
        torch.zeros_like(selected_values),
    )
    output = torch.zeros_like(positive)
    output.scatter_(1, selected_indices, selected_values)
    return output


class StructuredMultiViewSparseTranscoder(nn.Module):
    """Six-view sparse MLP surrogate with explicit feature-capacity roles."""

    BLOCK_NAMES = ("shared", "tuned", "private")

    def __init__(self, config: StructuredMultiViewTranscoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict(
            {
                view: nn.Linear(config.input_dim, config.num_features)
                for view in config.views
            }
        )
        self.decoder_raw = nn.ParameterDict(
            {
                view: nn.Parameter(
                    torch.empty(config.output_dim, config.num_features)
                )
                for view in config.views
            }
        )
        self.output_bias = nn.ParameterDict(
            {
                view: nn.Parameter(torch.zeros(config.output_dim))
                for view in config.views
            }
        )
        self.inference_threshold = nn.ParameterDict(
            {
                view: nn.Parameter(
                    torch.zeros(len(self.BLOCK_NAMES)),
                    requires_grad=False,
                )
                for view in config.views
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for view in self.config.views:
            self.encoders[view].reset_parameters()
            nn.init.normal_(
                self.decoder_raw[view],
                mean=0.0,
                std=self.config.output_dim**-0.5,
            )
            nn.init.zeros_(self.output_bias[view])
            nn.init.zeros_(self.inference_threshold[view])

    def _validate_view(self, view: str) -> None:
        if view not in self.config.views:
            raise KeyError(f"unknown view {view!r}")

    def decoder(self, view: str) -> torch.Tensor:
        self._validate_view(view)
        return F.normalize(
            self.decoder_raw[view],
            p=2.0,
            dim=0,
            eps=self.config.eps,
        )

    def block_slices(self, view: str) -> tuple[tuple[str, slice, int], ...]:
        """Return active blocks and their average BatchTopK budgets."""

        self._validate_view(view)
        shared_active = self.config.active_features // 2
        if view in self.config.tuned_views:
            return (
                ("shared", self.config.shared_slice, shared_active),
                (
                    "tuned",
                    self.config.tuned_slice,
                    self.config.active_features // 4,
                ),
                (
                    "private",
                    self.config.private_slice,
                    self.config.active_features // 4,
                ),
            )
        return (
            ("shared", self.config.shared_slice, shared_active),
            (
                "private",
                self.config.private_slice,
                self.config.active_features - shared_active,
            ),
        )

    def raw_scores(self, view: str, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_view(view)
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_dim:
            raise ValueError("inputs must be [batch, input_dim]")
        return self.encoders[view](inputs)

    def _threshold_index(self, block: str) -> int:
        return self.BLOCK_NAMES.index(block)

    def encode(
        self,
        view: str,
        inputs: torch.Tensor,
        *,
        force_batch_topk: bool | None = None,
    ) -> torch.Tensor:
        """Encode with BatchTopK in training and fixed-L0 row TopK in evaluation."""

        scores = self.raw_scores(view, inputs)
        use_batch_topk = self.training if force_batch_topk is None else force_batch_topk
        features = torch.zeros_like(scores)
        for block, block_slice, active in self.block_slices(view):
            block_scores = scores[:, block_slice]
            threshold_index = self._threshold_index(block)
            if use_batch_topk:
                selected, threshold = batch_topk(block_scores, active)
                features[:, block_slice] = selected
                if torch.isfinite(threshold):
                    momentum = self.config.threshold_momentum
                    with torch.no_grad():
                        current = self.inference_threshold[view][threshold_index]
                        current.mul_(momentum).add_(threshold * (1.0 - momentum))
            else:
                features[:, block_slice] = row_topk(block_scores, active)
        return features

    @torch.no_grad()
    def calibrate_thresholds(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> dict[str, dict[str, float | int | bool | None]]:
        """Freeze block thresholds to the exact full-training BatchTopK boundary."""

        if set(inputs) != set(self.config.views):
            raise ValueError("calibration inputs must contain every view")
        audit: dict[str, dict[str, float | int | bool | None]] = {}
        for view in self.config.views:
            scores = self.raw_scores(view, inputs[view])
            for block, block_slice, active in self.block_slices(view):
                selected, threshold = batch_topk(scores[:, block_slice], active)
                positive_count = int(torch.count_nonzero(selected).cpu())
                inactive = not bool(torch.isfinite(threshold))
                self.inference_threshold[view][
                    self._threshold_index(block)
                ].copy_(threshold)
                audit[f"{view}:{block}"] = {
                    "active_positive_entries": positive_count,
                    "inactive": inactive,
                    "diagnostic_threshold": (
                        None if inactive else float(threshold.detach().cpu())
                    ),
                }
        return audit

    def decode(
        self,
        view: str,
        features: torch.Tensor,
        *,
        include_bias: bool = True,
    ) -> torch.Tensor:
        self._validate_view(view)
        if features.ndim != 2 or features.shape[1] != self.config.num_features:
            raise ValueError("features must be [batch, num_features]")
        output = features @ self.decoder(view).T
        if include_bias:
            output = output + self.output_bias[view]
        return output

    def forward(
        self,
        view: str,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(view, inputs)
        return features, self.decode(view, features)

    def selected_feature_write(
        self,
        view: str,
        features: torch.Tensor,
        selected: Sequence[int],
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.num_features:
            raise ValueError("features must be [batch, num_features]")
        indices = torch.as_tensor(
            tuple(int(index) for index in selected),
            dtype=torch.long,
            device=features.device,
        )
        if indices.ndim != 1 or indices.numel() != torch.unique(indices).numel():
            raise ValueError("selected feature indices must be unique")
        if indices.numel() == 0:
            return torch.zeros(
                (features.shape[0], self.config.output_dim),
                dtype=features.dtype,
                device=features.device,
            )
        if torch.any(indices < 0) or torch.any(indices >= self.config.num_features):
            raise ValueError("selected feature index outside dictionary")
        return features[:, indices] @ self.decoder(view)[:, indices].T


def paired_difference_mse(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Variance-normalized reconstruction loss on ordered paired differences."""

    if reconstructed.shape != target.shape or reconstructed.ndim != 2:
        raise ValueError("reconstructed and target must be same-shaped matrices")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must be [pairs, 2]")
    indices = pair_indices.to(device=target.device, dtype=torch.long)
    if indices.numel() == 0:
        return target.new_zeros(())
    if torch.any(indices < 0) or torch.any(indices >= target.shape[0]):
        raise ValueError("pair index outside batch")
    predicted_delta = reconstructed[indices[:, 0]] - reconstructed[indices[:, 1]]
    target_delta = target[indices[:, 0]] - target[indices[:, 1]]
    scale = target_delta.detach().square().mean().clamp_min(eps)
    return (predicted_delta - target_delta).square().mean() / scale


def standardized_consistency_loss(
    features: Mapping[str, torch.Tensor],
    views: Sequence[str],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize disagreement of paired feature activations across named views."""

    names = tuple(views)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("consistency requires at least two unique views")
    missing = set(names) - set(features)
    if missing:
        raise KeyError(f"missing feature views: {sorted(missing)}")
    tensors = [features[name] for name in names]
    if any(tensor.ndim != 2 or tensor.shape != tensors[0].shape for tensor in tensors):
        raise ValueError("paired feature matrices must have the same 2D shape")
    stacked = torch.stack(tensors, dim=0)
    consensus = stacked.mean(dim=0)
    scale = consensus.detach().square().mean().clamp_min(eps)
    return (stacked - consensus).square().mean() / scale


def view_difference_mse(
    reconstructed: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    contrast_pairs: Sequence[tuple[str, str]],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Prioritize checkpoint shifts on identical ordered examples.

    Each pair is ``(reference, changed)``.  Native coordinates must be directly
    comparable for this loss; otherwise callers should transform both outputs
    into a common, outcome-blind coordinate system before invoking it.
    """

    pairs = tuple((str(left), str(right)) for left, right in contrast_pairs)
    if not pairs or any(left == right for left, right in pairs):
        raise ValueError("contrast_pairs must contain distinct view pairs")
    required = {name for pair in pairs for name in pair}
    missing = required - (set(reconstructed) & set(targets))
    if missing:
        raise KeyError(f"missing contrast views: {sorted(missing)}")
    terms = []
    for reference, changed in pairs:
        reference_prediction = reconstructed[reference]
        changed_prediction = reconstructed[changed]
        reference_target = targets[reference]
        changed_target = targets[changed]
        if not (
            reference_prediction.shape
            == changed_prediction.shape
            == reference_target.shape
            == changed_target.shape
        ):
            raise ValueError("contrasted views must have matching matrix shapes")
        predicted_delta = changed_prediction - reference_prediction
        target_delta = changed_target - reference_target
        scale = target_delta.detach().square().mean().clamp_min(eps)
        terms.append((predicted_delta - target_delta).square().mean() / scale)
    return torch.stack(terms).mean()


def decoder_alignment_loss(
    model: MultiViewSparseTranscoder,
    rotations_to_reference: Mapping[str, torch.Tensor],
    *,
    reference_view: str,
) -> torch.Tensor:
    """Align same-index writers after supplied outcome-blind rotations.

    A rotation maps row vectors from a source view into the reference view:
    ``row_source @ rotation``.  Decoder columns therefore transform as
    ``rotation.T @ decoder_source``.
    """

    names = model.config.aligned_views
    if reference_view not in names:
        raise ValueError("reference_view must be one of aligned_views")
    missing = set(names) - set(rotations_to_reference)
    if missing:
        raise KeyError(f"missing rotations: {sorted(missing)}")
    reference = model.decoder(reference_view)
    losses = []
    for view in names:
        rotation = rotations_to_reference[view].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        expected = (model.config.output_dim, model.config.output_dim)
        if rotation.shape != expected:
            raise ValueError(f"rotation for {view!r} must have shape {expected}")
        transported = F.normalize(
            rotation.T @ model.decoder(view),
            p=2.0,
            dim=0,
            eps=model.config.eps,
        )
        losses.append(1.0 - (transported * reference).sum(dim=0))
    return torch.stack(losses).mean()


def multiview_transcoder_loss(
    model: MultiViewSparseTranscoder | StructuredMultiViewSparseTranscoder,
    inputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    pair_indices: torch.Tensor | None = None,
    contrast_pairs: Sequence[tuple[str, str]] | None = None,
    reconstruction_weight: float = 1.0,
    difference_weight: float = 1.0,
    view_difference_weight: float = 0.0,
    sparsity_weight: float = 1e-3,
    consistency_weight: float = 0.0,
    consistency_indices: slice | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute label-free reconstruction, delta, and sparsity losses."""

    expected = set(model.config.views)
    if set(inputs) != expected or set(targets) != expected:
        raise ValueError("input and target mappings must contain every configured view")
    if min(
        reconstruction_weight,
        difference_weight,
        view_difference_weight,
        sparsity_weight,
        consistency_weight,
    ) < 0.0:
        raise ValueError("loss weights must be non-negative")

    features: dict[str, torch.Tensor] = {}
    reconstructed: dict[str, torch.Tensor] = {}
    reconstruction_terms = []
    difference_terms = []
    sparsity_terms = []
    for view in model.config.views:
        target = targets[view]
        feature_values, prediction = model(view, inputs[view])
        if target.shape != prediction.shape:
            raise ValueError("target must match reconstructed output shape")
        features[view] = feature_values
        reconstructed[view] = prediction
        centered = target.detach() - target.detach().mean(dim=0, keepdim=True)
        scale = centered.square().mean().clamp_min(model.config.eps)
        reconstruction_terms.append((prediction - target).square().mean() / scale)
        sparsity_terms.append(feature_values.abs().mean())
        if pair_indices is not None:
            difference_terms.append(
                paired_difference_mse(
                    prediction,
                    target,
                    pair_indices,
                    eps=model.config.eps,
                )
            )

    reconstruction = torch.stack(reconstruction_terms).mean()
    difference = (
        torch.stack(difference_terms).mean()
        if difference_terms
        else reconstruction.new_zeros(())
    )
    sparsity = torch.stack(sparsity_terms).mean()
    consistency_views = (
        model.config.aligned_views
        if isinstance(model.config, MultiViewTranscoderConfig)
        else model.config.tuned_views
    )
    consistency = (
        standardized_consistency_loss(
            (
                features
                if consistency_indices is None
                else {
                    view: values[:, consistency_indices]
                    for view, values in features.items()
                }
            ),
            consistency_views,
            eps=model.config.eps,
        )
        if consistency_weight > 0.0
        else reconstruction.new_zeros(())
    )
    view_difference = (
        view_difference_mse(
            reconstructed,
            targets,
            contrast_pairs,
            eps=model.config.eps,
        )
        if view_difference_weight > 0.0 and contrast_pairs is not None
        else reconstruction.new_zeros(())
    )
    if view_difference_weight > 0.0 and contrast_pairs is None:
        raise ValueError("positive view_difference_weight requires contrast_pairs")
    total = (
        reconstruction_weight * reconstruction
        + difference_weight * difference
        + view_difference_weight * view_difference
        + sparsity_weight * sparsity
        + consistency_weight * consistency
    )
    return {
        "loss": total,
        "reconstruction": reconstruction,
        "paired_difference": difference,
        "view_difference": view_difference,
        "sparsity": sparsity,
        "consistency": consistency,
        "features": features,
        "reconstructed": reconstructed,
    }


def feature_similarity_matrices(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    left_decoder: torch.Tensor,
    right_decoder: torch.Tensor,
    *,
    activation_weight: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compare independently indexed dictionaries by activation and writer geometry."""

    if (
        left_features.ndim != 2
        or right_features.ndim != 2
        or left_features.shape[0] != right_features.shape[0]
    ):
        raise ValueError("feature matrices must be 2D with paired rows")
    if (
        left_decoder.ndim != 2
        or right_decoder.ndim != 2
        or left_decoder.shape[0] != right_decoder.shape[0]
        or left_decoder.shape[1] != left_features.shape[1]
        or right_decoder.shape[1] != right_features.shape[1]
    ):
        raise ValueError("decoder columns must match feature dimensions")
    if not 0.0 <= activation_weight <= 1.0:
        raise ValueError("activation_weight must be in [0, 1]")

    left_centered = left_features - left_features.mean(dim=0, keepdim=True)
    right_centered = right_features - right_features.mean(dim=0, keepdim=True)
    activation_numerator = left_centered.T @ right_centered
    left_scale = left_centered.square().sum(dim=0).sqrt().clamp_min(eps)
    right_scale = right_centered.square().sum(dim=0).sqrt().clamp_min(eps)
    activation_correlation = activation_numerator / (
        left_scale[:, None] * right_scale[None, :]
    )

    left_directions = F.normalize(left_decoder, dim=0, eps=eps)
    right_directions = F.normalize(right_decoder, dim=0, eps=eps)
    decoder_cosine = left_directions.T @ right_directions
    combined = (
        activation_weight * activation_correlation.clamp(0.0, 1.0)
        + (1.0 - activation_weight) * decoder_cosine.clamp(0.0, 1.0)
    )
    return {
        "activation_correlation": activation_correlation,
        "decoder_cosine": decoder_cosine,
        "combined": combined,
    }


__all__ = [
    "LABEL_FREE_PACKAGE_FIELDS",
    "LABEL_FREE_SEQUENCE_PACKAGE_SCHEMA",
    "LABEL_FREE_SOURCE_FIELDS",
    "LABEL_FREE_STATE_FIELDS",
    "LABEL_BEARING_FIELDS",
    "MultiViewSparseTranscoder",
    "MultiViewTranscoderConfig",
    "StructuredMultiViewSparseTranscoder",
    "StructuredMultiViewTranscoderConfig",
    "assert_label_free_payload",
    "batch_topk",
    "decoder_alignment_loss",
    "feature_similarity_matrices",
    "label_free_package_binding",
    "multiview_transcoder_loss",
    "paired_difference_mse",
    "row_topk",
    "standardized_consistency_loss",
    "view_difference_mse",
    "validate_label_free_state_rows",
]
