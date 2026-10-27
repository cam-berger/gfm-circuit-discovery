"""A1 finite-stratum residual bootstrap; frozen v1 estimands/rules stay untouched."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from gfmcd.splicing import register_operator_statistics as original

METHOD = "finite-stratum-centered-residual-bootstrap-v2"
MDE_FACTOR = original.MDE_FACTOR
PSEUDO_SEEDS = original.PSEUDO_SEEDS
DIRECTIONS = original.DIRECTIONS
BRIDGES = original.BRIDGES
interval = original.interval
simultaneous = original.simultaneous
classify = original.classify
inside = original.inside
matching_endpoints = original.matching_endpoints
balanced_assignments = original.balanced_assignments
stratified_indices = original.stratified_indices


@dataclass
class CorrectedInferenceContext(original.InferenceContext):
    residual_scales: np.ndarray
    original_weights: np.ndarray
    inference_method: str = METHOD


def residual_scales(strata) -> np.ndarray:
    counts = Counter(strata)
    if not counts or min(counts.values()) < 2:
        raise ValueError("every stratum requires at least two independent blocks")
    return np.array([np.sqrt(counts[label] / (counts[label] - 1)) for label in strata])


def corrected_values(values, strata) -> np.ndarray:
    """Direct centered-residual map, useful for independent index memorials."""
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[0] != len(strata) or not np.isfinite(values).all():
        raise ValueError("invalid endpoint array")
    labels = np.asarray(strata, dtype=object)
    scales = residual_scales(strata)
    result = values.copy()
    for label in sorted(set(strata)):
        take = labels == label
        center = values[take].mean(axis=0)
        result[take] = center + scales[take][0] * (values[take] - center)
    return result


def make_inference_context(block_ids, strata, *, draws=20000, seed=20260803,
                           pseudo_seeds=PSEUDO_SEEDS) -> CorrectedInferenceContext:
    base = original.make_inference_context(block_ids, strata, draws=draws,
                                          seed=seed, pseudo_seeds=pseudo_seeds)
    return correct_inference_context(base)


def correct_inference_context(base: original.InferenceContext) -> CorrectedInferenceContext:
    """Pure affine correction; rejects accidentally applying the factor twice."""
    if isinstance(base, CorrectedInferenceContext):
        raise ValueError("context is already corrected")
    scales = residual_scales(base.strata)
    # These are affine replicate coefficients, not sampling probabilities.
    weights = base.weights * scales[None, :] + (1 - scales[None, :]) / len(base.block_ids)
    return CorrectedInferenceContext(base.block_ids, base.strata, base.indices,
        weights, base.assignments, base.coefficients, base.draws, base.seed,
        base.pseudo_seeds, scales, base.weights)


def analyze(reduced: dict, *, context: CorrectedInferenceContext | None = None) -> dict:
    context = context or make_inference_context(reduced["block_ids"], reduced["strata"])
    if not isinstance(context, CorrectedInferenceContext) or context.inference_method != METHOD:
        raise ValueError("v2 inference requires the A1 corrected context")
    # The untouched producer forms every endpoint, including paired pseudo excess,
    # BEFORE contracting this single corrected context. Raw matrices are not scaled.
    result = original.analyze(reduced, context=context)
    primary = result["primary_psi"]
    primary["uncorrected_analytic_empirical_mde80"] = primary["analytic_empirical_mde80"]
    values = matching_endpoints(reduced["matrices"]["cc_max"])["psi"]
    centered = values.copy()
    labels = np.asarray(reduced["strata"], dtype=object)
    for label in sorted(set(labels)):
        take = labels == label
        centered[take] -= centered[take].mean()
    primary["analytic_empirical_mde80"] = float(MDE_FACTOR * np.sqrt(
        np.sum((context.residual_scales * centered) ** 2)) / len(values))
    primary["analytic_mde_method"] = METHOD
    result["schema"] = "gfmcd.register_operator.analysis.v2"
    result["inference_method"] = METHOD
    result["bootstrap_correction"] = {
        "method": METHOD,
        "scale_by_stratum": {str(label): float(context.residual_scales[i])
                             for i, label in enumerate(reduced["strata"])},
        "endpoint_stage": "after raw readout reduction and endpoint formation",
        "observed_point_estimates_unchanged": True,
        "replicate_coefficients_may_be_negative": True,
    }
    for name in ("directions", "mismatch_equivalence", "bridge_diagnostics",
                 "historical_psi_ac_minus_at_linear", "historical_phase_blind_ac_minus_at_linear",
                 "historical_phase_blind_ac_minus_at_max"):
        result[name]["method"] = "finite-stratum centered-residual max-absolute-studentized whole-block bootstrap"
    return result
