# Head–Context Atlas analysis amendment A1

Status: **FROZEN BEFORE THE FULL DISCOVERY MATRIX WAS READ** (2026-07-19).

This amendment supplements `ANALYSIS_PLAN_V1.md` (SHA-256
`21e696e360c259c9a1f7ecd6b09c5447fc5f63b3f1d5251497744f7ce69de5ef`). It was written after the
four-packet/two-pair plumbing smoke only. No full-screen component outcome, discovery rank, or internal
validation outcome had been loaded. It changes no estimand, effect floor, correction family, stability
threshold, shortlist cap, or ranking rule.

## A1.1 Masked-health interpretation

`broad_MLM_flag` is an annotation, not an exclusion. This follows the original plan's instruction to keep
target and generic reconstruction effects visible side by side without post-treatment adjustment. A
general candidate passes the health *instrument* when both role-specific health estimates exist and are
finite and the upstream aggregate masked-health gate passed. A large generic effect is scientifically
informative evidence of broad use; silently deleting it would defeat the general-use atlas estimand.

## A1.2 Dyadic uniqueness precondition

The frozen meat `sum_g(S_g^2) - sum_p(s_p^2)` assumes one observation per unordered endpoint-gene dyad
within a contrast. The selected v7 cohort satisfies this condition. The analyzer and independent verifier
must now fail closed if a contrast contains a repeated unordered endpoint dyad. A future cohort with
repeated dyads requires an explicitly generalized intersection-cluster estimator; it may not silently use
the v1 formula.

## A1.3 Empty stability halves and strict JSON

An empty fixed-hash half is `stability_instrument_limited`: its mean is serialized as JSON `null`, never
non-standard `NaN`, and it cannot pass the direction gate. This affects only deliberately tiny smoke
subsets; the full discovery stability rule remains at least 40 within-half pairs in each half.

## A1.4 Independent audit gate

The full discovery analysis is not interpretable until
`validate_atlas_discovery_analysis.py` independently reconstructs packet grids, joins, gene-clustered and
gene-dyadic estimates, multiplicity corrections, classifications, stability gates, shortlist selection,
split boundaries, and artifact hashes with zero failures.
