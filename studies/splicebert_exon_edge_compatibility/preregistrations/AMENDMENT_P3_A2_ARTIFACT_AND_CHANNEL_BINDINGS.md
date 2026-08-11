# Amendment P3-A2 — artifact, batching, smoke-coverage, and channel bindings

> **Tier: T2 — prospective provenance. Frozen on 2026-08-04 after the
> mechanical-only smoke v1 and before smoke v2 or any development-preflight or
> target outcome from the direct-edge P3 family.**

This amendment resolves executable details exposed while implementing P3-A1. It
does not change the biological hypothesis, handedness, carrier band, supports,
coordinates, cohorts, sample sizes, null families, inferential thresholds, or
decision table. The only checkpoint output from this intervention family before
this amendment was the mechanical-only smoke v1, which reconstructed all six
assigned blocks but forwarded the first block. Smoke v2 had not run. No `G`,
`S_r`, `Omega`, mediation, Q/K, preflight-effect, or target-effect result had
been produced.

## 1. Active smoke artifact

The original immutable smoke artifact is retained. It reconstructed all six
assigned blocks but forwarded only the first. The active pre-freeze smoke is a
new `...development_smoke_v2.json` artifact. It requires at least one carrier
forward, finite Q/K/V capture, same-state round trip, and eager-attention audit
for every one of the six assigned smoke blocks. It remains mechanical-only and
may not serialize raw margins or any effect summary.

## 2. Batch semantics

The maximum masked-language request batch is 16. Carrier rows are inseparable
three-request records (donor/intact, donor/AC, acceptor/intact), so carrier
execution uses five records or 15 requests per forward. A/V and Q/K state
execution uses at most 16 states per forward. The package serializes all three
quantities; “batch 16” does not imply splitting a carrier record.

## 3. Immutable stage envelopes

Every stage reserves one fixed attempt before model loading. Re-entry requires
an explicit `--resume`; an existing attempt without that flag, or `--resume`
without an attempt, fails closed. A stage writes one SHA-256-named shard per
block and one arm manifest only after all exact shards validate. Shard
`content_sha256` is the canonical strict-JSON hash of the shard with that one
self-hash field omitted. Conflicting files, unknown files, stale temporary files,
wrong schemas, wrong row counts, wrong coordinates, wrong block identities, or
wrong package/attempt hashes fail closed.

The exact stage families are:

- development preflight: baseline, learned carrier, protected probes, L3H1 A/V,
  and L3H1 local Q/K;
- target carrier: baseline, learned carrier, and protected probes;
- signed-axis specificity: all 31 frozen seeds, anchor center 6 and both
  orientations only;
- target A/V: L3H1 anchor A/V;
- target Q/K: L3H1 local true/decoy Q/K;
- matched heads: the exact ordered 31-head family on the 24-block subset.

Raw runners serialize margins and recorder checks only. Coupling and every
inferential endpoint are analyzer-derived.

## 4. First-16 preflight audit

The first preflight block is the first frozen preflight ID. Before any analyzed
preflight forward, take the first 16 requests from the deterministic learned
carrier order (three requests per carrier record, truncated after request 16).
Score them twice for determinism and once each for eager-attention normalization.
Thus the audit has 48 model-request exposures but 16 unique request identities.
On the same first execution, hooks must observe the Q, K, and V input projections
for all six encoder layers: exactly 18 named modules, each with shape
`[16,1024,512]`, and every projection value must be finite. No audit margin or
projection value is serialized. The two repeat differences must be at most
`1e-8`; predictions must agree; the exact projection-module count, shapes, and
finiteness must pass; all attention values must be finite; and every attention
row must sum to one within `1e-5`.

## 5. Development channel-alive point gates

P3-A1's channel-alive language is bound to the following point estimates, with
no target confidence threshold imported into preflight:

- A/V: `D_total > 0`, `D_route > 0`, and every per-register routing contrast
  has point estimate above zero;
- local Q/K: `Omega_QK > 0`; every `S_r`, common-mode score, and each of the four
  rescue point estimates is above zero; mean rescue is above zero; and the
  true-key-minus-decoy point estimate is above zero.

`D_value` has only the frozen projected-resolution gate at preflight. These are
instrument/channel checks, not target evidence and not mechanism claims.

For simultaneous inference, the A/V mediation-contrast family is frozen as
exactly `{D_total, D_route, D_value, interaction}`. The three per-register
routing contrasts `{r_0,r_1,r_2}` form their own simultaneous family. For local
Q/K, the rescue family is frozen as the four directional rescues plus
`mean_rescue`; no endpoint may be added to or removed from one of these families
after exposure.

## 6. Target chronology

The provisional target-carrier analysis and its independent audit authorize the
signed-axis specificity family when construction, capability, health, numerical,
resolution, and provenance gates remain intact, regardless of the provisional
carrier sign. This prevents outcome-conditioned exposure of a prespecified null
family. It does not authorize route localization. The full carrier analysis adds
the separate signed-axis gate; only a fully positive carrier result authorizes
the complete A/V, Q/K, and matched-head package. Once authorized, those three
route stages run in order without inspecting A/V to decide whether to run Q/K or
matched heads.

## 7. Specificity inference and audit boundary

Signed-axis and matched-head paired excess use their own one-endpoint
max-absolute-studentized simultaneous intervals on the strand-stratified paired
24-block bootstrap. The exact add-one comparison and target-above-all-controls
gate remain separate. These two null families and the 99-label family are never
pooled.

An analysis status beginning `AWAITING_INDEPENDENT_..._AUDIT` is not an
authorization. A successor stage requires a separate model-free audit with
`status=PASS`, exact analysis/manifests hashes, and the relevant literal
authorization boolean. Audit `PASS` means the artifact, calculation, chronology,
and provenance were independently verified; a scientific gate may still be
false.

## 8. Opposite-handed and bounded-null precedence

A materially positive opposite-handed diagnostic requires both its target
`Omega_opposite` ordinary 95% interval to have lower bound above `+0.05` and all
three opposite-handed `S_r` simultaneous lower bounds above zero. This mirrored
cyclic-relocation requirement prevents the shared `r=0,delta=0` cell from
creating a false orientation verdict. A passing opposite-handed family is a
preregistered orientation falsification and takes precedence over bounded-null
wording. It may not be relabeled as the declared positive ridge.

Bounded-null eligibility requires the declared `Omega` interval wholly inside
`[-0.05,+0.05]`, no materially positive opposite-handed diagnostic, the passing
development preflight, and all target capability/instrument gates: no-surgery
baseline Psi and pooled coupling, baseline donor/acceptor site health, q-zero
fidelity, all 54 carrier-state coupling and donor/acceptor site-health endpoints,
protected-probe log-probability and retention, target resolution, finite
numerics, exact counts/batching, chronology, and provenance. Effect-specific
99-label and signed-axis superiority gates are not capability requirements for a
bounded null. Any missing capability gate yields `INSTRUMENT_UNRESOLVED` or
`INCONCLUSIVE`, never a bounded null.
