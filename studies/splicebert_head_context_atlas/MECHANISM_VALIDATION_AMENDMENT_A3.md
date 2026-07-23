# Head–Context Atlas mechanism-validation amendment A3 — numerical recovery

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Status: **FROZEN AFTER THE TARGET FORWARDS AND THEIR MECHANICAL REFERENCE-GATE FAILURE, BUT BEFORE
ANY STAGE-TWO QK/V/MLP FAMILY WAS MERGED, ANALYZED, CLASSIFIED, OR READ.** The failed runtime manifest
and all target shards remain immutable. This amendment changes no biological estimand, intervention,
candidate, endpoint, sample, family, or decision threshold.

## A3.1 Trigger and preserved record

The full Stage 2 target executor wrote all 4,019 expected shards, then exited fail-closed because its
cross-stage whole-effect agreement gate used an absolute tolerance of `1e-6`. The immutable target
manifest records exactly one failure:

- gate: `target_whole_stage_reference`;
- observed maximum absolute discrepancy: `2.9802322387695312e-05` nats;
- frozen runtime threshold: `1e-6` nats;
- every other runtime gate: pass.

No Stage 2 mechanism label existed at this point. Before this amendment, inspection was restricted to
packet counts, the failed gate, and the `candidate_whole`/`control_whole` reference discrepancies that
define that mechanical gate. Conditional QK, V/OV, MLP-recovery, exact-group, and path results were not
summarized or interpreted.

## A3.2 Why the original gate was miscalibrated

Stage 1 and Stage 2 evaluate the same whole interventions in different deterministic packet orders
(`mechanism-validation-order` versus `stage2`). That changes CUDA batch composition. Float32 matrix
reductions are deterministic for a fixed batch but are not bit-identical across different batch
compositions, even with TF32 and AMP disabled.

The observed discrepancy distribution is quantized at float32-scale and tiny relative to the scientific
scale: median `1.4305e-06`, p90 `4.7684e-06`, p99 `9.3275e-06`, maximum `2.9802e-05` nats. The maximum is
about `0.06%` of the `0.05`-nat SESOI. This is a numerical agreement issue, not evidence that an
intervention, coordinate, packet, or source state changed.

Amendment A2 already froze `1e-4` as the numerical tolerance for native-source identities and QKV-to-
whole bridges. A3 applies that already-preregistered numerical scale to the cross-batch whole-reference
agreement check. It does not choose a tolerance from a mechanism effect.

## A3.3 Narrow admissibility rule

An A3 independent audit may pass only if all of the following hold:

1. the original target manifest is preserved with `status=FAIL`, `n_failures=1`;
2. its sole failure is `target_whole_stage_reference`, with original threshold exactly `1e-6`;
3. the recorded and independently reconstructed maximum discrepancy is greater than `1e-6` and no
   greater than `1e-4`;
4. all 4,019 target shards are present and match the immutable manifest hashes and sizes;
5. the complete expected whole-reference grid is present;
6. every other runtime, coordinate, candidate, endpoint, finite-value, QKV bridge, health, algebra, and
   split/provenance check passes under the original rules; and
7. the new audit records the full comparison count, original/effective tolerances, median, p90, p99,
   maximum, and counts above each tolerance.

A3 cannot waive a second failed gate. It cannot repair a missing shard, changed hash, QKV mismatch,
health failure, non-finite value, endpoint-grid error, or analysis discrepancy.

## A3.4 Recovery and evidence status

The A3 audit is written to a new immutable path, `validation_audit.a3.json`. Merge and analysis use new
`*_a3` prefixes and pin that audit. The original runtime manifest is not rewritten, and the expensive
forwards are not repeated.

If the A3 execution audit and the existing independent analysis reconstruction both pass, Stage 2 may be
interpreted as an **outcome-blind conditional-mechanism analysis with a transparently amended numerical
gate**. The amendment weakens only the claim of cross-batch bit-level agreement; it does not weaken the
causal, health, multiplicity, materiality, or inference requirements.

## A3.5 Prevention

Future cross-stage executors should either reuse the same packet batching/order or preflight a cross-
batch float32 agreement distribution and freeze a tolerance appropriate to that numerical comparison.
A same-batch deterministic-repeat gate must not be treated as proof of cross-batch bit identity.

The Stage 2 executor now reuses the whole-validation `mechanism-validation-order` rank namespace, with a
regression test pinning that namespace. This prevention change applies to future executions only; the
completed A3 run remains tied to its immutable original executor and preserved numerical audit.

## A3.6 First-audit portability correction

The first A3 audit attempt was preserved as `validation_audit.a3.json` with `status=FAIL`. It found no
new numerical, shard, or mechanism failure. It stopped because the prebuilt health manifest recorded the
producer checkout's absolute `/Users/.../results/...` paths, which do not exist inside Colab even though
the A3 overlay contains the byte-identical, hash-pinned files under the same repository-relative
`results/...` suffix.

The auditor is therefore corrected to resolve a missing frozen absolute producer path through its
repository-relative `{results,data,studies}/...` suffix and then require the original SHA-256. This is a
path relocation rule, not a hash waiver: an absent candidate or any byte mismatch still fails. The new
audit is written as `validation_audit.a3_v2.json`; the failed first audit remains immutable. No mechanism
family, conditional label, or scientific result was read between the two attempts.
