# FROZEN M6 — held-out confirmation of the donor-class result

> **Tier: T1 — frozen protocol.** This spends the validation reserve. There is no second one.

> **Status: FROZEN. Authorises exactly one run.** Runner SHA-256
> `bdbf9c953b23c245734802890db88081ef8a3b1f58c6e4403004f4debaf36b08`
> (`scripts/splicebert_acceptor_coupling/run_m6_validation_confirmation.py`). The runner refuses
> to execute without `--protocol-sha256` matching this document. Re-running after any edit to
> either file requires a new protocol and a cohort that no longer exists.

## 1. What is being confirmed, and what is not

Development work on the train split established that this checkpoint's masked-acceptor readout
depends far less on a real minor-class `GC-AG` donor than on a matched canonical `GT-AG` donor
(`rho = 0.242 [0.173, 0.326]`), and that `L3H1` attends to the minor-class donor far less
(mass ratio `0.364`). Everything supporting those numbers is development tier.

M6 confirms **the direction and rough magnitude of those two facts on held-out data.** It
confirms nothing about mechanism. The layer-0/1 class-specific heads, `MLP1`, the transport
nulls, and the address × payload decomposition are **not** on trial here and remain
development tier after this run whatever it returns.

## 2. Cohort — pinned by an outcome-blind census before any model call

Validation split (chr11/16/19) only. `_dataset` is built with
`allowed_chromosomes=CouplingConfig().val_chromosomes`, so the sealed chr8/9 rows are never
materialised from FASTA — see `ACR18` for why "not opened" is a claim this repository has
previously made too loosely.

| quantity | frozen value |
|---|---|
| matched pairs | **67** |
| treatment genes | **56** |
| unmatched treatment rows dropped | 0 |
| ordered treatment `example_id` SHA-256 | `0dff7e6fd314d237407c926fd13e5784bfbbd9138b24b78ef74e288a922301d9` |
| ordered treatment `gene_id` SHA-256 | `cb0fe02d5fda8ecbdcae454e2647e16fee43f259184de41f26c1541ced609b51` |

Construction is identical to the train arc: operator `GG` (the only edit-size-matched,
motif-clean alt across `GT` and `GC`), comparators matched on `(-1, +3, +5)` flank bases,
donor transcript-support bin, and nearest donor–acceptor gap, one comparator per treatment row,
never reused.

## 3. Endpoints, frozen

Bonferroni over exactly two, `alpha = 0.025` each. Each decision inverts its own one-sided
bootstrap test, so test and interval cannot disagree.

**PRIMARY.** `rho = mean(Y_GC) / mean(Y_GT)`, `Y = clean − donor-destroyed` on the masked
acceptor `G` margin. `H0: rho >= 0.60`. Reject — and record `CONFIRMED` — if the 97.5th
bootstrap percentile of `rho` falls below `0.60`.

**SECONDARY.** `L3H1` donor attention mass from the readout token, ratio `GC / GT`.
`H0: ratio >= 0.70`. Same rule.

Gene-clustered throughout, resampling the union of both arms' genes so rows sharing a gene move
together; 4,000 draws; seed 0; minimum 30 clusters.

## 4. Power, stated honestly

Projected gene-clustered half-width at 56 genes is about **0.25**, scaling AC1's `0.0733` at
645 genes. With a train point estimate near `0.24` the expected upper bound is near `0.49`,
which clears `0.60` but not by much. **This cohort can replicate a large effect once and
nothing subtler.** It cannot distinguish `rho = 0.24` from `rho = 0`, and no such distinction
may be read out of it.

## 5. Disclosure

Both thresholds were chosen after seeing train. That is what a development split is for, and
they are deliberately set well above the development intervals rather than at the observed
values — but this is a confirmation of a pre-stated direction and magnitude, not a discovery.
Any report must say so.

## 6. Memorials

Each can fail and each is on the measurement path. Any failure sets both verdicts to
`MEMORIALS_FAILED` and licenses nothing.

1. **Cohort identity** — realised pairs, genes, and both SHA-256 values equal §2 exactly.
2. **Seal never materialised** — no chr8/9 row present in the built dataset.
3. **Native motifs as declared** — treatment rows are real `GC-AG`, comparators real `GT-AG`.
4. **Edit size matched** — `GG` is one substitution on both arms.
5. **Matching exactly balanced** — `-1`, `+3`, `+5` composition identical across arms.

## 7. Interpretation table

| outcome | what it licenses |
|---|---|
| both `CONFIRMED` | Held-out support that the minor class is substantially less coupled and less attended. Behaviour and attention-geometry rungs only. |
| primary only | The coupling result replicates; the attention ratio does not, at this power. No mechanism claim either way. |
| secondary only | The attention geometry replicates; the coupling magnitude is unconfirmed. |
| neither | The train result does not replicate at this power. Given §4 this is **not** evidence of absence, and the development findings revert to unconfirmed rather than refuted. |

## 8. Provenance

Runner SHA-256 above. `uv.lock` SHA-256 `c2d496796825b54d85e4f871e492baea1e0dea79333bcdd27aef745e3f3399a4`. GTF `83fba3e9…a453`, FASTA
`5be01555…f5e51`, checkpoint `2ad91428…0149d`. Seed 0, 4,000 bootstrap draws. Eager attention,
verified equivalent to sdpa at `3.5e-06` on the margin. The result writes a sibling
`*.manifest.json`.
