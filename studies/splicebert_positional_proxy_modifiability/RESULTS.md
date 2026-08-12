# Results index — SpliceBERT positional-proxy modifiability

> **Tier: T2 — prospective successor.** One audited construction-stage row lives in `CLAIMS.md`. This
> index carries exactly one row of its own, a **development-stage pilot that backs no claim** and has
> **not** been independently audited — see the audit note below before citing it anywhere.

| Run / result | Question | Status | Artifact | Manifest | Claim IDs | Notes |
|---|---|---|---|---|---|---|
| Paired-lineage pilot v1 (2026-08-12) | Does phase-decoupled continued pretraining change `Ψ` at all, by any route? | `PILOT` — development, no claim row | [`results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.json`](../../results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.json) | [`manifest`](../../results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.manifest.json) | — | **ONE SEED PER ARM — not a training-process claim.** Base stop rule passed first (`Ψ` +0.25625 against the recorded +0.256 on 479 cells). `Ψ`: base +0.25625, L1 (phase-preserving) +0.64208, L2 (phase-decoupled) +0.29748; `ΔΨ` = **+0.34460**, gene-clustered [+0.25755, +0.44128] by donor gene and [+0.25929, +0.43347] by acceptor gene. Carried by **L1 rising** (L1−base +0.386 [+0.288, +0.493]); **L2−base +0.041 [−0.028, +0.122] includes zero**. No change in carrier **amplitude** (period-3 band power +0.71%/+0.62%, equal across arms) — which does **not** show the effect bypassed the carrier, since band power is provably blind to a phase rotation; and not degradation (masked MLM top-1 flat at 0.473). Both arms shifted the coupling mean negative, so neither is the base instrument. Full index and every caveat: [`RESULTS_INDEX.md`](../../results/splicebert_positional_proxy_modifiability/RESULTS_INDEX.md) |

`CLAIMS.md` already carries **PPM1**, backed by
[`results/splicebert_positional_clock/real_panel_capacity_v1.json`](../../results/splicebert_positional_clock/real_panel_capacity_v1.json)
and its independent audit; those artifacts were produced under the (still in-flight) PBC v2 successor
line, not under this study's own name, so this index does not duplicate that row — see `CLAIMS.md`
directly. Artifacts are never overwritten. Every future row here must link to the exact indexed
artifact, manifest, and independent audit that support it.

**Audit note — the pilot row above has no independent audit, deliberately and explicitly.** The
standing requirement in the paragraph above is met for claim-backing rows; the pilot backs no claim
(`Claim IDs: —`, `status: development`) and is recorded for provenance, not for citation. Two
conditions must both be satisfied before any `CLAIMS.md` row may cite it:

1. **a multi-seed run.** One training seed per arm gives no degrees of freedom for the training
   process, which is the replicate a corpus intervention is measured against. The reported `ΔΨ`
   interval covers evaluation-cell resampling only.
2. **an independent audit** on its own code path, in the manner of
   `scripts/splicebert_positional_clock/audit_real_panel_capacity.py`.

Until both exist, this row licenses no statement about SpliceBERT, about training, or about biology.
