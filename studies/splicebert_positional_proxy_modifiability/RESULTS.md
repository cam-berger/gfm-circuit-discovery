# Results index — SpliceBERT positional-proxy modifiability

> **Tier: T2 — prospective successor.** One audited construction-stage row lives in `CLAIMS.md`. This
> index carries two rows of its own, both **development-stage pilots that back no claim** and
> neither independently audited — see the audit note below before citing either anywhere.

| Run / result | Question | Status | Artifact | Manifest | Claim IDs | Notes |
|---|---|---|---|---|---|---|
| Paired-lineage pilot v1 (2026-08-12) | Does phase-decoupled continued pretraining change `Ψ` at all, by any route? | `PILOT` — development, no claim row | [`results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.json`](../../results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.json) | [`manifest`](../../results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.manifest.json) | — | **ONE SEED PER ARM — not a training-process claim.** Base stop rule passed first (`Ψ` +0.25625 against the recorded +0.256 on 479 cells). `Ψ`: base +0.25625, L1 (phase-preserving) +0.64208, L2 (phase-decoupled) +0.29748; `ΔΨ` = **+0.34460**, gene-clustered [+0.25755, +0.44128] by donor gene and [+0.25929, +0.43347] by acceptor gene. Carried by **L1 rising** (L1−base +0.386 [+0.288, +0.493]); **L2−base +0.041 [−0.028, +0.122] includes zero**. No change in carrier **amplitude** (period-3 band power +0.71%/+0.62%, equal across arms) — which does **not** show the effect bypassed the carrier, since band power is provably blind to a phase rotation; and not degradation (masked MLM top-1 flat at 0.473). Both arms shifted the coupling mean negative, so neither is the base instrument. Full index and every caveat: [`RESULTS_INDEX.md`](../../results/splicebert_positional_proxy_modifiability/RESULTS_INDEX.md) |

| Three-arm carrier dependence v1 (2026-08-12) | Is the trained preference about phase, and does it depend on the learned period-3 carrier? | `ENGINEERING PILOT` — development, no claim row | [`results/splicebert_positional_proxy_modifiability/lineage_carrier_dependence_v1.json`](../../results/splicebert_positional_proxy_modifiability/lineage_carrier_dependence_v1.json) | [`manifest`](../../results/splicebert_positional_proxy_modifiability/lineage_carrier_dependence_v1.manifest.json) | — | **TWO TRAINING SEEDS, three arms each — observed in two exploratory lineage triples, not demonstrated.** Adds the native no-edit arm **L0** the pilot above lacked, at `k = 10`. Base stop rule passed first (`Ψ` +0.25625 on 479 cells). **`L0 > L1 > L2` in both seeds** (L0 +0.656/+0.732, L1 +0.579/+0.604, L2 +0.387/+0.456 against base +0.256): the rise is **generic continuation**, not phase-preservation, and `L1 − L0` is negative and excludes zero in both seeds. The edit-matched `L1 − L2` remains material (+0.191/+0.148). **Carrier: unmoved** — per-bin rotation < 1.1° (< 0.009 nt), coherence > 0.998, per-dimension energy TV ≤ 0.0021, band power within +0.85%. **Carrier dependence: yes** — `Ψ` retention under period-3 attenuation at dose 0.50 is 15.8%/17.8% for L1 (base 10.7%) against 97–117% for three power-matched control bands; the difference-of-differences endpoint on `L1 − L2` is +0.162 [+0.103, +0.219] and +0.130 [+0.074, +0.187] for period-3 against −0.025 to +0.018 for controls. R12/R13 applied: the 0.25 and 0.00 doses are **not read**. Full index and every caveat: [`RESULTS_INDEX.md`](../../results/splicebert_positional_proxy_modifiability/RESULTS_INDEX.md) |

`CLAIMS.md` already carries **PPM1**, backed by
[`results/splicebert_positional_clock/real_panel_capacity_v1.json`](../../results/splicebert_positional_clock/real_panel_capacity_v1.json)
and its independent audit; those artifacts were produced under the (still in-flight) PBC v2 successor
line, not under this study's own name, so this index does not duplicate that row — see `CLAIMS.md`
directly. Artifacts are never overwritten. Every future row here must link to the exact indexed
artifact, manifest, and independent audit that support it.

**Audit note — neither row above has an independent audit, deliberately and explicitly.** The
standing requirement in the paragraph above is met for claim-backing rows; both pilots back no claim
(`Claim IDs: —`, `status: development`) and are recorded for provenance, not for citation. Two
conditions must both be satisfied before any `CLAIMS.md` row may cite either:

1. **a multi-seed run.** One training seed per arm gives no degrees of freedom for the training
   process, which is the replicate a corpus intervention is measured against; two seeds give a range
   over two draws, which bounds nothing. Every interval in both rows covers evaluation-parent
   resampling only.
2. **an independent audit** on its own code path, in the manner of
   `scripts/splicebert_positional_clock/audit_real_panel_capacity.py`.

Until both exist, these rows license no statement about SpliceBERT, about training, or about biology.

**The 2026-08-12 pilot row is bounded by the row beneath it.** Its `ΔΨ` numbers stand as
measurements of a two-arm design, but the attribution its prose drew from them — that
phase-preserving continuation *strengthened* the preference — does not survive the native no-edit
control arm, which reproduces and exceeds the rise. See `DECISIONS.md`, 2026-08-12 (three-arm carrier
dependence).
