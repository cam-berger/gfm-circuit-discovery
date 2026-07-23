# results/origin-siblings — sibling-checkpoint carrier origin

**Tier:** T1. Controlling claim: **C39**. Natural experiment on the published SpliceBERT stage-1 siblings
(`SpliceBERT.510nt` 72-species vs `SpliceBERT-human.510nt` human-only; same 6L×16H×512 architecture,
stage, and 510-nt length — corpus is the only intended difference). Synthesis:
[`docs/CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md`](../../docs/CARRIER_ORIGIN_SYNTHESIS_2026-07-23.md).

## canonical/

| Artifact | What it establishes |
|---|---|
| `sibling_pspectrum.json` (+manifest) | Leakage-free P-spectrum of both carriers in all three checkpoints. Corpus contrast: period-3 72-sp 435× vs human-only 6.2× (undetected); ≈10.63 369× vs 2.4× (undetected). Validation gate on the final 1024-nt checkpoint reproduces the known 289×/bin-96 results. Both carriers already present at end of stage-1. |
| `sibling_carrier_bandtransfer.json` (+manifest) | Band-deletion (carrier-transfer) in the 72-species sibling: period-3 used coding-specifically (exon vs intron damage +0.170 vs +0.032, DiD +0.138; intron health 87.6%), reproducing C8. `verdict_mechanical` = VOID_unhealthy is the R13 gate artifact (exon headroom is the outcome); corrected `verdict` = USED_coding_specific via the intron non-outcome channel. |

## development/

Smoke run (`sibling_bandtransfer_smoke.json`) — preflight only, not cited by C39.

## Notes

- N=512 makes period-3 a non-exact Fourier bin (energy across bins 170–172); the estimator is
  leakage-free (direct cos/sin projection) and the band-deletion uses bins 169–173. Validated on N=1026
  before interpretation.
- Human-only sibling is not band-deleted: its carriers are already absent in P (nothing to delete).
