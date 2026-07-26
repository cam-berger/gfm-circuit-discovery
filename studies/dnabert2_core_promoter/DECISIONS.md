# Decisions — DNABERT-2 core promoter

> **Tier: T2 — provenance.** Dated decisions and deviations. Append only.

## 2026-07-26 — study opened

**D1. Model choice.** DNABERT-2-117M, for **architectural contrast** with SpliceBERT (BPE vs
single-nucleotide; ALiBi vs learned position table; MosaicBERT internal unpadding), per
`docs/TRANSFER_GUIDE.md`. Reproducing the SpliceBERT finding is explicitly not a goal.

**D2. Vendored modelling code rather than `trust_remote_code=True`.** The published remote code does not
load on either available transformers version (5.13 fails the remote-code import scanner on
`flash_attn_triton.py`; 4.44.2 fails `AutoModel.register` with `config_class not consistent`). The three
modelling modules are vendored verbatim at revision `7bce263b…abda0`, hash-pinned, and imported directly.
`flash_attn_triton.py` is deliberately not vendored — `bert_layers.py` already guards its import and
falls back to the PyTorch attention path, which is the path that runs on this machine regardless.
Rationale: reproducibility, no run-time network dependency, and a readable/hookable implementation for
later phases. Recorded in `src/gfmcd/adapters/dnabert2/vendor/README.md`.

**D3. Study-local pinned environment.** `transformers==4.44.2` is pinned in
`requirements-dnabert2.txt`, **not** as a root `pyproject` extra: pinning it at the root would re-resolve
the shared SpliceBERT environment and perturb a frozen record. Every runner asserts the pin and records
it in its manifest.

**D4. Native MLM head confirmed usable.** The checkpoint carries a complete trained MLM head
(`cls.predictions.decoder [4096, 768]`; `missing=[]`, `unexpected=[]`). This was checked **before** any
design work, because the entire primary assay is impossible without it.

**D5. Coordinate convention decoded, then verified.** EPD `DR UCSC` lines are **not** a single
convention (0-based for `+`, 1-based for `−`) — the exact trap rival R6 names. Coordinates are therefore
taken from the BED `thickStart`/`thickEnd` + strand and verified against the EPD `SE` line
(3,000/3,000). See `G0_GATE_PACKET.md` §3.

**D6. TATA definition.** JASPAR **MA0108.3** (TBP, 7 nt), motif-**START** anchor, relative threshold
0.85, functional interval −33…−27 ⇒ 3,297 promoters (11.1%). Chosen over the strict `TATAWAWR`
consensus (n = 518) on the gate owner's instruction to prefer a PWM with score-bin strata; the strict
subset remains recoverable as the top bin. A second, decisive reason emerged in red-team round 2: only
**44.8%** of native words admit a composition-matched alternative under the strict consensus, versus
**97.9%** under the PWM.

**D7. Estimands deviate from the supplied phase plan** in three places — content null, geometry design,
and the null-family structure. Each deviation, and the reason, is recorded in
`preregistrations/PREREGISTRATION_TATA_TSS.md` §11. Approved by the gate owner before drafting.

**D8. Design red-team run twice before any compute.** Round 1 found 2 FATAL and 6 MAJOR issues against
the first draft; round 2 found one regression introduced *by* a round-1 fix (edit-count asymmetry, now
memorial **M9**), one overstatement in a round-1 finding (the ALiBi issue was labelled FATAL; it is
MAJOR — ALiBi can shape a decay profile but cannot create a nonzero `V(d)`), and one control that was
**infeasible under this study's own rules** (the Inr-flank positive control violates M2 by construction).
Conclusions are folded into the preregistration; the working drafts are not part of the record.

**D9. Two G0-conditional blockers raised rather than assumed away.**
- **B1** — "focused" is asserted but EPD carries no shape score. Resolve by acquiring FANTOM5 CAGE and
  freezing a shape index, **or** by deleting "focused" from the cohort definition and every licensed
  claim. Gate owner's choice; it changes the claim text.
- **B2** — the −96…+64 window is a candidate, frozen only after the window-length preflight.

**D10. Relationship to `origin/dnabert-circuit-discovery`.** That branch holds an earlier
infrastructure-first scaffold (block ablation, SVD difference subspace) created before a biological
computation was chosen. It is **not** built on, because its exploratory-first ordering conflicts with the
gate discipline here. It is left untouched pending a gate-owner decision on whether to retire it.
