# `tex/` — workshop paper source

> Not authoritative for any claim. Where this directory and a claim ledger
> disagree, **the ledger wins** and the paper is wrong.

## What is here

| File | What it is |
|---|---|
| `neurips_2026_gfm_causal_discovery.tex` | "Causal Mechanistic Discovery in Genomic Foundation Models" — the current draft. **Incomplete:** empty abstract, four live `[TODO:]` markers, two stub figure captions, and an empty §5. |
| `refs.bib` | Bibliography, ported from branch `jeyashree-ajay`. **12 entries; the draft cites ~21 keys.** |
| `checklist.tex` | NeurIPS responsible-use checklist, ported from the same branch. |

A different and fuller paper, `tex/splicebert_neurips_2026.tex`, exists on `jeyashree-ajay` (it has an
abstract, introduction, and related work, and all of its citations resolve). **Decide which draft is
canonical before investing more effort** — they are not the same manuscript.

## Building

```bash
cd tex && pdflatex neurips_2026_gfm_causal_discovery && bibtex neurips_2026_gfm_causal_discovery \
  && pdflatex neurips_2026_gfm_causal_discovery && pdflatex neurips_2026_gfm_causal_discovery
```

**It will not build yet.** `neurips_2026.sty` is not in this repository and is not in its history.
Download it from the NeurIPS 2026 author kit and place it in `tex/`. It is **deliberately gitignored**:
the conference style file carries its own licence terms and should not be redistributed here without
checking them. No TeX toolchain is installed on the development machine, so no build has been executed
against this draft — the defect list below is from source inspection, not from a compiler.

`\graphicspath` was repaired on import: the draft arrived with a literal `{{TODO:}{./figures/}}`, which
resolves nothing from `tex/`. It now points at `../figures/splicebert/`, where both included figures
(`fig_full_circuit.pdf`, `fig5_p3_phase_surgery_confirmation.pdf`) actually live. **That is the only
edit made to the draft.** Nothing else — no number, no sentence, no caption — was altered.

## Audit against the claim ledger — READ BEFORE SUBMITTING

The draft was checked against `studies/splicebert_partner_selection/CLAIMS.md`, `CORRECTIONS.md`,
`docs/NOMENCLATURE.md`, `docs/EVIDENCE_LADDER.md`, and the controlling artifacts under `results/`.

**Most numbers are correct.** Verified reproducing exactly from the artifacts: `+0.872 [0.771,0.969]`
and the `+0.007`/`+0.005` controls (C1/C2); `81.6%` / `99.2%` (C3); joint `+0.365 [0.328,0.399]`,
address-only `+0.01`, payload-only `+0.05` (C23, `a2_address_payload.json`); `0.946→0.027` (C8);
`0.363`/`0.348` and the `0.042` one-head null max (C9); `+0.284 [0.192,0.386]` and `−0.005
[−0.013,0.003]` (C35); `435×`/`6.2×` and `369×`/`2.4×` (C39); the eight Level-A reader heads and the
`27–44 / 89–97 / ~100 / 93–107%` retention series; `−0.033 [−0.061,−0.005]` (C37); `0.002–0.003` (C12);
the 60-Mb spectrum; and the `0.33`-nt gap at "about seven times the larger measured wavelength spread"
(R37-compliant — the draft correctly avoids the withdrawn carrier-precision figure). The draft is also
clean on the three retired labels `docs/NOMENCLATURE.md` guards: neither superseded Fourier-bin
wavelength appears, and the payload-semantics term withdrawn under R22 does not appear — line 160
correctly reads "Its semantic identity remains unresolved."

**The following do not survive checking.** None is fixed here; all are the author's to resolve.

### Blocking — would misstate the record

1. **`L3H1 0.830 [0.654,0.994]`, `L2H15 0.515 [0.396,0.625]` described as "held-out denoising".**
   Verified: these come from `results/splicebert_coupling/development/ablation_scan.json`, whose
   manifest reads `"status": "EXPLORATORY"` and `"Exhaustive per-head ABLATION (necessity)"`. Three
   problems at once — it is an **ablation, not denoising** (the opposite operation); it is a
   **development-tier selection screen**, not a result; and the head rows carry only
   `drop_partner`/`drop_control`/`specificity`, so **the two bracketed intervals are not in the
   artifact at all** (they appear only in prereg prose). A pilot number must never appear as
   confirmatory. Use the sealed C3 figures, or `faithfulness.json` sufficiency, and label the split.

2. **`complement 0.005 [0.003,0.008]` inverts its referent.** Verified: `0.005394
   [0.003150,0.007794]` is `F_C` in `a6_circuit_faithfulness.json` — the **circuit's** faithfulness
   when the complement is scrubbed. It is not a measurement of the complement. As written a reader
   concludes the complement recovers nothing; the record says the opposite. Write it as `F(C)`.

3. **`≈10.298` nt appears without naming the organism.** `docs/NOMENCLATURE.md` makes this a
   *mandatory standing phrase*: whenever ≈10.63 and ≈10.298 appear together, the same sentence must
   carry the "0.33 nt apart / powered refutation" qualifier **and** name ≈10.298 as the **mouse ES-cell**
   measurement (Voong 2016, mm9). The draft names no organism, and the surrounding sentences are about
   the human genome — the exact misreading R30 exists to prevent.

4. **First use of "circuit" is unqualified.** `NOMENCLATURE.md` requires the phrase *structured causal
   core within an irreducibly distributed computation* at first use in any top-level document. The
   draft's first use is the §3.2 heading; the phrase appears nowhere in the manuscript.

5. **"we use it to resolve implementation of acceptor-donor coupling"** (Introduction) outruns the
   ledger, and the draft contradicts it itself 100 lines later with "Its semantic identity remains
   unresolved." The Conclusion is correctly scoped; the Introduction is the outlier.

6. **C34's failed gate is not disclosed.** The span-locus result is stated flat. R24 records that the
   confirm split missed its frozen M1 threshold (5.5% > 5%), and the draft's "approached one half at
   32 nt" is that gate-qualified split's own value (confirm 0.50 vs discovery 0.41). A paper whose
   Introduction promises to report failed gates must report this one.

### Important

7. **Checkpoint attribution collapses in §3.6.** The eight reader heads are stage-1 **SpliceBERT.510nt**;
   the `−0.033` coupling bound is **SpliceBERT.1024nt** (a different, nine-head reader set); the
   `369×`/`2.4×` contrast is the 510nt sibling pair. All three sit in one paragraph with no checkpoint
   named, and Table 2 mixes them in one unlabelled column. `NOMENCLATURE.md` makes checkpoint
   attribution mandatory disclosure.
8. **`0.944–0.953` truncates the control dose series.** The arm actually matched to the `0.027`
   full-deletion figure retains `0.970`; the full range is `0.944–0.973`. The record calls this band
   "mildly non-inert", which the truncation hides.
9. **The period-3 "energy-matched control" and the ≈10.63 carrier are the same position-table bins.**
   The draft says deleting them leaves coupling unchanged (§3.5) and moves coupling by `−0.033` with a
   CI excluding zero (§3.6), two paragraphs apart, without noting they are the same band.
10. **C35 quotes only the confirm split, unlabelled**, in an explicitly discovery/confirm design. On
    discovery the value arm is `−0.011 [−0.020,−0.000]`, which marginally *excludes* zero — so
    "recovered none" is cleaner than the record supports.
11. **`−0.033` is described as a "bound"** although the artifact's own verdict is
    `INCONCLUSIVE_underpowered`, and the **direction is dropped**: deletion *raises* coupling, i.e. the
    effect is suppressive.
12. **`98.9% destroyed`** is `N(C\embed)=0.989`, a post-hoc diagnostic installed after the preregistered
    `N(C)=1.00` was retracted as architecture-forced (R21). It has no CI and no null anywhere in the
    record; rendering it as a percentage beside CI-bearing results promotes it.
13. **`C_intron`** is ED1's symbol, and ED1 is a `GATES_FAILED — no interpretation` arc. The donor
    study's threshold is `coupling >= 0.3` (`MIN_EFFECT`, ratio denominators only).
14. **The 60-Mb non-detection omits its sensitivity floor** (5–10% modulation) and is conjoined with the
    RNA-FM/RNA-BERT tables, which C40 records as *powered* nulls. Two different power statements are
    merged into one.
15. **`checklist.tex` says GENCODE v43; the paper says v50.** Both are `\input` into one PDF, which
    therefore asserts two annotation releases and cites two GENCODE papers. The repo config is **v50**
    (`configs/splicebert_partner_selection/base.json`). The checklist also describes an abstract,
    an introduction, and a §4 Discussion that this draft does not contain.

### Build defects

16. **~16 of ~21 citation keys are undefined in `refs.bib`** and will render as `[?]`. Note two
    near-misses: the draft cites `mudge2025gencode` (bib has `mudge2023gencode`) and `wang2023ioi`
    (bib has `wang2022ioi`). `zhang2024patching` is absent entirely and is easy to miss because the
    `\citep` spans two source lines.
17. **`\setcounter{section}{1}`** makes the Introduction §2 and shifts every section number, so the
    checklist's "§3", "§4", "§3.3" references all point at the wrong place.
18. **Figure captions are stubs** (`\textbf{A }`, `\textbf{Clean}`), the text cites `Figure~\ref{fig:core}B`
    though no panels are defined, and `\ref{app:artifacts}` points at a label that does not exist.
19. **The author block is de-anonymized with placeholder `Address` and `email`**, while `checklist.tex`
    promises code release "upon de-anonymization". Under `[final]`/`[preprint]` the placeholders print
    verbatim. Also `\usepackage{neurips_2026}` uses the main-conference track, so the footer will not
    name the workshop.

### Scope note

The draft predates the 2026-08-04 consolidation and does not mention **C43/C44** (coding-boundary phase;
R45). That omission is defensible for a short workshop paper, but note that C44 is the direct test of
whether the period-3 carrier implements a coding-phase preference, and it returned
`INSTRUMENT_UNRESOLVED`. If the paper's period-3 discussion is expanded toward codon phase, C44 must be
cited — the draft's own `[TODO: fill in results of the experimental run]` sits exactly there.
