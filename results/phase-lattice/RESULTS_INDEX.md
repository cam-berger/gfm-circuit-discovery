# The 10.67 → 10.63 carrier investigation — every result, and what it says

**Repo:** `~/Research/projects/gfm-circuit-discovery` (branch `splicebert-coupling`)
**Claims registry:** `studies/splicebert_partner_selection/CLAIMS.md`  ← **read this first**
**Tripwires:** `tests/test_chromatin_refutation_memorials.py` (13 tests pinning R17)
**Figures:** `figures/splicebert/` — start at `EVIDENCE.md` (the P → QK → Attention →
OV → Behaviour ladder), then `fig1`–`fig4`. All four are generated from the single data package
`fig_report_data.json`, which self-checks against the recorded runs before writing.

---

## The headline

**The chromatin-source hypothesis is REFUTED, with power.** The nucleosomal signal is real and
enormous (z = +249) — and it is at **10.298 nt**. SpliceBERT's carrier is at **10.630 nt**.
They are different objects. See **R17**.

| | wavelength |
|---|---|
| nucleosomal WW periodicity (measured, mouse ESC chemical map) | **10.298** |
| SpliceBERT's carrier (measured sub-bin, two independent grids) | **10.630** |
| ~~"10.67"~~ | **never a measurement — a bin label, 512/48** |

---

## Results files

| file | what it holds |
|---|---|
| `results/phase-lattice/synth_grid.json` | **Exp 1.** The grid does NOT snap (D27); the phase-reset null (D29). One run marked **VOID** — do not cite it. |
| `results/phase-lattice/dyad_real.json` | **Exp 2.** The biological source: z = +249, depth 4.87%, ρ ≈ 1, offset control, WW/SS counterphase (D28). |
| `results/phase-lattice/dyad_wavelength.json` | **Exp 2.** The wavelength: fine scan 10.298, phase-rotation 10.252, **genic 10.296** (D28). |
| `results/phase-lattice/carrier_subbin.json` | **SpliceBERT's own carrier, sub-bin: 10.630.** Plus the human-only checkpoint at bin 50 (~10.2). |
| `results/phase-lattice/dyad_fpr.json` | **Exp 0.** FPR 4.6% [4.1, 5.2], no jitter dependence. The instrument, frozen. |
| `results/phase-lattice/fig_phase_data.json` | The period-3 clock + L0H11's two-tone fit (D21–D23). |
| `results/phase-lattice/synth_quantization.json` | ⚠️ **Superseded and partly clobbered.** Holds a single depth-1.0 control, NOT the pilot D26 describes. |

## Scripts (all in `scripts/splicebert/`)

| script | what it does |
|---|---|
| `analyze_synth_design.py` | **Zero GPU.** The paper math that re-anchored Exp 1: headroom, the 60-nt coherence length, the faithful Q_win null. Run this before any training. |
| `run_synth_grid.py` | Exp 1: stages A (phase-reset), P (readout control), Q (grid-snap). |
| `run_carrier_subbin.py` | Measures SpliceBERT's true carrier wavelength by inverting the sinc² leakage. |
| `run_dyad_real.py` | Exp 2: the real dyad assay. Imports the FROZEN instrument from `run_dyad_fpr.py`. |
| `run_dyad_wavelength.py` | Exp 2: fine scan, phase rotation, genic restriction. |
| `run_dyad_fpr.py` | Exp 0: the frozen instrument + its FPR. |
| ⛔ `run_synth_quantization.py` | **Do not run.** Superseded by `run_synth_grid.py`. |

## Data

| | |
|---|---|
| Voong chemical dyad map | `~/Research/data/nucleosome/GSM2183909_unique.map_95pc.txt.gz` (10,668,011 dyads, **mm9**) |
| mm9 reference | `~/Research/data/genomes/mm9.fa` |
| mm9 refGene | `~/Research/data/nucleosome/mm9_refGene.txt.gz` |
| provenance + traps | **`data/NUCLEOSOME_MAP.md`** (mm9 not mm10; chroms 20/21/22 = X/Y/M) |

## Docs

| | |
|---|---|
| `studies/splicebert_partner_selection/CLAIMS.md` | D21–D30, R13–R17. The record. |
| `studies/splicebert_partner_selection/EXP3_DESIGN.md` | Exp 3's design — now **moot** (R17). |
| `data/NUCLEOSOME_MAP.md` | The map's provenance and its two silent-corruption traps. |
| `figures/splicebert/README.md` | The figures and the evidence hierarchy. |

---

## ⛔ THREE nulls that are NEVER evidence

1. **R16** — the WW helical-phasing null. Detection floor 5–10% vs a few-percent effect.
2. **Assay 1** — the unanchored corpus spectrum (p ≥ 0.39). Report its sensitivity limit.
3. **D30** — the unanchored WW power spectrum. Flat because it is **blind**, calibrated by injection.

All three are **quadratic** detectors (SNR ∝ depth²·√N) pointed at a few-percent signal.
The anchored estimator is **linear** (SNR ∝ depth·√N) and finds the same signal at **z = +249**.

**R17 is NOT one of these.** It rests on a POSITIVE detection at the wrong frequency.

## The open question

**Something in vertebrate pre-mRNA is at 10.63, and it is not chromatin. What?**

Leading candidate: **Kogan & Trifonov's splice-junction RR/YY phasing, 10.58 ± 0.35.** SpliceBERT is
a splice model, and RR/YY were the STRONGEST channels at dyads (z ≈ +289 vs WW's +249) — but the
wavelength scan was only ever run on WW, and only ever anchored on dyads.
**Next: anchor on splice junctions, scan RR/YY.** Same frozen instrument, human genome.
