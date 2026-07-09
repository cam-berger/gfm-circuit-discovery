# Implementation plan

Every phase has a gate. A phase that fails its gate stops the ladder; it does not get worked
around. Rationale for each choice is in `DECISIONS.md`.

Legend: `[x]` done · `[ ]` not started · `[~]` partial

---

## Phase 0 — Rung 0: the tracer's ground truth

The only place in this program where the answer is exactly known. Nothing downstream is
interpretable until this passes.

- [x] `models/tiny.py` — LayerNorm-free transformer, all activations cached
- [x] `data/synthetic.py` — phase task; marker-anchored frame; phase-biased emissions
- [x] `models/planted.py` — compositional twin (exact) + heuristic twin (local Bayes)
- [x] `tests/test_planted.py` — twins behaviourally matched (|Δacc| = 0.0034), separable by
      marker ablation (1.000 → 0.383 vs 0.583 → 0.583)
- [x] `nulls/sequence.py` — dinucleotide shuffle (Altschul–Erikson), G4-negative decoys
- [x] `config.py` — contamination / tokenisation / strand firewalls
- [x] `dicts/base.py`, `dicts/train.py`, `dicts/matching.py`, `dicts/toy.py`
- [x] `tracer/graph.py` — direct-path attribution through the frozen-attention replacement model
- [x] `tracer/patch.py` — `edge_patch` (frozen/real gate), `node_ablate_head_dim`, `routing_share`
- [x] **Test A** — conservation exact (<1e-21); OV recall = precision = 1.0; QK edge invisible
      (attribution 0.000 on an edge whose ablation costs 0.617 accuracy); faithfulness gap 2x (D12, D13)
- [x] **Test B** — 64 atoms in 32 dims, k=3: EV 0.966, |cos| min 0.998, **MCC 0.945** (D14, D15)
- [x] **Test C** — EV 1.0000, edge recall 1.0, precision 0.6; phase recoverable from features only
      0.662 (vs 1.000 from neurons). Bias absorbs a concept; a feature is a difference vector (D17, D18)
- [x] **Test D** — AUC(comp>bag) 1.0, sep +0.0637; AUC(comp>shortcut) 1.0, sep +0.0063.
      H3 survives only as "not a bag of weak features", NOT as "composes" (D16)

> **GATE 0 — PASSED, with two hypotheses narrowed.** Tests A–D all green (53 tests).
> H3 is restricted (D16): the discriminator tests against bags, not against single shortcuts.
> Feature-level claims now require an identifiability check (D17). Edge-level structure is sound.

- [ ] Plant a *multiplicative* interaction; measure how far edge recall falls
      → this number is the faithfulness/nonlinearity result, and answers Risk #1
- [ ] QK attribution instrument — Test A showed frozen attention scores the marker edge at 0.000
      while its ablation costs 0.617 accuracy. Every real rung has a "where to look" step.

---

## Phase 1 — Rung 1: reading frame

- [ ] `data/genome.py` — hg38 + GENCODE CDS, **random-offset windows**, chromosome splits
- [ ] `models/gfm_nano.py` — single-nt transformer, ~6L × d=512
- [ ] Train `gfm-nano-cds` (AR) and `gfm-nano-dna` (both strands)
- [ ] Linear probe: is period-3 phase decodable, layer by layer, vs position-shuffled control?
- [ ] Trace; edge-patch; check sign-correctness under frameshift minimal pairs
- [ ] Port harness to `splicebert` (`yangheng/SpliceBERT-510nt`; 6L, d=512, absolute pos-emb)

> **GATE 1.** A compact frame circuit in GFM-nano with sign-correct edge patches. SpliceBERT may
> legitimately have no frame circuit — that is a finding, not a failure, *because* Gate 0 passed.

Preliminary, weak, reported as motivating only: SpliceBERT's masked triplet completes to a STOP
with P = 0.0395 in-frame vs 0.0511 off-frame (ratio 0.773, Mann-Whitney p = 0.356, n = 40). Direction
matches H0; **not significant**. The probe, not the behavioural proxy, is the instrument.

---

## Phase 2 — Rung 2: splice sites

- [ ] GENCODE donor/acceptor/branch-point/polypyrimidine annotations
- [ ] Sequence nulls: GT/AG present, canonical spacing destroyed
- [ ] Trace sub-signal → composition → readout; JSAE sparsity (threshold from Test D)
- [ ] Edge-patch vs spacing biophysics; routing share vs matched control edges

> **GATE 2.** Composition beyond the matched sequence null, or an interpretable negative.

Splice sites are SpliceBERT's *training distribution*, so the anchor is strongest here.

---

## Phase 3 — Rung 3: G-quadruplex

- [ ] G4-seq / G4 ChIP-seq maps; G4Hunter loop-length grammar labels
- [ ] G-density-matched, G4-negative decoys (`nulls.sequence.g_density_matched`)
- [ ] Trace G-run × loop-window × strand composition on `gfm-nano-dna`
- [ ] Anchor on `dnabert2` (window-level readout; token→nt span mapping required)

> **GATE 3.** Composition over and above a G-density counter.

`assert_cross_strand_capable("splicebert")` raises. That is the point.

---

## Phase 4 — H2: the payload

- [ ] Four arms at matched architecture/scale: AR · MLM · MLM+phyloP · phyloP-only
- [ ] Control arm 3′: phyloP scrambled *within codons* (mean preserved, period-3 destroyed)
- [ ] Common frozen linear eval head; frame labels never touch a backbone
- [ ] Select checkpoints at **matched eval-head accuracy**
- [ ] Completeness (within-model ratio) and routing share per arm, per rung

> **GATE 4.** Monotone ladder at matched accuracy, or the confound wins and we say so.

---

## Phase 5 — Extensions (contingent)

- [ ] Enhancer/TF grammar on ENCODE cCREs
- [ ] SSMs: Caduceus (RC-equivariant, therefore the *right* G4 substrate and the *hardest* to trace)
- [ ] Causal scrubbing on the surviving pathway

---

## Environment

```bash
conda activate gfm-thrust1        # torch 2.12, MPS; transformers 4.44
pip install -e ".[dev]"
pytest -q
```

M5 Pro, 20 cores, 68 GB unified memory. Transcoder sizing for SpliceBERT (6L × d=512):
4096 features ≈ 44M params; full CLT at 16384 features ≈ 175M. Both fit.
