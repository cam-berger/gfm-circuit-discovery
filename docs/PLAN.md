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

> **GATE 0 — PASSED, with two hypotheses narrowed.** Tests A–D all green (53 tests at the time).
> H3 is restricted (D16): the discriminator tests against bags, not against single shortcuts.
> Feature-level claims now require an identifiability check (D17). Edge-level structure is sound.

- [x] **Test A2 / QK instrument** — naive pattern-gradient is 3.7e-15 (saturated softmax); integrated
      gradient from a bias-only baseline recovers the marker edge, 100% of key attribution on the
      marker token, zero on the bias-driven twin. **OV + QK edge recall = 1.0** (D20)
- [ ] Plant a *multiplicative* interaction; measure how far edge recall falls
      → this number is the faithfulness/nonlinearity result, and answers Risk #1

---

## Phase 1 — Rung 1: reading frame

- [x] `data/genome.py` — hg38 + GENCODE v50 CDS, **random-offset windows**, chromosome splits,
      test split locked behind `unlock_test_split()`. Mutation-tested: CDS-anchored windows leak
      phase at 1.000, randomised at 0.372 (fixture) / **0.3446 on real hg38** (chance 0.3333)
- [x] `scripts/validate_genome.py` — biology as external ground truth: **0 in-frame stop codons in
      117,481**, vs 4.19% / 5.54% in the two shifted frames. Phase and strand arithmetic confirmed
- [x] `models/gfm_nano.py` — single-nt transformer, 6L × d=256 × 8 heads, 4.80M params, LayerNorm,
      absolute pos-emb. `causal=True` is the AR arm, `causal=False` the MLM arm; identical parameter counts
- [~] Train the AR and MLM arms (`scripts/train_nano.py`, 8000 steps, ~29 min/arm on MPS)
- [x] `scripts/probe_frame.py` — layer-by-layer phase probe with four controls: untrained model at
      random init, nucleotide-identity baseline, `index % 3`, label-shuffled floor
- [ ] Linear probe: is period-3 phase decodable, layer by layer, vs position-shuffled control?
- [ ] Trace; edge-patch; check sign-correctness under frameshift minimal pairs
- [ ] Port harness to `splicebert` (official weights `chenkenbio/SpliceBERT`; verify any mirror; 6L, d=512, absolute pos-emb)

> **GATE 1 — PASSED, and it reframed the project.** Real-DNA GFM-nano has **no** anchored frame
> circuit, on either data regime, and the cue-conflict assay proves it (D24). The reason is
> quantified: the circuit buys 0.4% of the loss. The **forcing task** (uniform codons, amino-acid
> readout) installs the circuit abruptly at step ~3000 and it verifies exactly: phase probe 1.0000 at
> L2, start-codon ablation 1.0000 -> 0.3552, cue conflict 1.0000 anchor / 0.0000 content at 250 nt
> (D25, D26). That model is the tracing substrate.

The claim is no longer "does an objective install a circuit" but **a circuit appears when the objective
makes it necessary, not when it is merely useful** — and real genomic self-supervision falls on the
wrong side of that line.

Preliminary, weak, reported as motivating only: SpliceBERT's masked triplet completes to a STOP
with P = 0.0395 in-frame vs 0.0511 off-frame (ratio 0.773, Mann-Whitney p = 0.356, n = 40). Direction
matches H0; **not significant**. The probe, not the behavioural proxy, is the instrument.

---

## Phase 2 — H2: the payload, as a dial rather than four points

**Reframed by D26/D27.** The original H2 asked whether an objective installs a circuit, with four
discrete arms. What we can now ask is sharper: *how much local shortcut does it take to stop the
circuit from forming?* The λ dial interpolates codon usage from uniform (no local phase cue; the
anchor-relative modulo-3 circuit is the only solution) through real human usage to beyond it, while the task -- predict
the amino acid of the containing codon -- never changes.

```
lambda   I(nt;phase)   % of real hg38 CDS
  0.00     0.00001            0%
  0.50     0.00383           16%
  1.00     0.01516           63%     (mixture alone cannot reach real; real DNA also
  1.50     0.03110          129%      carries inter-codon and amino-acid structure)
  2.00     0.05665          235%
```

The y-axis is the **circuit**, not the accuracy: ATG-ablation loss increase, and far-field
cue-conflict anchor share. Both were validated on the planted twins before ever being pointed at a
trained model.

- [~] `scripts/sweep_lambda.sh` — six arms, 8000 steps each
- [ ] `scripts/eval_lambda.py` — the dose-response curve
- [ ] Report the threshold, and where real hg38 sits relative to it

---

## Phase 2b — H2 on real objectives (deferred)

- [ ] Four arms at matched architecture/scale: AR · MLM · MLM+phyloP · phyloP-only
- [ ] Control arm 3′: phyloP scrambled *within codons* (mean preserved, period-3 destroyed)
- [ ] Common frozen linear eval head; frame labels never touch a backbone
- [ ] Select checkpoints at **matched eval-head accuracy**
- [ ] Completeness (within-model ratio) and routing share per arm, on the frame circuit

> **GATE 4.** Monotone ladder at matched accuracy, or the confound wins and we say so.

---

## Phase 3 — Circuit transplantation (proposed; Cam, 2026-07-09)

*Can the learned mechanism be moved onto a real-DNA system, and how does it perform against baseline?*

Transplantation is direct **by construction**: the forcing-task model and the real-DNA arms share one
backbone (6L × d=256 × 8 heads, 4.80M params). Only the output head differs (21-way amino acid vs
6-way nucleotide). That was the point of building a single architecture.

**A prediction we can make before running anything.** The synthetic task guarantees the start codon is
the *first* ATG (`_utr_without_atg`), so the model learned "first ATG = start". On real DNA that rule
is wrong:

```
real anchored windows with >=1 ATG upstream of the true start:  35.1%
mean upstream ATGs per window:                                   0.60
synthetic task, by construction:                                  0.0%
```

That is biology, not an artifact — 5' UTRs carry upstream ORFs, and real initiation weighs Kozak
context rather than taking the first ATG. So the transplanted detector should misfire on ~35% of
windows, in a diagnosable way.

- [ ] **3a. Zero-shot.** Run the circuit-bearing model on real anchored DNA, no retraining. Prediction:
      the *anchor-relative frame circuit* transfers (content-independent arithmetic; whether this trained
      model sequentially counts or does anchor/position residue lookup is unresolved — see
      `docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`); the *detector* locks onto the upstream
      decoy. Verify per-window: does L1H3 attend to the true start or to the decoy? Failures should
      concentrate exactly in the 35.1%.
- [ ] **3b. Transplant + freeze.** Initialise a real-DNA MLM from the circuit-bearing backbone, swap
      the head, freeze L0H0 and L1H3, train the rest. Compare loss to random init. **Hard prediction
      from D24's information calculation: improvement ≤ 0.005 nats (~0.4%).** Control: freeze a matched
      set of causally irrelevant heads, since freezing anything can hurt for unrelated reasons.
- [ ] **3c. Transplant + full fine-tune.** Hand the model the circuit for free, then train on real DNA.
      Does it survive or erode? Track far-field cue-conflict anchor share across steps.
      Prediction: falls 1.00 -> <0.10 within a few thousand steps.

**Why 3c matters most.** Today we can say *the objective does not build the circuit*. If the circuit
erodes when handed over for free, we can say **the objective will not even keep it** — a claim about
the loss landscape rather than about optimisation difficulty. That distinguishes "the bag is the
default" from "the bag is an attractor."

**Caveat to state explicitly.** All three work because the *whole backbone* is transplanted. A partial
graft — inserting only L1H3's weights into a separately trained real-DNA model — is meaningless without
representational alignment: the two residual streams have unrelated bases. That needs a learned
alignment (Procrustes or an adapter), and any positive result is then confounded by the adapter's
capacity. This is the version a reviewer will ask about.

Cost: ~1 hour of compute for all three.

---

## Future work (out of scope — see D19)

- Rung 2, splice sites: GENCODE donor/acceptor/branch-point/polypyrimidine, spacing nulls
- Rung 3, G-quadruplex: G4-seq maps, G-density-matched decoys, `gfm-nano-dna` + DNABERT-2 anchor
- H3 / JSAE: demoted to exploratory. D16 stands as a recorded caveat: Jacobian sparsity measures
  in-degree, so it separates composition from a *bag* but not from a lone shortcut
- SSMs: Caduceus, Evo 2. Causal scrubbing. Enhancer/TF grammar

---

## Environment

```bash
uv sync --extra lab --extra dev   # torch 2.12, MPS; transformers 4.44
uv run pytest -q
```

M5 Pro, 20 cores, 68 GB unified memory. Transcoder sizing for SpliceBERT (6L × d=512):
4096 features ≈ 44M params; full CLT at 16384 features ≈ 175M. Both fit.
