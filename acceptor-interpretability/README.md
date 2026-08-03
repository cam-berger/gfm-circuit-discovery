# Acceptor-Recognition Circuit — Extension to the SpliceBERT Proposal

Traces the reciprocal, unmapped half of the SpliceBERT donor-circuit account:
how the model recognizes the 3' **acceptor** from the branch point (BP) and
polypyrimidine tract (PPT). Motivated by the donor circuit's own result that
the BP/PPT region contributes ~0 to the *donor* payload — this project asks
where and how those 3' signals are actually used. The acceptor representation
this circuit builds is the payload the donor circuit currently takes as
given, so it slots upstream of L3H1 without disturbing that result.

Two metrics mirror the donor circuit's `m(x)`:

- **`m_acc`** — masked-AG logit margin: mask the acceptor's AG dinucleotide,
  score the model's confidence for the true base at each position against
  the other three nucleotides, sum the two per-position margins.
- **`Δppt`** — the drop in `m_acc` when the BP/PPT region is corrupted,
  against a matched non-splice control.

## Data: GUE `splice_reconstructed`

Phase 0 uses real human 3'/5' splice-site windows from
[`leannmlindsey/GUE`](https://huggingface.co/datasets/leannmlindsey/GUE)
(config `splice_reconstructed`, itself GENCODE/DeepSplice-derived) instead of
synthetic windows. Every row is a 400nt sequence with the splice-relevant
dinucleotide fixed at index 200 by construction:

| label | meaning | consensus |
|---|---|---|
| 0 | acceptor | `AG` at `[198, 200)` |
| 1 | donor (unused in phase 0; reserved for the shared-transport check) | `GT` at `[200, 202)` |
| 2 | non-splice negative | no consensus guaranteed at 200 |

Loaded directly via `pandas.read_csv` on the dataset's resolved CSV URL — no
`datasets` library dependency.

### Coordinates (`configs/experiment.yaml: coords`)

BP and PPT boundaries are **literature-typical positional approximations**
relative to the fixed AG at 198–200 (GUE carries no per-sequence
branch-point annotation — real coordinates are next-steps item 5):

- PPT: `[183, 198)` — ~3–17nt upstream of the AG
- BP: `[160, 183)` — ~18–40nt upstream of the AG

### Corruption method

Region corruption is **content-destroying**, not composition-preserving
shuffle: each corrupted base is redrawn from a balanced ACGU distribution
using a Markov process that provably cannot introduce an `AG` substring
anywhere in the region *or at either seam* with the flanking real sequence
(`acceptors/data.py: random_no_ag`, `corrupt_region`). This is the method
next-steps item 1 flagged as necessary to distinguish a real gating effect
from a floor artifact — it's used as the default here, not just in a later
decisive run.

The **random-AG** control destroys the *entire* window this way and forces
`AG` back in at the true position (bare AG, zero genuine context). The
**decoy AG** for the AG-scanning check is the first naturally-occurring `AG`
found strictly downstream of the true site (search window
`[201, 280)`) — testing whether the model prefers the correctly-positioned
site over an incidental AG rather than "any AG, however far downstream."

## Setup

```bash
pip install -r requirements.txt
python experiments/phase0_pilot.py
```

CPU-only (`configs/model.yaml: device: auto`); ~10 minutes for
n=200+200 samples on a CPU-only torch install. `phase0_colab.ipynb` runs the
same pipeline standalone on a GPU Colab runtime for larger sample sizes.

Writes `results/phase0_results.json` (summary + bootstrap CIs) and
`results/phase0_per_sample_{acceptor,negative}.csv` (raw per-condition
`m_acc` values, reusable by the 2×2 factorial in next-steps item 1 without
re-running the model).

## Phase 0 results (real data, n=200 acceptor / n=200 non-splice, 5000-resample bootstrap)

| comparison | Δ (mean) | 95% CI | acceptor-specific? |
|---|---|---|---|
| BP disruption alone | **+0.212** | [+0.061, +0.360] | yes — control: −0.105 [−0.201, −0.027] |
| PPT disruption alone | **+1.245** | [+0.875, +1.624] | yes — control: +0.046 [−0.171, +0.267] |
| BP+PPT both disrupted | **+1.386** | [+0.999, +1.775] | yes — control: −0.145 [−0.438, +0.125] |
| context required (clean − random-AG) | **+1.319** | [+0.919, +1.739] | — |
| AG-scanning (true − downstream decoy) | **+0.916** | [+0.478, +1.362] | — |

Three findings replicate the synthetic pilot directly:

- **Context is required.** A bare AG scores far below a real acceptor
  window — the largest effect measured.
- **AG-scanning is supported.** The true site clearly outscores a
  downstream decoy AG found in the same sequence.
- **BP disruption lowers `m_acc`, and it's acceptor-specific** — the
  matched non-splice control is an order of magnitude smaller (and
  slightly *negative*, not flat — see caveat below).

**One finding does not replicate, and it's the interesting part: PPT
disruption alone has a large, clearly acceptor-specific effect here**
(+1.245, CI excludes 0; control CI straddles 0), not the "does ~nothing"
result the synthetic pilot reported. BP+PPT together (+1.386) is close to
PPT-alone plus BP-alone (1.245 + 0.212 = 1.457) — i.e. **roughly additive,
not the AND-gate signature** (both-disrupted collapsing back toward zero)
the synthetic pilot's gating hypothesis predicted.

Read together with the donor circuit's original result — BP/PPT contributes
~0 to the donor payload, but PPT alone clearly drives the acceptor payload —
the simplest updated account is: **the polypyrimidine tract is the dominant,
largely independent driver of acceptor confidence** (consistent with U2AF65
binding the tract directly), with the branch-point region contributing a
smaller, roughly additive increment, rather than acting as a signal that's
only legible through an AND-gate with the tract. Whether this reflects a
real biological difference or an artifact of the cruder composition-preserving
corruption used in the original synthetic pilot (vs. the content-destroying
method used here) is exactly the question next-steps item 1 was designed to
resolve — this run answers it, using real rather than synthetic sequences.

## Phases 1-3: control fix, absolute floor, real coordinates

Phase 0's numbers above turned out not to be fully trustworthy on their own -
the non-splice control wasn't flat, and a Delta alone can't tell a real
interaction from a corruption method that's already hit its floor. Phases
1-3 (`experiments/phase{1,2,3}_*.py`, `visualization/visualization_{1,2,3}.py`)
resolve that, reusing the identical n=200 acceptor sample throughout (same
seed) so every number below is comparable to Phase 0's.

### Phase 1 — is the control clean, and is the tract effect composition or order?

`python experiments/phase1_control_fix.py` → `results/phase1_results.json`,
`results/figures/visualization_1.png`

| control | Δ | 95% CI | reads zero? |
|---|---|---|---|
| Far-intronic (content-destroying, deep in intron) | +0.046 | [+0.011, +0.085] | technically no, but ~5-27x smaller than every real effect - this is the noise floor |
| PPT shuffle-in-place (composition preserved) | **+1.207** | [+0.897, +1.530] | **no - nearly as large as full content-destroying corruption (+1.245)** |
| BP shuffle-in-place (composition preserved) | +0.168 | [+0.041, +0.305] | no, but small |

Verdict on the Phase 0 effects against the matched far-intronic floor:

| effect | Δ | vs. floor |
|---|---|---|
| Context required | +1.319 | SURVIVES |
| AG-scanning | +0.916 | SURVIVES |
| PPT-only disruption | +1.245 | SURVIVES |
| BP-only disruption (Δppt) | +0.212 | **MARGINAL - CI overlaps the floor's CI** |

Two outcomes, one expected and one not:

- The far-intronic control is *much* smaller than every real effect
  (0.046 vs. 0.9-1.3), so context/AG-scanning/tract are unambiguously real.
  It isn't exactly zero, though - there's a small, generic ~0.05-unit
  corruption bias anywhere in the sequence, which is now the honest noise
  floor rather than an assumed zero.
- **PPT shuffle-in-place was expected to read ~zero (composition preserved,
  same bases, same position) and instead came back at +1.207 - almost the
  full content-destroying effect.** That overturns the "tract acts by bulk
  pyrimidine density" reading: reordering the same bases already destroys
  almost the whole signal, so whatever SpliceBERT is reading in that window
  is order/identity-sensitive (a specific pattern or precise position-level
  dependency), not aggregate composition. This is a finding in its own
  right, not a control that needs fixing.
- The branch-point effect (+0.212) sits right at the edge of the corrected
  floor's CI - the *better* control didn't clear it, it just replaced "looks
  real" with "unresolved."

### Phase 2 — absolute m_acc vs. the random-AG floor (gate vs. artifact)

`python experiments/phase2_absolute_floor.py` → `results/phase2_results.json`,
`results/figures/visualization_2.png` (no model re-run - reuses Phase 0's
per-sample CSV, which already stored absolute `m_acc`, not just deltas)

| condition | absolute m_acc | 95% CI | vs. floor |
|---|---|---|---|
| Clean | +0.655 | [+0.254, +1.072] | - |
| BP disrupted | +0.444 | [+0.037, +0.855] | **clearly above floor** |
| PPT disrupted | -0.590 | [-0.842, -0.331] | **at floor** |
| BP+PPT both disrupted | -0.730 | [-0.935, -0.515] | **at floor** |
| random-AG floor | -0.664 | [-0.728, -0.595] | - |

This resolves the gate-vs-floor question the delta-only view couldn't:
**PPT disruption alone is already enough to crash `m_acc` down to the same
level as having no real context at all** (random-AG floor). BP-only leaves
real signal well above the floor. So "both disrupted" collapsing to the
floor isn't evidence of a BP-PPT interaction (an AND-gate) - PPT alone
already gets there. The mechanistic picture this supports is simpler than a
gate: **the tract is close to sufficient on its own for the model's acceptor
signal; the branch point contributes a smaller, dispensable increment on
top of it.**

### Phase 3 — relocate BP/tract per sequence instead of a fixed offset

`python experiments/phase3_real_coordinates.py` → `results/phase3_results.json`,
`results/figures/visualization_3.png`

Per-sequence branch-point localization via a `yUnAy`-consensus scan
(`acceptors.data.locate_branch_point` - see "Known approximations" below for
what this is and isn't) over the canonical 18-45nt upstream search window,
with the tract then defined as everything between the located branch point
and the AG (so tract length now varies per sequence: mean 27nt, vs. the
fixed 15nt window Phase 0 used). Only 2/200 sequences had no candidate
branch-point adenosine in the window.

| effect | fixed-offset (Phase 0) | real-coordinate (Phase 3) |
|---|---|---|
| BP-only disruption | +0.212 [+0.061, +0.360] | **+0.090 [-0.007, +0.186] - CI now includes zero** |
| PPT-only disruption | +1.245 [+0.875, +1.624] | +1.433 [+1.076, +1.802] - larger, not smaller |
| BP+PPT both disrupted | +1.386 [+0.999, +1.775] | +1.498 [+1.083, +1.914] |

Better-located coordinates did **not** rescue the branch-point effect - if
anything it gets weaker and loses significance entirely, while the tract
effect strengthens further. This rules out "the branch-point window was
just mislocated" as the explanation for the earlier small effect. It does
**not** rule out that the branch point matters and this PWM-style locator
simply isn't finding the real one (see caveat below) - but the burden has
shifted: there is now no positive evidence of a standalone, legible
branch-point signal in this model, only a shrinking, marginal one.

### Phase 4 — head content-control, retargeted to tract-readers

`python experiments/phase4_head_content_control.py` → `results/phase4_results.json`,
`results/phase4_per_sample.csv`, `results/figures/visualization_4.png` (reuses
the identical n=200 acceptor sample; adds attention extraction on top of the
existing masked-AG forward pass, so it's directly comparable to Phases 0-3).

Next-steps item 4 asked whether the six candidate heads carried over from the
donor-circuit proposal (`L0H2, L1H14, L2H7, L2H0, L3H10, L1H1`) are genuine
content readers of the BP/PPT regions, or just attend there by positional
locality (both regions sit immediately upstream of the masked-AG query, so
any generic "attend to the last N tokens" head would look like a reader
without this check). For each head this masks the AG exactly as `m_acc`
does, reads that query position's attention into the region, and checks
whether the mass **drops** when the region's content is destroyed
(content-destroying corruption, as before) or scrambled (shuffle-in-place -
included because Phase 1 showed the tract's `m_acc` effect is order-driven,
so a genuine tract reader should be order-sensitive too, not just
composition-sensitive).

A first pass (the 6 candidates only) also recorded, from the same forward
passes at no extra cost, every (layer, head)'s clean-condition attention
mass into each region - a ranking, not a re-run - to check whether the
candidate list still covers the heads actually attending there. It didn't:
the top-10 PPT-attenders were almost entirely the same 6 candidates (plus
near-duplicates), but the **top-10 BP-attenders were a completely different,
non-overlapping set of heads**, none of which were on the original list. A
second run added those 15 discovered heads (10 top BP-attenders + 5 top
PPT-attenders not already covered) to the same script, for 21 heads x 2
regions = 42 tests total.

**Result: 0/6 donor-circuit candidates pass content-control, at either
region.**

| head | PPT clean mass | PPT verdict | BP clean mass | BP verdict |
|---|---|---|---|---|
| L0H2 | 1.000 | positional-locality artifact | ~0 | no attention |
| L0H14 | 1.000 | positional-locality artifact | ~0 | no attention |
| L1H14 | 0.999 | positional-locality artifact | ~0 | no attention |
| L2H7 | 0.993 | positional-locality artifact | ~0 | no attention |
| L2H0 | 0.968 | positional-locality artifact | 0.009 | no attention |
| L1H1 | 0.903 | positional-locality artifact | 0.028 | no attention |
| L3H10 | 0.815 | positional-locality artifact | 0.041 | no attention |

All six spend 80-100% of the masked-AG query's entire attention budget on
the 15nt immediately upstream of it (the PPT window) - but that mass is
statistically unchanged (sometimes even *higher*) when the region's content
is replaced with random bases. This is the textbook signature of a
previous-token / local-window attention head, not a tract reader: it looks
there regardless of what's there. None of the six attend meaningfully to
the (more distant) BP region at all.

**Among the 15 discovered heads, five show genuine, statistically robust
content-sensitive attention into the PPT region** (delta = clean − corrupted
attention mass, 95% CI excludes 0 and is positive - attention really does
drop when content is destroyed):

| head | clean PPT mass | Δ(clean−corrupt) | Δ(clean−shuffle) | order-sensitive? |
|---|---|---|---|---|
| L5H8 | 0.076 | +0.0223 [0.017, 0.027] | +0.0080 [0.005, 0.011] | yes |
| L4H12 | 0.063 | +0.0119 [0.008, 0.016] | +0.0032 [0.0002, 0.006] | yes |
| L3H5 | 0.081 | +0.0087 [0.0045, 0.013] | +0.0023 [-0.001, 0.006] | no (CI incl. 0) |
| L0H13 | 0.383 | +0.0009 [0.0004, 0.0014] | +0.0003 [-0.0001, 0.0008] | no (CI incl. 0) |
| L0H14 | 1.000 | +0.00001 [0.0000075, 0.0000142] | +0.00001 [0.0000088, 0.0000145] | yes, but negligible in magnitude (already saturated) |

L5H8 and L4H12 are the clearest finding: real, order-sensitive drops (not
just composition), consistent with Phase 1's order/identity-driven tract
signal - the first positive evidence in this project of heads that plausibly
*read* the tract rather than merely sit near it. Their clean attention mass
(6-8% of budget) is far below the saturated positional-lock heads (80-100%),
so they are not the dominant attention sink at that query position, but the
content-sensitivity is real and well-powered (n=200, 5000-resample
bootstrap).

**Zero of the 21 tested heads pass content-control for the branch-point
region** - not the 6 original candidates, and not the top-10 heads
independently ranked by how much raw attention they send there. Several of
those top BP-attenders (`L2H12`, `L5H8`, `L5H14`) even show attention
*increasing* under corruption or shuffle (CI excludes 0 in the wrong
direction) rather than dropping - the opposite of a content-reading
signature. Read together with Phases 1-3, this closes out next-steps item 4
for the branch point specifically: there is now no positive evidence of a
legible, content-gated BP-attending circuit at all, by either the `m_acc`
route (Phases 1-3) or the attention route (Phase 4) - only the tract has one.

### Phase 5 — the intervention ladder: necessity, mediation, routing vs. content

`python experiments/phase5_intervention_ladder.py` → `results/phase5_results.json`,
`results/phase5_per_sample.csv`, `results/figures/visualization_5.png` (same
n=200 sample; adds causal interventions - mean-ablation, restoration
patching, and the query/value split from path patching - on top of the
Phase-4 attention readout, all via PyTorch forward hooks on
`SpliceBertSelfAttention`, numerically verified against a self-patch
no-op check before running at scale).

Phase 4 found heads whose *attention* tracks tract content. That's a
behavioral signature of the head's own softmax, not proof the head's output
actually moves `m_acc`. This phase runs the standard causal ladder
(mean-ablation over zero-ablation, to keep patched components on-distribution;
donor-sequence restoration patching for mediation; Q-patch/V-patch - Wang et
al. 2022, Heimersheim & Nanda 2024, Goldowsky-Dill et al. 2023 - for routing
vs. content) on `L5H8`/`L4H12` (the order-sensitive hits), `L3H5`/`L0H13`
(ablation only), and a negative control, `L0H2` (Phase 4's most
attention-saturated donor-circuit head - 0/6 that passed content-control).

**Step 1 - mean-ablation (necessity):** replace each head's output at the
masked-AG query positions with its dataset mean, on the clean sequence.

| head | Δ m_acc (clean − ablated) | 95% CI | significant? |
|---|---|---|---|
| L5H8 | -0.004 | [-0.015, +0.007] | no |
| L4H12 | +0.001 | [-0.009, +0.012] | no |
| L3H5 | -0.014 | [-0.030, +0.0003] | no (edge) |
| L0H13 | +0.003 | [-0.005, +0.011] | no |
| L0H2 (control) | -0.015 | [-0.031, +0.0003] | no (edge) |
| joint: 4 readers together | -0.015 | [-0.041, +0.010] | no |
| joint: all 5 together | **-0.030** | **[-0.060, -0.002]** | **yes, barely** |

**Zero of the five heads individually pass necessity** - every CI straddles
0, including the control. Jointly ablating the four real readers together
is still not significant. Jointly ablating all five (adding back the
control) barely clears 0 - but at −0.030, that's about **2% of the size of
the real PPT-corruption effect (+1.2 to +1.4)**, confirming these five heads'
direct output at the AG position, even summed, are nowhere near sufficient
to explain the tract effect on their own.

**Step 2 - restoration patching (mediation):** on the PPT-corrupted (and
separately, shuffled) sequence, patch in the *same sequence's* clean-run
output for one head only; recovery fraction = how much of the
clean-vs-degraded `m_acc` gap that one patch restores.

| head → recipient | recovery fraction | 95% CI |
|---|---|---|
| L5H8 → corrupted | +0.1% | [-0.6%, +0.8%] |
| L5H8 → shuffled | -0.2% | [-0.7%, +0.3%] |
| L4H12 → corrupted | +0.3% | [-0.3%, +0.9%] |
| L4H12 → shuffled | +0.0% | [-0.2%, +0.4%] |

**All four recovery fractions are indistinguishable from 0%.** Patching a
single head's true output back in, even for a head Phase 4 showed reads the
tract's content, recovers essentially none of the corruption effect.

**Step 3 - Q/V split (routing vs. content):** still on the corrupted
recipient, Q-patch feeds the head clean routing (attention pattern) over
corrupted values; V-patch feeds it corrupted routing over clean values.

| head | Q-patch (routing) recovery | V-patch (content) recovery |
|---|---|---|
| L5H8 | -0.1% [-0.4%, +0.2%] | -0.2% [-0.8%, +0.5%] |
| L4H12 | -0.2% [-0.4%, +0.03%] | +0.2% [-0.4%, +0.8%] |

Both are null too - there's no effect to decompose into routing vs. content
components when the whole-head patch (step 2) was already null.

**Verdict: single- and small-group causal necessity is a clean null,
contradicting a naive reading of Phase 4.** These heads' attention genuinely
tracks tract content (real, well-powered, order-sensitive), but that
correlation doesn't cash out as a large individual causal channel to
`m_acc` through this one layer's direct output at the AG position. Read
plainly, this means the tract-reading computation is **not concentrated in
a handful of identifiable heads** - it's either (a) genuinely distributed
across many more components than these five (a "hydra effect" - ablate one
contributor, others compensate or were never dominant to begin with), or
(b) routed indirectly (a head could write useful tract information to a
position other than the AG query, which a later layer then reads into the
AG representation - invisible to an intervention that only touches this
one layer's AG-row output). Phase 4's attention-based filter was still the
right first move (it's what separated these five from the six saturated
donor-circuit heads), but attention content-sensitivity and causal
importance are evidently different questions in this model - the former
does not imply the latter at the single-head level.

### Where this leaves the mechanistic headline

Steps 1-5 are done, and they converge on one story rather than clearing the
way for the branch point:

- Context and AG-scanning are robust, unambiguous, and unaffected by any of
  this - **GO** on the overall direction.
- **The polypyrimidine tract is the dominant, close-to-sufficient driver**
  of acceptor confidence, and the signal looks order/identity-driven rather
  than bulk composition.
- **Five heads (`L5H8`, `L4H12`, `L3H5`, `L0H13`, `L0H14`) show genuine,
  content-sensitive attention into the tract** - a first positive,
  behavioral hit on the tract-reading circuit; `L5H8`/`L4H12` are
  order-sensitive, matching the `m_acc` signal's own order-dependence. None
  were on the original candidate list.
- **That content-sensitivity does not translate into individual causal
  importance.** Phase 5's intervention ladder found a clean null: no single
  head (including a saturated-attention negative control) passes
  necessity or mediation, and even jointly ablating all five barely clears
  significance at ~2% of the real tract-corruption effect's size. The
  tract-reading computation is genuinely distributed across more
  components than these five, routed indirectly through positions this
  layer-local intervention can't see, or both - Phase 4's attention filter
  was the right first move, but attention content-sensitivity ≠ causal
  necessity at the single-head level in this model.
- **The branch-point-alone effect has not survived four independent
  attempts to firm it up** (a matched floor, an absolute-vs-floor test,
  per-sequence relocation, and now a full attention-based content-control
  sweep across 21 heads, not just the original 6) - it should be treated as
  unresolved/likely weak in this model, not as a secondary-but-real effect,
  pending either a real branch-point annotation (branchpointer/LaBranchoR
  proper, not the PWM stand-in here) or a larger sample.
- The original synthetic pilot's AND-gate framing is not supported: Phase 2
  shows PPT-alone already reaches the floor, which is sufficient on its own
  to explain "both disrupted" without invoking an interaction.
- The six donor-circuit candidate heads themselves are a **dead end**: all
  six are saturated positional-locality heads (80-100% of their attention
  budget on the region regardless of content) - none read content at either
  region, and even the five Phase-4 replacements don't individually or
  jointly explain much of `m_acc`. The circuit is more distributed (or more
  indirect) than a small head list can capture - next-steps item 6 now
  targets a broader search (more heads, MLPs, and non-AG-position paths),
  not a fixed candidate list.

### Known approximations / caveats

- **The original GUE `label==2` non-splice control (used in Phase 0) is
  superseded by Phase 1's far-intronic control** - it wasn't reliably
  intronic (label 2 just means "not a splice site," not "matched genomic
  context"), which is the likely reason it read a small but significant
  −0.105 instead of zero. Treat Phase 0's non-splice-control numbers as
  historical; the far-intronic control is the one to trust going forward.
- **The branch-point locator (`acceptors.data.locate_branch_point`) is a
  hand-set `yUnAy`-consensus PWM scan, not branchpointer or LaBranchoR.**
  Those tools need either genomic coordinates GUE doesn't carry or a
  separate pretrained model; this is a documented, self-contained stand-in.
  Phase 3's null result on the branch point is therefore a claim about
  *this locator's* candidate positions, not proof the branch point is
  irrelevant to the model - a real annotation (next-steps item 5) could
  still turn up a legible signal this heuristic misses.
- The decoy-AG check only uses samples where a downstream AG exists within
  the search window (not all 200 samples) — `n` for that comparison is
  reported in `results/phase0_results.json`.
- All of Phase 0-3 run on the same n=200 acceptor sample (fixed seed) for
  comparability. None of this has been checked for stability at larger n
  yet - CIs are wide enough (e.g. BP-only real-coordinate: [-0.007,+0.186])
  that a larger sample could still move the marginal calls either way.

## Layout

```
configs/                model.yaml (checkpoint/device), experiment.yaml (data/coords/phase0-3 params)
acceptors/               config.py, data.py (GUE + corruption/shuffle/BP-locator), model.py (SpliceBERT), metrics.py (m_acc), stats.py (bootstrap CI)
experiments/
  phase0_pilot.py         Phase 0: the six-condition pilot described above
  phase1_control_fix.py   Phase 1: far-intronic control + shuffle-in-place controls
  phase2_absolute_floor.py Phase 2: absolute m_acc vs. the random-AG floor (no model re-run)
  phase3_real_coordinates.py Phase 3: per-sequence BP/tract localization
  phase4_head_content_control.py Phase 4: attention-based head content-control + head discovery
  phase5_intervention_ladder.py Phase 5: mean-ablation, restoration patching, Q/V split
visualization/
  visualization_0.py       Phase 0 forest plot
  visualization_1.py       Phase 1: effects vs. floor + order-vs-composition panels
  visualization_2.py       Phase 2: absolute m_acc vs. floor
  visualization_3.py       Phase 3: fixed-offset vs. real-coordinate comparison
  visualization_4.py       Phase 4: clean attention mass vs. content-control delta, per head
  visualization_5.py       Phase 5: ablation / mediation / Q-V-split forest plots
results/                  phase{0,1,2,3,4,5}_results.json + per-sample CSVs + figures/
phase0_colab.ipynb        standalone GPU-runnable copy of Phase 0 for larger-n runs
```

`acceptors/` also has `attention.py` (per-head attention-mass readout, Phase
4) and `patching.py` (causal hooks - capture, replacement, mean-ablation,
Phase 5).

## Next steps

Steps 1-5 below are done (Phases 1-5 above); 6-7 proceed with the tract, not
the branch point, as the primary object, but now aim wider than the five
Phase-4 heads, since Phase 5 showed those don't individually or jointly
explain much of `m_acc`:

1. ~~Flatten the control~~ **done (Phase 1).** Far-intronic floor is
   +0.046 - two orders smaller than the tract effect, though not exactly
   zero (that residual is now the documented noise floor, not swept under
   "flat"). Bonus finding: PPT shuffle-in-place shows the tract signal is
   order-driven, not composition-driven.
2. ~~Absolute m_acc vs. the floor~~ **done (Phase 2).** PPT-only reaches
   the floor by itself; BP-only stays clearly above it. "Both collapses"
   is explained by PPT alone, not a BP×PPT interaction - the AND-gate
   framing is dropped.
3. ~~Re-locate BP/tract from content, not fixed offsets~~ **done (Phase
   3).** Real-coordinate placement makes the branch-point effect *smaller*
   and non-significant (CI now includes 0) while the tract effect grows -
   mislocation was not hiding a bigger branch-point signal.
4. ~~Head content-control, retargeted to tract-readers~~ **done (Phase
   4).** All 6 original candidate heads turned out to be saturated
   positional-locality heads (0/6 pass at either region). A ranking pass
   discovered 15 new candidate heads from actual clean-attention data (10
   top BP-attenders, 5 top PPT-attenders); of those, 5 pass content-control
   for the tract (`L5H8`, `L4H12`, `L3H5`, `L0H13`, `L0H14` - `L5H8`/`L4H12`
   order-sensitive). Zero heads, old or new, pass for the branch point.
5. ~~Intervention ladder~~ **done (Phase 5).** Mean-ablation, restoration
   patching, and the Q/V split all came back null for `L5H8`/`L4H12`
   individually; jointly ablating all 5 discovered heads barely clears
   significance, at ~2% of the real tract-corruption effect's size.
   Content-sensitive attention does not imply single- or small-group causal
   necessity in this model - the computation is more distributed, more
   indirect (routed through non-AG-query positions this layer-local
   intervention can't see), or both.
6. **Broaden the causal search**, now that a 5-head list is confirmed
   insufficient: (a) path-patch backward from `m_acc` itself (not from
   Phase-4's attention ranking) to let the model nominate which components
   actually matter, the way Wang et al.'s IOI methodology does, rather than
   filtering by attention first; (b) include MLP outputs and other token
   positions (a head could write tract information somewhere other than the
   AG query for a later layer to read), not just each head's own AG-row
   output; (c) a true whole-circuit scrub (ablate the full attention layer
   or several layers jointly) as an upper bound on how much *any* causal
   story could explain, to calibrate how far off 2% actually is. A legible
   branch-point component remains a possible but unsupported opportunistic
   finding, not the headline.
7. **Shared-transport question** — does whatever *does* turn out to mediate
   the tract effect (once item 6 finds it) reuse the donor spine's transport
   heads (L4H0, L4H2, L5H11) and a `U_info`-like subspace, or run on its
   own? Use GUE's `label==1` (donor) rows, already loaded by the same
   pipeline - deferred until there's a mediating component to ask the
   question about.
8. **Real branch-point annotations** (branchpointer / LaBranchoR / Leman et
   al.) in place of the PWM-consensus stand-in used in Phase 3, plus
   gene-disjoint held-out splits - the one thing that could still revive a
   standalone branch-point finding, now that the `m_acc` route (Phases 1-3)
   and the attention route (Phase 4) have both come up empty for it.

**One-line status:** real data is a GO on the overall direction (context
and AG-scanning are robust); the mechanistic headline has moved from branch
point to polypyrimidine tract; Phase 4 found heads whose *attention*
genuinely tracks tract content, but Phase 5's intervention ladder found
that content-sensitivity doesn't cash out as individual or small-group
causal necessity - the real circuit is more distributed or more indirect
than five heads, and finding it is now the open problem; and the
branch-point-alone effect has not survived four independent checks - treat
it as unresolved, not secondary-but-real, until a proper branch-point
annotation says otherwise.
