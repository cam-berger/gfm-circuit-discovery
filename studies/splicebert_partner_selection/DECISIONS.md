# Decisions

Append entries; do not rewrite history.

## 2026-07-10 — Initial implementation

- Chose partner-aware internal-exon acceptor–donor selection as the first real-GFM target.
- Chose official SpliceBERT as the primary checkpoint.
- Registered `additive-frozen` as the primary mechanistic condition.
- Classified bilinear readouts as diagnostic because they can synthesize interactions.
- Reserved chromosome-held-out test data for one-time confirmatory evaluation.
- Classified endpoint masking and exon-core shuffling as instrumentation, not sufficient partner-specific proof.

## 2026-07-11 — Behavioral gate: GO

- Fixed the `additive-random-frozen` control (transformers ≥5 `init_weights()` no-op → `from_config`); this converted a spurious NO-GO into a true GO. All 5 control seeds re-run.
- Ran the 25-run matrix on Colab GPU with scale caps (train 12k / val 4k / eval 4k); gates, controls, and metric unchanged.
- **Behavioral gate met (recommended CALL: GO):** additive-frozen conflict PMR 0.807 vs randomized-backbone control 0.056 (non-overlapping, 5/5 seeds); gene-grouped bootstrap margin-difference CI excludes 0 on every seed (worst CI low 2.84); conflict PMR ≥ 0.60 and replication 5/5 both met. Interaction readout adds nothing over additive → partner info is linearly accessible in frozen states.
- Froze the preregistration + method + config + control-fix script + gate analysis (hashes in `FREEZE.md`). Test split (chr19–chr22) remains sealed.
- **Human elected to STOP before circuit tracing** for review; the mechanistic stage is not yet authorized (it opens new compute and must freeze the component set first).
- Scope discipline: this stage establishes decodable, pretraining-derived partner information — not yet a localized circuit the model causally uses. That is the tracing stage's job.

## 2026-07-11 — Interpretive reframe: "conflict" is an identifiability device, not the phenomenon (C4 working theory)

Context: seed-0 tracing landed and we stepped back from the conflict framing. This entry
records the shift in argument — it does **not** rewrite the frozen behavioral gate (C1–C3,
which stand as run) nor C4's pre-registered advancement bar. It is the evolving proposal
narrative, logged as it develops.

- **Seed 0 (n=1 of 5, NOT a gate call).** Necessity on the held-out confirmation split V_c:
  convergent head-set ablation effect **0.644** vs matched-random mean **0.068**,
  `p_add_one = 0.000999` (the floor of a 1000-set null — zero random sets reached it).
  Activation patching localizes: `site-windows-only` recovery **0.763**. Specificity:
  conflict effect **0.644** < matched-ordinary effect **1.289**, so `conflict_gt_ordinary =
  false`. AtP shortlist clusters in **L2–L4** (top `4-0`, `2-13`, `3-4`).
- **Reframe (working theory).** The conflict set was always an *identifiability device*: on
  examples where the local PWM+length shortcut points at a decoy, any above-baseline
  performance is forced to use relational information. It is a hand-built shortcut-insufficiency
  zone (aligns with the repo's shortcut-gap hypothesis), **not** evidence of a dedicated
  "conflict-resolution" module. The seed-0 `specificity=false` is therefore expected, not a
  failure: the discovered heads are **general partner-selection machinery** the conflict probe
  merely makes visible. C4's working theory is downgraded from "conflict-specific circuit" to
  "**localized partner-selection circuit that the conflict set exposes**." Final wording awaits
  ≥3/5-seed replication (is `specificity=false` stable across seeds, or noise?).
- **Conflict strength is a continuum, not a binary** (`scripts/splicebert/conflict_strength.py`
  → `behavior/conflict_strength.json`, over all 10,034 conflicts; 0 unexpected true-wins).
  Baseline decoy-preference in log-score units: median **1.59**, mean **2.02**; **33% below
  1.0**, **60% below 2.0**, ~9% near-ties (<0.25); broad with a real hard tail (p90 **4.38**,
  max **13.5**). Consequence for the write-up: "conflict" lumps genuine hard cases with
  barely-preferred decoys, so PMR 0.807 should be characterized *across* the strength range
  (a natural follow-up: stratify necessity/PMR by conflict-strength bin — dose-response), not
  reported as a single binary-conflict number. This justifies demoting conflict to Methods.

## 2026-07-12 — v1 and v2 retracted; pivot to cross-site coupling

**The forced-choice partner task is dead, and the reason is biological, not technical.**

- **v1 retracted (R2).** Two independent fatal confounds, both verified against the real
  data. The window was cut *relative to the label*, so the true acceptor sat at position
  200 in **100.0%** of examples and a **geometry oracle reading zero nucleotides scores
  PMR 1.000**. And the decoys were spurious AG/GT motifs, so an **oracle site detector
  also scores 1.000** — the task never required partner information at all. The
  randomized-backbone control catches neither.
- **v2 retracted (R3).** Real-site decoys turned out to be *minor* sites (median 1
  supporting isoform vs 18). "Pick the partner" became "pick the better donor"; a plain
  PWM scores 0.687, and the bilinear readout beat the additive control by **+0.008**.
- **Why no third attempt.** In a real genome, a *strong* donor at a *plausible* exon
  distance from an acceptor generally **is** its partner. Clean negatives barely exist,
  so partner identity is nearly determined by site strength + distance and a
  forced-choice task has almost no residual to expose. This is not fixable by better
  decoy engineering.

**The replacement asks for a causal dependence instead of a task score** — no candidate
list, no label, no decoy, hence nothing to leak. Native MLM readout; perturbations
substitute real bases (never `[MASK]`, which is off-manifold).

Established on train (D1–D3): non-local dependence (+0.380 partner vs −0.000 for a
motif-matched non-partner AG), partner-selectivity (own donor +0.380 vs other donors
+0.003 in the same window, paired Wilcoxon p = 9.3e-18), bidirectional.

**We retracted our own first interpretation.** The sharp decay of coupling with exon
length looked like exon definition; it was **distance decay**. The distance-matched
exon-vs-intron contrast showed coupling is **4–7× stronger across the intron** — the
model represents the *intron* (the spliced-out unit), not the exon. Hard limitation:
only short introns fit the 1024nt context, so this cannot be generalised to typical
(long) human introns.

**D4/D5 are NOT reportable yet.** L3H1 was *discovered* on train at n=48 with 60 null
sets; quoting its effect on that same data is circular. `run_confirm.py` re-runs the
whole ladder on gene-disjoint held-out chromosomes with 1000 null sets, fitting the
subspace direction on train and scoring it on val. Test (chr8, chr9) stays SEALED.

---

## 2026-07-12 — the confirmation run landed. D4/D5 promoted; D4's "dedicated" retracted (R4).

`run_confirm.py` executed the full ladder on gene-disjoint held-out chromosomes
(chr11/16/19, n = 400, 1000 null sets), fitting the subspace direction on train and
scoring it on held-out. Artifact: `results/splicebert_coupling/confirmation.json`.

**It replicates, and it is not circular.** L3H1 was selected on train; on data it never
saw it ranks **1st of 96** heads, recovers **0.808** of the coupling alone (the next head
manages 0.186), and beats 1000 matched-random single-head sets (observed 1.193 vs null
mean 0.012, max 0.794 → **p_add_one = 0.000999**, the floor). The partner-dependent
factorial contrast is **1.110 [1.018, 1.205]**. A rank-2 projection of the head's output
removes 80% of the coupling where a random rank-2 removes 2%. MLM health barely moves
under ablation (0.9958 → 0.9953), so the head is not simply breaking the model.

This is the program's **first held-out-confirmed circuit in a pretrained genomic model**,
and it comes from the design we adopted *because* the previous one was leaking.

**And the same run killed half of D4.** We had written that L3H1 is *dedicated* to the
coupling, on the strength of a train-split partner-independent role of −0.005 (n_eval=50).
Held-out, that role is **−0.200 [−0.272, −0.128]** — the CI excludes zero. The head carries
the coupling; it does not carry only the coupling. Retracted as **R4** rather than quietly
reworded, because the flattering number was the one from the discovery split, which is
exactly the failure mode this study already paid for once.

**Claim label: `CIRCUIT_CANDIDATE`.** Not `CAUSALLY_VALIDATED_CIRCUIT` — we have necessity
and mediation, but no path account (QK/OV decomposition, edge patching), only one ablation
null, and an unexplained 19% residual. Those three are the next work. Test stays SEALED.

## 2026-07-12 — Mechanism: a conjunctive, content-addressed read (and two retractions)

The result is now **mechanistic**, not merely localisation. See `MECHANISM.md`.

**Established (held-out, gene-disjoint):** L3H1 implements an acceptor-side,
content-addressed read. Key alone recovers 0.035, value alone 0.104, **key+value 0.711** —
87% of the 0.819 recovered by patching the whole head output. Paired factorial interaction
**I = +0.573 [+0.522, +0.624]** (per-example, gene-grouped). Acceptor corruption collapses
L3H1's attention to the partner by **95%**. The acceptor must be both *findable* (key) and
*informative* (value); neither channel is worth anything alone.

**L2H15 is an upstream feeder** (92% nested inside L3H1; zero patch-recovery). Route not yet
causally established — nesting + cosine is inference. The K/V rescue is the pending test.

### Two retractions (permanent: R4, R5)

- **R4 — "single dedicated head, zero local role" is dead.** The −0.005 partner-independent
  estimate was *noise at n=50*; at n=400 it is **−0.200 [−0.272, −0.128]**, CI excluding
  zero. And the **exhaustive ablation scan** — which the denoise ranking had missed
  entirely — found **L2H15 at 0.542**. L3H1 is the largest contributor, not the circuit.
- **R5 — "33% of causal mass" was statistically invalid.** Those were summed *marginal*
  ablation effects, which double-count serial/redundant components. Directly falsified:
  0.944 + 0.542 = 1.486 > the 1.121 baseline. Never report summed marginals as shares.

### Methodological corrections that cost us real conclusions

- **Denoise ≠ ablation.** L2H15 is necessary (0.542) but *invisible* to denoising. Inferring
  the second component from the denoise ranking would have pointed at L4H2 and missed it.
  Sufficiency-under-restoration and necessity-under-removal must be measured separately.
- **The matched-random head "null" was never a null.** If other heads belong to the circuit,
  its heavy tail is *mechanism*, not noise. Replaced by (a) the exhaustive causal
  distribution with IDs and (b) a per-head **specificity** null: the same ablation scored on
  a distance-matched NON-site.
- **`query@donor = 0.003` is a null about the corruption, not the query.** Acceptor
  corruption never perturbed the donor query, so a clean-query patch had nothing to restore.
  Query *necessity* is untested (D8). Do not cite 0.003 as evidence the query is unimportant.

### The two-base bug, and the permanent safeguard

The acceptor's AG sits at `[a-2, a-1]`; the corruption destroys exactly those bases. The K/V
patch was applied at `a` — a position the corruption never touched. Recovery: **0.037**,
which reads as *"the acceptor's key and value barely matter, so the effect enters
elsewhere."* Patching the window that actually changed gives **0.711**. **Same code, same
data, opposite mechanistic conclusion — two bases.**

Now enforced, not reviewed (`src/gfmcd/circuits/guards.py`, `tests/test_patch_window_guard.py`):
1. every patch window must **overlap the corruption mask**, or it raises;
2. the **exact token indices** are recorded in provenance (an off-by-two is invisible in a
   mean and obvious in a recorded index);
3. **adjacent-window controls** are generated automatically, and a "control" that overlaps
   the corruption raises too.
