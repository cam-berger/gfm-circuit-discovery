# Methods audit — what I did wrong, why, and the gate that stops it

**Written 2026-07-13, after the third retraction of the session, with no code running.**
Nothing new gets measured until this is agreed.

---

## 1. The ledger

Every instrument failure this session, and how it was caught. The "caught by" column is the
important one.

| # | What broke | Cost | **Caught by** |
|---|---|---|---|
| 1 | `run_synth_quantization.py` rewrote its whole output file each run, **destroying the pilot D26 cites** | D26's artifact is gone; its numbers survive only in a console log | reading the JSON, by luck |
| 2 | The stage-P "control" had **no `shuffle_mod`** — it regenerated the *treatment* corpus | A bit-identical duplicate sat in the record looking like a control that agreed | **two runs matching to the last decimal** |
| 3 | The same clobbering bug **again** — a stale in-memory dict erased my VOID marking | A void run silently un-voided | **the tripwire test, 30 ms after I wrote it** |
| 4 | Splice calibration injected a 5% tone, recovered it, and I called that "calibrated" | 18 of 32 discovery peaks landed **exactly on a scan boundary** | **looking at where the peaks landed** |
| 5 | Scanned wavelengths with a **51-nt window** (resolution ΔT ≈ 4 nt) over a range needing 0.3 nt | Reported "interior peaks cluster near 10.3" — **retracted** | computing the resolution, *after* reporting |
| 6 | Splice estimator was **~1000× slower** than necessary (re-summing 232k windows per wavelength) | ~90 min of wasted compute; the slowness *hid* the artifact | noticing the runtime |
| 7 | Hard-coded token IDs `A=5` when the tokenizer says `A=6` | Would have fed the model **shifted nonsense, silently, with plausible losses** | **checking the tokenizer** |
| 8 | The "energy-matched **INERT**" control band was **not inert** — bins 90–92 are read at amp **2.2** vs the carrier's **1.4** | **R18: D34 retracted in full** | **the user asking "how are you deleting it?"** |
| 9 | Fed 510 **bare** nucleotides; the model was trained on `[CLS]`+510+`[SEP]` | **0.0144 nats — 100× the effect being measured**; flipped H1's sign | the same question |
| 10 | Post-softmax attention measurement — **three consecutive attempts**, each returning >100% modulation of a **probability** | Nothing (not reported) | **arithmetic impossibility** |

---

## 2. The root cause — and it is a bias, not a slip

Read the "caught by" column again.

**Almost nothing was caught by procedure. It was caught by *surprise*.** I noticed the bug when the
number was ugly, impossible, or inconvenient.

That is the worst possible trigger, because **surprise only fires when the answer is one I did not
want.** When the instrument produced a number I found *plausible*, I stopped checking.

The clearest instance is #8. The "inert" control band **out-damaged the carrier it was controlling
for** (0.00040 vs 0.00013 nats). That is an anomaly screaming that the control is broken. Instead I
constructed a *scientific story* to accommodate it — *"the human-only carrier is causally weak"* —
and wrote that story into a claim. The story was plausible. It was also a way of not looking.

**And the tell is decisive: the one load-bearing claim in this whole body of work that rests on an
unvalidated instrument (D32's "14 readers") is the claim I was most excited about.** That is not
a coincidence. That is what the bias *does*.

> **I have been running an error-detection process whose sensitivity depends on whether I like the
> answer.** That is **result-dependent scrutiny**, and it is worse than any individual bug.

⚠️ **CORRECTED (review, 2026-07-13).** My first draft said *"my nulls are more trustworthy than my
positives."* **That was itself an overreach, and the history refutes it** — R16, Assay 1, D30 and the
splice scan-edge peaks were all **FALSE NULLS**. The correct rule is symmetric:

> **Neither a positive nor a null is admissible until the instrument has passed sensitivity,
> specificity, resolution, bounds, and intervention audits.**

The problem was never excessive trust in positives. It was scrutiny that varied with the result.

---

## 3. The four failure modes

Each needs a different gate. "Be careful" fixes none of them — I have written *"validate the
instrument"* into `CLAUDE.md` **three times** and then not done it.

### A. Asserted controls
I *call* something a control and never verify it has the property that makes it one.
- the stage-P "control" (not order-destroyed) — #2
- the "energy-matched **inert** band" (not inert) — #8

> **A control is a HYPOTHESIS ABOUT THE INSTRUMENT. It must be tested, with a number, before use.**

### B. Sensitivity without specificity
I calibrate by injecting a signal and recovering it. That proves the instrument can find something
that **is** there. It says **nothing** about whether it returns null when nothing is there.
- the splice calibration — #4. Injected 5%, recovered it on all 8 arms, declared "PASS". Then the
  discovery scan pinned 18 of 32 peaks to a scan edge.

> **Every calibration needs BOTH arms. Inject → must detect. Inject NOTHING → must return null.
> The null arm is the one I skip, every time.**

### C. No plausibility bounds on the output
I read numbers without asking whether the number is *possible*.
- 207% / 184% modulation of a **probability distribution** — #10
- mean attention **exactly 1/121** for every one of 96 heads — should have screamed

> **Before interpreting any number, state its physical bounds. An out-of-bounds value is a HARD
> FAILURE of the instrument, never a surprising result.**

### D. No resolution / power arithmetic before measuring
- the 51-nt window scan — #5. ΔT ≈ 4 nt, used to distinguish 10.298 from 10.630 (Δ = 0.33 nt).

> **Compute resolution and power BEFORE the measurement, not after the null.**
> *(I did exactly this for D26 and D33, and both times it killed a bad experiment before it burned
> a GPU. So I know how. I just don't do it unless something prompts me.)*

---

## 4. The gate

Aspirational rules have failed three times. This one is **code**, and the measurement **cannot run
without it**.

### `preflight()` — mandatory, gating, and it writes an artifact

Every estimator that will touch real data must expose a `preflight()` that runs **first**, in the
same process, prints its numbers, writes them to the results JSON, and **returns False** on any
failure — at which point the script **exits without measuring anything**.

It has four checks. Each must produce **a number**, not an assertion:

1. **POSITIVE CONTROL** — recover a *known* answer. Inject a signal of known size, wavelength and
   phase; it must come back within a stated tolerance, declared in advance.
2. **NEGATIVE CONTROL** — with the signal structurally destroyed, the instrument must return the
   null. **Report the false-positive rate.** This is the arm I skip.
3. **CONTROL VALIDITY** — every object I *call* a control (a band, a stratum, a null set, a shuffled
   corpus) must be **measured** to have the property that makes it a control, and the measurement
   printed. A band asserted inert reports its **max reader amplitude** and **kept-fraction**.
4. **BOUNDS & RESOLUTION** — declare the physical bounds of every output and the instrument's
   resolution/power. Out-of-bounds → hard failure. Any distinction finer than the resolution → not
   reportable.

### The two rules that address the *bias*, not the bugs

**5. THE ANOMALOUS-CONTROL TRIPWIRE — an ANALYSIS FREEZE, not a verdict.**
> An anomalous control **freezes scientific interpretation**. Diagnose the **control** and the
> **instrument** first. **No biological explanation may be promoted until the anomaly is resolved
> independently.**

⚠️ **CORRECTED.** My first draft said an anomalous control *proves* the instrument is broken. It does
not — sometimes it reveals genuine biology (the "inert" band really was a functional feature). The
enforceable rule is the freeze, not the verdict.

**AND NAME CONTROLS BY MEASURED PROPERTY ONLY:** "energy-matched band" (if energy is matched),
"carrier-deleted model", "low-leak band". **Never "inert" before inertness is demonstrated.**
*"Energy-matched inert band"* was two claims wearing one name and only one was ever tested.

**6. THE SYMMETRY RULE.**
> The preflight must be run and recorded **before the real number is seen** — not before it is
> *believed*. And when a result *confirms* what I expected, it gets the **same** scrutiny a
> disconfirming one would get. Concretely: **for every positive result, I must state what would
> have made it an artifact, and check that thing.**

Because right now, my nulls are more trustworthy than my positives, and that is backwards.

---

## 5. What the audit does to the existing record

| claim | instrument validated? | how |
|---|---|---|
| **D21** period-3 clock | ✅ | model-free boxcar cross-check (100.4% agreement); leak floor **measured** on adversarial confusers |
| **D27** the grid does not snap | ✅ | faithful Q_win **frozen from the exact DFT before the run**; two on-grid positives at different bins; genuine negative |
| **D28** biology at 10.298 | ✅ | FPR 4.6% over 4,800 replicates; offset control (dead at ±500); WW/SS counterphase 181.5°; split-half phase 0.4°; **three independent wavelength estimates** |
| **D29** phase-reset unlearnable | ✅ | readout validated by stage P (Q_win 0.966) |
| **D30** unanchored spectrum is blind | ✅ | calibrated by injection at the true effect size |
| **D31** Kogan | ⚠️ **partial** | intron null's floor calibrated by injection — but at **one channel, one wavelength**. Exon-arm numbers are **not usable** (resolution). |
| **carrier = 10.630** (sub-bin) | ✅ | recovers a known off-grid mode 47.500 → 47.505. ⚠️ but the **human-only** sub-bin fit has **R² = 0.810** — its "10.165" is shaky; the bin label (10.24) and QK free-T (10.14–10.24) are the sounder numbers |
| **D32** "14 readers" | ❌ **NOT VALIDATED** | **This is a LOGIT-ONLY claim.** Post-softmax was never validly measured — three broken attempts. And D34b's 0.00003-nat main effect is **exactly what you would see if the comb does not survive the softmax.** |
| **D34b** Exp 3 | ✅ instrument / ❌ result | strata validated (4.65× tone contrast, GC-matched); inert band now verified. But **underpowered and OOD-confounded** — uninformative either way |

**⇒ D32 must be downgraded now**, before any further work: *the human-only model has a real,
causally-deletable ≈10.2 carrier in its position table, read by 14 heads **in their QK logits**.
Whether that comb survives the softmax — i.e. whether it modulates attention at all — is
**UNMEASURED**.* R14 is the memorial for precisely this mistake, and I walked into it anyway.

---

## 6. The post-softmax instrument — declared BEFORE it is built

Per the gate, here is the preflight **before** any code:

- **POSITIVE CONTROL, with its answer stated in advance:** the **period-3 codon comb**. D21
  established it post-softmax: **dominant phase mass 0.794 against a chance level of 0.333**. Any
  post-softmax estimator must recover ≈0.79 on those heads (L2H15, L1H12, …). **If it cannot
  recover a comb we already know is there, it does not get pointed at the carrier.**
- **NEGATIVE CONTROL:** the same estimator on the period-3-deleted model must fall to ≈0.345
  (D21's measured value). And on heads with no comb, it must return ~chance.
- **BOUNDS:** attention is a probability. Modulation **cannot exceed 100% of the mean**. Any value
  >1 is a hard failure, not a result. *(All three of my attempts returned 184–207%.)*
- **WHY MY ATTEMPTS FAILED (diagnosed, not guessed):** attention decays sharply with offset. A
  low-order polynomial envelope cannot fit that decay, so the cosine term absorbs the residual and
  the "amplitude" is fitting the **decay**, not a comb. The fix is a **local** baseline — a boxcar
  of width ≈T, which *annihilates* a period-T tone exactly (the same trick `wave.py` already uses
  for period 3) — and restricting to heads that actually place attention mass in the analysed
  offset band (several of mine had mean attention of **0.00001**, so "95% modulation" was 95% of
  nothing).
- **WHAT WOULD MAKE A POSITIVE AN ARTIFACT** (rule 6): if the "comb" survives deleting the carrier
  band; if it appears equally on heads that do not read the band; if it appears on a
  content-shuffled sequence.

**None of this gets written until the plan is agreed.**


---

# 7. Consolidated status — after the review's five amendments

## The gate now exists and it is holding

`scripts/splicebert/run_postsoftmax.py` runs `preflight()` **before any experimental data is
loaded**, and exits on failure. Current state:

| check | result | frozen limit |
|---|---|---|
| positive control — period-3 intact | **0.8092** | D21: 0.794, band [0.72, 0.86] ✅ |
| negative control — period-3 deleted | **0.3419** | D21: 0.345 (chance 0.333) ✅ |
| deletion audit | target retained **0.0**, protected retained 1.0, checkpoint **unmutated** | ✅ |
| probability invariants | max row-sum error **0.0** | ✅ |
| window leakage at target | **0.0003** | < 0.02 ✅ |
| **injection: amplitude bias** | **0.204** | **< 0.20 ❌ FAIL** |
| injection: phase error | **2.89°** | < 20° ✅ |
| FPR (per-head permutation) | **0.0521** | ≤ 0.07 ✅ |
| null location | median **0.059** | < 0.30 ✅ |
| m_T bounded | max **0.9967** | ≤ 1 ✅ |

**IT FAILS BY 0.004, AND THE TOLERANCE WILL NOT BE MOVED.** Changing `0.20` to `0.25` is one
character and is trivially rationalisable. It is also precisely the failure this audit exists to
prevent. The number was frozen *before* the run so that this moment would be impossible to argue
with. The fix goes in the **instrument** — diagnosis: per-depth bias, almost certainly the
`a = 0.05` arm, where the baseline is refit on the injected profile and the cubic absorbs a larger
*fraction* of a small tone.

## What the amendments caught, in order

1. **The boxcar does not notch a non-integer period.** I had asserted it did, generalising from
   `1 + 2cos(2π/3) = 0` — exact only for **integer** period 3. Measured: M=10 leaks **2.4%** at
   10.24, M=11 leaks **3.4%** at 10.63. A false statement would have gone into a docstring.
2. **The content-shuffle criterion was backwards — R15 again.** A positional comb built from
   `(P, W_Q, W_K)` **must** survive content shuffling. I had listed shuffle-survival as an
   *artefact* criterion, conflating QK with OV.
3. **`amplitude / mean` is unbounded** once a fitted baseline is involved — which is how I produced
   184–207% modulation of a **probability**, three times. `m_T` is bounded in [0,1] by the triangle
   inequality for **any** r ≥ 0.
4. **`classify()` had to be IMPORTED, not reimplemented.** Its `SPAN` is 24, not my 60. A
   reimplementation would have silently used my window, missed 0.794, and tempted me to tune the
   definition until it matched. **That is the sin itself.**
5. **A split offset band with one window across the discontinuity** made `m_T` measure the
   *window's own transform*: the tone-free null saturated at **0.98**. Fixed by windowing each
   contiguous half.
6. **The FPR check PASSED on a broken statistic** — it measured the rate at the null's own 95th
   percentile and never asked **where the null sits**. Specificity-without-calibration. Two new
   gates: `window_leakage` and `null_location`.
7. **`m_T` is a COHERENCE, not an amplitude** — it saturates when the residual is spiky. So the
   null must be **per-head** (permuting that head's own residual), and injection recovery must use
   the **complex increment** (`c' − c = (a/2)e^{iφ}` exactly, for multiplicative injection).

## Claim-level firewall — the human-only ≈10.2 carrier

| level | status |
|---|---|
| **P** — feature exists in the position table | **established** |
| **QK** — a logit circuit reads it (14 heads) | **established, provisional on the intervention audit** |
| **A** — survives softmax into attention | **OPEN — instrument not yet passing preflight** |
| **OV** — head output carries relevant content | **OPEN** |
| **BEHAVIOUR** — performance causally depends on it | **OPEN** (D34b: 0.00003 nats, underpowered, OOD) |

Evidence never auto-promotes between levels.


---

# 8. ⛔ SUPERSEDED — DO NOT IMPLEMENT

Preserved as history. **Both of these appear in earlier commits and both are WRONG:**

1. ~~"A boxcar of width ≈T annihilates a period-T tone exactly."~~ **FALSE for non-integer T.**
   `1 + 2cos(2π/3) = 0` is exact **only** for integer period 3. A length-M boxcar has zeros only at
   `T = M/n`. **Measured: M=10 leaks 2.4% at T=10.24; M=11 leaks 3.4% at T=10.63.**
2. ~~"A comb appearing on content-shuffled sequence is an artefact."~~ **BACKWARDS — this is R15.**
   A positional comb built from `(P, W_Q, W_K)` **MUST** survive content shuffling; that is what makes
   it positional. Correctly: **carrier deletion must kill the attention comb; content shuffling need
   not; phase-scrambling must kill any OV/behavioural effect.**

---

# 9. THE STOPPING RULE — this is bounded, not an endless sequence

**Two experiments close the human-only claim. No more.**

1. **Pass the post-softmax preflight and measure Level A.** *(Preflight PASSES as of `run_postsoftmax.py`:
   injection bias 0.0016, phase 0.16°, FPR 5.33%, null median 0.007.)*
   - **If the ≈10.2 modulation does NOT survive the softmax → STOP.** Close the human chromatin
     circuit claim at **Level QK**. No rescue experiments.
   - If it survives **and collapses specifically under carrier deletion** → promote to **Level A**.
2. **Only after a positive Level A**, run **one** species-matched causal experiment: human-only model,
   **human** sequence, **human** dyads, with MNase-bias controls.
   - Positive carrier × dyad-phase interaction → evidence the model uses nucleosomal phasing.
   - **Powered null → close the hypothesis.** No rescue experiments.

**The unanchored ≈10.630 carrier is a SEPARATE research project.** Interesting, but not required to
finish this story.

# 10. Claim-level firewall — human-only ≈10.2 carrier

| level | status |
|---|---|
| **P** — feature exists in the position table | **established** |
| **QK** — a logit circuit reads it | **provisional / supportive** |
| **A** — survives softmax into attention | **unmeasured** |
| **OV** — head output carries relevant content | **unmeasured** |
| **BEHAVIOUR** — performance causally depends on it | **unmeasured** |

# 11. D27 — narrowed

D27 shows the 512-row table does **not mechanically quantize** frequencies onto Fourier bins, so the
sharp carrier is **not a DFT-grid artefact**. It does **NOT** show the sharpness came from biological
data — training dynamics or optimization could select a sharp mode just as well. The admissible
statement is:

> **The frequency was SELECTED DURING TRAINING rather than IMPOSED by the position-table grid.
> What training signal selected it remains OPEN.**

*(My consolidated report said "sharpness in P is evidence about the data." That does not survive.)*
