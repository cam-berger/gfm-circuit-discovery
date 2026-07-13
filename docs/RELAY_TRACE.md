# Tracing the relay

**[RECORD]** — the Phase-C deliverable: an operational definition of the three mechanisms
(counter / relay / bag), and the retraction of four of the five "relays" we had claimed.
**Read this if** you want the deepest single trace in the program, or an object lesson in how a probe
read at the wrong layer manufactures a mechanism that is not there.
**Status:** the *retraction* is SETTLED (two independent interventions, a positive control, a null
control, and an edit control, all agreeing). Every *per-cell number* is one seed and is therefore
PENDING — nothing here is settled at the cell level until Phase D's three seeds.

**Pre-registration:** `PREREGISTRATION.md`, addendum 2026-07-09 (b), written before any measurement.
**Full reasoning:** `DECISIONS.md` D34, D35. **Plan (executed, archived):**
`archive/PLAN_reading_frame_executed.md` step 10.

---

## The headline

The plan opened: *"the single most valuable unclaimed result is to trace the relay — bag and counter
were hand-buildable in advance, the relay was not."*

**Four of the five models we called relays or mixtures are anchored counters.** The relay was an
artifact of reading the cue-conflict intervention off a linear probe at `L2.resid_post`, a layer that on
real DNA does not yet carry the anchor. Read at the model's own output, the mechanism inverts.

```
                        anchor share under cue conflict, far field (>= 64 nt)
cell                    at L2 (D33 reported)     at the model's output      verdict
A · uniform, easy              1.0000                   1.0000              counter
B · uniform, hard              0.3859  "mixed"          0.8817              counter
C · real CDS, easy             0.0389  "relay"          0.7106              counter
D · real CDS, natural          0.1077                   0.2314              RELAY
μ=0 · real CDS, LM             0.0925                   -- (nt head)        bag
```

**The relay is real, and there is exactly one of it: cell D.**

---

## `SETTLED` — an operational definition of the three mechanisms

Two interventions, both scored against the model's own predictions, both carrying a paired decoy-Met
control (an in-frame ATG ≥ 30 nt downstream, replaced by the same CTC, so the edit *type* is held fixed
and only the *position* varies).

```
ABLATION      replace the start ATG with CTC.        Does the far field still get the frame right?
CUE CONFLICT  insert one nt after the ATG, so the anchor and the local codon structure
              disagree about the frame by exactly one. Which one does the far field follow?
```

|            | far-field ablation Δ | anchor share under conflict | reading |
|---|---|---|---|
| **counter** | large | > 0.5 | the frame is *counted from* the anchor |
| **relay**   | large | < 0.5 | the anchor is *used* but not counted from |
| **bag**     | ~0    | ~0    | the anchor is *unused* |

Ablation alone cannot separate counter from relay (both lose the anchor and both collapse). Conflict
alone cannot separate relay from bag (both follow content). **Neither axis is sufficient; the pair is.**
That is the operational definition, and it is why the 2×2 needed both.

### The measured 2×2

```
cell   shortcut   detector   aa_acc   far-field ablation Δ (ctl)   anchor share (output)   mechanism
A       absent      easy     1.0000        0.5891 (0.0193)               1.0000            counter
B       absent      hard     0.9605        0.5553 (0.0219)               0.8817            counter
C      present      easy     0.9999        0.3264 (0.0242)               0.7106            counter
D      present      hard     0.9892        0.1126 (0.0335)               0.2314            relay
μ=0    present      n/a           --       0.0025 (probe)                0.0925 (probe)    bag
```

All five score 0.96–1.00 on their own objective. **Accuracy sees none of this.** That result, from D29,
survives the retraction intact — it is the only claim in the program that never depended on a probe.

### The pre-registered attribution logic resolves: the two factors interact

```
A -> B  (detector hard)     1.0000 -> 0.8817     an unreliable anchor is still counted from
A -> C  (shortcut on)       1.0000 -> 0.7106     a pooled route competes, and loses
A -> D  (both)              1.0000 -> 0.2314     only together do they displace the counter
```

Neither factor alone displaces the circuit. A hard-to-find anchor is worth counting from when nothing
else works. A pooled shortcut loses to a reliable anchor. **The model abandons counting only when the
anchor is unreliable *and* something else is on offer** — which is, exactly, the condition of real
genomic self-supervision.

---

## `SETTLED` — the retraction, and why it survives scrutiny

Four independent reasons to believe the corrected number rather than the old one.

1. **A second intervention, in the opposite direction.** The conflict moves the *content* and leaves the
   ATG. **T3 (anchor relocation)** destroys the ATG and rewrites it one or two bases over, leaving all
   downstream content untouched. Far-field output, `follows anchor / follows content`:

   ```
                shift+1            shift+2            shift+3 (in-frame move)
   A       1.0000 / 0.0553    0.9890 / 0.0608              0.9920
   B       0.8306 / 0.1312    0.8590 / 0.1338              0.8091
   C       0.6507 / 0.4025    0.6904 / 0.3229              0.9996
   D       0.3591 / 0.6990    0.1899 / 0.7525              0.9880
   ```

   Same ordering, `A > B > C > D`. Move the content and the model follows the anchor; move the anchor and
   the model follows the anchor.

2. **The `shift+3` control clears the edit.** The ATG is moved three bases, so the frame it announces is
   unchanged. Cost: ≤ 0.02 everywhere. The edit itself is harmless; only its *phase* matters.

3. **T3's OOD branch does not fire.** The registered falsifier was: *if the far field collapses to chance
   under both deletion and relocation, the edit was an out-of-distribution shock and D33's whole ablation
   column measures distribution shift.* It does not collapse — cell A goes to 1.0000 on the relocated
   anchor. **The ablation column stands.**

4. **A positive control and a null control both pass.** Cell A, a hand-verified counter, reads 1.0000 at
   every readout and at every distance. And on the uniform-codon cells the pooled oracle is at chance
   (`oracle-right` fraction 0.331, 0.341), where the T2 partition returns a ratio of 1.05 and 1.00 — the
   partition does not manufacture an effect.

---

## `SETTLED` — what cell C actually does, and it is stranger than a relay

Cell C is an anchored counter that *also builds a pooled estimate it then overrules*.

```
cell C, by depth        phase decodable   under cue conflict, follows...
  L0–L2                    0.83 -> 0.95        content (0.91 at L2)
  L4                       0.9999              the anchor (0.69)
  output                   0.9999              the anchor (0.71)
```

By L2 the model has reached **0.9515**, which is the anchor-free pooled ceiling (0.9509, measured
independently in `gfmcd/oracle.py`) — without the anchor, which does not enter until L4. Then L4
overrides it.

**And the pooled estimate survives the ablation.** With the start codon destroyed, a linear probe *fit
on intact data* still reads the frame off cell C's L2 at **0.9453** — while the model's output has fallen
to **0.6492**. The model is carrying a 0.945-accurate representation of the answer and discarding it.

This kills D34's "anchor-corrected pooling" reading, which said the anchor supplies only the residual
~5% that pooling cannot reach. **T2** tests that directly: partition far-field positions by whether the
anchor-free oracle already recovers the frame, then ablate.

```
cell C, far field        fraction   ablation Δ   Δloss (nats)
  oracle-right             0.922       0.3153       3.385
  oracle-wrong             0.078       0.4529       5.105
  ratio wrong/right                    1.44
```

Under a correction term, ablation on `oracle-right` should have cost **nothing**. It costs 0.3153. The
anchor is enriched where pooling fails (1.44× in C, 2.57× in D) but it is load-bearing **everywhere**.
The registered `H3-strong` criterion (`Δ(right) < 0.10` and ratio ≥ 3) fails for C and D — as the
pre-registration predicted, naming outcome (ii) in advance.

So the pooled route is a **scaffold, not a summand.** Why the model builds it is the open question below.

---

## `WITHDRAWN` — T1, the sawtooth test, discriminates nothing

The plan asked for the intervention that distinguishes a *global count from the anchor* from a *local
re-sync chain*: cue conflict at graded distances, looking for a sawtooth rather than a monotone decay.
Registered as: `slope in log d -> counter`, `periodic -> re-sync chain`, `flat -> correction`.

It failed three ways, and all three are worth recording.

1. **Aliasing.** The first run binned Δ(d) at 16 nt and reported a fitted period of **17 nt** — a period
   at the sampling interval, which is unidentifiable. Re-run at 1-nt resolution with the period floored
   at 8 nt and the sinusoid tested *on top of* the log-d trend.
2. **A miscalibrated threshold.** The periodic term then survived BIC for both C and D (ΔBIC 11.1 each)
   — at periods of **88 nt and 21.5 nt**, a 4× disagreement between two cells of the same putative
   mechanism. Δ(d) is autocorrelated (AR(1) ρ = 0.49, 0.62); BIC assumes it is not. Against 2000 AR(1)
   surrogates: **p = 0.14 and p = 0.36**, and the null's own 95th percentile of ΔBIC is **17.1 and 25.8**,
   far above the registered threshold of 6. **The threshold was wrong before it was ever used.**
   There is no periodicity. **H2 — the re-sync chain, the hypothesis "relay" was named for — is dead.**
3. **It cannot even find a counter.** Cell A slopes (+0.0264 per log2, ΔBIC 110.8). Cell B — also a
   counter, by both T3 and the output-level conflict — does not (+0.0096, ΔBIC −2.8). A test that calls
   one counter a counter and the other flat is not a test.

The anchor's effect is **flat in distance** out to 255 nt for every model (`fig_decay_output.png`).
Distance is not what separates the mechanisms. Everything decisive here is behavioural.

---

## `PENDING` — what is not done

- **T4, component localisation.** Not run. Which heads carry the anchor → far-field signal in cells C and
  D, and is it the same head the forcing-task counter uses (`L1H3`)? The registered rule stands: a head
  is `CARRYING` only if `Δloss > 0.01` nats **and** the effect is ≥ 10× its decoy-Met counterpart, and
  every nominated head is reported including those that fail. `L1H2` in the forcing task had 406× decoy
  specificity by attribution and a 2×10⁻⁶ loss effect. Attribution ranks; only intervention decides.
- **Seeds.** One per cell. No number in this document may be tagged `SETTLED` at the *cell* level until
  Phase D's three seeds. What is settled is the *retraction* — it rests on two interventions, a positive
  control, a null control, and an edit control, all in the same direction.
- **The LayerNorm faithfulness gap.** Still unmeasured (`RESULTS.md` §11). It gates the causal claims.

---

## The question this opens

If cell C's output is anchored, why does it build the pooled estimate at all — and then keep carrying it
after the anchor is gone?

The obvious guess is **training order**: pooling is the fast route, cheap and learnable early, and the
counter is layered on top of it later. That is a claim about optimisation, not architecture, and we have
the checkpoints to test it. Probe L2 (content) and L4 (anchor) across training steps and see which
appears first.

If the scaffold precedes the counter, then the λ-sweep's grokking curves — where the anchored arms find
the circuit *faster* as the shortcut gets stronger (3250 → 3000 → 1500 → 750 steps as λ rises) — stop
being a curiosity and become a mechanism. **The shortcut would not be the circuit's competitor. It would
be its scaffold.**

That is the experiment I would run next.
