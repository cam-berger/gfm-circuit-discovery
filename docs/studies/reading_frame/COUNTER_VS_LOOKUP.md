# Counter vs anchor/position lookup — preregistration

**Status: PREREGISTRATION (no experiment run yet).** Written before the discriminating runs so the
prediction cannot be narrated after the fact. Data generators, configs, and results land later
(`TO_FIX.md` §5.4, commit "preregister counter-versus-lookup tests").

## The question

The trained forcing-task model (`nano_uniform`) and the 2×2 cells use the anchor and carry a linearly
recoverable relative phase. The docs call this a **counter**. But an anchor-relative computation of the
form `phase(i) = f(i − a)` can be implemented two ways that no experiment on record distinguishes:

- **Sequential counter** — an iterative state passed position to position, `phase(i+1) = T(phase(i))`,
  seeded at the anchor `a`.
- **Anchor/position residue lookup** — the model reads `(i − a) mod 3` (or `a mod 3` and `i mod 3`)
  through learned absolute-position embeddings and a compact residue-class map, with no iterated state.

Both satisfy cue conflict (they follow the anchor, not local content), both survive the D21 randomised-
offset guard (which only rules out an *absolute* `pos % 3` lookup relative to the true frame), and the
withdrawn T1 sawtooth test could not separate them. GFM-nano uses **learned absolute position
embeddings**, which makes the lookup implementation available. So "counter" is currently licensed only
as **"anchor-relative modulo-3 mechanism"**.

Scope: this concerns the **trained** model. The hand-built planted twin implements a counter *by
construction* and is not in question.

## Pre-registered discriminators

Each is registered with the prediction it makes under each hypothesis. Run each with a paired
no-edit / decoy control; report the continuous statistic regardless of which side of any threshold it
lands on.

1. **Held-out anchor positions.** Train with a subset of absolute anchor positions; test on unseen
   positions. — *Counter:* generalises (the count does not depend on absolute position). *Lookup:*
   degrades on unseen absolute-position/anchor combinations it never indexed.
2. **Joint translation.** Translate anchor and query together, relative distance fixed. — *Counter:*
   invariant. *Lookup keyed to absolute position:* shifts.
3. **Length extrapolation.** Test beyond trained lengths / absolute-position ranges (architecture
   permitting). — *Counter:* extrapolates. *Lookup:* fails outside the trained position range.
4. **Position-encoding intervention.** Retrain the same task with learned-absolute vs relative vs rotary
   position encodings. — If the mechanism only forms (or only extrapolates) with a position code that
   supports iterated relative state, that is evidence for counting; if a residue-class lookup off
   absolute embeddings suffices, evidence for lookup.
5. **Transition consistency.** Test whether the phase representation at `i+1` is a *stable*
   transformation of the representation at `i`. — *Counter:* a single reusable transition `T`.
6. **Composition test.** Apply the inferred transition repeatedly; test whether it composes modulo 3
   (`T³ ≈ identity`). — *Counter:* composes. *Lookup:* no single composing transition exists.
7. **Patch-forward.** Patch the putative phase state from one position into a downstream position; test
   for the predicted phase shift. — *Counter:* the patched state drives the downstream phase by the
   expected offset. *Lookup:* downstream phase is recomputed from position, unmoved by the patch.

Also relevant (from `TO_FIX.md` §C in the earlier audit notes): **graded insertion/deletion** of
`k ∈ {1,2,3,6,9}` nt between anchor and a fixed distal readout — a counter re-derives `(i − a)` and
preserves the anchor-frame for every `k`; a distance-indexed lookup keyed to pre-edit geometry misfires
in a size-dependent way.

## Pre-registered verdict

```
SEQUENTIAL COUNTER            passes transition, composition, held-out-position, and extrapolation
ANCHOR/POSITION LOOKUP        solves cue conflict but fails transition or extrapolation
UNRESOLVED ANCHOR-RELATIVE    behaviour is anchored but the implementation is not separable with
                             current evidence (report as UNRESOLVED, do not narrate)
```

Until this runs, active docs must say **anchor-relative modulo-3 circuit / phase-maintenance
mechanism**, not "sequential counter," for the trained model.
