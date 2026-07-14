# ⛔ `phase_function.json` IS VOID — R13 (2026-07-12)

**Do not cite any number in `phase_function.json`. Do not re-run `run_phase_function.py` and
expect a different answer — the design, not the run, is what failed.**

## What it claimed

That the codon-phase lattice is **not** what produces SpliceBERT's codon-periodic predictions:
ablating all 42 lattice heads did not reduce period-3 enrichment (3.34× → **4.52×**), and 9 of 15
size-matched random 42-head ablations did as much (add-one **p = 0.625**).

## Why it is void

The ablation drove the model to **chance-level nucleotide prediction**. Straight from the file:

| arm | `logprob` (mean log-prob of the true base) | probability |
|---|---|---|
| `intact` → exon | −1.1200 | 0.326 |
| **`lattice_ablated` → exon** | **−1.3766** | **0.2525** |
| `nulls` (15 random 42-head sets) | −1.2278 … −1.4043 | ~0.25–0.29 |
| **chance = ln(¼)** | **−1.3863** | **0.2500** |

The intact model's entire headroom over chance is **0.076 nats**. The ablation removes **0.074 of
it — 97%.** Treatment and null alike were dead.

So the period-3 statistic was computed on the output of a model that was no longer predicting
anything. And the *direction* is the tell: P3 **rose** (3.34 → 4.52). That is what a wrecked model
does — it falls back on its position prior, which is trivially periodic. It is also why the null
"passed": all 15 random sets were equally destroyed, so they too showed elevated P3.

**The experiment had no power. It could not have detected a true effect had one been there.**

`run_phase_function.py` never gated on MLM health. `grep -n health run_phase_function.py` returns
nothing. The log-prob was computed, printed in the script's own output table, and read as a sanity
number instead of as ln(¼).

## The failure mode — third instance in this project

| | what was measured on a model that could not do the task |
|---|---|
| **R8** | "MLM health" read the argmax at *unmasked* positions → scored the model copying its own input → returned 0.99 for everything, **including a model with 94 of 96 heads ablated (0.998, *above* baseline)** |
| **R12** | "Layers 4–5 are specifically required" → died to a size-matched null; at matched health the circuit-preserving ablation was *gentler* than random |
| **R13** | this file |

> **An intervention that destroys the behaviour cannot measure the mechanism.**
> Every arm must be shown to retain the behaviour *before* its metric is interpreted.

(Standing beside: **a control that cannot fail is not a control.**)

## What is NOT void

- `phase_inventory.json` — the **structure** result (42/96 heads, all relative, three phases
  covered, absent at random init). Untouched. That is D19 and it stands.
- The observation that the model's predictions **are** period-3 enriched in exons (3.34×) and much
  less so in introns (1.33×). That is a property of the intact model and does not depend on any
  ablation.

**What is now unknown is whether the lattice causes it.** That question is **open**.

## The replacement

`studies/codon_phase_lattice/PREREGISTRATION_PHASE_SHIFT.md` — a **phase-shift** intervention that
rolls each lattice head's attention along the key axis by δ and leaves its values, weights, and
every other component intact. The model survives, so the behaviour stays measurable; damage is
scanned over δ = 0…6 and the *phase* effect (dips at δ = 3, 6) is identified separately from the
*displacement* effect (a monotone slope in δ).
