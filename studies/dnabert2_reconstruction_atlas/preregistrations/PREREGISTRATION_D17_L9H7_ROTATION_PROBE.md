# D17 preregistration — is L9H7 a content detector, a fixed-offset reader, or conjunctive?

> **Tier: T1 — frozen protocol.** Frozen 2026-07-29; a defect in its own decision rule is
> recorded in `CLAIMS.md` and `DECISIONS.md` rather than repaired here.

> **Status: FROZEN 2026-07-29.** Thresholds and the trichotomy were fixed in
> `scripts/dnabert2_atlas/run_l9h7_rotation_probe.py` before any rotation was scored.
>
> **This is a descriptive measurement, not a sealed confirmatory test.** `[CLS]` attention on this
> cohort was already measured in D16, so there is no seal to spend. The full-cohort run buys
> precision, not confirmation. Said plainly rather than dressed up as a confirmation.

---

## 1. Intent

> **Does L9H7 attend to TSS-proximal sequence *content*, or to a fixed window *offset* that always
> contains the TSS?**

D16 reported L9H7's `[CLS]` mass moving onto `proximal_upstream` (−50:−2) and called it a "TSS
reader." That is not established. The task geometry is degenerate — the TSS sits at index 249 in
every one of the 384 cohort records, GUE `prom_300_all` is 47,356 windows of exactly 300 nt — and
because `[CLS]` is at index 0 and ALiBi encodes distance from the query, **absolute window position
is recoverable**. `−50:−2` is also window positions **199–247**, a constant offset.

---

## 2. Manipulation

No reference genome is present in this checkout, so windows cannot be re-cut from the genome. The
available manipulation is stronger for this question anyway: **circular rotation**,
`s' = s[k:] + s[:k]`. Content is preserved **exactly** — same 300 nt, same composition — and only
its offset changes. Verified empirically before freezing: composition identical at every `k`, token
count stable (64.30 → 64.35–64.50), and **93.6% of BPE token boundaries survive** in original
coordinates, so re-segmentation is minor and the residual is absorbed by proportional
nucleotide-overlap allocation.

Two competing readouts per rotation, both computed from the same `[CLS]` attention vector:

| symbol | definition |
|---|---|
| **F(k)** | attention on the **nucleotides** originally at 199–247, now at (199−k)–(247−k) |
| **P(k)** | attention on whatever now occupies **window positions** 199–247 |

At `k=0` they coincide by construction.

---

## 3. What the design red-team changed

A four-lens adversarial review was run before any rotation was scored. **Only one lens returned a
report** — three produced no output and the synthesis consequently received empty inputs; this is
disclosed rather than presented as a completed four-lens review. The lens that did return forced
three changes, each verified independently against the arithmetic before adoption:

1. **The dichotomy was over a trichotomy.** "Conjunctive addressing" — the head responds to
   promoter-like content *only when it sits at the trained offset* — is a third hypothesis, and
   given a training geometry where `P(content | position)` is a point mass it is arguably the most
   likely one. It predicts **both** F and P collapse, which was exactly the draft rule's
   "inconclusive" branch. The rule is now an explicit trichotomy with a discriminator.
2. **Small rotations are degenerate and were removed from the decision.** At `k=10` the F and P
   bands overlap 39 of 49 nt, so the *positional* hypothesis itself predicts `F ≈ 0.444` — which
   would have passed a `F ≥ 0.35` "content" threshold and fired for the wrong reason. Bands are
   disjoint only for `k ≥ 49`. Only `k ∈ {125, 149, 175, 199}` decide; `k=25` is retained as a
   **declared non-discriminating** OOD-dose anchor.
3. **The junction bias ran toward the expected answer.** The junction sits between new indices
   `299−k` and `300−k`. Since `300−k > 247−k` always, it can **never** fall inside F but **can**
   fall inside P — so junction-attracted attention inflates P and never F, biasing toward the
   positional verdict. `k=50` puts it exactly on the canonical TSS index and `k=99` puts it inside
   the P band; both are excluded. `J(k)` is reported as a memorial with a 0.10 ceiling.

**One lens claim was rejected on checking:** it asserted `k=149` places the F band at 99–147, only
3 nt from the junction. The F band at `k=149` is `199−149 = 50` to `247−149 = 98`, and the junction
is at 150/151 — 52 nt clear. `k=149` is retained.

---

## 4. Estimands

The retention statistic `F(k)` alone is confounded with generic degradation in one direction
forever, so the primary endpoint is an **OOD-severity-matched difference**. Each rotation is run in
two arms:

- **`intact`** — plain rotation.
- **`shuffled`** — the 199–247 band is mononucleotide-shuffled **in place before rotation**
  (composition exactly preserved), then rotated identically. Same junction, same displacement, same
  re-segmentation class; content destroyed.

**Primary: `ΔF(k) = F_intact(k) − F_shuffled(k)`**, averaged over the discriminating rotations.
Generic OOD collapse is common to both arms and differences out. Cluster-bootstrapped over
`homology_cluster` (8000 draws, seed 20260729).

**Co-primary: `P(k)`** under the intact arm.

**Global stability** (separates head-specific loss from network collapse): mean promoter margin per
rotation, and mean absolute drift in P-band mass across all 144 heads relative to `k=0`.

---

## 5. Frozen trichotomy

| verdict | condition |
|---|---|
| **CONTENT_ADDRESSED** | `mean ΔF ≥ 0.15` over discriminating k, with every CI lower bound > 0 |
| **INDEX_ADDRESSED** | `mean P ≥ 0.35` and `mean ΔF < 0.05` — attention stays at the offset regardless of what is there |
| **CONJUNCTIVE** | `mean P < 0.20` and `mean ΔF < 0.05` **with the network still working** (promoter margin positive at max k) — selectivity lost only in this head |
| **MIXED_OR_UNINTERPRETABLE** | anything else, including joint collapse with global instability |

---

## 6. Memorials

| id | check | can it fail? |
|---|---|---|
| **M1** identity at `k=0` | rotation by 0 is the identity | **architecture-forced** — flagged, not counted |
| **M2** rotation preserves composition | sorted(rotated) == sorted(native), every locus and k | **yes** |
| **M3** shuffle preserves composition | sorted(shuffled) == sorted(native) | **yes** |
| **M5** L11H2 stays pinned | L11H2 (reach ~8 tokens, ALiBi-pinned near `[CLS]`) must keep negligible P mass under every rotation | **yes — and it is the framing check.** If L11H2's attention started following content, the whole positional account would be wrong |
| **M6** junction below ceiling | `J(k) ≤ 0.10` for all k | **yes** |

---

## 7. Scope

1. **`[CLS]`-row attention only.** Other query positions are not measured.
2. **Descriptive.** This says where L9H7 looks, not what it contributes; D15 already shows it
   contributes ~19× less than the unchanged carriers.
3. **Rotation is out of distribution** by construction. That is why `ΔF` (matched) and the global
   stability readout carry the inference, not raw `F`.
4. **A positional verdict bounds D16's wording; it does not overturn D16's measurement.** The
   attention shift from `special` to `proximal_upstream` is real either way — only its
   *interpretation* as TSS recognition is at stake.
5. Single third-party checkpoint pair, n=1; nothing here speaks to what a different fine-tuning run
   would produce.
