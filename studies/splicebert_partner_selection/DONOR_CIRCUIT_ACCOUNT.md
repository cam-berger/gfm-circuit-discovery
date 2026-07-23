# The SpliceBERT donor circuit — original C16–C19 synthesis

> **Tier: T3 — historical.** Superseded or point-in-time; retained for the record. Not current guidance.

> **Historical scope notice (2026-07-20):** this document predates the A0→A12 generator/address work and
> corrections R22–R24. It is retained for provenance, not as the current one-file mechanism account.
> Read [`../../MECHANISM.md`](../../MECHANISM.md) and [`CLAIMS.md`](CLAIMS.md) for current wording.

*A consolidated, start-to-finish account of the reasoning and evidence. Branch
`splicebert-donor-graph`. Every number below is reproducible from a frozen pre-registration and
a committed runner; the controlling artifacts are named inline and the claims are registered as
**C16–C19** in [`CLAIMS.md`](CLAIMS.md).*

---

## 0. The result, first

SpliceBERT, asked to fill a masked nucleotide at a splice **donor** (the G of the intronic
`GT`), scores G more probable **when the donor's own paired acceptor is intact**. We traced the
computation that does this end to end, as a causal, information-specific graph:

```
   paired acceptor (AG)
          │  read by
          ▼
   L3H1 ──writes at donor±5──► MLP3 · MLP4 ──assemble──► U_info ──transported by──► L4H0 · L4H2 · L5H11
   (reader)                    (nonlinear transform)     (low-rank              (value/OV, to the donor)
                                                          residual object)              │
                                                                                        ▼
                                                                                   U_info at the
                                                                                   masked-donor
                                                                                   residual
                                                                                        │ read by the
                                                                                        ▼ MLM head (exact)
                                                                                   logit(G) − max(A,C,T)
```

What flows through this path — the content of `U_info` — is **paired-intron consistency plus
donor-identity confidence**: it is specific to the donor's *own* partner acceptor (not any AG),
specific to the *donor's* G (not G globally), and it is **phase-independent** (it does not encode
reading frame). The period-3 positional clock is *necessary* for the effect but does not make it
phase-sensitive, so its role here is positional addressing, not frame computation.

Each stage was established behind a pre-registered, falsifiable gate:

| Stage | Claim | Key evidence | ID |
|---|---|---|---|
| Output object | Coupling carried by a low-rank final-residual direction `U_info` | rank-8 patch → **0.92** recovery; random rank-8 **≤0.05** (p=0.005); baseline injection collapses +4.65→+0.33 | **C16** |
| L3H1 → object | L3H1 delivers `U_info`, assembled at MLP3/MLP4, localized to donor±5 | **93%** of L3H1's recovery routes through `U_info`; donor±5 **96%** vs elsewhere **6%** | **C17** |
| Attention → heads | Three **transporter** heads carry it via value/OV | L5H11/L4H2/L4H0 value-noising **0.13/0.12/0.08**, QK-noising **≈0** | **C18** |
| Content | Paired-intron consistency + donor identity, phase-independent | partner **37×**, donor **74×**; flat across intron-mod-3 *and* annotated phase | **C19** |

---

## 1. The question, and the strategy

Prior work had established a **behaviour** — corrupting a donor's paired acceptor lowers the
model's score for that donor — and a **dominant subcircuit** `C = {L3H1, L2H15}` (C1–C6). But the
*downstream* half was only a scalar curve: path-patching L3H1's clean output into the corrupt run
and un-freezing the stack recovered the effect distributively, with no single mediating block
(`readout_path.json`):

| Route available after L3H1 | direct | +MLP3 | +L4-attn | +MLP4 | +L5-attn | +MLP5 |
|---|--:|--:|--:|--:|--:|--:|
| coupling recovered | 0.023 | 0.181 | 0.258 | 0.445 | 0.506 | 0.818 |

"The head matters" and "recovery accrues through the stack" are compatible with many circuits.
The goal of this work was to stop searching for more heads and instead build a **causal,
information-specific graph from L3H1 to the masked-donor logit** — resolving *which directions*,
*which heads*, and *what information*. The strategy was **meet-in-the-middle**: anchor the output
end, trace L3H1 forward, and join them, then ask what the joined object encodes.

---

## 2. Where we started (established before this program)

From the sealed-test and coupling work (`CLAIMS.md` C1–C8, `sealed_test.json`,
`readout_path.json`, `carrier_*.json`):

- **C1–C2** The behaviour is real and partner-specific: sealed coupling **+0.872 [0.771, 0.969]**;
  a non-partner AG moves the donor **+0.007**, a distance-matched other donor **+0.005**.
- **C3** `C = {L3H1, L2H15}` is a dominant necessary subcircuit — ablating it removes **81.6%** of
  the coupling while retaining **99.2%** masked-MLM health (matched two-head p = 0.0025).
- **C4** L3H1's acceptor read is **conjunctive**: key alone 0.032, value alone 0.101, key+value
  **0.663** (interaction +0.531).
- **C5** L3H1 reads at the acceptor and **writes at the donor ±5 neighbourhood**, followed by a
  distributed downstream chain (the curve above).
- **C8** A rank-2 learned **period-3 position code** supports the coupling: deleting it drives the
  coupling **0.946 → 0.027** and costs coding-exon MLM **0.159 nats** vs **0.042** in introns.

The open problem was everything downstream of L3H1's write: its *direction*, its *carriers*, and
its *meaning*.

---

## 3. Method and discipline

Four conventions govern every result and are worth stating because they are what make the
numbers trustworthy.

- **The interpretive firewall.** Evidence is tracked at levels P → QK → A → OV → Behaviour →
  Biology and never auto-promoted. This account operates at the OV/Behaviour levels; it makes no
  biological claim.
- **Pre-registration with controls that can fail.** Every experiment's design, interventions, and
  pass/fail thresholds are frozen in a `preregistrations/PREREGISTRATION_DONOR_*.md` *before* the run. The standing
  rule "a control that cannot fail is not a control" is enforced: descriptive decompositions (an
  SVD, a head scan) are licensed only by controls that are reachable-falsifiable — a size-matched
  random null and a destructive alive-check.
- **The frozen readout, imported not redefined.** Mask the donor's G; score
  `margin = logit(G) − max(logit A,C,T)`; corrupt by substituting real bases (`AG→CC`), never
  `[MASK]`; coupling = margin(clean) − margin(corrupt). Recovery is **ratio-of-means** on the
  unconditional set (gene-grouped bootstrap CIs), which avoids conditioning on the outcome.
- **Exact, cheap interventions.** The masked-donor logit is exactly `cls(final_residual)` (verified
  bit-exact), so the output-side causal tests run by re-applying the MLM head to modified cached
  residuals — no re-forward. Interventions are one-way substitutions (copies), never index-
  permuting swaps (which are algebraically invariant and measure nothing).

---

## 4. The five experiments

### 4.1 Anchor the output — the information object `U_info` (1a → C16)

**Reasoning.** The backward arm cannot start until we know *which scalar* the graph explains and
*which residual direction* produces it. The prompt's own recipe: isolate a low-rank clean−corrupt
subspace at the readout; if patching only it preserves most recovery, it is "the actual
information object you follow through the network." The trap: an SVD always returns a dominant
direction, so the decomposition itself cannot fail. It is therefore licensed only by causal tests
that can.

**Method.** Cache, per eligible val exon, the final-residual at the masked donor in the clean and
corrupt runs (`h_cl`, `h_co`) and their margins. Define `U_info` = top-8 SVD of `h_cl − h_co`.
Test it with four gates, each an application of the exact MLM head to a modified residual.

**Evidence** (`donor_anchor.json`, val n = 250):

| Gate | Result |
|---|---|
| G0 — the logit-lens is exact | `max|cls(final) − logits| = 0.0` |
| Coupling (sanity) | **+0.990 [0.856, 1.126]** — matches the known unconditional val coupling |
| Donor margin vs base-rate floor | **+4.65** vs **+0.25** (masked non-site G) → +4.40 nats of donor-specific signal |
| **C-Suff** (patch only `U_info`) | rank-1 **0.74**, rank-8 **0.92** of the coupling recovered |
| **C-Null** (random rank-k, size-matched) | random rank-8 **≤0.05**; add-one **p = 0.005** |
| **C-Nec** (replace `U_info` with corrupt) | rank-8 **0.94** of the coupling destroyed |
| **C-Alive** (inject no-info baseline) | +4.65 → **+0.33** (collapses to floor) |

**Finding.** The coupling is carried by a low-rank, specific, causally-read, destructible
direction in the final residual — `U_info`. **Honest caveat:** it is only *moderately* low-rank
(12 dims to 80% of the energy); and the *subspace* overlap between the gradient read-direction
and `U_info` is weak (0.15, vs 0.002 random). That looked alarming but is a nonlinear-head effect
(the per-example gradient's top-variance directions differ from the info directions), not the
object going unread — the per-example alignment is strong (the local-linear read explains 82% of
the coupling) and C-Suff proves `U_info` is causally read. It is far from the random floor, so it
is not the near-orthogonal case that would force a reframing.

### 4.2 The forward arm — L3H1 delivers `U_info` (1b → C17)

**Reasoning.** 1a anchored the *output* object; `readout_path` showed L3H1's restoration recovers
most of the coupling through a distributed chain. Do they meet? Is the margin L3H1 restores
carried by `U_info`, or delivered through directions the anchor never saw? The near-tautology
(a *full* restoration would make `h_re ≈ h_cl`, whose `U_info`-component is clean by
construction) is defused by (a) L3H1's restoration being **partial** (~0.83), so the test is
whether the *part it restores* lies in `U_info`, and (b) a size-matched random-subspace null.

**Method.** Restore L3H1's clean output into the corrupt run (`patch_heads`), then *route* the
restored final residual through `U_info` (patch only its rank-8 component into `h_co`) and read
the margin. Localize by restoring L3H1 only at donor±5 vs everywhere-else.

**Evidence** (`donor_forward.json`, val n = 250):

| Measurement | Result |
|---|---|
| Total L3H1 recovery `R_L3` | **0.829 [0.783, 0.868]** — matches readout_path's 0.818 |
| `U_info`-routed recovery `R_U` | **0.770 [0.726, 0.809]** → **93%** of L3H1's recovery routes through `U_info` |
| Random rank-8 null | **≤0.047**; add-one **p = 0.005** |
| Localization | donor±5 restoration → **96%** of routing; everywhere-else → **6%** |
| Assembly (`U_info`-write per block, clean/corrupt/restored) | MLP3 3.71/**2.60**/3.63, MLP4 5.00/**4.02**/4.93; MLP5 5.75/**5.76**/5.63 |

**Finding.** `U_info` is what L3H1 delivers: 93% of its recovered margin flows through the
anchor's object, localized to exactly where L3H1 writes. The acceptor-dependent `U_info` write is
**assembled at MLP3 and MLP4** (corruption suppresses their `U_info`-write ~30%/~20%, restoration
recovers it); MLP5 writes `U_info` *acceptor-independently*. **Honest caveats:** a real **7%** of
L3H1's recovery routes *outside* rank-8 `U_info` (a small second route, open); and the direction
alignment (0.905) is partly by-construction, so the falsifiable weight sits on the routing and
localization, not the alignment.

### 4.3 Resolving the attention into heads — transporters (1c → C18)

**Reasoning.** L3H1 writes at donor±5 but the readout is at the donor; something carries the
signal across positions, and `readout_path` credited the L4/L5 attention *blocks*. Which heads,
and how — do they *route* (their attention pattern gathers the signal: QK) or *transport* content
(their value carries it: OV)? To find heads that carry the *acceptor-dependent* signal (not heads
merely present on the path), the intervention is **noising-to-corrupt** — replace a head's clean
output with the value it takes when the acceptor is destroyed — not mean-ablation, which would
flag any load-bearing head.

**Method.** Noise each of the 32 L4/L5 heads (plus L3H1/L2H15/L1H12 as references) and measure
the fraction of coupling destroyed, with gene-bootstrap CIs. For the top heads, split the
intervention into query+key (routing) vs value (content).

**Evidence** (`donor_heads.json`, val n = 120):

| Result | Value |
|---|---|
| Sanity (reference reader) | **L3H1 = 0.782** — the max of all heads, confirming the estimator |
| L4/L5 relays (CI-low > 0.05) | **L5H11 0.125**, **L4H2 0.122**, **L4H0 0.081** (selective: 3 of 32) |
| Router vs transporter | value-noising **0.14 / 0.12 / 0.08** vs query+key-noising **≈0** for all three |
| Head character (mean-ablation MLM) | all ~**0.51** → general-purpose heads, not dedicated |

**Finding.** The L4/L5 relay is three **transporter** heads — L5H11, L4H2, L4H0 — that carry
`U_info` *content* through their value/OV path; their attention routing is acceptor-independent.
They are **general-purpose** heads (the model reuses shared positional-mixing machinery), and
there is a **distributed tail** of smaller contributors below the relay threshold, so "three
heads" is the sparse core, not the whole story. **Honest caveat:** the QK/OV split was noised at
all positions (coarse), so the attention *geometry* (do they attend donor→flank) is not directly
shown.

### 4.4 Infrastructure — the CDS/splice-phase bridge

To ask whether `U_info` encodes reading frame (and to enable the period-3 thread), we built
splice-phase annotation from the GTF CDS `frame` column (`gfmcd.splicing.phase`), with the
strand-agnostic identity *intron phase = ((3 − frame) + L) mod 3*, plus per-base codon position.
It is **validated**: known-answer unit-tested (including ±strand and consistency with the phase
formula), and on real val data 90% of internal exons resolve to a clean CDS, phases 0/1/2 are
well-distributed, and — crucially — annotated phase agrees with intron-length-mod-3 only 32% of
the time (chance), confirming the two are genuinely distinct constructs.

### 4.5 The causal signature — what `U_info` encodes (1e → C19)

**Reasoning.** The prompt's central methodological demand: never characterize an edge by `AG→CC`
corruption alone. Because C-Suff (C16) tied the coupling to `U_info`, the coupling's signature
across many conditions *is* `U_info`'s signature. We stratified and re-corrupted the coupling
across seven conditions to discriminate generic splice evidence / paired-intron consistency /
donor-identity confidence / coding-context / phase-sensitivity.

**Evidence** (`donor_signature.json`, val n = 250):

| Condition | Result | Reading |
|---|---|---|
| **Partner vs non-partner AG** | **+0.966 [0.827,1.107]** vs **+0.026 [0.009,0.045]** (~37×) | paired-intron consistency (its *own* partner) |
| **Donor vs non-site G** | **+0.964** vs **+0.013 [−0.006,0.036]** (~74×) | donor-identity confidence (the *donor's* G) |
| **Strand** | + **1.067**, − **0.848** (overlapping) | symmetric — a real effect, not an artifact |
| **Intron length mod 3** | 0/1/2 = 1.05 / 0.84 / 0.99 (overlapping) | **not** raw cross-site phase alignment |
| **Annotated splice phase** | 0/1/2 = 1.06 / 0.93 / 0.99 (overlapping) | **not** biological boundary-phase |
| **Coding vs noncoding** | 1.01 vs 0.62 (overlapping; noncoding n=29) | mildly coding-enriched, but present on noncoding boundaries |

**Finding.** `U_info` encodes **paired-intron consistency + donor-identity confidence**, and it is
**phase-independent**. A donor GT is scored more probable when its *specific* paired acceptor is
intact — not any AG, not a global G shift, and not conditioned on reading frame. Both phase probes
being flat rules out *both* "raw cross-site alignment" and "biological boundary-phase
representation": the object carries site **consistency**, not frame. **Honest caveats:** measured
on the coupling rather than re-routed through `U_info` per condition; noncoding n = 29 is small.

---

## 5. The assembled circuit

Reading the five results as one mechanism:

1. **The acceptor is read.** L3H1 performs a conjunctive (key+value) read at the paired acceptor
   (C4) and writes a paired-intron-consistency signal into the donor±5 neighbourhood (C5, C17).
2. **The signal is transformed into `U_info`.** MLP3 and MLP4 nonlinearly assemble the
   acceptor-dependent content into the low-rank residual object `U_info` (C17); the write is
   suppressed under corruption and restored when L3H1 is restored.
3. **`U_info` is transported to the donor.** Three general-purpose layer-4/5 heads — L4H0, L4H2,
   L5H11 — carry `U_info`'s content to the masked-donor position through their value/OV path, not
   through routing (C18).
4. **`U_info` is read out.** At the donor residual, the exact MLM head maps `U_info` to the G-vs-
   {A,C,T} margin; the coupling is a low-rank, specific, destructible function of that direction
   (C16).
5. **The positional substrate.** The learned period-3 clock is *necessary* for the whole effect
   (C8), but since the readout is phase-independent (C19), its role here is best read as
   positional addressing — locating the partner acceptor at a codon-phased stride — rather than
   reading-frame computation. (The clock's detailed computation is pre-registered but not yet
   run; see §7.)

---

## 6. What SpliceBERT is doing (interpretation)

SpliceBERT was trained only to reconstruct masked nucleotides, never to select splice partners.
The most defensible reading of this circuit, consistent with all six signature conditions: **a
donor-like `GT` is scored more probable when the broader sequence contains a coherent, specific
paired acceptor.** The circuit distinguishes a genuine donor boundary from an incidental `GT` by
checking its own intron's acceptor — paired-intron consistency — and raises P(G) at the masked
donor accordingly. It does this without representing reading frame: the effect is indifferent to
splice phase and to intron length mod 3, and it operates (somewhat weakened) even on noncoding
boundaries. This is site-relational evidence, not a frame computation.

---

## 7. Honest limitations, and what remains open

- **Firewall.** Every result is OV/Behaviour-level. Nothing here reaches **Biology**: this is a
  faithful account of a computation *inside the model*, not a claim about splicing in cells.
- **Completeness is bounded.** The downstream carriers are a head-sparse *core* with a distributed
  tail (C18); a residual 7% of L3H1's recovery routes outside rank-8 `U_info` (C17). The circuit
  is dominant, not exhaustive.
- **Shared machinery.** The transporter heads are general-purpose, so this is not a set of
  dedicated splice units; head-level completeness is inherently bounded.
- **Measured on the coupling.** The signature (C19) inherits, via C-Suff, that the coupling is
  what `U_info` carries, rather than re-routing each condition through `U_info` — a reasonable
  economy that a per-condition routing check would tighten.
- **Open next steps.** (a) Period-3's *computation* — the discriminating truth-table (whole-codon
  vs single-base masking, CDS-specificity, codon phase) is pre-registered
  (`preregistrations/PREREGISTRATION_PERIOD3_COMPUTATION.md`) with its infrastructure built and validated, not yet
  run. (b) The attention *geometry* of the three transporters. (c) The 7% off-`U_info` route.
  (d) Whether an analogous object appears in another genomic foundation model.

---

## 8. Reproducibility

- **Branch** `splicebert-donor-graph` (not merged, not pushed). Environment: `uv sync --extra
  splicebert --extra dev`; MPS; `SpliceBERT.1024nt`.
- **Artifacts per experiment:** frozen `preregistrations/PREREGISTRATION_DONOR_{ANCHOR,FORWARD,HEADS,SIGNATURE}.md`
  → runner `scripts/splicebert/run_donor_{anchor,forward,heads,signature}.py` → result
  `results/splicebert_coupling/donor_{anchor,forward,heads,signature}.json` (each with a
  `REFLECTION` block of self-critique and an `ExperimentManifest`).
- **Reusable machinery** (unit-tested): `gfmcd.splicing.{unembed,subspace,anchor_gates,phase}`,
  16 tests green.
- **Run:** `PYTHONPATH=src uv run --extra splicebert --extra dev python
  scripts/splicebert/run_donor_anchor.py --device mps --n 250` (and the other three likewise).
- **Two pre-run methodological corrections**, made transparently before any analysis run and
  documented in the preregs: C-Nec was changed to the dual-of-sufficiency (bounded), and F-Health
  → F-Selective (mean-ablation was the wrong control for general-purpose heads).

---

*Claims registered: C16 (anchor), C17 (forward), C18 (heads), C19 (signature) in
[`CLAIMS.md`](CLAIMS.md). Prior context: C1–C8. This document is a synthesis; the frozen
pre-registrations and result JSONs are the primary record.*
