# Decisions — A13/A14 automated circuit discovery (ACDC)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Engineering and execution decisions behind
[`PREREGISTRATION_A13_ADAPTIVE_MINIMALITY.md`](preregistrations/PREREGISTRATION_A13_ADAPTIVE_MINIMALITY.md)
and [`PREREGISTRATION_A14_EDGE_ACDC.md`](preregistrations/PREREGISTRATION_A14_EDGE_ACDC.md). The
scientific protocol lives in those two documents; what follows is why the machinery is shaped the way
it is.

---

## D1 — This extends the donor-graph arc; it is not a new study

A13/A14 use the frozen `CouplingTask`, the frozen corruption operator (`CORRUPT_ALT = "CC"` at the
intron-mate acceptor AG), the frozen eligibility filter, and the A6 faithfulness `F`. They report into
`studies/splicebert_partner_selection/CLAIMS.md` and can change C33's status. So they are experiments
17 and 18 of the A0→A12 arc, with runners in `scripts/splicebert/` and preregistrations in
`preregistrations/` — not a new `studies/` directory. The reusable kernel goes to
`src/gfmcd/circuits/acdc.py` per `AGENTS.md`'s rule that reusable circuit primitives live in the
circuit package.

**Spec location.** The brainstorming workflow's default is a dated file under `docs/superpowers/specs/`.
`AGENTS.md` forbids dated documents in the tree and names `DECISIONS*.md` and the preregistrations as
the home for durable design records. The repository's own rule wins; this file and the two preregs are
the design record.

---

## D2 — Why ACDC's standard edge patching cannot be used unmodified

ACDC's subgraph evaluation assumes residual additivity, `r_v = Σ_{u ∈ pa(v)} write(u)`. That holds in
the pre-LayerNorm decoder models ACDC was built on. **SpliceBERT is post-LayerNorm**: HF `BertLayer`
applies `LayerNorm(dense(context) + input)` and then `LayerNorm(dense(intermediate) + attn_output)`, so
the stream is renormalized twice per layer and is not a sum of upstream writes at any depth.

Three options were considered:

| Option | Verdict |
|---|---|
| **Fold / freeze LayerNorm** — treat LN scale as a constant from a reference run, recovering additivity | Rejected. It makes the subgraph evaluation an approximation whose error is uncontrolled at exactly the small-`ΔF` decisions ACDC's threshold rule turns on. |
| **Per-edge path patching** (Goldowsky-Dill et al.) — exact, but needs a separate forward per destination to build each input | Rejected on cost: ~103 forward passes per subgraph evaluation, ≈ 2 min per candidate edge. |
| **Exact replay of the LN recursion per destination** | **Chosen.** See A14 §2. |

The chosen construction is exact, not approximate: it reduces to the true clean forward at `H = ALL`
and the true corrupt forward at `H = ∅`, and both identities gate the run. The additional cost is that
heads within a layer no longer share an input, so the kernel runs 96 per-head attention computations
instead of 6 batched ones. Since QK^T is already per-head, the loss is batching efficiency, not
asymptotic work — measured at roughly 2–3× a plain forward pass, which the benchmark cell confirms
before any target run launches.

Memory: 103 cached writes at `[B, S, 512]` fp32. At `B = 4`, ≈ 0.9 GB — the reason the kernel's default
batch is 4 and not 8.

---

## D3 — τ is derived from a measured device floor

ACDC's rule is `ΔF < τ`. A τ below the hardware's numerical reproducibility floor yields a circuit that
depends on which accelerator ran it, with nothing in the artifact disclosing that.

This repository has already measured such a floor twice: `colab_atlas_package/CHECKLIST.md` gates its
cross-device smoke at **absolute tolerance 0.002**, and `MECHANISM_VALIDATION_AMENDMENT_A3.md` accepts a
cross-batch reference discrepancy in `(1e-6, 1e-4]`. An a-priori grid starting at 3e-4 — the natural
first guess — would have sat **below** the Atlas floor.

So the grid is not chosen. It is measured (A14 §4), gated by `tau_above_device_floor`, and recorded in
the manifest. If the floor leaves no informative range, the run stops and reports an instrument limit.

`decision_determinism` covers the adjacent failure: ACDC is a sequential greedy, so a single
nondeterministic kernel call early in a sweep changes every downstream decision. The gate re-runs the
first 200 decisions on one device and requires a bit-identical accept/reject sequence.

---

## D4 — Execution boundary: what runs where

Local reference device is Apple MPS, measured at **18.4 ms/example**, flat across batch sizes 8–32
(compute-bound at S = 1024). One subgraph evaluation at n = 64 ≈ 1.2 s.

| Arm | Evaluations | Est. local | Runs on |
|---|---|---|---|
| A13 adaptive node greedy | ~5,356 | ~1.8 h | **local** |
| A14 reference smoke (n = 24, truncated) | ~200 | ~5 min | **local** — this is the artifact Colab is diffed against |
| A14 τ sweep, head-granular | 5 × ~4,600 | ~20 h | **Colab GPU** |
| A14 calibration (planted + non-partner) | 2 × ~4,600 | ~8 h | **Colab GPU**, before the target arm |
| A14 Q/K/V arm on survivors | ~2,000–6,000 | ~4–8 h | **Colab GPU** |

Everything over three hours goes to Colab. A13 staying local is deliberate: it validates the metric,
the splits, and both calibration constructions that A14 reuses, before any bundle is built.

---

## D5 — Colab package design

`studies/splicebert_partner_selection/colab_acdc_package/`, following
`studies/splicebert_head_context_atlas/colab_atlas_package/` rather than inventing a second pattern:

- **`build_bundle.py`** — zip containing all of `src/gfmcd/**.py`, the A13/A14 runners, the relevant
  tests, the cohort artifact, the local reference smoke artifact, and the 75 MB checkpoint. Emits an
  in-zip `BUNDLE_MANIFEST.json` with per-file sha256 plus an external `.sha256.json` sidecar, and
  **refuses to overwrite an existing bundle**.
- **`requirements-colab.txt`** — pins `transformers`, `numpy`, `scipy`, `pytest`. **Never `torch`** —
  Colab's preinstalled CUDA build must survive.
- **`README.md`**, **`CHECKLIST.md`**, **`run_acdc_colab.ipynb`**.

**No GTF or FASTA on Colab.** The cohort ships as
`results/splicebert_coupling/acdc_cohort_v1.sequences.jsonl` (~500 KB), built locally by
`scripts/splicebert/build_acdc_cohort.py` and deterministic given `(GTF, FASTA, split, tag, seed)`.
This follows the Atlas package ("the outcome-independent sequence windows are bundled") and the
convention of committing the cohort artifact that makes a claim auditable. It also improves on
rebuilding from a 3.2 GB genome: the exact cohort is sha256-pinned rather than regenerated.

**The seal is enforced by absence.** Only `val` sequences enter the bundle. chr8/9 are not in the file,
so a remote runtime cannot spend the seal by accident — the same mechanism the Atlas package uses
("chr8/9 and historically touched chr11/16/19 are absent from the packaged sequences").

**Resume granularity is the destination node.** ACDC state is `(live edge set, node index, parent
index, decision log, RNG state)`, checkpointed to Drive after each of the 103 destination nodes as
`acdc_state_<tau>_<node>.json`. At ~4 h per τ that is roughly five-minute granularity, so a Colab
disconnect costs one node rather than a run. This is the Atlas shard-resume with the node as the shard.

**Notebook phase order** (Atlas order, adapted):

0. Mount Drive → verify bundle sha256 → assert CUDA → assert torch was not reinstalled
1. Run tests; run the cross-device reference smoke against the committed local artifact →
   **this is what measures `device_floor`** and therefore sets the τ grid
2. Benchmark 32 conditions: GPU model, peak memory, s/condition, projected hours per τ
3. Calibration arms — planted-circuit recovery and non-partner null. Both must pass.
4. **Owner reads the benchmark and sets `LAUNCH_TARGET = True`.** The agent does not set it.
5. Target τ sweep, resumable per node, writing to Drive
6. Validate → merge → compact JSON + manifest → returned into `results/splicebert_coupling/`

---

## D6 — Deferred, and deliberately not decided here

- **Whether to spend the sealed chr8/9 test.** A14 reports at verification tier on `val`, which A12 and
  A13 have already seen for the adjacent question. If A14 contradicts C33, opening the seal becomes a
  decision for the owner at the REFLECTION gate. This protocol does not license it.
- **Whether `R_core` is the right reference edge set.** It is assembled from C29/C31/C35 and the frozen
  `CIRCUIT` constant; no single ledger row states it as one circuit. It is frozen in A14 §6 before the
  run precisely so that it cannot be tuned after seeing the TPR — but its composition is a judgement
  made here, and a reviewer may reasonably contest it. Contesting it after the run requires a new
  reference set and a re-scored, separately reported arm.
- **Promotion of any unreferenced edge ACDC recovers.** Those are attribution-level candidates. The
  evidence ladder is not crossed by an edge surviving a greedy search.
