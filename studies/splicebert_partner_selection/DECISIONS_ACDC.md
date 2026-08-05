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
and the true corrupt forward at `H = ∅`, and both identities gate the run.

**Measured cost, 2026-08-05, real SpliceBERT.1024nt on MPS at S = 1024** (exactness confirmed in the
same run: `kernel_exact_full` 3.05e-5, `kernel_exact_empty` 1.53e-5, `per_head_input_identity` 2.86e-6,
all inside their gates):

| Implementation | Overhead vs plain forward | ms/example |
|---|---|---|
| per-head Python `bool(mask[...])` selection | **9.2×** | 101 |
| vectorised tensor selection over a destination axis | **15.3×** | 144 |

**The 2–3× written here at design time was wrong**, and so was the first diagnosis of why. The cost is
not per-head attention and it is not Python synchronisation: it is memory traffic. Edge granularity
requires a *separate residual per destination*, so the LayerNorm recursion runs over `[n_dst, B, S, 512]`
tensors roughly 528 times per forward against 12 in a plain pass. That is intrinsic to exact edge-level
evaluation in a post-LN model, not an implementation defect.

The vectorised form is kept even though it is slower **on MPS**, because per-element `bool()` selection
forces a device synchronisation per head and that is far more punishing on CUDA than on MPS. **That
ordering is an expectation, not a measurement** — the Colab benchmark cell measures both paths on the
run device and the faster one is selected there, with the choice recorded in the manifest.

Memory: 103 cached writes at `[B, S, 512]` fp32, ≈ 216 MB × B, plus a `[n_dst, B, S, 512]` transient
per stage. D7.5 auto-tunes B from free VRAM rather than fixing it at the 16 GB-era default of 4.

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

Local reference device is Apple MPS. A **plain** forward measures 18.4 ms/example, flat across batch
sizes 8–32 (compute-bound at S = 1024), so an A13 node-level condition at n = 64 ≈ 1.2 s. The **A14
edge kernel** measures 144 ms/example (D2), so an A14 condition at n = 64 ≈ 9.2 s.

| Arm | Conditions | Est. local (measured) | Runs on |
|---|---|---|---|
| A13 adaptive node greedy | ~5,356 | ~1.8 h | **local** |
| A14 reference smoke (n = 24, truncated) | ~200 | ~10 min | **local** — the artifact Colab is diffed against |
| A14 τ sweep, head-granular | 5 × 4,636 | **~60 h** (was estimated ~20 h) | **Colab GPU** |
| A14 calibration (planted + non-partner) | 2 × 4,636 | ~24 h | **Colab GPU**, before the target arm |
| A14 Q/K/V arm on survivors | ~2,000–6,000 | ~5–15 h | **Colab GPU** |

**The A14 estimates tripled** once the kernel was measured rather than assumed. The local numbers above
are what makes the Colab requirement load-bearing rather than a convenience: at MPS speed the τ sweep
alone is two and a half days. Whether a GPU closes the gap is the question the benchmark cell answers
before `LAUNCH_TARGET` is set — and if it does not, the honest responses are to cut the τ grid, cut
discovery `n`, or report an instrument limit, **not** to relax the τ floor or drop calibration arms.

A13 staying local is unchanged and still cheap: it uses the A6 node scrub, not this kernel.

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

---

## D7 — CUDA efficiency, and the one place it must not be bought

The τ sweep is ~4,600 sequential subgraph evaluations per threshold. Six decisions shape what the GPU
buys, and one of them is a place where a routine optimization would silently damage the result.

**D7.1 — TF32 and reduced precision are OFF on the decision path.** On Ampere and later, TF32 carries a
10-bit mantissa. Enabling it (or bf16 autocast) raises `device_floor`, which by A14 §4 raises the
minimum admissible τ, which coarsens the recovered circuit — a scientific cost paid for a throughput
gain, invisible in the artifact. The runner sets `torch.backends.cuda.matmul.allow_tf32 = False` and
`torch.backends.cudnn.allow_tf32 = False` explicitly (not by relying on a library default, which has
changed across versions) and records both in the manifest. The benchmark cell additionally *measures*
what bf16 and TF32 would cost in floor terms and reports it, so the trade is quantified rather than
assumed. **This is the only optimization in this section that is refused rather than tuned.**

**D7.2 — The edge set is a boolean mask tensor, not a Python set.** `mask[src, dst]` of shape
`[103, 103]`. Removing an edge is a mask flip, so the graph is structurally static across the whole
sweep and the selection `sel_v(b) = where(mask[b, v], live[b], corrupt[b])` is a fused tensor op
evaluated for all destinations at once. This removes Python from the inner loop and is the
precondition for everything below.

**D7.3 — Per-head attention is batched, not looped.** The kernel's cost is that heads in a layer no
longer share an input, so a naive implementation issues 16 small attention calls per layer. Stacking
per-head inputs to `[H, B, S, D]` and contracting against sliced weights
(`einsum('hbse,hde->hbsd', r_heads, W_q_h)`) collapses that to one batched matmul per projection per
layer. This is the difference between the ~2–3× slowdown estimated at design time and something close
to parity.

**D7.4 — SDPA for the attention math.** At S = 1026 the materialized attention tensor is
`[B, 16, 1026, 1026]` — 2.2 GB at B = 32 in fp32. `scaled_dot_product_attention` avoids materializing
it, which is what makes large batches affordable. `load_adapter`'s `eager=True` requirement exists
because SDPA silently returns zero attention weights when `output_attentions=True`; A14's head-granular
arm never reads attention weights, so the requirement does not bind. Numerical agreement with the eager
path is verified against the device floor before the SDPA path is used.

**D7.5 — Batch size is auto-tuned from free VRAM, not hardcoded.** Corrupt write caches are constant
across the entire sweep and cost `103 × B × 1026 × 512 × 4 B ≈ 216 MB × B`, with a transient of the
same order for live writes. So B = 4 → ~0.9 GB, B = 16 → ~3.5 GB, B = 32 → ~6.9 GB. The runner queries
`torch.cuda.mem_get_info()` and selects B under a safety factor, recording the chosen value. The design
default of B = 4 was set by a 16 GB assumption; on a 40 GB A100 it wastes most of the device.

**D7.6 — Candidate-parallel evaluation is an arm with an equivalence gate, not a substitution.** Within
a destination node, ACDC tests each parent edge in turn and *recomputes* after each accepted removal,
so later candidates are tested against an already-pruned `H`. Evaluating all ~45 in-edges of a node in
one batched forward — replicating along a candidate axis, which is exactly the parallelism a GPU
supplies and MPS does not — reduces the sweep from ~4,600 sequential steps to ~103, roughly 4 h to
~10 min per τ.

But it is **not Algorithm 1**: it tests every candidate against the node's entry `H`. So:

- `--edge-eval sequential` (exact Algorithm 1) is the default and produces the reported circuit unless
  the gate below licenses otherwise.
- `equivalence_check` memorial: run 8 destination nodes spanning layers 0–5 under both paths. The
  retained edge sets must be **identical**. If they are, the parallel path is licensed for this task
  with that evidence recorded in the manifest; if they are not, the discrepancy is reported and the
  sequential path stands.
- The manifest records which path produced the reported circuit. A speedup taken without the gate
  passing is a protocol violation, not a tuning choice.

**D7.7 — Deferred and measured, not assumed.** CUDA graphs and `torch.compile` are plausible further
wins (the shapes are fixed across the sweep, which suits both). Both can perturb numerics, and D7.2
already removes most launch overhead. They stay off by default; the benchmark cell reports what they
would give, and adopting either requires re-measuring `device_floor` under it.
