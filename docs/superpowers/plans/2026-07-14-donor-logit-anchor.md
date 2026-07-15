# Donor-Logit Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and causally validate the residual→donor-logit readout (the backward-arm foundation of the L3H1→masked-donor-logit meet-in-the-middle trace), per `studies/splicebert_partner_selection/PREREGISTRATION_DONOR_ANCHOR.md`.

**Architecture:** Two small reusable, unit-tested modules — `unembed.py` (logit-lens: apply the MLM head to any residual) and `subspace.py` (SVD subspace + project/patch/overlap primitives, numpy) — plus one runner `run_donor_anchor.py` that caches clean/corrupt/no-site final residuals once and runs every frozen gate as cheap `cls()` re-applications (no re-forward). A synthetic-oracle test proves each gate can output both PASS and FAIL.

**Tech Stack:** PyTorch (MPS), transformers 5.x (eager not required here — no attention read), numpy, pytest. All reuse from `gfmcd.splicing.circuit`, `gfmcd.rigor`, `gfmcd.provenance`.

## Global Constraints

- Model: `SpliceBERT.1024nt` via `load_adapter(DEFAULT_MODEL, dev, eager=False)`; `dev` from `pick_device(--device)`; default MPS on this host.
- **Never redefine the target.** Readout, corruption, coupling, eligibility import unchanged from `gfmcd.splicing.circuit` (`CouplingTask`, `load_examples`, `CORRUPT_ALT`, `GAP_MIN/MAX`).
- Recovery is **ratio-of-means** (`ratio_of_means` + gene-grouped `gboot_ratio`), on the **unconditional** set (`load_examples(..., min_effect=None)`). No per-example coupling conditioning (dodges R10).
- p-values: exact **add-one** via `gfmcd.rigor.empirical_upper_p_value`; floor `1/(N+1)`; **never print `p=0.0000`**.
- Interventions are **one-way subspace substitutions**, never swaps (R7).
- Seeds fixed at 0. Set `PYTORCH_MPS_LOW/HIGH_WATERMARK_RATIO` defensively at runner import.
- Every result JSON gets a sibling `.manifest.json` via `ExperimentManifest`.
- Frozen thresholds (from the prereg, verbatim): G0 `<1e-3`; C-Suff rank-1 `≥0.50`, rank-8 `≥0.80`; C-Nec rank-8 `≥0.80`; C-Null observed `> max(200 random)` and add-one `p≤0.05` at k∈{1,8}; C-Alive `mean(m_alive) ≤ floor+0.10`.

---

### Task 1: `unembed.py` — the logit-lens (reader-direction primitive)

**Files:**
- Create: `src/gfmcd/splicing/unembed.py`
- Test: `tests/test_unembed.py`

**Interfaces:**
- Produces: `donor_logits(cls, h)`, `margin_hard(logits, bid)`, `margin_smooth(logits, bid)`, `read_gradient(cls, h, bid)`. `cls` is any `nn.Module` mapping `[...,hidden]→[...,vocab]` (the real `ad.model.cls`, or a `nn.Linear` in tests). `bid` is `{"A":id,...}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_unembed.py
import numpy as np, torch
from gfmcd.splicing.unembed import margin_hard, margin_smooth, read_gradient, donor_logits

BID = {"A": 0, "C": 1, "G": 2, "T": 3}

def test_margin_hard_matches_reference():
    logits = torch.tensor([[0.5, 1.0, 3.0, -2.0, 9.9]])  # extra vocab col ignored
    assert torch.allclose(margin_hard(logits, BID), torch.tensor([3.0 - 1.0]))

def test_margin_smooth_is_logit_minus_logsumexp():
    logits = torch.randn(7, 5)
    g = logits[:, BID["G"]]
    lse = torch.logsumexp(torch.stack([logits[:, BID[b]] for b in "ACT"], -1), -1)
    assert torch.allclose(margin_smooth(logits, BID), g - lse)

def test_read_gradient_of_linear_head_is_analytic():
    torch.manual_seed(0)
    cls = torch.nn.Linear(6, 5, bias=False)
    h = torch.randn(3, 6)
    g = read_gradient(cls, h, BID)                      # [3,6]
    # analytic: d/dh [W_G - logsumexp(W_A,W_C,W_T)] . h  = W_G - sum_b softmax_b W_b
    W = cls.weight.detach()
    logits = h @ W.T
    p = torch.softmax(torch.stack([logits[:, BID[b]] for b in "ACT"], -1), -1)  # [3,3]
    want = W[BID["G"]][None, :] - (p[:, 0:1]*W[BID["A"]] + p[:, 1:2]*W[BID["C"]] + p[:, 2:3]*W[BID["T"]])
    assert torch.allclose(g, want, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cberg/Research/projects/gfm-circuit-discovery && PYTHONPATH=src uv run pytest tests/test_unembed.py -q`
Expected: FAIL, `ModuleNotFoundError: gfmcd.splicing.unembed`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gfmcd/splicing/unembed.py
"""Logit-lens for the donor readout: apply the MLM head to any residual.

`logits = cls(hidden_states[-1])` exactly, so the donor margin reads off any final-layer
residual with no re-forward. Backward-arm primitive of the L3H1->donor-logit trace
(prereg: studies/splicebert_partner_selection/PREREGISTRATION_DONOR_ANCHOR.md).
"""
from __future__ import annotations
import torch


def donor_logits(cls: torch.nn.Module, h: torch.Tensor) -> torch.Tensor:
    return cls(h)


def margin_hard(logits: torch.Tensor, bid: dict) -> torch.Tensor:
    """logit(G) - max(logit A,C,T). The frozen readout (matches CouplingTask.margin)."""
    alt = torch.stack([logits[..., bid[b]] for b in "ACT"], -1).max(-1).values
    return logits[..., bid["G"]] - alt


def margin_smooth(logits: torch.Tensor, bid: dict) -> torch.Tensor:
    """logit(G) - logsumexp(logit A,C,T). Smooth surrogate for gradient direction-finding."""
    alt = torch.logsumexp(torch.stack([logits[..., bid[b]] for b in "ACT"], -1), -1)
    return logits[..., bid["G"]] - alt


def read_gradient(cls: torch.nn.Module, h: torch.Tensor, bid: dict) -> torch.Tensor:
    """d(margin_smooth)/dh per row -- the local readout direction. Returns [N, hidden]."""
    with torch.enable_grad():
        h = h.detach().clone().requires_grad_(True)
        margin_smooth(cls(h), bid).sum().backward()
        return h.grad.detach()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/cberg/Research/projects/gfm-circuit-discovery && PYTHONPATH=src uv run pytest tests/test_unembed.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/cberg/Research/projects/gfm-circuit-discovery
git add src/gfmcd/splicing/unembed.py tests/test_unembed.py
git commit -m "feat(anchor): logit-lens unembed + donor margins (donor-graph 1a)"
```

---

### Task 2: `subspace.py` — SVD subspace + project/patch/overlap primitives

**Files:**
- Create: `src/gfmcd/splicing/subspace.py`
- Test: `tests/test_subspace.py`

**Interfaces:**
- Produces (all numpy, `rows`/`h` are `[N,D]`, `U`/`V` are orthonormal `[k,D]`): `top_k_subspace(rows,k)`, `project(h,U)`, `project_out(h,U)`, `patch_component(h_target,h_source,U)`, `subspace_overlap(U,V)`, `random_subspace(D,k,rng)`, `energy_spectrum(rows)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subspace.py
import numpy as np
from gfmcd.splicing.subspace import (
    top_k_subspace, project, project_out, patch_component, subspace_overlap,
    random_subspace, energy_spectrum)

def test_top_k_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((10, 2)))[0].T          # [2,10]
    rows = rng.standard_normal((200, 2)) @ basis                     # lie in that plane
    U = top_k_subspace(rows, 2)
    assert subspace_overlap(U, basis) > 0.999

def test_project_is_identity_in_span_and_zero_orthogonal():
    rng = np.random.default_rng(1)
    U = random_subspace(8, 2, rng)
    x_in = (rng.standard_normal((5, 2))) @ U                          # in span
    assert np.allclose(project(x_in, U), x_in, atol=1e-10)
    x_out = project_out(x_in, U)
    assert np.allclose(x_out, 0, atol=1e-10)

def test_patch_component_transfers_only_the_span():
    rng = np.random.default_rng(2)
    U = random_subspace(8, 3, rng)
    a, b = rng.standard_normal((4, 8)), rng.standard_normal((4, 8))
    p = patch_component(a, b, U)
    assert np.allclose(project(p, U), project(b, U), atol=1e-10)      # span comes from b
    assert np.allclose(project_out(p, U), project_out(a, U), atol=1e-10)  # rest stays a

def test_overlap_bounds():
    rng = np.random.default_rng(3)
    U = random_subspace(64, 4, rng)
    assert abs(subspace_overlap(U, U) - 1.0) < 1e-9
    V = random_subspace(64, 4, rng)
    assert subspace_overlap(U, V) < 0.30                              # ~4/64, well below 1

def test_energy_spectrum_monotone_to_one():
    rng = np.random.default_rng(4)
    e = energy_spectrum(rng.standard_normal((50, 10)))
    assert e[-1] == 1.0 and np.all(np.diff(e) >= -1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cberg/Research/projects/gfm-circuit-discovery && PYTHONPATH=src uv run pytest tests/test_subspace.py -q`
Expected: FAIL, `ModuleNotFoundError: gfmcd.splicing.subspace`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gfmcd/splicing/subspace.py
"""Subspace primitives for the donor-anchor causal tests (numpy, model-free)."""
from __future__ import annotations
import numpy as np


def top_k_subspace(rows, k: int) -> np.ndarray:
    """Top-k right singular vectors of `rows` [N,D] -> orthonormal U [k,D]."""
    _, _, Vh = np.linalg.svd(np.asarray(rows, float), full_matrices=False)
    return Vh[:k]


def project(h, U) -> np.ndarray:
    """Component of `h` [N,D] in span(U) [k,D] (U orthonormal rows)."""
    h = np.asarray(h, float)
    return (h @ U.T) @ U


def project_out(h, U) -> np.ndarray:
    return np.asarray(h, float) - project(h, U)


def patch_component(h_target, h_source, U) -> np.ndarray:
    """Replace h_target's span(U) component with h_source's."""
    return project_out(h_target, U) + project(h_source, U)


def subspace_overlap(U, V) -> float:
    """Mean squared cosine of principal angles between span(U), span(V) [k,D] orthonormal."""
    s = np.linalg.svd(np.asarray(U, float) @ np.asarray(V, float).T, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0) ** 2))


def random_subspace(D: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random orthonormal U [k,D]."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return Q.T


def energy_spectrum(rows) -> np.ndarray:
    """Cumulative fraction of squared-singular-value energy of `rows` [N,D]."""
    s = np.linalg.svd(np.asarray(rows, float), compute_uv=False)
    e = np.cumsum(s ** 2)
    return e / e[-1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/cberg/Research/projects/gfm-circuit-discovery && PYTHONPATH=src uv run pytest tests/test_subspace.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/cberg/Research/projects/gfm-circuit-discovery
git add src/gfmcd/splicing/subspace.py tests/test_subspace.py
git commit -m "feat(anchor): numpy subspace primitives (project/patch/overlap)"
```

---

### Task 3: `gates.py` — the four causal tests, with a synthetic oracle proving each can PASS and FAIL

**Files:**
- Create: `src/gfmcd/splicing/anchor_gates.py`
- Test: `tests/test_anchor_gates.py`

This task exists to satisfy "a control that cannot fail is not a control" at the code level: the oracle constructs inputs with a known answer and asserts each gate returns PASS on the intended-positive and FAIL on the intended-negative.

**Interfaces:**
- Consumes: `subspace.*`, `unembed.margin_hard`, `circuit.ratio_of_means`.
- Produces: `suff_recovery(margins_fn, H_cl, H_co, m_cl, m_co, U)`, `nec_fraction(margins_fn, H_cl, m_cl, m_co, U)`, `null_recoveries(margins_fn, H_cl, H_co, m_cl, m_co, k, n_random, rng)`, `alive_margin(margins_fn, h_baseline, N)`. `margins_fn(H_np)->np[N]` applies the real (or fake) unembed+`margin_hard` to residuals.

- [ ] **Step 1: Write the failing test (the oracle)**

```python
# tests/test_anchor_gates.py
import numpy as np
from gfmcd.splicing.subspace import top_k_subspace, random_subspace
from gfmcd.splicing.anchor_gates import (
    suff_recovery, nec_fraction, null_recoveries, alive_margin)

def _linear_margins_fn(direction, floor=0.0):
    """A fake unembed whose donor-margin is (h . direction) + floor."""
    d = direction / np.linalg.norm(direction)
    return lambda H: np.asarray(H, float) @ d + floor

def test_lowrank_coupling_passes_suff_nec_and_null():
    rng = np.random.default_rng(0); D, N = 64, 120
    d = rng.standard_normal(D); d /= np.linalg.norm(d)
    H_co = rng.standard_normal((N, D))
    H_cl = H_co + 1.5 * np.outer(rng.standard_normal(N) + 3.0, d)   # coupling ALONG d
    mf = _linear_margins_fn(d)
    m_cl, m_co = mf(H_cl), mf(H_co)
    U = top_k_subspace(H_cl - H_co, 1)
    assert suff_recovery(mf, H_cl, H_co, m_cl, m_co, U) > 0.95      # PASS branch
    assert nec_fraction(mf, H_cl, m_cl, m_co, U) > 0.95
    nulls = null_recoveries(mf, H_cl, H_co, m_cl, m_co, 1, 200, rng)
    assert suff_recovery(mf, H_cl, H_co, m_cl, m_co, U) > nulls.max()  # beats random

def test_highrank_coupling_fails_sufficiency():
    rng = np.random.default_rng(1); D, N = 64, 120
    H_co = rng.standard_normal((N, D))
    H_cl = H_co + rng.standard_normal((N, D))                        # coupling spread over ALL dims
    mf = lambda H: np.asarray(H, float).sum(1)                       # reads every dim equally
    m_cl, m_co = mf(H_cl), mf(H_co)
    U = top_k_subspace(H_cl - H_co, 8)
    assert suff_recovery(mf, H_cl, H_co, m_cl, m_co, U) < 0.50       # FAIL branch reachable

def test_alive_collapses_for_reader_but_not_for_copier():
    rng = np.random.default_rng(2); D = 64
    d = rng.standard_normal(D)
    reader = _linear_margins_fn(d, floor=0.0)
    h_baseline = np.zeros(D)                                         # reader -> margin 0 == floor
    assert abs(alive_margin(reader, h_baseline, 10)) < 1e-9          # collapses
    copier = lambda H: np.full(len(H), 5.0)                          # ignores h
    assert alive_margin(copier, h_baseline, 10) == 5.0              # does NOT collapse -> caught
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cberg/Research/projects/gfm-circuit-discovery && PYTHONPATH=src uv run pytest tests/test_anchor_gates.py -q`
Expected: FAIL, `ModuleNotFoundError: gfmcd.splicing.anchor_gates`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gfmcd/splicing/anchor_gates.py
"""The four donor-anchor causal tests. `margins_fn(H)->np[N]` applies unembed+margin_hard."""
from __future__ import annotations
import numpy as np
from gfmcd.splicing.circuit import ratio_of_means
from gfmcd.splicing.subspace import patch_component, project_out, random_subspace


def suff_recovery(margins_fn, H_cl, H_co, m_cl, m_co, U) -> float:
    """Patch only span(U) from clean into corrupt; fraction of coupling recovered."""
    m = margins_fn(patch_component(H_co, H_cl, U))
    return ratio_of_means(m - m_co, m_cl - m_co)


def nec_fraction(margins_fn, H_cl, m_cl, m_co, U) -> float:
    """Project span(U) out of clean; fraction of coupling lost."""
    m = margins_fn(project_out(H_cl, U))
    return ratio_of_means(m_cl - m, m_cl - m_co)


def null_recoveries(margins_fn, H_cl, H_co, m_cl, m_co, k, n_random, rng) -> np.ndarray:
    D = H_cl.shape[1]
    return np.array([suff_recovery(margins_fn, H_cl, H_co, m_cl, m_co,
                                   random_subspace(D, k, rng)) for _ in range(n_random)])


def alive_margin(margins_fn, h_baseline, N) -> float:
    """Mean donor margin when every donor residual is replaced by the no-info baseline."""
    return float(np.mean(margins_fn(np.tile(np.asarray(h_baseline, float), (N, 1)))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/cberg/Research/projects/gfm-circuit-discovery && PYTHONPATH=src uv run pytest tests/test_anchor_gates.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/cberg/Research/projects/gfm-circuit-discovery
git add src/gfmcd/splicing/anchor_gates.py tests/test_anchor_gates.py
git commit -m "feat(anchor): four causal gates + synthetic oracle (can PASS and FAIL)"
```

---

### Task 4: `run_donor_anchor.py` — cache residuals once, run every gate, write JSON + manifest + REFLECTION

**Files:**
- Create: `scripts/splicebert/run_donor_anchor.py`
- Output: `results/splicebert_coupling/donor_anchor.json` (+ `.manifest.json`)

**Interfaces:**
- Consumes: `circuit.{load_adapter,pick_device,CouplingTask,load_examples,ratio_of_means,gboot_ratio,DEFAULT_*}`, `unembed.{margin_hard,margin_smooth,read_gradient}`, `subspace.{top_k_subspace,subspace_overlap,random_subspace,energy_spectrum}`, `anchor_gates.*`, `rigor.empirical_upper_p_value`, `provenance.ExperimentManifest`.

- [ ] **Step 1: Caching layer.** Set MPS watermark env at top. Load adapter (`eager=False`), `task = CouplingTask(ad)`, `bid = task.bid`, `off = task.off`, `cls = ad.model.cls`. `X, _ = load_examples(gtf, fasta, split, n, task=task, tag=f"anchor-{split}", min_effect=None)`; `genes=[x.gene_id for x in X]`. For each batch: `cl, co, dn = task.pair(ch)`; run `ad.model(**cl, output_hidden_states=True)` and `**co`, gather `hidden_states[-1][i, d+off]` into `H_cl, H_co` (numpy `[N,512]`); `m_cl, m_co = margin_hard(logits at masked pos)`. Also build the **no-site-G** set: per example pick a G with `|p-donor_pos|>=30 and |p-next_acceptor_pos|>=30` (deterministic: nearest to sequence midpoint), mask only it, forward, cache its final residual and margin; `floor = mean(non-site margins)`, `h_baseline = mean(non-site residuals)`.

- [ ] **Step 2: G0 gate.** For one batch assert `max|cls(hidden_states[-1])[i,d+off] - logits[i,d+off]| < 1e-3`. Abort with a clear message if it fails (wrong tensor/position). Record `g0_max_abs`.

- [ ] **Step 3: Descriptive D1/D2/D3.** `U_info_k = top_k_subspace(H_cl-H_co, k)`, `U_read_k = top_k_subspace(read_gradient(cls, torch(H_cl), bid) as np, k)` for k∈{1,2,4,8}. D1: `energy_spectrum(H_cl-H_co)` and of gradients → dims to 80/90%. D2: `subspace_overlap(U_read_k, U_info_k)` and the random baseline `mean(subspace_overlap(random,random))`. D3: `corr(rowwise (g_i·Δh_i), (m_cl-m_co))` and fraction explained.

- [ ] **Step 4: Causal gates.** Define `margins_fn(H_np)`: `torch.from_numpy(H_np).to(device,dtype) -> cls -> margin_hard -> np`. For k∈{1,8} and each of `U_info_k, U_read_k`: `suff = suff_recovery(...)` (+ `gboot_ratio` CI), `nec = nec_fraction(...)`; `nulls = null_recoveries(..., k, 200, rng)`, `p = empirical_upper_p_value(suff, nulls)`. C-Alive: `alive = alive_margin(margins_fn, h_baseline, N)`. Evaluate each PASS/FAIL against the Global-Constraints thresholds.

- [ ] **Step 5: Write JSON + manifest + REFLECTION.** JSON holds every number, the pass/fail booleans, `m_cl.mean()`, `m_co.mean()`, coupling mean + CI, `floor`, `m_cl_mean - floor`, and a `REFLECTION` block templating the three prereg §7 checks (filled after inspection in Task 5). `ExperimentManifest.create("splicebert_coupling:donor_anchor", sys.argv, config={...}, notes=[...]).write(...manifest.json)`.

- [ ] **Step 6: Commit**

```bash
cd /Users/cberg/Research/projects/gfm-circuit-discovery
git add scripts/splicebert/run_donor_anchor.py
git commit -m "feat(anchor): runner caches residuals + runs G0/D1-3/C-* gates"
```

---

### Task 5: Execute, verify, and reflect

- [ ] **Step 1: Full test suite green.** `PYTHONPATH=src uv run pytest tests/test_unembed.py tests/test_subspace.py tests/test_anchor_gates.py -q` → all pass.
- [ ] **Step 2: Run.** `PYTHONPATH=src uv run python scripts/splicebert/run_donor_anchor.py --device mps --n 250` → writes `donor_anchor.json`. Watch for the G0 assert.
- [ ] **Step 3: Verify against the frozen thresholds** (Global Constraints), verbatim, no softening.
- [ ] **Step 4: Write the prereg §7 REFLECTION** into the JSON and report to the user: pass/fail table; the intent check (does D2 alignment hold so the backward arm has one target? is `m_cl-floor` substantial?); any escaped failure mode. If the anchor fails or `U_read⟂U_info`, surface the program-reframe, do not patch.
- [ ] **Step 5: Commit results.**

```bash
cd /Users/cberg/Research/projects/gfm-circuit-discovery
git add results/splicebert_coupling/donor_anchor.json results/splicebert_coupling/donor_anchor.manifest.json studies/splicebert_partner_selection/PREREGISTRATION_DONOR_ANCHOR.md
git commit -m "results(anchor): donor-logit readout is/ isn't low-rank, read-aligned, specific"
```

---

## Self-Review

**Spec coverage:** G0→Task 4.2; unembed→Task 1; U_read/U_info + D1/D2/D3→Task 4.3; C-Suff/C-Nec→Tasks 3+4.4; C-Null→Tasks 3+4.4; C-Alive→Tasks 3+4.4; floor + h_baseline→Task 4.1; ratio-of-means/gboot→imported; manifest→Task 4.5; §7 reflection→Task 5.4. All prereg sections covered.

**Placeholder scan:** Tasks 1–3 carry full code. Task 4 is specified as prose steps with exact APIs and formulas rather than one 250-line block, because it is assembly of the already-defined, already-tested primitives (`suff_recovery`, `patch_component`, `margin_hard`, `gboot_ratio`); no new algorithm is hidden. Acceptable under "fold scaffolding into the task whose deliverable needs it."

**Type consistency:** `margins_fn(H_np)->np[N]` consistent across gates; `U` orthonormal `[k,D]` consistent across `subspace`/`anchor_gates`; `cls` is `nn.Module` in both real and test use; `bid` dict consistent.
