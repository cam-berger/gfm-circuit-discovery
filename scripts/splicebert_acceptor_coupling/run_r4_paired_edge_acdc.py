#!/usr/bin/env python
"""R4 — paired edge-level causal core: do the two directions share their WIRING?

Executes `studies/splicebert_acceptor_coupling/preregistrations/
PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md` (DRAFT — freeze before the target run).

R3/AC4 partitions the 96 heads into 73 SHARED / 17 UNRESOLVED / 4 FORWARD_SPECIFIC /
2 REVERSE_SPECIFIC.  Shared components do not imply shared wiring: two directions can recruit
the same heads and route between them differently, which a component partition cannot separate.
R4 asks the same question over the 4,636 head-granular EDGES, on R3's cohort, paired in one
process so cohort, distance, composition and gene structure cancel inside every contrast.

TWO IDIOMS, NEVER POOLED (D9).  R3 chose mean-ablation as primary because the two disagree 45x
on forward L2H15 (+0.0115 noising vs +0.5146 ablation) -- a noising-primary sweep returns a
false null for the addresser.  ACDC's H(x, x') sources out-of-H edges from a cached activation,
so the idiom is entirely a choice of WHICH cache:

    noising        out-of-H edges take their PARTNER-CORRUPTED activation
    mean-ablation  out-of-H edges take the cohort MEAN activation      <- primary, matches R3

An addresser edge dropped under the noising arm is NOT evidence against C22/C32.  It is the
expected behaviour of that idiom on that component (D9), and is reported as a metric limitation.

    uv run python scripts/splicebert_acceptor_coupling/run_r4_paired_edge_acdc.py --smoke
    uv run python .../run_r4_paired_edge_acdc.py --n 64 --reference-smoke --out <path>
    uv run python .../run_r4_paired_edge_acdc.py --n 974 --tau-from-floor --out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file      # noqa: E402
from gfmcd.circuits.acdc import acdc_run, equivalence_check                    # noqa: E402
from gfmcd.circuits.acdc_device import (                                       # noqa: E402
    assert_decision_determinism, autotune_batch, measure_device_floor,
    set_precision_policy, tau_grid_from_floor,
)
from gfmcd.circuits.acdc_graph import EdgeGraph                                # noqa: E402
from gfmcd.circuits.acdc_kernel import AcdcForward, NodeWrites, capture_node_writes  # noqa: E402
from gfmcd.splicing.circuit import (                                           # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot, load_adapter,
    load_examples, matched_decoy_acceptor, matched_decoy_donor, pick_device, ratio_of_means,
    substitute,
)
from gfmcd.splicing.coupling import MOTIF_CLEAN_DONOR_ALT, planted_donor_motifs  # noqa: E402
from gfmcd.splicing.reverse_coupling import (                                  # noqa: E402
    FORWARD_ACCEPTOR_ALT, acceptor_readout, donor_alt_admissible, matched_operators,
)

DRAW_TAG = "r1b-d0"          # R3's cohort tag -- same draw, so R4 is paired with AC4
SEED = 0
N_BOOT = 2000
SESOI = 0.05
AC1_Y_A, AC1_Y_D, ANCHOR_TOL = 0.9763, 0.9682, 0.15
FLOOR_PANEL = 32
DETERMINISM_PREFIX = 200
EQUIVALENCE_NODES = 8


# ======================================================================================
# One direction's ACDC metric.
# ======================================================================================
class Direction:
    """A readout position, a corruption position, and the two caches the idioms need."""

    def __init__(self, name, task, graph, exs, mask_pos, corrupt_pos, alt):
        self.name, self.task, self.graph, self.exs = name, task, graph, exs
        self.mask_pos, self.alt = mask_pos, alt
        self.seq = [x.sequence for x in exs]
        self.cor = [substitute(s, p, alt) for s, p in zip(self.seq, corrupt_pos)]
        self.forward = AcdcForward(task.ad.model, graph)
        self.batch = task.batch
        self._prepare()

    # ---------------------------------------------------------------------------------
    def _enc(self, seqs, sl):
        return self.task.encode(list(seqs[sl]), list(self.mask_pos[sl]))

    def _margin(self, logits, sl):
        return self.task.margin(logits.float(), list(self.mask_pos[sl]))

    def _prepare(self):
        """Baselines and the two out-of-H caches, computed once for the whole sweep."""
        clean, corrupt = [], []
        with torch.no_grad():
            for i in range(0, len(self.mask_pos), self.batch):
                sl = slice(i, i + self.batch)
                clean += self._margin(
                    self.task.ad.model(**self._enc(self.seq, sl)).logits, sl).tolist()
                corrupt += self._margin(
                    self.task.ad.model(**self._enc(self.cor, sl)).logits, sl).tolist()
        self.m_clean = np.asarray(clean)
        self.m_corrupt = np.asarray(corrupt)
        self.coupling = self.m_clean - self.m_corrupt

    def _writes(self, seqs, sl) -> NodeWrites:
        return capture_node_writes(self.task.ad.model, self._enc(seqs, sl))

    @staticmethod
    def _mean_writes(w: NodeWrites) -> NodeWrites:
        """Cohort-mean activation, broadcast back -- R3's mean-ablation, lifted to edges."""
        return NodeWrites(
            embed=w.embed.mean(dim=0, keepdim=True).expand_as(w.embed),
            attention={k: v.mean(dim=1, keepdim=True).expand_as(v) for k, v in w.attention.items()},
            mlp={k: v.mean(dim=0, keepdim=True).expand_as(v) for k, v in w.mlp.items()},
            attention_bias=w.attention_bias, mlp_bias=w.mlp_bias,
        )

    # ---------------------------------------------------------------------------------
    def metric(self, idiom: str):
        """Return ``metric_fn(mask) -> divergence`` (0 at H=ALL, rising as edges are cut).

        ``noising``: recovery of the clean margin with out-of-H edges sourced from the
        partner-corrupted run.  One kernel evaluation per candidate.

        ``meanabl``: loss of the COUPLING with out-of-H edges sourced from the cohort mean --
        the quantity R3 measured, so R4's partition is commensurable with AC4's.  Two kernel
        evaluations per candidate, because the coupling is a difference of two readouts.
        """
        if idiom not in ("noising", "meanabl"):
            raise ValueError(f"unknown idiom {idiom!r}")

        def evaluate(mask, seqs, source_of):
            out = []
            for i in range(0, len(self.mask_pos), self.batch):
                sl = slice(i, i + self.batch)
                enc = self._enc(seqs, sl)
                src = source_of(sl)
                out += self._margin(self.forward.logits(enc, src, mask), sl).tolist()
            return np.asarray(out)

        if idiom == "noising":
            def source_of(sl):
                return self._writes(self.cor, sl)

            denominator = float(self.coupling.mean())

            def metric_fn(mask: torch.Tensor) -> float:
                got = evaluate(mask, self.seq, source_of)
                recovery = ratio_of_means(got - self.m_corrupt, self.coupling)
                return 1.0 - float(recovery)
        else:
            def source_of(sl):
                return self._mean_writes(self._writes(self.seq, sl))

            denominator = float(self.coupling.mean())

            def metric_fn(mask: torch.Tensor) -> float:
                y = evaluate(mask, self.seq, source_of) - evaluate(mask, self.cor, source_of)
                return float((denominator - y.mean()) / max(abs(denominator), 1e-9))

        metric_fn.denominator = denominator          # type: ignore[attr-defined]
        return metric_fn


# ======================================================================================
def partition(delta_ci: dict, band: float = SESOI) -> dict[str, str]:
    """AC4's vocabulary, applied to edges.  UNRESOLVED is NOT SHARED."""
    out = {}
    for key, (lo, hi) in delta_ci.items():
        if lo > 0:
            out[key] = "REVERSE_SPECIFIC"
        elif hi < 0:
            out[key] = "FORWARD_SPECIFIC"
        elif -band <= lo and hi <= band:
            out[key] = "SHARED"
        else:
            out[key] = "UNRESOLVED"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=974)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=0, help="0 = autotune from free VRAM (D7.5)")
    ap.add_argument("--idiom", default="both", choices=["meanabl", "noising", "both"])
    ap.add_argument("--mode", default="sequential", choices=["sequential", "parallel"])
    ap.add_argument("--tau", type=float, default=None,
                    help="explicit tau; omit to derive the grid from the measured device floor")
    ap.add_argument("--qkv-split", action="store_true")
    ap.add_argument("--state-dir", default=None, help="checkpoint per destination node (Colab)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="cap destination nodes visited (verification runs only; 0 = all). "
                         "A capped run is NOT a circuit and its artifact says so.")
    ap.add_argument("--floor-n", type=int, default=32,
                    help="cohort prefix the device-floor panel uses; must match the reference")
    ap.add_argument("--reference-smoke", action="store_true",
                    help="emit the committed reference panel Colab is diffed against")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None and not a.smoke:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    policy = set_precision_policy()
    torch.set_grad_enabled(False)
    dev = pick_device(a.device)
    ad = load_adapter(a.model, dev, eager=False)

    graph = EdgeGraph(ad.num_layers, ad.num_heads, qkv_split=a.qkv_split)
    batch = a.batch or autotune_batch(graph.n_nodes, 1026, ad.hidden_size, device=dev)
    task = CouplingTask(ad, batch=batch)

    n = 16 if a.smoke else a.n
    X, _ = load_examples(a.gtf, a.fasta, "val", n, task=task, tag=DRAW_TAG, min_effect=None)

    # R3's cohort, built the same way so R4 is paired with AC4 rather than merely similar.
    dd = [matched_decoy_donor(x) for x in X]
    da = [matched_decoy_acceptor(x, 40) for x in X]
    keep = [i for i in range(len(X)) if dd[i] is not None and da[i] is not None]
    P = [X[i] for i in keep]
    genes = [x.gene_id for x in P]
    cohort_hash = hashlib.sha256("|".join(x.example_id for x in P).encode()).hexdigest()

    fwd = Direction("forward", task, graph, P, [x.donor_pos for x in P],
                    [x.next_acceptor_pos - 2 for x in P], FORWARD_ACCEPTOR_ALT)
    rev = Direction("reverse", task, graph, P, [acceptor_readout(x) for x in P],
                    [x.donor_pos for x in P], MOTIF_CLEAN_DONOR_ALT)
    print(f"cohort n={len(P)} genes={len(set(genes))} dev={dev} batch={batch} "
          f"edges={graph.n_edges}\n  Y_D={fwd.coupling.mean():+.4f}  Y_A={rev.coupling.mean():+.4f}",
          flush=True)

    # ---- memorials ---------------------------------------------------------------------
    M: dict = {}

    def mem(k, ok, forced=False, gating=True, **d):
        """``gating=False`` records a memorial without letting it decide the run.

        Used only where a gate is underpowered by construction -- the behavioural anchors at
        smoke n, where a 16-exon cohort cannot support a +/-0.15 tolerance.  A gate that cannot
        pass at smoke scale validates nothing, which is the R21/R7 forced-memorial pathology
        with the sign flipped.  It is recorded, reported, and excluded from PASS, never widened.
        """
        M[k] = dict(PASS=bool(ok), forced=bool(forced), gating=bool(gating), **d)

    full = graph.full_mask()
    empty = torch.zeros_like(full)
    ref_noise = fwd.metric("noising")
    err_full, err_empty = abs(ref_noise(full)), abs(1.0 - ref_noise(empty))
    mem("kernel_exact_full", err_full < 1e-4, max_abs_err=float(err_full))
    mem("kernel_exact_empty", err_empty < 1e-4, max_abs_err=float(err_empty))
    mem("edge_count", graph.n_edges == (12988 if a.qkv_split else 4636),
        n_edges=graph.n_edges, n_nodes=graph.n_nodes)
    mem("behavioural_anchors",
        abs(fwd.coupling.mean() - AC1_Y_D) <= ANCHOR_TOL
        and abs(rev.coupling.mean() - AC1_Y_A) <= ANCHOR_TOL,
        gating=not (a.smoke or a.max_nodes),
        Y_D=float(fwd.coupling.mean()), Y_A=float(rev.coupling.mean()), tol=ANCHOR_TOL,
        note=(None if not (a.smoke or a.max_nodes) else
              "underpowered at this cohort size; recorded, not gating. A smoke or node-capped "
              "run is marked is_partial_sweep and is NOT a circuit, so gating it on an anchor "
              "it cannot meet licenses nothing. Gating is restored for any full sweep."))
    plants = sum(1 for x in P if not donor_alt_admissible(x, MOTIF_CLEAN_DONOR_ALT))
    mem("construction_zero_planted_motifs", plants == 0, n_plants=plants, n=len(P))
    mem("edit_size_matched", matched_operators(MOTIF_CLEAN_DONOR_ALT, FORWARD_ACCEPTOR_ALT))
    mem("pairing_identity_same_cohort", True, forced=True, cohort_sha256=cohort_hash,
        note="both directions are built from one example list in one process; forced")
    ckpt = Path(a.model) if Path(a.model).is_absolute() else REPO / a.model
    mem("checkpoint_recorded", True, forced=True, sha256=sha256_file(ckpt / "pytorch_model.bin"))
    mem("chr8_9_absent", all(x.chrom not in ("chr8", "chr9") for x in P), forced=False,
        chroms=sorted({x.chrom for x in P}))

    # ---- device floor sets tau (R4 section 3) ------------------------------------------
    # The panel is evaluated on a FIXED prefix of the cohort, independent of --n.  The floor
    # is a property of the device and the kernel; if the panel moved with --n, a reference
    # measured at one size and a run at another would be differencing two different
    # quantities and calling it device noise.
    reference_path = REPO / "results/splicebert_acceptor_coupling/r4_reference_smoke.json"
    reference = json.loads(reference_path.read_text()) if reference_path.is_file() else None

    # The floor is only a DEVICE property if both sides evaluate the same cohort with the same
    # panel.  A run whose cohort is smaller than floor_n cannot supply that, so the gate is
    # marked not-evaluable rather than differencing two different quantities.
    floor_evaluable = len(P) >= a.floor_n
    floor_metric = Direction(
        "floor", task, graph, P[:a.floor_n], [x.donor_pos for x in P[:a.floor_n]],
        [x.next_acceptor_pos - 2 for x in P[:a.floor_n]], FORWARD_ACCEPTOR_ALT,
    ).metric("noising") if floor_evaluable else ref_noise

    # Panel size must match the reference exactly, or the comparison is undefined.
    n_panel = len(reference["panel_values"]) if reference else FLOOR_PANEL
    rng = np.random.default_rng(SEED)
    panel = []
    for _ in range(n_panel):
        mask = full.clone()
        cut = rng.choice(graph.n_edges, size=max(1, graph.n_edges // 50), replace=False)
        for c in cut:
            mask[graph.edge_slot_index(graph.edges[int(c)])] = False
        panel.append(mask)
    panel_values = [float(floor_metric(m)) for m in panel]

    if a.reference_smoke:
        out = Path(a.out) if a.out else REPO / "results/splicebert_acceptor_coupling/r4_reference_smoke.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "acceptor_coupling.r4_reference_panel/1",
            "device": dev, "cohort_sha256": cohort_hash, "n": len(P),
            "floor_n": a.floor_n,
            "panel_values": panel_values, "memorials": M,
            "precision_policy": policy.to_dict(),
        }, indent=1))
        ExperimentManifest.create(
            "r4_reference_smoke", sys.argv, repo=REPO, status="EXPLORATORY",
            config={"n": len(P), "floor_n": a.floor_n, "n_panel": len(panel_values),
                    "device": dev, "batch": batch},
            seeds={"seed": SEED},
            inputs=[{"path": str(ckpt), "role": "checkpoint"}],
            outputs=[{"path": str(out), "role": "reference_panel"}],
            notes=["Cross-device reference panel. The floor is only a DEVICE property if the "
                   "run reproduces this panel at the same floor_n on the same cohort prefix."],
        ).write(out.with_suffix(".manifest.json"))
        print(f"reference panel -> {out}  ({len(panel_values)} subgraphs on {dev})")
        return

    tau_grid: list[float] = []
    if reference and floor_evaluable and reference.get("floor_n") == a.floor_n:
        if reference["device"] == dev:
            # Same device: the difference is 0 by construction and bounds nothing.  A floor
            # derived here would be a grid-shaped hole (acdc_device refuses it outright).
            mem("tau_above_device_floor", a.tau is not None,
                note=f"reference was measured on {dev}; a same-device floor bounds nothing, "
                     "so --tau must be supplied explicitly for a local run",
                reference_device=reference["device"], run_device=dev)
        else:
            floor = measure_device_floor(floor_metric, panel, reference["panel_values"],
                                         reference_device=reference["device"], run_device=dev)
            mem("tau_above_device_floor", floor.passes, **floor.to_dict())
            tau_grid = list(floor.tau_grid)
    else:
        why = ("no committed reference panel" if not reference
               else f"cohort n={len(P)} < floor_n={a.floor_n}" if not floor_evaluable
               else f"reference floor_n={reference.get('floor_n')} != {a.floor_n}")
        mem("tau_above_device_floor", a.tau is not None,
            note=f"{why}; --tau must be supplied explicitly")
    if a.tau:
        tau_grid = [a.tau]

    pf = {"commit": git_commit(REPO), "n": len(P), "memorials": M,
          "PASS": all(v["PASS"] for v in M.values() if v.get("gating", True))}
    print(f"preflight PASS={pf['PASS']}")
    for k, v in M.items():
        tag = " [forced]" if v["forced"] else ("" if v.get("gating", True) else " [not gating]")
        print(f"    {'ok ' if v['PASS'] else 'XX '}{k}{tag}")
    if a.smoke:
        print("SMOKE: memorials only; no circuit reported.")
        return
    if not pf["PASS"]:
        raise SystemExit("preflight FAILED — not reporting (R4 §4).")
    if not tau_grid:
        raise SystemExit("no tau grid: supply --tau or commit a reference panel first.")

    # ---- the sweep ---------------------------------------------------------------------
    idioms = ["meanabl", "noising"] if a.idiom == "both" else [a.idiom]
    state_dir = Path(a.state_dir) if a.state_dir else None
    if state_dir:
        state_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    for idiom in idioms:
        metrics = {"forward": fwd.metric(idiom), "reverse": rev.metric(idiom)}
        eq = equivalence_check(metrics["forward"], graph, tau_grid[0],
                               graph.reverse_topological()[:EQUIVALENCE_NODES])
        mem(f"equivalence_check_{idiom}", eq["PASS"] or a.mode == "sequential", **eq)

        for tau in tau_grid:
            for name, metric_fn in metrics.items():
                def checkpoint(node, res, _t=tau, _i=idiom, _n=name):
                    if state_dir:
                        (state_dir / f"r4_{_i}_{_n}_{_t:.3e}_{str(node).replace(' ', '')}.json"
                         ).write_text(json.dumps({"n_edges_kept": res.n_edges_kept}))

                node_filter = (set(graph.reverse_topological()[:a.max_nodes])
                               if a.max_nodes else None)
                res = acdc_run(metric_fn, graph, tau, mode=a.mode, on_node=checkpoint,
                               node_filter=node_filter)
                results[f"{idiom}/{name}/{tau:.3e}"] = res.to_dict(graph)
                print(f"  {idiom:8s} {name:7s} tau={tau:.3e}  "
                      f"kept {res.n_edges_kept}/{graph.n_edges}  "
                      f"evals={res.n_evaluations}", flush=True)

    # ---- paired edge partition ---------------------------------------------------------
    partitions: dict = {}
    for idiom in idioms:
        for tau in tau_grid:
            f_key, r_key = f"{idiom}/forward/{tau:.3e}", f"{idiom}/reverse/{tau:.3e}"
            f_kept = {tuple(e) for e in results[f_key]["kept_edges"]}
            r_kept = {tuple(e) for e in results[r_key]["kept_edges"]}
            both = f_kept | r_kept
            partitions[f"{idiom}/{tau:.3e}"] = {
                "n_forward": len(f_kept), "n_reverse": len(r_kept),
                "n_shared": len(f_kept & r_kept),
                "n_forward_specific": len(f_kept - r_kept),
                "n_reverse_specific": len(r_kept - f_kept),
                "jaccard": len(f_kept & r_kept) / max(len(both), 1),
                "forward_specific": sorted(map(list, f_kept - r_kept))[:64],
                "reverse_specific": sorted(map(list, r_kept - f_kept))[:64],
            }

    payload = {
        "schema": "acceptor_coupling.r4_paired_edge_acdc/1",
        "cohort": {"n": len(P), "n_genes": len(set(genes)), "sha256": cohort_hash,
                   "draw_tag": DRAW_TAG, "split": "val"},
        "graph": {"n_nodes": graph.n_nodes, "n_edges": graph.n_edges,
                  "qkv_split": a.qkv_split},
        "baselines": {"Y_D": float(fwd.coupling.mean()), "Y_A": float(rev.coupling.mean())},
        "tau_grid": tau_grid, "mode": a.mode, "idioms": idioms,
        "precision_policy": policy.to_dict(),
        "memorials": M, "preflight_PASS": pf["PASS"],
        "max_nodes": a.max_nodes or None,
        "is_partial_sweep": bool(a.max_nodes),
        "runs": results, "partitions": partitions,
        "scope": (
            ("PARTIAL SWEEP - node-capped verification run, NOT a circuit. " if a.max_nodes else "")
            + "Verification tier; validation split; chr8/9 not opened. Attribution RANKS, "
            "intervention DECIDES (AC4). Greedy => the retained edge count bounds the minimal "
            "faithful edge set from ABOVE. UNRESOLVED is NOT SHARED. Idioms reported "
            "separately and never pooled (D9): an addresser edge dropped under the noising arm "
            "is not evidence against C22/C32."
        ),
    }
    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"refuse to overwrite an existing artifact: {out}")
    out.write_text(json.dumps(payload, indent=1))
    ExperimentManifest.create(
        "r4_paired_edge_acdc", sys.argv, repo=REPO, status="EXPLORATORY",
        config={"n": len(P), "tau_grid": tau_grid, "mode": a.mode, "idioms": idioms,
                "qkv_split": a.qkv_split, "batch": batch, "device": dev},
        seeds={"seed": SEED},
        inputs=[{"path": str(ckpt), "role": "checkpoint"}],
        outputs=[{"path": str(out), "role": "result"}],
        notes=["R4 DRAFT protocol; freeze before the target run."],
    ).write(out.with_suffix(".manifest.json"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
