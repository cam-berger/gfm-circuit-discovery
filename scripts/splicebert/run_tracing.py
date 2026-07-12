#!/usr/bin/env python
"""End-to-end partner-circuit tracing for ONE additive-frozen seed (plan steps 0-5
+ specificity). Runnable on MPS (smoke) and CUDA (Colab). Mini-batched so full
D/V_c/O fit in GPU memory.

Pipeline (all on validation splits; test stays sealed):
  0. AtP fast screen on D -> shortlist for necessity and for each partner mask.
  1-2. Exact ablation (zero/mean/resample) on D over shortlist -> necessity set N
       (positive under >=2/3 nulls).
  2b. Exact single-head denoising patch on D for acceptor- and donor-mask over
      shortlist -> mediation set M (positive recovery on BOTH masks).
  2c. Candidate circuit C = N ∩ M, capped at component-set size by combined rank.
  3. Grouped necessity of C on V_c: joint mean-ablation effect vs 1000 matched-random
     sets -> add-one p.
  4. Mediation of C on V_c: joint patch recovery for each partner mask and for a
     context-only corruption (site-windows-only).
  5. Token-localized patch of C on V_c (endpoints, donor-mask).
  Specificity: mean-ablation effect of C on V_c vs on matched-ordinary O.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics as st
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.circuits.attribution import head_attribution_scan  # noqa: E402
from gfmcd.circuits.hf_bert import (  # noqa: E402
    capture_head_outputs, edit_heads, matched_random_head_sets, patch_heads,
)
from gfmcd.splicing.config import load_config  # noqa: E402
from gfmcd.splicing.context_controls import mask_true_partner_neighborhood, retain_site_windows  # noqa: E402
from gfmcd.splicing.io import load_jsonl  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.rigor import benjamini_hochberg, empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.metrics import matched_random_pvalue  # noqa: E402
from gfmcd.splicing.readout import collate_pair_indices, true_pair_margin  # noqa: E402
from gfmcd.splicing.training import load_pair_checkpoint, set_seed  # noqa: E402


@dataclass
class Chunk:
    tok: object
    pair_batch: object
    n: int
    clean_cache: dict | None = None  # per-chunk clean head outputs (for patching)


class Scorer:
    """Mini-batched true-pair-margin scoring with optional head interventions."""

    def __init__(self, adapter, readout, rep_layer):
        self.a, self.r, self.rep = adapter, readout, rep_layer

    def _margin_sum(self, tok, pb, grad=False):
        out = self.a.forward(tok, output_hidden_states=True, output_attentions=False, grad=grad)
        return true_pair_margin(self.r(out.hidden_states[self.rep], pb), pb).sum()

    def mean(self, chunks, *, ablate=None, patch=None, token_mask_fn=None):
        """ablate=(heads,mode) | patch=heads (uses chunk.clean_cache) | neither."""
        total, n = 0.0, 0
        with torch.no_grad():
            for c in chunks:
                if ablate is not None:
                    with edit_heads(self.a, ablate[0], mode=ablate[1]):
                        m = float(self._margin_sum(c.tok, c.pair_batch))
                elif patch is not None:
                    tm = token_mask_fn(c) if token_mask_fn else None
                    with patch_heads(self.a, c.clean_cache, patch, token_mask=tm):
                        m = float(self._margin_sum(c.tok, c.pair_batch))
                else:
                    m = float(self._margin_sum(c.tok, c.pair_batch))
                total += m; n += c.n
        return total / n

    def attribution(self, clean_chunks, corrupt_chunks):
        """Sum per-head AtP denoising attribution across chunks (clean->corrupt)."""
        agg: dict[tuple[int, int], float] = {}
        for cc, xc in zip(clean_chunks, corrupt_chunks):
            res = head_attribution_scan(
                self.a, cc.clean_cache,
                lambda: self._margin_sum(xc.tok, xc.pair_batch, grad=True))
            for e in res:
                agg[(e.layer, e.head)] = agg.get((e.layer, e.head), 0.0) + e.attribution
        return agg


def build_chunks(adapter, examples, batch, *, corrupt=None, want_cache=False):
    chunks = []
    n = len(examples)
    i = 0
    while i < n:
        j = min(i + batch, n)
        if n - j == 1:  # absorb a trailing singleton so resample (needs >=2) is valid
            j = n
        ex = examples[i:j]
        seqs = corrupt(ex) if corrupt else [x.sequence for x in ex]
        tok = adapter.tokenize(seqs)
        pb = collate_pair_indices(ex, token_offset=tok.prefix_tokens, device=adapter.device)
        chunks.append((ex, Chunk(tok=tok, pair_batch=pb, n=len(ex))))
        i = j
    return chunks


def add_clean_cache(scorer, chunks):
    for c in chunks:
        with torch.no_grad():
            c.clean_cache = capture_head_outputs(
                scorer.a, lambda c=c: float(scorer._margin_sum(c.tok, c.pair_batch)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--splits-dir", default="results/splicebert_partner_selection/circuits/discovery")
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/circuits/tracing")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-d", type=int, default=None)
    p.add_argument("--max-vc", type=int, default=None)
    p.add_argument("--max-o", type=int, default=None)
    p.add_argument("--shortlist", type=int, default=16)
    p.add_argument("--set-size", type=int, default=None)
    p.add_argument("--random-sets", type=int, default=1000)
    p.add_argument("--fdr-null-draws", type=int, default=1000,
                   help="single-head random null for per-head necessity FDR; must be "
                        ">= required_null_draws(96, q=0.1)=960 so a true hit can pass BH")
    p.add_argument("--flank", type=int, default=12)
    p.add_argument("--min-gap", type=float, default=0.05)
    args = p.parse_args()

    set_seed(args.seed)
    cfg = load_config(REPO / "configs/splicebert_partner_selection/base.json")
    rep = int(cfg["readout"]["representation_layer"])
    set_size = args.set_size or int(cfg["circuit"].get("default_component_set_size", 5))
    mc = cfg["model"]
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        REPO / f"colab_results/splicebert_run/runs/additive-frozen_seed{args.seed}/checkpoint.pt")
    adapter = SpliceBertAdapter.from_pretrained(
        REPO / mc["path"], tokenizer_mode=mc.get("tokenizer_mode", "auto"), device=args.device,
        trust_remote_code=False, local_files_only=True, model_class=mc.get("model_class", "masked_lm"))
    readout, checkpoint = load_pair_checkpoint(ckpt, adapter=adapter)
    if checkpoint.get("extra_metadata", {}).get("regime") != "additive-frozen":
        raise SystemExit("trace target must be additive-frozen")
    S = Scorer(adapter, readout, rep)

    sd = REPO / args.splits_dir
    D = load_jsonl(sd / "discovery.jsonl")[: args.max_d]
    Vc = load_jsonl(sd / "confirmation.jsonl")[: args.max_vc]
    O = load_jsonl(sd / "ordinary.jsonl")[: args.max_o]

    flank = args.flank
    masks = {
        "acceptor-mask": lambda exs: [mask_true_partner_neighborhood(x, kind="acceptor", flank=flank) for x in exs],
        "donor-mask": lambda exs: [mask_true_partner_neighborhood(x, kind="donor", flank=flank) for x in exs],
        "site-windows-only": lambda exs: [retain_site_windows(x, flank=flank, include_all_candidates=True) for x in exs],
    }

    def chunks_of(examples, corrupt=None, cache=False):
        pairs = build_chunks(adapter, examples, args.batch, corrupt=corrupt)
        cs = [c for _e, c in pairs]
        if cache:
            add_clean_cache(S, cs)
        return cs

    # clean chunks (with cache for patching)
    D_clean = chunks_of(D, cache=True)
    Vc_clean = chunks_of(Vc, cache=True)
    O_clean = chunks_of(O)
    all_heads = [(l, h) for l in range(adapter.num_layers) for h in range(adapter.num_heads)]

    # ---------- Step 0: AtP shortlists on D ----------
    # necessity AtP: (act-0)*grad at clean  ==  -attribution(zeros_cache, clean margin)
    zeros = [Chunk(tok=c.tok, pair_batch=c.pair_batch, n=c.n,
                   clean_cache={l: torch.zeros_like(c.clean_cache[l]) for l in c.clean_cache})
             for c in D_clean]
    neg = S.attribution(zeros, D_clean)  # (0-act)*grad
    nec_attr = {k: -v for k, v in neg.items()}  # necessity effect ~ act*grad
    nec_short = [k for k, _ in sorted(nec_attr.items(), key=lambda kv: kv[1], reverse=True)[: args.shortlist]]

    med_short = {}
    for mask in ("acceptor-mask", "donor-mask"):
        D_cor = chunks_of(D, corrupt=masks[mask])
        attr = S.attribution(D_clean, D_cor)
        med_short[mask] = [k for k, _ in sorted(attr.items(), key=lambda kv: kv[1], reverse=True)[: args.shortlist]]
    shortlist = sorted(set(nec_short) | set(med_short["acceptor-mask"]) | set(med_short["donor-mask"]))

    # ---------- Steps 1-2: exact necessity on D over shortlist ----------
    base_D = S.mean(D_clean)
    nec_effects = {}
    for head in shortlist:
        eff = []
        for mode in ("zero", "mean", "resample"):
            eff.append(base_D - S.mean(D_clean, ablate=([head], mode)))
        nec_effects[head] = eff
    N = {h: e for h, e in nec_effects.items() if sum(x > 0 for x in e) >= 2 and st.median(e) > 0}

    # Per-head multiplicity control (finding RRR-2): calibrate each shortlisted head's
    # mean-ablation effect against a single-head random null, then BH-FDR across the
    # shortlist. NOTE: the shortlist itself came from an AtP prescreen over all 96 heads,
    # so the effective search space is 96 -- reported for honesty.
    single_null_heads = matched_random_head_sets(
        adapter, [shortlist[0]], n=args.fdr_null_draws, seed=1, exclude_selected=False)
    single_null = [base_D - S.mean(D_clean, ablate=(list(rs), "mean")) for rs in single_null_heads]
    head_mean_eff = {h: (base_D - S.mean(D_clean, ablate=([h], "mean"))) for h in shortlist}
    head_p = {h: empirical_upper_p_value(head_mean_eff[h], single_null) for h in shortlist}
    q_vals = benjamini_hochberg([head_p[h] for h in shortlist])
    head_q = {h: float(q) for h, q in zip(shortlist, q_vals)}

    # ---------- Step 2b: exact mediation on D over shortlist, both masks ----------
    recovery_D = {m: {} for m in ("acceptor-mask", "donor-mask")}
    for mask in ("acceptor-mask", "donor-mask"):
        D_cor = chunks_of(D, corrupt=masks[mask])
        clean_s, corr_s = base_D, S.mean(D_cor)
        gap = clean_s - corr_s
        for head in shortlist:
            # patch this head's clean output into the aligned corrupt chunks
            recovery_D[mask][head] = single_head_recovery(S, D_clean, D_cor, head, clean_s, corr_s, gap)
    M = [h for h in shortlist
         if recovery_D["acceptor-mask"].get(h, -1) > 0 and recovery_D["donor-mask"].get(h, -1) > 0]

    # ---------- Step 2c: convergent candidate C = N ∩ M ----------
    conv = [h for h in M if h in N]
    conv.sort(key=lambda h: (st.median(N[h]) + 0.5 * (recovery_D["acceptor-mask"][h] + recovery_D["donor-mask"][h])),
              reverse=True)
    C = conv[:set_size]

    report = {
        "seed": args.seed, "checkpoint": str(ckpt),
        "n_D": len(D), "n_Vc": len(Vc), "n_O": len(O),
        "shortlist": shortlist,
        "necessity_set_N": [{"head": list(h), "median_effect": st.median(N[h])} for h in N],
        "necessity_fdr": {
            "single_head_null_draws": len(single_null),
            "effective_search_space": adapter.num_layers * adapter.num_heads,
            "per_head": [{"head": list(h), "mean_effect": head_mean_eff[h],
                          "p_add_one": head_p[h], "q_bh": head_q[h]} for h in shortlist],
        },
        "mediation_set_M": [list(h) for h in M],
        "candidate_C": [list(h) for h in C],
        "recovery_D": {m: {f"{l}-{hh}": v for (l, hh), v in d.items()} for m, d in recovery_D.items()},
    }

    if not C:
        report["verdict"] = "NO_CONVERGENT_SET (necessity without mediation, or vice versa)"
    else:
        # ---------- Step 3: grouped necessity of C on Vc vs matched-random ----------
        base_Vc = S.mean(Vc_clean)
        obs = base_Vc - S.mean(Vc_clean, ablate=(C, "mean"))
        rand_sets = matched_random_head_sets(adapter, C, n=args.random_sets, seed=0, exclude_selected=True)
        rand_eff = [base_Vc - S.mean(Vc_clean, ablate=(list(rs), "mean")) for rs in rand_sets]
        p_nec = matched_random_pvalue(obs, rand_eff)

        # ---------- Step 4: mediation of C on Vc (both masks + context-only) ----------
        med = {}
        for mask in ("acceptor-mask", "donor-mask", "site-windows-only"):
            Vc_cor = chunks_of(Vc, corrupt=masks[mask])
            clean_s, corr_s = base_Vc, S.mean(Vc_cor)
            gap = clean_s - corr_s
            rec = set_recovery(S, Vc_clean, Vc_cor, C, clean_s, corr_s, gap) if gap >= args.min_gap else None
            med[mask] = {"gap": gap, "recovery": rec}

        # ---------- Specificity: C effect on Vc vs O ----------
        base_O = S.mean(O_clean)
        eff_O = base_O - S.mean(O_clean, ablate=(C, "mean"))

        report.update({
            "necessity_Vc": {"observed_effect": obs, "random_mean": st.mean(rand_eff),
                             "p_add_one": p_nec, "n_random": len(rand_eff)},
            "mediation_Vc": med,
            "specificity": {"conflict_effect_Vc": obs, "ordinary_effect_O": eff_O,
                            "conflict_gt_ordinary": obs > eff_O},
            "verdict": "SEE_GATES",
        })

    out = REPO / args.out_dir / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # provenance manifest (finding RRR-3): params, package versions, input hashes
    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:tracing:seed{args.seed}", sys.argv, repo=REPO, config=cfg,
        seeds={"tracing": args.seed},
        notes=["Discovery/confirmation/ordinary splits are gene-disjoint and frozen.",
               "Candidate circuit C = necessity ∩ mediation; per-head FDR reported.",
               "Test split was not opened."])
    for name in ("discovery.jsonl", "confirmation.jsonl", "ordinary.jsonl"):
        manifest.add_input(sd / name)
    manifest.add_input(ckpt)
    manifest.add_output(out / "trace_report.json")
    manifest.write(out / "run_manifest.json")
    print(json.dumps(report, indent=2))


def single_head_recovery(S, clean_chunks, corrupt_chunks, head, clean_s, corrupt_s, gap):
    if abs(gap) < 1e-9:
        return 0.0
    # align clean cache into corrupt chunks
    total, n = 0.0, 0
    with torch.no_grad():
        for cc, xc in zip(clean_chunks, corrupt_chunks):
            with patch_heads(S.a, cc.clean_cache, [head]):
                total += float(S._margin_sum(xc.tok, xc.pair_batch)); n += xc.n
    patched = total / n
    return (patched - corrupt_s) / gap


def set_recovery(S, clean_chunks, corrupt_chunks, heads, clean_s, corrupt_s, gap):
    if abs(gap) < 1e-9:
        return 0.0
    total, n = 0.0, 0
    with torch.no_grad():
        for cc, xc in zip(clean_chunks, corrupt_chunks):
            with patch_heads(S.a, cc.clean_cache, heads):
                total += float(S._margin_sum(xc.tok, xc.pair_batch)); n += xc.n
    return ((total / n) - corrupt_s) / gap


if __name__ == "__main__":
    main()
