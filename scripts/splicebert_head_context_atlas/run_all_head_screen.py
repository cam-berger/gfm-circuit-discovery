#!/usr/bin/env python
"""All-96-head A/B/C/D exact-effect screen (EXPLORATORY head-effect matrix), scrub-to-corrupt.

Measures, for every one of the 6x16 = 96 SpliceBERT heads, its EXACT intervention effect on splice-site
reconstruction under four lenses, across the tagged intron-mate discovery cohort:

  A  general reconstruction   : native-base margin / logprob / entropy under the joint mask (+ optional
                                A_zero_effect = zero-ablation necessity, labeled SEPARATELY)
  B  partner coupling         : scrub head output present -> partner-destroyed
  C  competitor-site pert.     : scrub head output present -> competitor-destroyed
                                (relabeled per the failed specificity pilot: "competitor-site perturbation
                                 response" — NOT competition-specific recruitment)
  D  decoy contrast           : scrub head output present -> matched-decoy-destroyed

Behavioral B/C/D are RECORDED DIAGNOSTICS, not launch gates. MECHANICAL gates are mandatory.
DISCOVERY only: chr8/9 are HARD-EXCLUDED unless --unlock-sealed AND a separate confirmation config.
No credit spend here — this is the local/Colab CLI; the notebook only orchestrates.

Deviations (documented): single canonical destroy variant (index 0) for the head cache (behavioral pilots
averaged 4 variants; a single variant keeps 96-head compute tractable — the mean-over-variants margin is
still exported at condition level). Per-head collateral health = unmasked true-base logprob at K fixed
non-site positions (cheap); proper masked mlm_health is exported per CONDITION.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.gtf import load_transcripts
from gfmcd.splicing.competition import CompetitionConfig, build_competition_index, enumerate_competition_examples
from gfmcd.splicing.competition_seq import build_window, WindowedCompetition
from gfmcd.splicing.specificity import find_matched_decoy  # extracted decoy selector (see module)
from gfmcd.splicing.coupling import (destroy_donor, destroy_acceptor, destroy_donor_clean,
                                     destroy_acceptor_clean, planted_donor_motifs,
                                     planted_acceptor_motifs, MOTIF_CLEAN_ALT)

from gfmcd.splicing.dataset import PyfaidxGenome
from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads, edit_heads

SEALED = {"chr8", "chr9"}
VALIDATION = {"chr11", "chr16", "chr19"}
BASES = "ACGT"
# R44 run-mode switches.  DEFAULTS PRESERVE THE COMMITTED BEHAVIOUR byte-for-byte: the
# planted-motif rate is always RECORDED, but only enforced/replaced when asked.
USE_CLEAN_ALT = False        # --clean-alt        : destroy with MOTIF_CLEAN_ALT ('AC')
REQUIRE_MOTIF_CLEAN = False  # --require-motif-clean : make a planted motif fatal
PLANTED_TOTALS: dict[str, int] = {}   # per-condition planted-motif counts, always recorded


# --------------------------------------------------------------------------- metrics
def metrics_from_logits(logits_t: torch.Tensor, native_id: int, base_ids: list[int]) -> dict:
    """logits_t: [vocab] at the masked target. Returns margin/logprob/entropy over {A,C,G,T}."""
    v = logits_t[base_ids].float()
    lp = torch.log_softmax(v, dim=-1)
    p = lp.exp()
    ni = base_ids.index(native_id)
    others = [j for j in range(4) if j != ni]
    margin = float(v[ni] - v[others].max())
    return {"margin": margin, "logprob": float(lp[ni]), "entropy": float(-(p * lp).sum())}


def collateral_health(logits_all: torch.Tensor, positions: list[int], native_ids: list[int],
                      base_ids: list[int], tok_offset: int) -> float:
    """Mean unmasked true-base logprob at fixed non-site positions (cheap per-head collateral proxy)."""
    vals = []
    for p, nid in zip(positions, native_ids):
        v = logits_all[p + tok_offset, base_ids].float()
        vals.append(float(torch.log_softmax(v, dim=-1)[base_ids.index(nid)]))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- forward
class ScreenModel:
    def __init__(self, adapter, batch_size=8):
        self.a = adapter
        self.bs = batch_size
        self.mask_id = adapter.tokenizer.mask_token_id
        self.base_ids = [adapter.tokenizer.convert_tokens_to_ids(b) for b in BASES]
        self.tok_offset = adapter.nucleotide_to_token(0)  # CLS/prefix offset
        for p in adapter.model.parameters():
            p.requires_grad_(False)
        adapter.model.eval()

    def enc(self, seqs, targets):
        a = self.a
        e = a.tokenizer([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(a.device)
        for j, p in enumerate(targets):
            e["input_ids"][j, p + self.tok_offset] = self.mask_id
        return e

    @torch.no_grad()
    def logits(self, enc):
        return self.a.model(**enc).logits.float()  # [B,T,vocab]


# --------------------------------------------------------------------------- gates
def gate_destroy_real_bases(seq, dseq, site_pos, is_donor):
    """Length, footprint and alphabet.

    NOTE (R44): this gate validates what the edit WRITES, never what it CREATES.  A
    substitution that destroys the site while planting a splice motif in an adjacent
    register passes it.  That blind spot is why the atlas ran for its whole life with a
    planted motif on ~88% of donor-site destructions.  The creation check is
    ``gate_no_planted_motif`` below -- kept separate so this function's verdict, and the
    artifacts it gated, remain byte-reproducible.
    """
    if len(seq) != len(dseq):
        return False, "length changed"
    diffs = [i for i in range(len(seq)) if seq[i] != dseq[i]]
    lo = site_pos if is_donor else site_pos - 2
    if any(not (lo <= i < lo + 2) for i in diffs):
        return False, f"edit outside motif footprint: {diffs}"
    if set(dseq) - set(BASES):
        return False, "non-ACGT base introduced"
    return True, ""


def gate_no_planted_motif(seq, dseq, site_pos, is_donor):
    """Reject a 'destruction' that CREATES a splice motif in a straddling register (R44).

    Writing ``XY`` over a site touches three registers -- ``seq[q-3]+X``, ``XY``, and
    ``Y+seq[q]``.  A spurious ``AG`` is a planted acceptor; a spurious ``GT``/``GC`` is a
    planted donor.  In this study the corrupted site is never the masked readout site, so
    a planted motif is fully visible to the model and the counterfactual silently becomes
    "site MOVED one base" rather than "site removed".

    Measured on this study's own discovery cohort
    (``results/splicebert_head_context_atlas/atlas_planted_motif_census_v1.json``):
    ``DONOR_ALTS[0]="CC"`` plants on 87.8% of condition-B donor sites and ``TA`` plants a
    CANONICAL ``GT`` on 95.5%.  ``AC`` (``MOTIF_CLEAN_ALT``) plants on 0.0%.
    """
    alt = dseq[site_pos:site_pos + 2] if is_donor else dseq[site_pos - 2:site_pos]
    hits = (planted_donor_motifs(seq, site_pos, alt) if is_donor
            else planted_acceptor_motifs(seq, site_pos, alt))
    if hits:
        return False, f"alt {alt!r} plants {hits} (use MOTIF_CLEAN_ALT)"
    return True, ""


def check_tokenization(adapter, seq):
    tk = adapter.tokenize(seq)  # validates 1-token-per-nt internally; raises otherwise
    return True


# --------------------------------------------------------------------------- screen one chunk
CONDITIONS = [("B", "partner", "partner_is_donor"),
              ("C", "competitor", "competitor_is_donor"),
              ("D", "decoy", "decoy_is_donor")]


def target_base_position(w: WindowedCompetition) -> int:
    """Return the sequence coordinate of the native G used by the MLM readout.

    ``target_win`` is the G of a donor ``GT`` but the coordinate immediately
    after an acceptor ``AG``.  Acceptor reconstruction therefore masks
    ``target_win - 1``.  Centralizing this conversion prevents silent coordinate
    drift between donor and acceptor screens.
    """
    return w.target_win if w.target_is_donor else w.target_win - 1


def screen_chunk(model: ScreenModel, windows, decoys, *, self_test=False):
    """Screen a chunk of equal-length windows. Returns per-example records + gate status."""
    a = model.a
    L, H = a.num_layers, a.num_heads
    heads = [(l, h) for l in range(L) for h in range(H)]
    B = len(windows)
    seqs = [w.seq for w in windows]
    targets = [target_base_position(w) for w in windows]
    native_ids = [model.base_ids[BASES.index(w.seq[p])] for w, p in zip(windows, targets)]
    # fixed collateral probe positions per example (deterministic, non-site)
    col_pos, col_ids = [], []
    for w in windows:
        ps = [q for q in range(30, len(w.seq) - 30, 137)][:6]
        col_pos.append(ps)
        col_ids.append([model.base_ids[BASES.index(w.seq[q])] for q in ps])

    def cond_metrics(logits, which):  # per-example metrics at target + collateral health
        out = []
        for j in range(B):
            tt = targets[j] + model.tok_offset
            m = metrics_from_logits(logits[j, tt], native_ids[j], model.base_ids)
            m["collateral"] = collateral_health(logits[j], col_pos[j], col_ids[j], model.base_ids, model.tok_offset)
            out.append(m)
        return out

    # baseline (present)
    enc_present = model.enc(seqs, targets)
    base_logits = model.logits(enc_present)
    base_m = cond_metrics(base_logits, "A")

    records = {w.example_id: {"example": _example_meta(w, decoys[j]), "conditions": {}}
               for j, w in enumerate(windows)}
    for j, w in enumerate(windows):
        records[w.example_id]["conditions"]["A_baseline"] = base_m[j]

    gate_msgs = []
    # source-to-source no-op (once)
    if self_test:
        cache0 = capture_head_outputs(a, lambda: model.logits(enc_present))
        with patch_heads(a, cache0, heads):
            sp = model.logits(enc_present)
        d = float((sp - base_logits).abs().max())
        gate_msgs.append(("source_to_source_noop", d < 1e-3, f"|max diff|={d:.2e}"))

    metric_keys = ["margin", "logprob", "entropy", "collateral"]
    for cond, site_attr, isdon_attr in CONDITIONS:
        # build destroyed seqs (canonical variant 0)
        dseqs = []
        ok_all = True
        for j, w in enumerate(windows):
            pos = getattr(w, f"{site_attr}_win") if site_attr != "decoy" else decoys[j]
            isdon = getattr(w, isdon_attr) if isdon_attr != "decoy_is_donor" else w.competitor_is_donor
            if USE_CLEAN_ALT:
                dseq = (destroy_donor_clean if isdon else destroy_acceptor_clean)(w.seq, pos)
            else:
                dseq = (destroy_donor if isdon else destroy_acceptor)(w.seq, pos)[0]
            g_ok, g_msg = gate_destroy_real_bases(w.seq, dseq, pos, isdon)
            if not g_ok:
                ok_all = False
                gate_msgs.append((f"destroy_real_bases_{cond}", False, f"{w.example_id}: {g_msg}"))
            # R44: always REPORT what the edit created; fatal only under --require-motif-clean.
            m_ok, m_msg = gate_no_planted_motif(w.seq, dseq, pos, isdon)
            if not m_ok:
                PLANTED_TOTALS[cond] = PLANTED_TOTALS.get(cond, 0) + 1
                if REQUIRE_MOTIF_CLEAN:
                    ok_all = False
                    gate_msgs.append((f"no_planted_motif_{cond}", False,
                                      f"{w.example_id}: {m_msg}"))
            dseqs.append(dseq)
        enc_cf = model.enc(dseqs, targets)
        cf_logits = model.logits(enc_cf)
        cf_m = cond_metrics(cf_logits, cond)
        cache = capture_head_outputs(a, lambda: model.logits(enc_cf))
        # 96-head scrub-to-corrupt, batched over examples
        eff = {k: np.empty((B, len(heads))) for k in metric_keys}
        for hi, (l, h) in enumerate(heads):
            with patch_heads(a, cache, [(l, h)]):
                pl = model.logits(enc_present)
            pm = cond_metrics(pl, cond)
            for j in range(B):
                for k in metric_keys:
                    eff[k][j, hi] = base_m[j][k] - pm[j][k]
        for j, w in enumerate(windows):
            recs = [{"layer": heads[hi][0], "head": heads[hi][1],
                     **{f"effect_{k}": float(eff[k][j, hi]) for k in metric_keys}} for hi in range(len(heads))]
            finite = all(math.isfinite(r["effect_margin"]) for r in recs)
            gate_msgs.append((f"96_finite_{cond}_{w.example_id[:20]}", len(recs) == 96 and finite, ""))
            records[w.example_id]["conditions"][cond] = {
                "baseline": base_m[j], "counterfactual": cf_m[j],
                "behavioral_effect": {k: base_m[j][k] - cf_m[j][k] for k in metric_keys},
                "head_effects": recs,
            }

    # A: zero-ablation necessity (labeled separately), margin only
    eff_a = np.empty((B, len(heads)))
    for hi, (l, h) in enumerate(heads):
        with edit_heads(a, [(l, h)], mode="zero"):
            al = model.logits(enc_present)
        am = cond_metrics(al, "A")
        for j in range(B):
            eff_a[j, hi] = base_m[j]["margin"] - am[j]["margin"]
    for j, w in enumerate(windows):
        records[w.example_id]["conditions"]["A_zero_ablation"] = [
            {"layer": heads[hi][0], "head": heads[hi][1], "A_zero_effect_margin": float(eff_a[j, hi])}
            for hi in range(len(heads))]
    return records, gate_msgs


def _example_meta(w: WindowedCompetition, decoy_pos):
    return {
        "example_id": w.example_id, "gene_id": w.gene_id, "chrom": w.chrom, "strand": w.strand,
        "split": w.split, "site_type": w.site_type,
        "target_win": w.target_win, "target_base_position": target_base_position(w),
        "target_is_donor": w.target_is_donor,
        "target_native_base": w.seq[target_base_position(w)],
        "partner_win": w.partner_win, "competitor_win": w.competitor_win, "decoy_win": int(decoy_pos),
        "cluster_size": w.cluster_size, "n_usable_competitors": w.n_usable_competitors,
        "target_partner_distance": w.target_partner_distance,
        "target_competitor_distance": w.target_competitor_distance,
        "claim_boundary": "exploratory train-only all-head screen; C = competitor-site perturbation response, NOT competition-specific",
    }


# --------------------------------------------------------------------------- IO / manifest / resume
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cohort_example_allowed(example, *, unlock_sealed: bool) -> bool:
    """Enforce disjoint discovery and sealed-confirmation cohorts."""
    if unlock_sealed:
        return example.chrom in SEALED
    return example.split == "train" and example.chrom not in (SEALED | VALIDATION)


def write_shard(out_dir: Path, example_id: str, record: dict) -> tuple[str, str]:
    body = json.dumps(record, sort_keys=True)
    name = f"shard_{_sha(example_id)[:16]}.json"
    tmp = out_dir / (name + ".tmp")
    tmp.write_text(body)
    (out_dir / name).unlink(missing_ok=True)
    tmp.rename(out_dir / name)  # atomic
    return name, _sha(body)


def load_cohort(cfg_json, min_sep, unlock_sealed):
    from maxentpy.maxent import load_matrix5, load_matrix3
    m5, m3 = load_matrix5(), load_matrix3()
    ccfg = CompetitionConfig(min_separation=min_sep)
    txs = load_transcripts(cfg_json["gtf"], transcript_types=("protein_coding",),
                           allowed_chromosomes={f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"})
    idx = build_competition_index(txs)
    exs = [e for e in enumerate_competition_examples(idx, ccfg) if e.usable]
    # Discovery means train, not merely "not sealed". Historical v1 omitted this
    # filter and touched chr11/16/19; AMENDMENT_HEAD_SCREEN.md records the correction.
    # A separately authorized confirmation reads ONLY the sealed chromosomes.
    exs = [e for e in exs if cohort_example_allowed(e, unlock_sealed=unlock_sealed)]  # GATE 12
    exs.sort(key=lambda e: int.from_bytes(hashlib.sha256(e.example_id.encode()).digest()[:8], "big"))
    return exs, ccfg, m5, m3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(REPO / "results/splicebert_head_context_atlas/head_screen"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--per-gene-cap", type=int, default=3)
    ap.add_argument("--min-separation", type=int, default=30)
    ap.add_argument("--unlock-sealed", action="store_true")
    ap.add_argument("--confirm-sealed", default=None, help="required separate config to actually run sealed")
    ap.add_argument("--clean-alt", action="store_true",
                    help="R44: destroy with MOTIF_CLEAN_ALT (AC) instead of variants[0]")
    ap.add_argument("--require-motif-clean", action="store_true",
                    help="R44: fail the run if any destruction plants a splice motif")
    ap.add_argument("--self-test", action="store_true", help="run source-to-source no-op gate on chunk 1")
    args = ap.parse_args()
    global USE_CLEAN_ALT, REQUIRE_MOTIF_CLEAN
    USE_CLEAN_ALT = bool(args.clean_alt)
    REQUIRE_MOTIF_CLEAN = bool(args.require_motif_clean)
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("WARN PYTHONHASHSEED!=0", file=sys.stderr)
    if args.unlock_sealed and not args.confirm_sealed:
        sys.exit("REFUSED: --unlock-sealed requires --confirm-sealed <config> (sealed chr8/9 guard)")

    cfg_json = json.loads(Path(args.config).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from gfmcd.splicing.circuit import load_adapter, pick_device
    dev = pick_device(None)
    mp = cfg_json["model"] if Path(cfg_json["model"]).is_absolute() else str(REPO / cfg_json["model"])
    adapter = load_adapter(mp, dev, eager=False)
    model = ScreenModel(adapter, batch_size=args.batch)

    exs, ccfg, m5, m3 = load_cohort(cfg_json, args.min_separation, args.unlock_sealed)
    genome = PyfaidxGenome(cfg_json["fasta"])
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # build windows + matched decoys deterministically
    from collections import defaultdict
    windows, decoys, per_gene = [], [], defaultdict(int)
    for e in exs:
        if args.n and len(windows) >= args.n:
            break
        if per_gene[e.gene_id] >= args.per_gene_cap:
            continue
        w = build_window(e, genome, ccfg)
        if w is None:
            continue
        dq, _ = find_matched_decoy(w, m5, m3)
        if dq is None:
            continue
        check_tokenization(adapter, w.seq)  # GATE 1
        windows.append(w); decoys.append(dq); per_gene[e.gene_id] += 1
    print(f"[screen] {len(windows)} windows w/ matched decoy across {len(per_gene)} genes", file=sys.stderr)

    manifest = {"expected_shards": [], "completed_shards": {}, "n_examples": len(windows),
                "n_heads": adapter.num_layers * adapter.num_heads, "sealed_excluded": not args.unlock_sealed,
                "cohort_policy": "sealed_chr8_9_only" if args.unlock_sealed else "train_only_excluding_val_and_sealed",
                "gate_failures": [], "config": cfg_json, "min_separation": args.min_separation,
                "precision": "float32 (TF32/AMP off)", "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    t0 = time.time()
    done = 0
    for ci in range(0, len(windows), args.batch):
        chunk_w = windows[ci:ci + args.batch]
        chunk_d = decoys[ci:ci + args.batch]
        # RESUME: skip if all shards in this chunk already exist
        pending = [(w, d) for w, d in zip(chunk_w, chunk_d)
                   if not (out_dir / f"shard_{_sha(w.example_id)[:16]}.json").exists()]
        for w in chunk_w:
            manifest["expected_shards"].append(f"shard_{_sha(w.example_id)[:16]}.json")
        if not pending:
            continue
        recs, gates = screen_chunk(model, [w for w, _ in pending], [d for _, d in pending],
                                   self_test=(args.self_test and ci == 0))
        for gname, gok, gmsg in gates:
            if not gok:
                manifest["gate_failures"].append({"gate": gname, "msg": gmsg})
        if manifest["gate_failures"]:
            sys.exit(f"MECHANICAL GATE FAILURE — refusing to write. {manifest['gate_failures'][:3]}")
        for eid, rec in recs.items():
            name, sha = write_shard(out_dir, eid, rec)
            manifest["completed_shards"][name] = sha
            done += 1
        if done % 40 == 0:
            print(f"[screen] {done}/{len(windows)} ({(time.time()-t0)/max(done,1):.2f}s/ex)", file=sys.stderr)

    # R44: the planted-motif census of THIS run, recorded whether or not it was enforced.
    manifest["planted_motif_counts"] = dict(PLANTED_TOTALS)
    manifest["planted_motif_enforced"] = REQUIRE_MOTIF_CLEAN
    manifest["destroy_alt_mode"] = ("MOTIF_CLEAN_ALT=" + MOTIF_CLEAN_ALT) if USE_CLEAN_ALT else "variants[0]"
    manifest["seconds"] = time.time() - t0
    # rebuild completed_shards from ALL shards on disk (resume-safe: includes prior-run shards)
    manifest["completed_shards"] = {p.name: _sha(p.read_text()) for p in sorted(out_dir.glob("shard_*.json"))}
    manifest["expected_shards"] = sorted(set(manifest["expected_shards"]))
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[screen] done: {done} shards, {len(manifest['completed_shards'])} total, "
          f"{manifest['seconds']:.1f}s. gate_failures={len(manifest['gate_failures'])}")


if __name__ == "__main__":
    main()
