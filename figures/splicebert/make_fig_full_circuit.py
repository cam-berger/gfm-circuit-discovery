#!/usr/bin/env python
"""The complete donor–acceptor mechanism, front to back, as ONE directed schematic.

This renders the frozen circuit statement in ``../../MECHANISM.md`` ("Computation" block) as a single
Graphviz drawing that threads the two branches the repository documents separately:

  * the ADDRESS branch (period-3 positional coordinate → L2H15/MLP2 → L3H1 attention address), and
  * the PAYLOAD branch (acceptor-local state → L0H7/H12/H10 → MLP0 → MLP1 → MLP2), which is the same
    sub-graph drawn in ``fig_generator_graph`` and is read here from the SAME artifact,

converging at L3H1 (binding + donor-side write of the low-rank ``U_info`` object), then delivered by
L4H0/L4H2/L5H11 to the masked-donor MLM margin.

PROVENANCE / WHAT IS DATA-DERIVED.  The payload-branch head set and every ``N`` value on the
``L0* → MLP0 → MLP1 → MLP2`` chain, the H12–H10 interaction, the MLP0 mediation pair, n, and the
generator ceiling are read out of ``results/splicebert_coupling/a10_generator_edges.json`` (C31,
held-out, Directed-circuit level) — nothing on that branch is placed by eye.  The address-branch,
binding, U_info-delivery and behavior nodes carry the exact ledger values from
``studies/splicebert_partner_selection/CLAIMS.md`` (C1, C3, C5, C8, C16, C17, C31) and are labelled with
their controlling C-number so the drawing is auditable against the record.

WHAT IT MUST NOT SAY.  The ``N`` values are CONDITIONAL causal effects on held-out splits; they do not
sum and are not additive fractions of the effect (core principle 6; the artifact's own REFLECTION says
so). The chain is strictly serial. This is a STRUCTURED TOP-RANKED CORE inside an irreducibly
distributed computation — not the smallest sufficient circuit (F(C)≈0.005 when the complement is
scrubbed; the held-out size curve has no knee). The figure states this in a footer and makes no
minimality or completeness claim.

DEPENDENCY.  Rendering PDF/PNG needs Graphviz ``dot`` on PATH.  The DOT source is written either way and
is itself the data-derived, reproducible artifact; if ``dot`` is missing this script says so and stops
rather than hand-placing coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from figstyle import (  # noqa: E402
    BLUE, INK, MIST, MUT, PANEL, PURPLE, RULE, SEC, SLATE, figure_provenance,
)

EDGES = REPO / "results/splicebert_coupling/a10_generator_edges.json"
OUT_GV = HERE / "fig_full_circuit.gv"
OUT_PDF = HERE / "fig_full_circuit.pdf"
OUT_PNG = HERE / "fig_full_circuit.png"

# --------------------------------------------------------------------------- ledger anchors (CLAIMS.md)
# Exact live wording is controlled by studies/splicebert_partner_selection/CLAIMS.md; each value below is
# quoted from the C-row named in the comment. If the ledger changes, this figure is stale.
BEHAVIOR = "+0.872 [0.771, 0.969]"          # C1  paired-acceptor corruption -> masked-donor margin
CORE_ABLATE = "81.6%"                        # C3  {L3H1,L2H15} removes 81.6% coupling ...
CORE_HEALTH = "99.2%"                        # C3  ... retaining 99.2% MLM health
P3_DELETE = "0.946 \u2192 0.027"            # C8  rank-2 period-3 deletion collapses coupling
UINFO_REC = "\u22480.92"                    # C16 rank-8 U_info recovers ~0.92 vs random <=0.05
L3H1_REC = "0.83"                            # C17 restoring L3H1 recovers 0.83 of coupling ...
UINFO_ROUTE = "0.93"                         # C17 ... 0.93 of that routes through U_info at donor +/-5


def check(ge: dict) -> None:
    """Refuse to draw an artifact that is not the confirmatory, preflight-passed generator (as fig_generator_graph does)."""
    problems = []
    if not ge.get("confirmatory"):
        problems.append("artifact is not marked confirmatory")
    if not ge.get("preflight_PASS"):
        problems.append("preflight did not pass")
    if not ge.get("serial_chain_confirmed"):
        problems.append("serial chain not confirmed")
    star = ge.get("L0_star_heldout") or []
    lb = [h for h in star]  # the held-out load-bearing set defines the payload heads
    if not lb:
        problems.append("no held-out load-bearing L0 head set")
    if problems:
        sys.exit("REFUSING TO DRAW fig_full_circuit: " + "; ".join(problems))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    ge = json.loads(EDGES.read_text())
    check(ge)

    # ---- payload branch, read from the artifact (identical source to fig_generator_graph) ----------
    # Match fig_generator_graph exactly: verify load_bearing_H set agrees with L0_star_heldout, then
    # sort heads by descending held-out necessity N_h_H and label with N_h_H + ci_H.
    heads = ge["l0_heads"]
    flagged = {h for h, v in heads.items() if v.get("load_bearing_H")}
    star_set = set(ge["L0_star_heldout"])
    if flagged != star_set:
        sys.exit(f"REFUSING TO DRAW: load_bearing_H {sorted(flagged)} disagrees with L0_star_heldout {sorted(star_set)}")
    star = sorted(ge["L0_star_heldout"], key=lambda h: -heads[h]["N_h_H"])   # descending necessity
    head_N = {h: (heads[h]["N_h_H"], heads[h]["ci_H"]) for h in star}
    top_N = heads[star[0]]["N_h_H"]
    dominant = [h for h in star if h in ("L0H7", "L0H12", "L0H10")]
    marginal = [h for h in star if h not in dominant]

    mlp = ge["mlp_chain"]
    def mv(key):
        v = mlp[key]
        return (v[0], v[1]) if isinstance(v, list) else (v, None)
    N_MLP0, CI_MLP0 = mv("N_MLP0_H")
    N_MLP1, CI_MLP1 = mv("N_MLP1_H")
    N_MLP2, CI_MLP2 = mv("N_MLP2_H")
    inter = ge["top_head_interactions_D"]["L0H12xL0H10"]      # -0.126 discovery-split redundancy
    med = ge["mediation"]
    R_heads_only = med["R_L0heads_only_H"]
    R_heads_mlp0 = med["R_L0heads_MLP0_H"]
    ceiling = ge["ceiling"]
    n_pairs = ge["n"]; n_genes = ge["n_genes"]

    def nlab(h):
        v = head_N.get(h)
        if not v or v[0] is None:
            return f"{h}"
        val, ci = v
        if ci:
            return f"{h}\\nN = {val:+.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        return f"{h}\\nN = {val:+.3f}"

    # ---------------------------------------------------------------- assemble DOT ------------------
    L = []
    A = L.append
    A('digraph full_circuit {')
    A('  rankdir=LR;')
    A('  bgcolor="white";')
    A(f'  graph [fontname="Helvetica", fontsize=11, labelloc="b", labeljust="l", nodesep=0.28, '
      f'ranksep=0.85, fontcolor="{SEC}"];')
    A(f'  node  [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", '
      f'fillcolor="white", color="{SEC}", fontcolor="{INK}", penwidth=1.4, margin="0.12,0.08"];')
    A(f'  edge  [fontname="Helvetica", fontsize=9.5, color="{SEC}", fontcolor="{SEC}", arrowsize=0.8];')

    # bottom caveat, hard-wrapped so Graphviz does not stretch the layout to fit one long line
    cav_lines = [
        f'<b>The donor&#8211;acceptor mechanism, front to back</b>   '
        f'<font color="{SEC}">a structured top-ranked causal core: address &#215; acceptor-local payload '
        f'&#8594; L3H1 write &#8594; distributed readout</font>',
        f'<font color="{SEC}">Payload-branch heads, N values, CIs, the H12&#8211;H10 interaction and the MLP0 '
        f'mediation are read from a10_generator_edges.json (C31); n = {n_pairs} pairs / {n_genes} genes; generator '
        f'ceiling {ceiling:.3f}.</font>',
        f'<font color="{SEC}">MLP0 mediates the layer-0 contribution: heads alone R = {R_heads_only:.3f}, heads + '
        f'MLP0 R = {R_heads_mlp0:.3f}. {{L3H1, L2H15}} removes {CORE_ABLATE} of coupling at {CORE_HEALTH} MLM health '
        f'(C3).</font>',
        f'<font color="{SEC}"><b>N are conditional causal necessities on held-out splits &#8212; they do NOT sum and '
        f'are not additive fractions; the chain is strictly serial.</b></font>',
        f'<font color="{SEC}">This is a structured top-ranked core inside an irreducibly distributed computation, NOT '
        f'the smallest sufficient circuit: F(C)&#8776;0.005 when the complement is scrubbed and the held-out size curve '
        f'has no knee &#8212; no minimality or completeness claim is made.</font>',
    ]
    A('  label=<' + '<br align="left"/>'.join(cav_lines) + '<br align="left"/>>;')

    # ----- scope-context source boxes (not measured nodes; grey dashed, matching generator fig) -----
    A(f'  seq   [label="acceptor-local\\nsequence state", shape=box, style="dashed,rounded", '
      f'color="{MUT}", fontcolor="{SEC}", penwidth=1.2];')
    A(f'  p3    [label="period-3 positional coordinate\\nacross the donor&#8211;acceptor span   (C8)", '
      f'shape=box, style="dashed,rounded", color="{BLUE}", fontcolor="{BLUE}", penwidth=1.4];')

    # ----- ADDRESS branch (blue thread = period-3 carrier routing) ----------------------------------
    A('  subgraph cluster_addr {')
    A(f'    label=<<b>Address</b>  <font point-size="9" color="{SEC}">position geometry &#8594; L3H1 attention</font>>;')
    A(f'    labeljust="l"; color="{BLUE}"; style="rounded"; bgcolor="#F4F8FC"; penwidth=1.2; fontcolor="{BLUE}";')
    A(f'    L2H15 [label="L2H15 / MLP2\\namplify + route address", color="{BLUE}", penwidth=1.6];')
    A('  }')

    # ----- PAYLOAD branch (the embedded generator sub-graph, data-derived) --------------------------
    A('  subgraph cluster_payload {')
    A(f'    label=<<b>Payload generation</b>  <font point-size="9" color="{SEC}">A10 / C31, held-out, Directed-circuit level</font>>;')
    A(f'    labeljust="l"; color="{PURPLE}"; style="rounded"; bgcolor="{PANEL}"; penwidth=1.2; fontcolor="{PURPLE}";')
    # load-bearing L0 heads
    A('    subgraph cluster_l0 {')
    A(f'      label=<<font point-size="9" color="{SEC}">layer-0 load-bearing heads</font>>; labeljust="l"; '
      f'color="{RULE}"; style="rounded"; bgcolor="#FBFCFD"; penwidth=1.0;')
    for h in dominant:
        A(f'      {h} [label="{nlab(h)}", color="{PURPLE}", penwidth=1.7];')
    for h in marginal:
        A(f'      {h} [label="{nlab(h)}", color="{MUT}", fontcolor="{SEC}", penwidth=1.1];')
    A('    }')
    A(f'    MLP0 [label="MLP0\\nN = {N_MLP0:+.3f} [{CI_MLP0[0]:.3f}, {CI_MLP0[1]:.3f}]", color="{PURPLE}", penwidth=1.7];')
    A(f'    MLP1 [label="MLP1\\nN = {N_MLP1:+.3f} [{CI_MLP1[0]:.3f}, {CI_MLP1[1]:.3f}]", color="{PURPLE}", penwidth=1.6];')
    A(f'    MLP2 [label="MLP2\\nN = {N_MLP2:+.3f} [{CI_MLP2[0]:.3f}, {CI_MLP2[1]:.3f}]", color="{PURPLE}", penwidth=1.6];')
    A('  }')

    # ----- BINDING / WRITE (L3H1) -------------------------------------------------------------------
    A(f'  L3H1 [label=<<b>L3H1</b><br/>bind address &#215; payload, write donor-side<br/>'
      f'<font point-size="9" color="{SEC}">restore recovers {esc(L3H1_REC)} of coupling (C17)</font>>, '
      f'color="{PURPLE}", penwidth=2.4, fillcolor="#F3EFF8"];')

    # ----- DELIVERY (U_info -> downstream heads) ----------------------------------------------------
    A(f'  Uinfo [label=<low-rank <b>U_info</b> donor-side object<br/>'
      f'<font point-size="9" color="{SEC}">rank-8 recovers {esc(UINFO_REC)} vs random &#8804;0.05 (C16)</font>>, '
      f'shape=box, style="rounded,filled", color="{PURPLE}", fillcolor="white", penwidth=1.5];')
    A(f'  deliver [label="L4H0 / L4H2 / L5H11\\ndonor-side transport", color="{SEC}", penwidth=1.4];')

    # ----- READOUT (behavior) -----------------------------------------------------------------------
    A(f'  readout [label=<<b>masked-donor MLM margin</b><br/>'
      f'<font point-size="9" color="{SEC}">paired-acceptor corruption: {esc(BEHAVIOR)} (C1)</font>>, '
      f'shape=box, style="rounded,filled", color="{PURPLE}", fillcolor="#F3EFF8", penwidth=2.0];')

    # ------------------------------------------------------ edges -----------------------------------
    # address thread (blue)
    A(f'  p3 -> L2H15 [color="{BLUE}", penwidth=2.0, xlabel="carrier routing"];')
    A(f'  L2H15 -> L3H1 [color="{BLUE}", penwidth=2.0, label="attention\\naddress"];')
    # period-3 also supports L3H1 attention pattern directly (C8/C35), drawn as a thin blue assist
    A(f'  p3 -> L3H1 [color="{BLUE}", style=dashed, penwidth=1.1, constraint=false, '
      f'xlabel="deletion 0.946\u21920.027 (C8)"];')

    # payload thread: seq -> L0 heads -> MLP0 (edge width matches fig_generator_graph: 1.0 + 3.0*min(N/top,1))
    def pw(h):
        v = head_N.get(h)
        if not v or v[0] is None:
            return 1.0
        return 1.0 + 3.0 * min(v[0] / max(top_N, 1e-9), 1.0)
    for h in dominant:
        A(f'  seq -> {h} [color="{MUT}", penwidth=1.0];')
        A(f'  {h} -> MLP0 [color="{PURPLE}", penwidth={pw(h):.2f}];')
    for h in marginal:
        A(f'  seq -> {h} [color="{MIST}", penwidth=0.8];')
        A(f'  {h} -> MLP0 [color="{MUT}", penwidth={pw(h):.2f}];')
    # H12-H10 redundancy (discovery split), dashed grey, non-constraining
    A(f'  L0H12 -> L0H10 [dir=none, style=dashed, color="{MUT}", constraint=false, '
      f'xlabel="interaction {inter:+.3f} (D)\\npartially redundant"];')
    # serial backbone
    A(f'  MLP0 -> MLP1 [color="{PURPLE}", penwidth=2.2];')
    A(f'  MLP1 -> MLP2 [color="{PURPLE}", penwidth=2.2];')
    A(f'  MLP2 -> L3H1 [color="{PURPLE}", penwidth=2.4, label="acceptor-local\\npayload"];')
    # binding -> write -> delivery -> readout
    A(f'  L3H1 -> Uinfo [color="{PURPLE}", penwidth=2.0, minlen=2, label="donor-side\\nwrite"];')
    A(f'  Uinfo -> deliver [color="{PURPLE}", penwidth=1.6, minlen=2, '
      f'label="{esc(UINFO_ROUTE)} of L3H1 effect\\nroutes here (C17)"];')
    A(f'  deliver -> readout [color="{PURPLE}", penwidth=1.8, minlen=2];')

    A('}')
    OUT_GV.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_GV}")

    dot = shutil.which("dot")
    if not dot:
        sys.exit("dot (graphviz) not on PATH; wrote DOT source only. `conda install -c conda-forge graphviz` and re-run.")
    for fmt, out in (("pdf", OUT_PDF), ("png", OUT_PNG)):
        args = [dot, f"-T{fmt}", str(OUT_GV), "-o", str(out)]
        if fmt == "png":
            args[2:2] = ["-Gdpi=300"]
        subprocess.run(args, check=True)
    print(f"wrote {OUT_PDF} / {OUT_PNG}")

    # provenance sidecar (figure_provenance(sources) -> dict; written next to the figure)
    try:
        prov = figure_provenance([EDGES])
        (HERE / "fig_full_circuit.provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
        print("wrote fig_full_circuit.provenance.json")
    except Exception as e:  # provenance is best-effort; the figure is the deliverable
        print(f"provenance note skipped: {e}")


if __name__ == "__main__":
    main()
