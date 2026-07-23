#!/usr/bin/env python
"""The A10 payload-generator graph, drawn FROM the edge artifact rather than by hand.

Every node, every edge and every number in this figure is read out of
``results/splicebert_coupling/a10_generator_edges.json`` (C31, held-out, Directed-circuit level).
Nothing is placed by eye: the load-bearing layer-0 heads are selected by the artifact's own
``load_bearing_H`` flag, edge weights are the recorded conditional necessities, and Graphviz does
the layout.  Re-run the experiment and the drawing follows; hand-placed coordinates would not.

WHAT THE GRAPH IS ALLOWED TO SAY.  This artifact measures the VALUE path with the address held
clean, so the graph shows how the acceptor-local payload is BUILT, not how it is addressed or
delivered.  The two grey dashed boxes at the ends are scope context, not measured nodes.

WHAT IT MUST NOT SAY.  The N values are CONDITIONAL causal effects on a held-out split.  They do
not sum and they are not additive fractions of the effect (core principle 6, and the artifact's
own REFLECTION says so).  The chain is strictly serial: each MLP is necessary, none is sufficient
alone.  The graph therefore labels edges with necessity, never with a share.

DEPENDENCY.  Rendering PDF/PNG needs Graphviz's ``dot`` on PATH (``brew install graphviz`` /
``apt-get install graphviz``).  The DOT source is written either way and is itself the
data-derived, reproducible artifact; if ``dot`` is missing this script says so and stops rather
than falling back to a hand-drawn imitation.
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
    COUPLING, INK, MIST, MUT, PANEL, RULE, SEC, figure_provenance,
)

EDGES = REPO / "results/splicebert_coupling/a10_generator_edges.json"
OUTBASE = HERE / "fig_generator_graph"

FONT = "Helvetica"


def _fmt(mean: float, ci: list[float]) -> str:
    return f"N = {mean:+.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]"


def check(edges: dict) -> None:
    """Refuse to draw an unadjudicated or internally inconsistent artifact."""
    if not edges.get("confirmatory"):
        raise RuntimeError("a10_generator_edges is not marked confirmatory")
    if not edges.get("preflight_PASS"):
        raise RuntimeError("refusing to draw a circuit whose preflight did not pass")
    if not edges.get("serial_chain_confirmed"):
        raise RuntimeError("the serial MLP chain is not confirmed in this artifact")
    flagged = {h for h, v in edges["l0_heads"].items() if v["load_bearing_H"]}
    if flagged != set(edges["L0_star_heldout"]):
        raise RuntimeError(
            f"load_bearing_H {sorted(flagged)} disagrees with L0_star_heldout "
            f"{sorted(edges['L0_star_heldout'])}")
    if not edges["mediation"]["mediated_by_MLP0"]:
        raise RuntimeError("the artifact does not record MLP0 mediation")


def build_dot(edges: dict) -> str:
    heads = sorted(edges["L0_star_heldout"], key=lambda h: -edges["l0_heads"][h]["N_h_H"])
    med, chain = edges["mediation"], edges["mlp_chain"]

    def head_node(name: str) -> str:
        rec = edges["l0_heads"][name]
        n, ci = rec["N_h_H"], rec["ci_H"]
        width = 1.0 + 3.0 * min(n / max(edges["l0_heads"][heads[0]]["N_h_H"], 1e-9), 1.0)
        return (f'  "{name}" [label=<<b>{name}</b><br/>'
                f'<font point-size="9">{_fmt(n, ci)}</font>>, '
                f'penwidth={width:.2f}];\n')

    def mlp_node(name: str, key: str) -> str:
        mean, ci = chain[key]
        return (f'  "{name}" [label=<<b>{name}</b><br/>'
                f'<font point-size="9">{_fmt(mean, ci)}</font>>, '
                f'shape=box, style="rounded,filled", fillcolor="{PANEL}", penwidth=2.0];\n')

    lines = [
        "digraph a10_generator {\n",
        "  // GENERATED FROM results/splicebert_coupling/a10_generator_edges.json -- do not hand-edit\n",
        "  rankdir=LR;\n",
        '  bgcolor="white";\n',
        f'  node [fontname="{FONT}", fontsize=11, shape=box, style=rounded, '
        f'color="{COUPLING}", fontcolor="{INK}"];\n',
        f'  edge [fontname="{FONT}", fontsize=9, color="{SEC}", fontcolor="{SEC}", '
        "arrowsize=0.8];\n",
        f'  graph [fontname="{FONT}", fontsize=10, labelloc=b, labeljust=l, '
        "nodesep=0.28, ranksep=0.85];\n",
    ]

    # ---- scope context: not measured nodes in this artifact
    ctx = (f'shape=box, style="rounded,dashed", color="{MUT}", fontcolor="{MUT}", '
           "fontsize=10")
    lines.append(f'  "acceptor-local state" [{ctx}];\n')
    lines.append(f'  "acceptor-local payload" [{ctx}];\n')

    # ---- the layer-0 attention heads
    lines.append("  subgraph cluster_l0 {\n")
    lines.append('    label=<<b>layer-0 attention</b>  held-out load-bearing heads>;\n')
    lines.append(f'    labeljust=l; style=filled; color="{RULE}"; fillcolor="{MIST}";\n')
    for h in heads:
        lines.append("  " + head_node(h))
    lines.append("  }\n")

    # ---- the MLP backbone
    lines.append(mlp_node("MLP0", "N_MLP0_H"))
    lines.append(mlp_node("MLP1", "N_MLP1_H"))
    lines.append(mlp_node("MLP2", "N_MLP2_H"))

    # ---- edges
    lines.append(f'  "acceptor-local state" -> "{heads[0]}" '
                 f'[style=dashed, color="{MUT}"];\n')
    top = edges["l0_heads"][heads[0]]["N_h_H"]
    for h in heads:
        n = edges["l0_heads"][h]["N_h_H"]
        lines.append(f'  "{h}" -> "MLP0" [penwidth={0.6 + 3.4 * n / top:.2f}];\n')
    lines.append('  "MLP0" -> "MLP1" [penwidth=2.2];\n')
    lines.append('  "MLP1" -> "MLP2" [penwidth=2.2];\n')
    lines.append(f'  "MLP2" -> "acceptor-local payload" [style=dashed, color="{MUT}"];\n')

    # ---- the redundancy the group test caught (discovery split, as recorded)
    inter = edges["top_head_interactions_D"]
    if "L0H12xL0H10" in inter:
        lines.append(
            f'  "L0H12" -> "L0H10" [dir=none, style=dashed, constraint=false, '
            f'color="{MUT}", fontcolor="{MUT}", '
            f'label="interaction {inter["L0H12xL0H10"]:+.3f} (D split)\\npartially redundant"];\n')

    # ---- mediation, stated as the artifact states it
    lines.append(
        f'  "mediation" [shape=note, color="{RULE}", fontcolor="{SEC}", fontsize=9, '
        f'style=filled, fillcolor="{PANEL}", label="MLP0 MEDIATES the layer-0 contribution\\n'
        f'L0 heads alone R = {med["R_L0heads_only_H"]:.3f}   '
        f'L0 heads + MLP0 R = {med["R_L0heads_MLP0_H"]:.3f}"];\n')
    lines.append('  { rank=same; "MLP0"; "mediation"; }\n')
    lines.append(f'  "MLP0" -> "mediation" [style=invis];\n')

    caption = (
        f"A10 / C31 payload generator, held-out (n = {edges['n']} pairs, "
        f"{edges['n_genes']} genes).  N = conditional causal necessity with a gene-bootstrap 95% CI.\\l"
        "STRICT SERIAL chain: each MLP is necessary on held-out, none is sufficient alone "
        f"(MLP0 only {chain['R_MLP0_only_H']:.3f}, MLP1 only {chain['R_MLP1_only_H']:.3f}, "
        f"MLP0+MLP1 {chain['R_MLP0_MLP1_H']:.3f}).\\l"
        "The N values are CONDITIONAL effects. They do not sum and are not additive fractions.\\l"
        f"Generator ceiling {edges['ceiling']:.3f}.  Value path only: the address is held clean, "
        "so this graph shows how the payload is BUILT, not how it is addressed or delivered.\\l"
        "Dashed grey boxes are scope context, not measured nodes.\\l")
    lines.append(f'  label="{caption}";\n')
    lines.append("}\n")
    return "".join(lines)


def main() -> None:
    if not EDGES.exists():
        raise SystemExit(f"missing {EDGES}")
    edges = json.loads(EDGES.read_text())
    check(edges)

    dot_path = OUTBASE.with_suffix(".gv")
    dot_path.write_text(build_dot(edges))
    OUTBASE.with_suffix(".provenance.json").write_text(
        json.dumps(figure_provenance([EDGES]), indent=2) + "\n")
    print(f"wrote {dot_path}  (+ provenance)")

    dot = shutil.which("dot")
    if dot is None:
        print("graphviz 'dot' is not on PATH, so no PDF/PNG was rendered.\n"
              "  install it (brew install graphviz / apt-get install graphviz) and re-run;\n"
              "  the .gv source above is the data-derived artifact and is complete.")
        return
    for ext in ("pdf", "png"):
        args = [dot, f"-T{ext}", str(dot_path), "-o", str(OUTBASE.with_suffix(f".{ext}"))]
        if ext == "png":
            args.insert(2, "-Gdpi=300")
        subprocess.run(args, check=True)
    print(f"wrote {OUTBASE.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    main()
