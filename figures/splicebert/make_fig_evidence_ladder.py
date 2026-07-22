#!/usr/bin/env python
"""THE METHODS FIGURE: the seven-stage workflow, the evidence ladder, and the tool-to-question map.

Rendered from `docs/METHODOLOGY.md` and `docs/EVIDENCE_LADDER.md`.  Every stage, level, tool and
licensed statement below is taken from those two documents; this script invents nothing, promotes
nothing, and asserts no component's status.  If the documents change, this figure is stale and
must be regenerated -- it is a rendering of a specification, not a second copy of it.

WHY IT USES NO COLOUR.  Blue, orange, purple and red are reserved across this figure set for the
period-3 carrier, the ~10.63 carrier, donor-acceptor coupling and the nucleosomal 10.298 signal.
A methods figure that describes the workflow carries no result, so it borrows none of those hues;
the thread stays reserved for figures that do carry results.

The one thing the figure is FOR: making it visible that each column of the ladder is a separate
experiment, and that the arrows between levels are promotions that have to be earned.
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from figstyle import Canvas, INK, MIST, MUT, PANEL, RULE, SEC  # noqa: E402

OUT = HERE / "fig_evidence_ladder.svg"
METHODOLOGY = REPO / "docs/METHODOLOGY.md"
LADDER = REPO / "docs/EVIDENCE_LADDER.md"

# ---------------------------------------------------------------- content (from the two docs)

# docs/METHODOLOGY.md, "The seven-stage workflow".  The second element of each pair is the
# stage's own stopping/licensing sentence, condensed but not softened.
STAGES = [
    ("1  Specify the biological\n   computation",
     ["What information must be", "detected or integrated, where,",
      "and which model-native output", "should change if it is used?",
      "", "Output: a behavioral assay and",
      "rival explanations, NOT a", "circuit hypothesis."]),
    ("2  Establish model\n   behavior",
     ["Native logits, losses, scores.", "Paired counterfactuals over",
      "aggregate correlations.", "",
      "If the model does not reliably", "perform the computation,",
      "CIRCUIT TRACING STOPS."]),
    ("3  Preflight the\n   instrument",
     ["Freeze positive and negative", "controls, sensitivity and",
      "resolution, invariants,", "thresholds, stopping rule.", "",
      "A null is informative only if", "the instrument could have",
      "found the effect."]),
    ("4  Generate and localize\n   candidates",
     ["Probes, geometry, spectra, QK", "decomposition, attribution,",
      "layer/head/context screens.", "",
      "These answer WHERE TO", "INTERVENE NEXT. They establish",
      "no necessity, sufficiency, or", "path."]),
    ("5  Test necessity and\n   specificity",
     ["Ablate, project out, delete a", "band, or substitute a matched",
      "counterfactual; score against", "matched-size nulls.", "",
      "Licenses only: this component", "contributes to this behavior",
      "under this intervention."]),
    ("6  Test mediation,\n   information, direction",
     ["Patching and path patching,", "rescue, transplant, dose",
      "response, QK and OV", "decomposition, edge",
      "interventions.", "",
      "A strong claim combines all", "four -- they are not one."]),
    ("7  Replicate, delimit,\n   interpret",
     ["Held-out genes, seeded draws,", "checkpoints, species, contexts;",
      "cross-example / cross-context", "transduction for a transferable",
      "state.", "",
      "State the strongest licensed", "reading and the FIRST MISSING",
      "BRIDGE."]),
]

# docs/EVIDENCE_LADDER.md, "General ladder" -- rows in the documented order.
LADDER_ROWS = [
    ("Behavior", "Does the model perform the computation?",
     "Native-output counterfactuals and baselines",
     "The model exhibits a scoped behavior."),
    ("Representation", "Is relevant information encoded?",
     "Probes, geometry, spectra, position or residual features",
     "Information is decodable or structurally present."),
    ("Selection / QK", "Does a component read or select it before normalization?",
     "Logit decomposition, QK bilinear analysis",
     "A component scores the candidate relation."),
    ("Routing / attention", "Does the signal survive competition?",
     "Post-softmax attention and controlled deletion",
     "The component routes attention using the signal."),
    ("Output / OV", "What is written downstream?",
     "OV analysis, output projection, downstream feature change",
     "The component writes relevant information."),
    ("Mediation", "Does the component carry the causal effect?",
     "Patching, rescue, transplant, dose response",
     "The component mediates a measured fraction of the effect."),
    ("Directed circuit", "Do connected components implement the computation?",
     "Edge/path interventions plus necessity and mediation",
     "A scoped directed causal pathway implements the behavior."),
    ("Biological interpretation", "Why does this mechanism exist and what does it mean biologically?",
     "Training-data source tests, external assays, cross-model/species evidence",
     "A biological origin or function is supported."),
]

# docs/EVIDENCE_LADDER.md, "Promotion requirements".
PROMOTIONS = [
    ("Representation -> use", "show that a model component reads or routes the feature under an "
                              "appropriate control"),
    ("Use -> necessity", "remove the feature or component and measure the declared behavior"),
    ("Necessity -> mediation", "restore or patch the candidate and recover the effect relative to "
                               "matched nulls"),
    ("Mediation -> directed circuit", "isolate the path and rule out simpler or redundant "
                                      "alternatives"),
    ("Model circuit -> biological interpretation", "connect the mechanism to training-data "
                                                   "provenance or biological function with an "
                                                   "independently capable assay"),
]

# docs/METHODOLOGY.md, "Tool-to-question map".
TOOLS = [
    ("Does the model perform the computation?",
     "Native logits/losses, paired biological counterfactuals, baselines",
     "Behavioral effect", "Internal mechanism"),
    ("Where might the information live?",
     "Probes, spectra, subspaces, attention, attribution",
     "Candidate components and locations", "Causality"),
    ("Is a component required?",
     "Ablation, projection removal, band deletion",
     "Necessity and collateral damage", "Directed mediation"),
    ("What information is carried?",
     "Patching, counterfactual swaps, QK/OV decomposition",
     "Information specificity", "Complete path"),
    ("Does it mediate the behavior?",
     "Rescue, transplant, dose response, path patching",
     "Mediation estimate", "Biological origin or function"),
    ("Is the pathway specific?",
     "Matched nulls, protected channels, rival-path interventions",
     "Comparative specificity", "Generality across models"),
    ("Does the interpretation generalize?",
     "Sealed replication, checkpoint/model contrasts, external data",
     "Scope and boundary conditions", "More than the tested domain"),
]

# ---------------------------------------------------------------- layout

W, H = 1560, 1180
c = Canvas(W, H)
L, R = 56, W - 56

c.txt(W / 2, 42, "How a claim is licensed", size=22, fill=INK, anchor="middle", weight="700")
c.txt(W / 2, 66, "The seven-stage workflow, the evidence ladder, and the tool-to-question map, "
                 "as specified in docs/METHODOLOGY.md and docs/EVIDENCE_LADDER.md",
      size=12, fill=MUT, anchor="middle")

# ============================================================ A. the seven-stage workflow
AY = 104
c.plabel(L, AY, "A", "The seven-stage workflow -- each stage licenses less than the next")

BW, GAP, BH = (R - L - 6 * 14) / 7, 14, 208
BTOP = AY + 22
for i, (title, body) in enumerate(STAGES):
    x = L + i * (BW + GAP)
    c.rect(x, BTOP, BW, BH, fill=PANEL, stroke=RULE, rx=5)
    c.rect(x, BTOP, BW, 3, fill=INK, rx=0)
    for j, line in enumerate(title.split("\n")):
        c.txt(x + 11, BTOP + 24 + j * 15, line.strip() if j else line,
              size=11.5, fill=INK, weight="700")
    for j, line in enumerate(body):
        c.txt(x + 11, BTOP + 68 + j * 14.5, line, size=9.5, fill=SEC)
    if i < 6:
        c.arrow(x + BW + 2, BTOP + BH / 2, x + BW + GAP - 2, BTOP + BH / 2,
                stroke=MUT, w=1.4, head=4)

c.txt(L, BTOP + BH + 20,
      "Attribution ranks; intervention decides.   Evidence never auto-promotes between levels.   "
      "If a control fails the status is INSTRUMENT UNRESOLVED, not positive or negative.",
      size=10.5, fill=INK, weight="700")

# ============================================================ B. the ladder
BY = BTOP + BH + 52
c.plabel(L, BY, "B", "The evidence ladder -- every column is a separate experiment")

COLS = [("Level", 168), ("Question", 372), ("Typical evidence", 452), ("Licensed statement", 412)]
ROW = 33
hy = BY + 24
x = L
for name, w in COLS:
    c.txt(x, hy, name, size=10, fill=MUT, weight="700")
    x += w
c.line(L, hy + 7, R, hy + 7, stroke=RULE, w=1)

for i, row in enumerate(LADDER_ROWS):
    y = hy + 20 + i * ROW
    if i % 2 == 0:
        c.rect(L - 6, y - 13, R - L + 12, ROW, fill=MIST, rx=3, op=.55)
    x = L
    for j, (cell, (_, w)) in enumerate(zip(row, COLS)):
        c.txt(x, y + 5, cell, size=10, fill=INK if j == 0 else SEC,
              weight="700" if j == 0 else None)
        x += w

PY = hy + 20 + len(LADDER_ROWS) * ROW + 14
c.txt(L, PY, "A study may enter the ladder at different observational levels, but it may not skip "
             "the causal bridge when making a causal claim.", size=10.5, fill=INK, weight="700")
for i, (name, rule) in enumerate(PROMOTIONS):
    c.txt(L, PY + 20 + i * 15, name + ":", size=9.5, fill=INK, weight="700")
    c.txt(L + 268, PY + 20 + i * 15, rule, size=9.5, fill=SEC)

# ============================================================ C. the tool-to-question map
CY = PY + 20 + len(PROMOTIONS) * 15 + 26
c.plabel(L, CY, "C", "The tool-to-question map -- and what each tool does NOT establish")

TCOLS = [("Question", 330), ("Main tools", 470), ("Output", 292), ("What it does not establish", 312)]
ty = CY + 24
x = L
for name, w in TCOLS:
    c.txt(x, ty, name, size=10, fill=MUT, weight="700")
    x += w
c.line(L, ty + 7, R, ty + 7, stroke=RULE, w=1)

for i, row in enumerate(TOOLS):
    y = ty + 20 + i * ROW
    if i % 2 == 0:
        c.rect(L - 6, y - 13, R - L + 12, ROW, fill=MIST, rx=3, op=.55)
    x = L
    for j, (cell, (_, w)) in enumerate(zip(row, TCOLS)):
        c.txt(x, y + 5, cell, size=10,
              fill=INK if j == 0 else (MUT if j == 3 else SEC),
              weight="700" if j == 0 else None)
        x += w

c.txt(L, H - 26, "Rendered from docs/METHODOLOGY.md and docs/EVIDENCE_LADDER.md. "
                 "No stage, level, tool or licensed statement is added here, and no component's "
                 "status is asserted.", size=9.5, fill=MUT)

for p in (METHODOLOGY, LADDER):
    if not p.exists():
        raise SystemExit(f"missing {p} -- this figure renders those documents")

c.render(OUT)
print(f"saved -> {OUT}")
