# fig_evidence_ladder — how a claim is licensed (methods figure)

The method, rendered rather than restated: the seven-stage workflow, the evidence ladder, and the
tool-to-question map, drawn directly from `docs/METHODOLOGY.md` and `docs/EVIDENCE_LADDER.md`.

**A. The seven-stage workflow.** Each stage carries its own stopping or licensing sentence, because
those sentences are the method: stage 2 stops circuit tracing if the behaviour is not there; stage 3
insists that a null is informative only when the instrument could have found the effect; stage 4's
observational tools answer *where to intervene next* and establish no necessity, sufficiency, or
path; stage 5 licenses only "this component contributes to this behaviour under this intervention";
stage 7 ends at the strongest licensed reading **and the first missing bridge**.

**B. The evidence ladder.** Eight levels, in the documented order, with the question each asks, the
evidence that answers it, and — the column that does the work — the statement it licenses and no
more. A study may enter at different observational levels, but it may not skip the causal bridge
when making a causal claim. The five promotion requirements beneath the table are what has to be
*earned* to move up.

**C. The tool-to-question map.** Every tool paired with what it **does not** establish. Attribution
ranks; intervention decides.

**No colour.** Blue, orange, purple and red are reserved across this figure set for the period-3
carrier, the ≈10.63 carrier, donor–acceptor coupling, and the nucleosomal ≈10.298 signal. A methods
figure carries no result, so it borrows none of those hues.

**Nothing here is asserted about any component.** The figure renders the specification only: no
stage, level, tool or licensed statement is added, and no component's evidence status appears. The
SpliceBERT status tables live in `EVIDENCE.md` and `docs/EVIDENCE_LADDER.md`, where a `?` (the
defining intervention has not been run) and a `failed`/`null` (it was run and rejected) are
different epistemic states and must not be collapsed.

**Sources:** `docs/METHODOLOGY.md` and `docs/EVIDENCE_LADDER.md`. This figure is a rendering of those
documents; if they change, regenerate it. Generator: `make_fig_evidence_ladder.py`.
