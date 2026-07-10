"""Generate the study template + sandbox notebooks from source.

Notebooks are hard to review in a diff and easy to leave in a half-run state, so they are generated
here rather than hand-edited. Run this, then execute with nbconvert:

    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute notebooks/TEMPLATE_study_a_circuit.ipynb \
        --output study_reading_frame_counter.ipynb   # (with CHECKPOINT set)

The template walks the five pillars in order and branches on the model family (`m.is_aa_head`) so the
same notebook studies both the reading-frame anchor-relative modulo-3 circuit and the repeat-copy
induction circuit.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text.strip("\n"))


# ============================================================================
# TEMPLATE: study a circuit -- the five pillars in order
# ============================================================================


def build_template() -> nbf.NotebookNode:
    cells = []

    cells.append(md(
        "# Study a circuit — the five-pillar template\n"
        "\n"
        "A parameterized walk through the standard circuit-study protocol. Set the **checkpoint** and "
        "**data** in the parameters cell, run top to bottom, and read the verdict off the last plot. "
        "Every section is one of the five pillars the group settled on:\n"
        "\n"
        "| # | pillar | question | the trap it guards |\n"
        "|---|--------|----------|--------------------|\n"
        "| 1 | **discovery** | which components look responsible? | attention *ranks*, it does not *decide* (L1H2) |\n"
        "| 2 | **testing** | is this a mechanism or a shortcut? | a task a pooled statistic can solve needs no circuit |\n"
        "| 3 | **ablation** | what breaks when you remove it? | accuracy saturates and lies — read Δ**loss** (D27) |\n"
        "| 4 | **baselines** | what could a bounded window do? | the measured ceiling is the control, not a guess |\n"
        "| 5 | **conflict** | which cue wins when they disagree? | only cue conflict sees a *redundant* circuit (D24) |\n"
        "\n"
        "The template branches on the model family, so it runs unchanged on the amino-acid reading-frame "
        "models and the causal repeat-copy models. The two executed instances "
        "(`study_reading_frame_counter.ipynb`, `study_repeat_copy.ipynb`) are this notebook with the "
        "parameters cell changed."
    ))

    cells.append(md(
        "## Problem statement — what does the model take in, and what should it put out?\n"
        "\n"
        "Before any circuit talk, state the task the model was trained on. A circuit is a mechanism for "
        "*this* input→output map; naming it fixes what 'the anchor', 'the frame', 'the copy' mean below.\n"
        "\n"
        "| | reading-frame models (aa head) | repeat-copy models (causal LM head) |\n"
        "|---|---|---|\n"
        "| **input** | a 256-nt DNA window with one ATG start codon, tokens in `{A,C,G,T,N}` | a 255-nt window containing a sequence that is repeated once (unit + its copy) |\n"
        "| **output** | per position, the **amino acid** of the codon that position falls in (21-way: 20 aa + stop) | per position, the **next nucleotide** (6-way LM over `ACGTN?`) |\n"
        "| **the hard part** | to name the amino acid you must know the reading **frame** — which of the 3 offsets from the ATG this position sits in | inside the second copy, the next token is only predictable by **copying** from the matching position in the first occurrence |\n"
        "| **the shortcut** | local codon composition is *almost* a sufficient statistic for frame — so a pooled bag nearly solves it | none: the copied content is arbitrary, so no bounded-window statistic beats the base rate |\n"
        "| **the circuit of interest** | an **anchor-relative modulo-3 circuit**: locate the ATG, then recover each position's phase relative to it (whether it iterates a count or reads a residue-class lookup is unresolved) | a **prefix-keyed copy (induction) circuit**: match the current position back to the earlier occurrence and copy |\n"
        "\n"
        "The cell below prints the specific statement for whichever checkpoint you loaded."
    ))

    cells.append(md("## Parameters\nEverything you change to study a different circuit lives here."))
    cells.append(code(
        '''
# --- parameters -----------------------------------------------------------
CHECKPOINT = "nano_uniform"   # a family stem (latest step) or a full path; see lab.list_checkpoints()
N_WINDOWS  = 256              # how many sequences to measure on
SEED       = 555             # window seed (reading-frame models only)
DEMANDED   = 1.0             # accuracy the objective demands (1.0 = perfect; sets the circuit-vs-shortcut test)
# --------------------------------------------------------------------------
'''
    ))

    cells.append(code(
        '''
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))  # notebooks/ -> ../src

import numpy as np
import matplotlib.pyplot as plt

import gfmcd.lab as lab
from gfmcd.lab import viz

m = lab.load_checkpoint(CHECKPOINT)
print(m)
print("family:", "amino-acid reading-frame" if m.is_aa_head else "causal repeat-copy")
'''
    ))

    cells.append(code(
        r'''
# Problem statement for THIS checkpoint.
if m.is_aa_head:
    print("TASK: reading-frame amino-acid readout")
    print("  input : 256-nt DNA window, one ATG start codon, tokens in {A,C,G,T,N}")
    print("  output: per position, the amino acid of its codon (21-way: 20 aa + stop)")
    print("  hard  : the readout requires the reading FRAME (offset mod 3 from the ATG)")
    print("  short : local codon composition is nearly a sufficient statistic for frame")
    print("  circuit: an anchor-relative modulo-3 circuit (find ATG -> phase relative to it; counting vs lookup unresolved)")
else:
    print("TASK: repeat-copy next-nucleotide prediction")
    print("  input : 255-nt window containing a sequence repeated once (unit + copy)")
    print("  output: per position, the next nucleotide (6-way LM over ACGTN?)")
    print("  hard  : inside the 2nd copy, the next token is only set by the 1st occurrence")
    print("  short : none - copied content is arbitrary, so no pooled statistic beats base rate")
    print("  circuit: a prefix-keyed copy/induction circuit (match back, then copy)")
'''
    ))

    cells.append(code(
        '''
# Load the windows for whichever family this checkpoint is.
if m.is_aa_head:
    from gfmcd.signatures import start_index
    tok, aa, phase = lab.load_uniform_windows(n=N_WINDOWS, seed=SEED)
    start = start_index(phase)
    print("reading-frame windows:", tok.shape)
else:
    batch = lab.load_repeat_windows(n=N_WINDOWS)
    print("repeat-copy batch:", batch.tokens.shape)
'''
    ))

    # ---- SECTION 0: training dynamics / grokking ------------------------------
    cells.append(md(
        "## 0 · Training dynamics — *when* does the circuit form?\n"
        "\n"
        "Before dissecting the trained model, look at how it got there. The circuit does not fade in "
        "with the loss — it **groks**: accuracy stays near the shortcut/base-rate floor for thousands of "
        "steps while the loss drifts down, then jumps. The step of that jump is when the mechanism snaps "
        "into place. The panel plots CE loss and accuracy against training step and marks the "
        "transition. (Needs `checkpoints/<family>_history.json`.)"
    ))
    cells.append(code(
        '''
family = m.name.split("_step")[0]
try:
    history = lab.load_history(family)
    if m.is_aa_head:
        acc_key, loss_key, floor = "aa_acc", "val", None
    else:
        acc_key, loss_key, floor = "copy_acc", "loss", None
        # base-rate floor for the copy task, measured
        floor = lab.copy_metrics(m, batch)["base_rate_ceiling"]
    trans = lab.grok_transition(history, acc_key)
    print(f"final {acc_key}: {trans['final_acc']:.4f}")
    print(f"first step with {acc_key} >= 0.5: {trans['step_at_0.5']}")
    print(f"first step with {acc_key} >= 0.9: {trans['step_at_0.9']}")
    print(f"steepest rise (circuit forms): step {trans['transition_from']} -> {trans['transition_to']}")
    viz.plot_grokking(history, acc_key, loss_key, transition=trans, base_floor=floor,
                      title=f"{family}: circuit formation (grokking)")
    plt.show()
except FileNotFoundError as e:
    print("no training history for this family:", e)
'''
    ))

    # ---- PILLAR 1 -------------------------------------------------------------
    cells.append(md(
        "## 1 · Discovery — which components look responsible?\n"
        "\n"
        "Survey attention to the circuit's anchor (the start codon for the reading-frame circuit; the copy "
        "source for induction). This **ranks** candidate heads and nothing more. On `nano_uniform` the "
        "head that attends the start codon *most* is L5H2 — yet its ablation Δloss (see the pillar-3 "
        "table below) is ~0, while the essential head L1H3 ranks lower in attention here and only "
        "reveals itself under ablation. Compare the two numbers when you reach pillar 3. This is the "
        "repo's founding lesson in one panel: **attribution ranks, intervention decides.** Read this "
        "ranking as a list of suspects, not a verdict."
    ))
    cells.append(code(
        '''
if m.is_aa_head:
    attn = lab.attention_survey(m, tok, start)          # attention to the ATG, downstream queries
    viz.plot_attention_survey(attn, title=f"{m.name}: attention to start codon")
    order = np.argsort(-attn.ravel())[:5]
    print("top heads by attention to start:",
          [(f"L{k//m.cfg.n_heads}H{k%m.cfg.n_heads}", round(float(attn.ravel()[k]),3)) for k in order])
else:
    # induction: attend from each copy position back to its source (the matching position in occurrence 1)
    attn = lab.copy_source_attention(m, batch)
    viz.plot_attention_survey(attn, title=f"{m.name}: attention to copy source")
    order = np.argsort(-attn.ravel())[:6]
    print("top heads by attention to source:",
          [(f"L{k//m.cfg.n_heads}H{k%m.cfg.n_heads}", round(float(attn.ravel()[k]),3)) for k in order])
plt.show()
'''
    ))

    # ---- PILLAR 2 -------------------------------------------------------------
    cells.append(md(
        "## 2 · Testing — mechanism or shortcut?\n"
        "\n"
        "Before tracing anything, ask whether a **bounded-window pooled statistic** could already "
        "solve the task. If it can reach what the objective demands, there is no reason for the model "
        "to build a circuit and nothing to trace — the shortcut suffices. A circuit is predicted "
        "only when *demanded accuracy exceeds the pooled ceiling*. This is the project's central working "
		"hypothesis — the shortcut-gap hypothesis — not a settled law."
    ))
    cells.append(code(
        '''
if m.is_aa_head:
    report = lab.ceiling_report(m, tok, phase, demanded=DEMANDED)
    print(report.note)
    print(f"pooled ceiling {report.pooled_oracle_acc:.4f} | model phase {report.model_phase_acc:.4f} | demanded {report.demanded}")
    viz.saturation_guard(report.model_phase_acc, "model phase accuracy")
    viz.plot_ceiling(report, title=f"{m.name}: shortcut ceiling vs model")
else:
    cm = lab.copy_metrics(m, batch)
    print(f"copy accuracy {cm['copy_accuracy']:.4f} vs base-rate ceiling {cm['base_rate_ceiling']:.4f}")
    print("the copy task is shortcut-proof: base rate is far below the model, so a circuit must exist.")
plt.show()
'''
    ))

    # ---- PILLAR 3 -------------------------------------------------------------
    cells.append(md(
        "## 3 · Ablation — what breaks when you remove it?\n"
        "\n"
        "Ablate each head and read **Δloss**, not accuracy. Accuracy saturates: L1H3 can move the loss "
        "by +8 while accuracy (pinned at 1.0) reports nothing. The panel plots the causal Δloss next to "
        "the attention rank from pillar 1 — where they disagree, the loss bar is right.\n"
        "\n"
        "Ablations here restore themselves (`with lab.ablate_heads(...)`), so a zeroed head never leaks "
        "into a later cell. For the start-codon ablation the decoy-Met control is attached by default "
        "(D24): the effect must be the anchor's *position*, not the edit."
    ))
    cells.append(code(
        '''
if m.is_aa_head:
    table = lab.head_ablation_table(m, tok, aa, target_pos=start)
    print("head            Δloss      Δacc     attn->start")
    for e in table[:6]:
        print(f"L{e.layer}H{e.head:<3}   {e.delta_loss:>+10.4f}  {e.delta_acc:>+7.4f}   {e.attn_to_target:.3f}")
    viz.plot_head_effects(table, title=f"{m.name}: attention rank vs causal effect")
else:
    table = lab.copy_head_ablation(m, batch)
    print("head            Δcopy-loss")
    for e in table[:8]:
        print(f"L{e.layer}H{e.head:<3}   {e.delta_loss:>+10.4f}")
    viz.plot_head_effects(table, title=f"{m.name}: copy-loss Δ per head")
plt.show()
'''
    ))
    cells.append(md(
        "### Start-codon ablation with the decoy control (reading-frame) · redundancy (copy)\n"
        "The decoy-Met control isolates the anchor's position from the edit. For a distributed circuit, "
        "compare each head's solo Δloss against the joint Δloss: sum ≫ joint means the heads are "
        "redundant, and *ablation alone cannot see the circuit* — which is exactly why pillar 5 exists."
    ))
    cells.append(code(
        '''
if m.is_aa_head:
    abl = lab.start_codon_ablation(m, tok, phase)
    print(f"far-field phase: intact {abl['far_intact_phase']:.4f} -> ablated {abl['far_ablated_phase']:.4f}")
    print(f"start-codon drop {abl['ablation_drop']:.4f} | decoy control {abl['decoy_control_drop']:.4f} | "
          f"anchor-specific {abl['anchor_specific_drop']:.4f}")
else:
    top_heads = [(e.layer, e.head) for e in table[:5]]
    red = lab.copy_redundancy(m, batch, top_heads)
    print("top-5 source-following heads:", top_heads)
    for k, v in red.singles.items():
        print(f"  L{k[0]}H{k[1]} solo Δloss {v:+.4f}")
    print(f"Σ singles = {red.sum_of_singles:.4f}   joint = {red.subset_delta:.4f}")
    print(red.interpretation)
    viz.plot_redundancy(red, title=f"{m.name}: single vs joint ablation")
    plt.show()
'''
    ))

    # ---- PILLAR 4 -------------------------------------------------------------
    cells.append(md(
        "## 4 · Baselines — what a bounded window could do, measured\n"
        "\n"
        "The comparison in pillar 2 is only as good as the ceiling it uses. The ceiling is **measured**, "
        "not asserted: the anchor-free pooled oracle is a codon-level naive-Bayes estimator over a ±120 "
        "nt window (D30's 0.9509 on real CDS). For copy, the base rate is the accuracy of always "
        "predicting the corpus-frequency token. Report the number and where it came from."
    ))
    cells.append(code(
        '''
if m.is_aa_head:
    pooled = lab.pooled_ceiling(m, tok, phase)
    print(f"anchor-free pooled oracle (codon naive Bayes, ±120 nt): {pooled:.4f}")
    print("interpretation: this is the best any start-codon-blind statistic does. The model can only "
          "beat it by building an anchored circuit.")
else:
    cm = lab.copy_metrics(m, batch)
    print(f"base-rate ceiling (predict most-frequent token): {cm['base_rate_ceiling']:.4f}")
    print(f"model copy accuracy: {cm['copy_accuracy']:.4f}  ->  gap the circuit must explain: "
          f"{cm['copy_accuracy'] - cm['base_rate_ceiling']:.4f}")
'''
    ))

    # ---- PILLAR 5 -------------------------------------------------------------
    cells.append(md(
        "## 5 · Conflict — which cue wins when they disagree?\n"
        "\n"
        "The decisive test, and the only one that sees a redundant circuit. Put the anchor and the "
        "local content in conflict (insert one nt after the ATG; edit a copied token) and measure which "
        "the model follows at range. A **counter** follows the anchor at every distance; a **bag** "
        "crosses to content beyond ~30 nt; a **relay** initialises from the anchor but propagates "
        "locally. The mechanism verdict is the far-field anchor share, read at the phase probe. The "
		"**counter** pole means the anchor is followed at every distance — it does not by itself settle "
		"whether the model iterates a sequential count or reads an anchor-relative modulo-3 lookup.\n"
        "\n"
        "For the reading-frame family we place the verdict in two contexts that the single-model "
        "signature can't show: the **λ sweep** (how the mechanism moves as the shortcut is made "
        "stronger) and the **2×2 grid** (shortcut × detector difficulty, the design that isolated "
        "which conditions force the circuit)."
    ))
    cells.append(code(
        '''
mech = lab.mechanism_signature(m, tok, phase) if m.is_aa_head else None
if m.is_aa_head:
    print(f"mechanism signature: anchor-dependence x={mech.anchor_dependence:.3f}, "
          f"anchor-share y={mech.anchor_share:.3f}")
    print("VERDICT:", mech.verdict)
    viz.plot_mechanism_map(mech, title=f"{m.name}: mechanism verdict")
    plt.show()
else:
    cm = lab.copy_metrics(m, batch)
    print(f"under a single distal edit: follows distal source {cm['conflict_follows_distal']:.4f}, "
          f"follows local continuation {cm['conflict_follows_local']:.4f}")
    print("VERDICT: prefix-keyed copy circuit — the model follows the distal source it copied from, "
          "not the locally-plausible continuation.")
'''
    ))

    # --- reading-frame only: lambda sweep + 2x2 grid ---
    cells.append(md(
        "### 5a · λ sweep — the mechanism moves as the shortcut strengthens *(reading-frame only)*\n"
        "\n"
        "Each arm trains on a data pool with a different shortcut strength: `I(nt; phase)`, the mutual "
        "information between local nucleotide content and the frame. Accuracy stays ~1.0 across the "
        "sweep, so accuracy tells you nothing — only the **mechanism** axis moves. As the shortcut "
        "carries more of the frame, the model slides from an anchored **counter** (anchor share → 1) "
        "toward a pooled **bag** (anchor share → 0). This is the shortcut-gap hypothesis as a dial."
    ))
    cells.append(code(
        '''
if m.is_aa_head:
    LAMBDA_ARMS = ["nano_sense", "nano_lam0.25", "nano_lam0.5", "nano_lam0.75",
                   "nano_lam1", "nano_lam1.5", "nano_lam2"]
    arms = [a for a in LAMBDA_ARMS if a in lab.list_checkpoints()]
    sweep = lab.lambda_sweep(arms, n=max(N_WINDOWS, 384))
    print(f"{'arm':14s}{'I(nt;phase)':>13}{'aa_acc':>9}{'anchor y':>10}   verdict")
    for p in sweep:
        print(f"{p.name:14s}{p.shortcut_nats:>13.5f}{p.aa_acc:>9.4f}{p.anchor_share:>10.3f}   {p.verdict}")
    viz.plot_lambda_sweep(sweep, title="mechanism vs shortcut information (all arms at aa_acc≈1.0)")
    plt.show()
else:
    print("λ sweep is a reading-frame construction; skipped for the copy family.")
'''
    ))

    cells.append(md(
        "### 5b · The 2×2 — which conditions actually force the circuit *(reading-frame only)*\n"
        "\n"
        "The necessity design: **shortcut** (absent / present) × **detector difficulty** (easy / hard). "
        "Each cell gets the frozen, pre-registered verdict criterion — a cell is scored as a circuit iff "
		"far-field anchor share > 0.50 **and** the "
        "start-codon ablation beats the decoy-Met control by > 0.10. Cell A (uniform, shortcut absent) "
        "forces the anchored circuit; the interesting cells are where the shortcut is present. Cells B/C/D need "
        "the cached hg38 windows; any that are missing show as unavailable rather than faked."
    ))
    cells.append(code(
        '''
if m.is_aa_head:
    CELLS = {"A": "nano_uniform", "B": "cellB_s0", "C": "cellC_s0", "D": "cellD_s0"}
    verdicts = {}
    for cell, stem in CELLS.items():
        if stem not in lab.list_checkpoints():
            print(f"cell {cell}: checkpoint {stem} absent — skipped")
            continue
        try:
            mc = lab.load_checkpoint(stem)
            if cell == "A":
                tk, a_, ph = lab.load_uniform_windows(n=1000, seed=4242)
            else:
                tk, a_, ph = lab.load_cell_windows(cell, n=1000, seed=4242)
            v = lab.verdict_2x2(mc, tk, a_, ph, cell=cell)
            verdicts[cell] = v
            print(f"cell {cell} ({stem}): intact {v.intact_acc:.3f}  ablated {v.ablated_acc:.3f}  "
                  f"ctrl {v.control_acc:.3f}  Δabl {v.ablation_effect:+.3f}  anchor {v.anchor_share:.3f}  "
                  f"-> {'CIRCUIT' if v.circuit_formed else 'no circuit'}")
        except Exception as e:
            # cells C/D need real hg38 windows (data/cache/*.npz or hg38.fa); skip cleanly if absent.
            print(f"cell {cell}: data unavailable — {type(e).__name__}: {str(e).splitlines()[0][:80]}")
    if verdicts:
        viz.plot_2x2_grid(verdicts, title=f"{m.name.split('_step')[0]}: 2×2 mechanism verdict")
        plt.show()
else:
    print("The 2×2 is a reading-frame construction; skipped for the copy family.")
'''
    ))

    # ---- REVERSE ENGINEERING --------------------------------------------------
    cells.append(md(
        "## 6 · Reverse-engineer the structure — trace the logic yourself\n"
        "\n"
        "The five pillars tell you *that* a circuit exists, which components are essential, and *what* "
        "mechanism it implements. This section asks the next question: **how is it wired across depth?** "
        "You reconstruct the flow of information from input to readout, layer by layer, using only "
        "quantities the notebook has already measured — then draw it and check the drawing against the "
        "model.\n"
        "\n"
        "The method is the same for any circuit:\n"
        "\n"
        "1. **Where does the answer become available?** Probe each layer's residual stream for the target "
        "(the frame; the copied token). The layer where it becomes decodable is where the circuit has "
        "finished computing it. Everything below is construction; everything above is propagation.\n"
        "2. **Which heads move it there?** You have the ablation table (pillar 3) and the attention "
        "survey (pillar 1). An essential head that writes *into* the layer where the answer appears is a "
        "construction head; an essential head *below* it is setup; heads above mostly propagate.\n"
        "3. **What is each head's job?** Setup heads have high Δloss but do not attend the anchor "
        "(they build local features); the anchor/induction head attends the anchor *and* has high Δloss. "
        "This is the difference between 'reads composition' and 'locates the anchor'.\n"
        "4. **Draw it and falsify it.** The flow diagram is a hypothesis. Every arrow is an ablation you "
        "can run in the sandbox: if L1H3 is 'locate the anchor', ablating it must destroy decodability "
        "at L2, not L1."
    ))
    cells.append(code(
        '''
# Step 1 — where does the target become decodable across depth?
if m.is_aa_head:
    decode = lab.phase_decodability_by_layer(m, tok, phase)
    print("phase (reading frame) decodability by layer:")
    for L, a in enumerate(decode):
        bar = "█" * int(a * 30)
        print(f"  L{L}.resid_post  {a:5.3f}  {bar}")
    formed_layer = next((L for L, a in enumerate(decode) if a > 0.95), None)
    print(f"\\n-> the frame is essentially in place by L{formed_layer} "
          f"(control / bag baseline sits near 1/3 = 0.333)")
else:
    # for the copy circuit, source-attention concentrates in the induction layer
    attn = lab.copy_source_attention(m, batch)
    print("max source-attention by layer (which layer looks back to copy):")
    for L in range(m.cfg.n_layers):
        h = int(attn[L].argmax()); a = float(attn[L].max())
        bar = "█" * int(a * 30)
        print(f"  L{L}  head {h}  {a:5.3f}  {bar}")
    formed_layer = int(attn.max(axis=1).argmax())
    print(f"\\n-> the copy is performed at L{formed_layer} (the induction layer); "
          f"L0 is the previous-token setup layer")
'''
    ))
    cells.append(code(
        '''
# Step 2-4 — assemble the flow from the notebook's own measurements and draw it.
if m.is_aa_head:
    ess = [e for e in lab.head_ablation_table(m, tok, aa, target_pos=start)][:8]
    # the anchor head: the essential head that attends the ATG most (top attention among high-Δloss heads)
    anchor_e = max((e for e in ess if e.delta_loss > 1.0), key=lambda e: e.attn_to_target, default=None)
    anchor_layer = anchor_e.layer if anchor_e else max(formed_layer - 1, 1)
    anchor = [f"L{anchor_e.layer}H{anchor_e.head}"] if anchor_e else ["anchor head"]
    # setup heads: essential heads strictly BELOW the anchor layer (exclude the anchor head itself)
    setup = [f"L{e.layer}H{e.head}" for e in ess
             if e.layer < anchor_layer and (e.layer, e.head) != (anchor_e.layer, anchor_e.head)][:4]
    stages = [
        {"layer": ("L0" if anchor_layer <= 1 else f"L0–L{anchor_layer-1}"), "role": "read local codon composition",
         "heads": " ".join(setup) or "L0 heads", "detail": "high Δloss, do NOT attend the ATG", "kind": "setup"},
        {"layer": f"L{anchor_layer}", "role": "locate the ATG anchor, fix the anchor-relative phase",
         "heads": " ".join(anchor), "detail": "attends the ATG AND high Δloss", "kind": "anchor"},
        {"layer": f"L{formed_layer}–L{m.cfg.n_layers-1}", "role": "propagate the frame to every position",
         "heads": "", "detail": f"phase fully decodable from L{formed_layer}", "kind": "propagate"},
        {"layer": "out", "role": "read out the amino acid", "heads": "", "detail": "21-way per position", "kind": "readout"},
    ]
    viz.plot_circuit_flow(stages, title=f"{m.name.split('_step')[0]}: reading-frame anchor-relative circuit — flow",
                          probe_by_layer=decode)
else:
    tbl = lab.copy_head_ablation(m, batch)
    setup = [f"L{e.layer}H{e.head}" for e in tbl if e.layer < formed_layer][:4]
    induct = [f"L{e.layer}H{e.head}" for e in tbl if e.layer == formed_layer and e.attn_to_target > 0.3][:3]
    stages = [
        {"layer": "L0", "role": "previous-token / setup heads",
         "heads": " ".join(setup) or "L0 heads", "detail": "high Δloss, source-attn ≈ 0", "kind": "setup"},
        {"layer": f"L{formed_layer}", "role": "induction: attend back to the source, copy it",
         "heads": " ".join(induct) or "induction heads", "detail": "source-attn 0.6–0.7", "kind": "propagate"},
        {"layer": "out", "role": "predict the next nucleotide", "heads": "", "detail": "6-way LM", "kind": "readout"},
    ]
    viz.plot_circuit_flow(stages, title=f"{m.name.split('_step')[0]}: repeat-copy induction — flow")
plt.show()
'''
    ))
    cells.append(md(
        "**Now falsify the diagram (open the sandbox).** Each arrow is a prediction:\n"
        "\n"
        "- *Reading frame* — if the anchor head really sets the count at its layer, ablating it should "
        "collapse phase decodability at the **next** layer (re-run step 1 inside `ablate_heads`). The "
        "setup heads should degrade composition features but not the anchor's location.\n"
        "- *Repeat copy* — if L0 is the previous-token setup and L1 is the induction step, then ablating "
        "an L0 head should blunt the L1 heads' source-attention (a two-hop dependency), while ablating "
        "an L1 head removes the copy directly. The induction motif is L0→L1; test the hop.\n"
        "\n"
        "This is the loop the sandbox exists for: read a number, form a wiring guess, break the wire, "
        "check the number moved where the guess said it would."
    ))

    cells.append(md(
        "## Verdict\n"
        "Fill in for the record:\n"
        "\n"
        "- **Circuit predicted?** (pillar 2) — demanded vs pooled ceiling\n"
        "- **Essential components** (pillar 3) — heads with large Δloss\n"
        "- **Inert look-alikes** (pillar 1 vs 3) — high attention, ~zero Δloss\n"
        "- **Mechanism** (pillar 5) — counter / relay / bag, with the signature coordinates\n"
        "- **Saturation checks passed?** — did any conclusion rest on a metric at its ceiling?"
    ))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "gfmcd (uv)", "language": "python", "name": "gfmcd-uv"}
    return nb


# ============================================================================
# SANDBOX: freeform model-poking for reverse engineering
# ============================================================================


def build_sandbox() -> nbf.NotebookNode:
    cells = []

    cells.append(md(
        "# Sandbox — poke the model\n"
        "\n"
        "A freeform surface for reverse-engineering a circuit by hand. The study template runs a fixed "
        "protocol; this is for the exploratory loop that comes *before* you know which protocol to run — "
        "look at a head, break it, watch what changes, form a guess, test it in the next cell.\n"
        "\n"
        "Each section is a self-contained recipe. Copy a cell, change the head or the checkpoint, re-run. "
        "Nothing here is sacred; overwrite it. The only rule the tooling enforces for you is that every "
        "ablation restores itself, so you can never leave a head silently zeroed.\n"
        "\n"
        "The lab API in one screen:\n"
        "```python\n"
        "m = lab.load_checkpoint('nano_uniform')      # any family stem or path\n"
        "lab.list_checkpoints()                        # what's on disk\n"
        "logits, cache = m.model.run_with_cache(tok)   # 56 cache keys: L{i}.attn.{scores,q,pattern,z}, resid_*\n"
        "with lab.ablate_heads(m.model, [(1,3)]): ...  # zero-ablate, auto-restore\n"
        "with lab.resample_heads(m.model, [(1,3)], src_tok): ...  # on-manifold baseline\n"
        "```"
    ))

    cells.append(code(
        '''
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
import numpy as np, torch
import matplotlib.pyplot as plt
import gfmcd.lab as lab
from gfmcd.lab import viz

print("checkpoints:", lab.list_checkpoints())
'''
    ))

    cells.append(md("## Load a model\nSwap the stem to study anything on disk. `.pool` tells you the data regime it was trained on."))
    cells.append(code(
        '''
m = lab.load_checkpoint("nano_uniform")
m   # repr shows family (mlm/causal), head (aa/lm), layers x heads, device
'''
    ))

    cells.append(md(
        "## Recipe 1 — look at one head's attention\n"
        "Pick a head, run a few sequences, and look at what it attends to. Here: does L1H3 attend the "
        "start codon? Change `(L, H)` and re-run to inspect any head."
    ))
    cells.append(code(
        '''
from gfmcd.signatures import start_index
tok, aa, phase = lab.load_uniform_windows(n=32, seed=1)
start = start_index(phase)

L, H = 1, 3
with torch.no_grad():
    _, cache = m.model.run_with_cache(torch.from_numpy(tok).long().to(m.device))
pattern = cache[f"L{L}.attn.pattern"][:, H].cpu().numpy()   # [batch, q, k]

row = 0
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(pattern[row], cmap="magma", aspect="auto")
ax.axvline(start[row], color="c", lw=0.8, ls="--", label="start codon")
ax.set_xlabel("key (attended-to)"); ax.set_ylabel("query"); ax.set_title(f"L{L}H{H} attention, seq {row}")
ax.legend(); plt.colorbar(im, ax=ax, fraction=0.046); plt.show()

# how much does this head attend to the start codon, from downstream positions?
print("attention to start (this head):", float(lab.attention_survey(m, tok, start)[L, H]))
'''
    ))

    cells.append(md(
        "## Recipe 2 — ablate and diff, live\n"
        "The core move. Measure something, turn a head off, measure again. Δloss is the arbiter — a head "
        "can carry huge attention and move the loss by nothing (the L1H2 lesson). Try `(1,2)` vs `(1,3)`."
    ))
    cells.append(code(
        '''
base_loss, base_acc = lab.aa_loss_acc(m, tok, aa)
print(f"intact:  loss {base_loss:.4f}  acc {base_acc:.4f}")

for (L, H) in [(1, 3), (1, 2)]:
    with lab.ablate_heads(m.model, (L, H)):
        loss, acc = lab.aa_loss_acc(m, tok, aa)
    print(f"ablate L{L}H{H}:  Δloss {loss-base_loss:+.5f}   Δacc {acc-base_acc:+.5f}   "
          f"({'ESSENTIAL' if loss-base_loss > 0.1 else 'inert'})")
'''
    ))

    cells.append(md(
        "## Recipe 3 — resample ablation (on-manifold baseline)\n"
        "Zero-ablation pushes the residual stream off-distribution. Resample ablation patches the head's "
        "output from a *shuffled* batch instead — the head still writes, just a plausible value. Compare "
        "the two: a large zero-vs-resample gap means part of the 'effect' was distribution shock, not the "
        "head's computation."
    ))
    cells.append(code(
        '''
L, H = 1, 3
idx = torch.randperm(tok.shape[0])
src = torch.from_numpy(tok[idx]).long()

with lab.ablate_heads(m.model, (L, H)):
    zero_loss, _ = lab.aa_loss_acc(m, tok, aa)
with lab.resample_heads(m.model, [(L, H)], src):
    resample_loss, _ = lab.aa_loss_acc(m, tok, aa)

print(f"intact       {base_loss:.4f}")
print(f"zero-ablate  {zero_loss:.4f}  (Δ {zero_loss-base_loss:+.4f})")
print(f"resample     {resample_loss:.4f}  (Δ {resample_loss-base_loss:+.4f})")
print("gap between the two baselines:", f"{abs(zero_loss-resample_loss):.4f}")
'''
    ))

    cells.append(md(
        "## Recipe 4 — one cue-conflict example, read the probabilities\n"
        "Insert a single nucleotide after the ATG so the anchor and the local content disagree, then read "
        "the model's amino-acid distribution at a far-downstream position. An anchor-relative circuit still reads the frame "
        "the anchor implies; a bag has drifted to the content frame. This is the mechanism test on one "
        "sequence, by hand."
    ))
    cells.append(code(
        '''
from gfmcd.signatures import frameshift
tok1, aa1, phase1 = lab.load_uniform_windows(n=4, seed=7)
start1 = start_index(phase1)

row = 0
# frameshift inserts one nt just after the ATG, so anchor and content now disagree downstream.
shifted = frameshift(tok1[row:row+1], np.array([start1[row] + 2]))
pos = start1[row] + 90   # far downstream

with torch.no_grad():
    p_intact  = m.model(torch.from_numpy(tok1[row:row+1]).long().to(m.device)).softmax(-1)[0, pos]
    p_shifted = m.model(torch.from_numpy(shifted).long().to(m.device)).softmax(-1)[0, pos]

top_intact  = int(p_intact.argmax());  top_shifted = int(p_shifted.argmax())
print(f"at position {pos} (distance {pos-start1[row]} from ATG):")
print(f"  intact:  argmax aa {top_intact}  p={float(p_intact[top_intact]):.3f}")
print(f"  shifted: argmax aa {top_shifted}  p={float(p_shifted[top_shifted]):.3f}")
print("  an anchor-relative circuit re-reads the frame from the moved anchor; a bag would not.")
'''
    ))

    cells.append(md(
        "## Recipe 5 — compare two checkpoints (e.g. across grokking)\n"
        "The uniform anchor-relative frame circuit groks abruptly (~step 3000). Load an early and a late checkpoint of the same "
        "family and watch a circuit metric jump. Pin a step with `step=` (or pass the full filename)."
    ))
    cells.append(code(
        '''
early = lab.load_checkpoint("nano_uniform", step=1200)
late  = lab.load_checkpoint("nano_uniform", step=12000)
tokc, aac, phc = lab.load_uniform_windows(n=64, seed=3)

for tag, mm in [("early (1200)", early), ("late (12000)", late)]:
    _, acc = lab.aa_loss_acc(mm, tokc, aac)
    sig = lab.mechanism_signature(mm, tokc, phc)
    print(f"{tag:14s}  aa_acc {acc:.4f}   anchor-share y={sig.anchor_share:.3f}   -> {sig.verdict}")
'''
    ))

    cells.append(md(
        "## Recipe 6 — the copy circuit, same tools\n"
        "The induction models are causal with a nucleotide head. Everything above works; the assay names "
        "swap to their copy variants. One cell to see the whole circuit."
    ))
    cells.append(code(
        '''
mc = lab.load_checkpoint("repeat_copy_s0")
batch = lab.load_repeat_windows(n=64)
print(mc)
print("copy metrics:", {k: round(v,4) for k,v in lab.copy_metrics(mc, batch).items()})

attn = lab.copy_source_attention(mc, batch)
order = np.argsort(-attn.ravel())[:5]
print("top source heads:", [(f"L{k//mc.cfg.n_heads}H{k%mc.cfg.n_heads}", round(float(attn.ravel()[k]),3)) for k in order])
viz.plot_attention_survey(attn, title="repeat_copy_s0: attention to source"); plt.show()
'''
    ))

    cells.append(md(
        "## Scratch\nYours. Add cells below. The lab API is stable; the model is not going to bite."
    ))
    cells.append(code("# scratch\n"))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "gfmcd (uv)", "language": "python", "name": "gfmcd-uv"}
    return nb


def main():
    NB.mkdir(exist_ok=True)
    nbf.write(build_template(), NB / "TEMPLATE_study_a_circuit.ipynb")
    print("wrote", NB / "TEMPLATE_study_a_circuit.ipynb")
    nbf.write(build_sandbox(), NB / "sandbox_reverse_engineering.ipynb")
    print("wrote", NB / "sandbox_reverse_engineering.ipynb")


if __name__ == "__main__":
    main()
