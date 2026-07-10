"""gfmcd.lab -- an interactive reverse-engineering layer over the tracer / signatures / oracle.

The scripts in `scripts/` are batch CLIs: they load a checkpoint, run one measurement, print a table,
exit. That is right for a settled result and wrong for the loop this layer is for -- load a model,
look at it, poke a head, watch what breaks, form a guess, test it. Everything here is importable and
returns data, so a notebook drives it interactively.

    import gfmcd.lab as lab

    m = lab.load_checkpoint("nano_uniform")          # one call, any checkpoint stem
    tok, aa, phase = lab.load_windows("A")           # cached or regenerated windows

    # the five pillars, each a call:
    lab.ceiling_report(m, tok, phase)                # 2 could a shortcut do this?
    lab.head_ablation_table(m, tok, aa, start_pos)   # 3 what breaks? (Δloss, not accuracy)
    lab.mechanism_signature(m, tok, phase)           # 5 counter / relay / bag?

Safe interventions restore themselves:

    with lab.ablate_heads(m.model, [(1, 3)]):
        ...                                          # L1H3 off only inside the block

The house rules the repo learned the hard way (D24, D27) are defaults here, not reminders: ablation
reports Δloss with the decoy-Met control attached, and the mechanism verdict comes from cue conflict,
not from an accuracy that saturates.
"""

from __future__ import annotations

from gfmcd.lab.load import (
    LoadedModel,
    grok_transition,
    load_cell_windows,
    load_checkpoint,
    load_history,
    load_repeat_windows,
    load_uniform_windows,
    load_windows,
    list_checkpoints,
    pick_device,
    resolve_checkpoint,
)
from gfmcd.lab.intervene import (
    ablate_all_but,
    ablate_heads,
    mean_heads,
    resample_heads,
)
from gfmcd.lab.assays import (
    CeilingReport,
    HeadEffect,
    LambdaPoint,
    MechanismSignature,
    RedundancyResult,
    Verdict2x2,
    aa_loss_acc,
    lambda_sweep,
    phase_decodability_by_layer,
    verdict_2x2,
    attention_survey,
    ceiling_report,
    copy_head_ablation,
    copy_loss,
    copy_metrics,
    copy_redundancy,
    copy_source_attention,
    cue_conflict_curve,
    head_ablation_table,
    mechanism_signature,
    pooled_ceiling,
    redundancy_test,
    start_codon_ablation,
)

__all__ = [
    # load
    "LoadedModel",
    "load_checkpoint",
    "load_windows",
    "load_uniform_windows",
    "load_repeat_windows",
    "load_cell_windows",
    "load_history",
    "grok_transition",
    "list_checkpoints",
    "resolve_checkpoint",
    "pick_device",
    # intervene
    "ablate_heads",
    "ablate_all_but",
    "resample_heads",
    "mean_heads",
    # assays
    "CeilingReport",
    "HeadEffect",
    "LambdaPoint",
    "MechanismSignature",
    "RedundancyResult",
    "Verdict2x2",
    "lambda_sweep",
    "phase_decodability_by_layer",
    "verdict_2x2",
    "aa_loss_acc",
    "attention_survey",
    "ceiling_report",
    "cue_conflict_curve",
    "head_ablation_table",
    "mechanism_signature",
    "pooled_ceiling",
    "redundancy_test",
    "start_codon_ablation",
    "copy_metrics",
    "copy_loss",
    "copy_head_ablation",
    "copy_source_attention",
    "copy_redundancy",
]
