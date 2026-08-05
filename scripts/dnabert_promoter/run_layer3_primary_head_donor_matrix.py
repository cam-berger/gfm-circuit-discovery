"""Run the donor-class matrix for the frozen DNABERT Layer-3 six-head set."""

from __future__ import annotations

from pathlib import Path

import torch

import run_component_donor_matrix as donor_matrix


PROBE_PATH = Path(
    "results/dnabert_promoter/frozen/"
    "layer_probes/all_layer_probes/layer_3_probe.joblib"
)

PRIMARY_HEADS = (
    (0, 8),
    (1, 8),
    (2, 3),
    (2, 8),
    (2, 9),
    (3, 11),
)


def pool_layer3(
    adapter,
    tokenized,
) -> torch.Tensor:
    """Masked-mean pooling of hidden_states[3]."""

    output = adapter.forward(
        tokenized,
        output_hidden_states=True,
        output_attentions=False,
    )

    if output.hidden_states is None:
        raise RuntimeError("No hidden states returned")

    hidden = output.hidden_states[3]

    biological_mask = adapter.biological_token_mask(
        tokenized
    ).to(
        device=hidden.device,
        dtype=hidden.dtype,
    )

    mask = biological_mask.unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)

    return (hidden * mask).sum(dim=1) / denominator


def configure_layer3_experiment() -> None:
    """Replace Layer-2 defaults while retaining donor-matrix logic."""

    donor_matrix.PROBE_PATH = PROBE_PATH
    donor_matrix.SELECTED_LAYER = 3
    donor_matrix.pool_selected_layer = pool_layer3

    donor_matrix.COMPONENT_CONDITIONS = {
        "PRIMARY_SIX_HEADS": {
            "heads": PRIMARY_HEADS,
            "mlp_layer": None,
        },
    }


if __name__ == "__main__":
    configure_layer3_experiment()
    donor_matrix.main()
