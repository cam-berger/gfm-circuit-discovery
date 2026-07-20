"""A2b -- localize L3H1's value payload to U_info (donor-graph experiment 6).

Kernel for ``preregistrations/PREREGISTRATION_A2b_PAYLOAD_LOCALIZATION.md``.  SpliceBERT is post-LayerNorm, so the
layer-3 value projection is LINEAR in the L3-input residual ``r = hidden_states[3]``:
``V = W_V r + b_V`` (no LayerNorm between the layer input and QKV).  So a residual edit ``δr`` at
the acceptor window gives an exact value edit ``δV = W_V δr``, and -- with the clean attention row
``A+`` held fixed -- L3H1's donor head output is reconstructed analytically:

    o_d(payload) = A+_d . V- + Σ_{j in acc} A+_{dj} (W_V δr_j)

and injected donor-only (``patch_heads``) with NO re-run of a patched forward.  The attention row
is therefore numerically identical across payloads by construction; the linearity is a preflight
memorial (``V+ - V- == W_V Δr``).
"""
from __future__ import annotations

import numpy as np
import torch

from gfmcd.circuits.hf_bert import patch_heads
from gfmcd.splicing.circuit import READER

L3, H3 = READER


def value_weight_head(ad, layer: int = L3, head: int = H3) -> torch.Tensor:
    """The value projection weight rows for one head: ``W_V^{head}`` [Dh, hidden]."""
    Dh = ad.hidden_size // ad.num_heads
    W = ad.attention_self_modules[layer].value.weight       # [hidden, hidden] (out, in)
    return W[head * Dh:(head + 1) * Dh, :].detach()          # [Dh, hidden]


def projector(U: np.ndarray, device) -> torch.Tensor:
    """Symmetric rank-k projector ``P = Uᵀ U`` [D,D] from orthonormal rows ``U`` [k,D]; row-vector
    convention ``P_U x = x @ P`` matches ``subspace.project``."""
    Ut = torch.tensor(np.ascontiguousarray(U, dtype=np.float32), device=device)
    return Ut.T @ Ut                                         # [D,D]


def deliver(Wv: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    """Value-space image of a residual edit: ``δV = δr W_Vᵀ``.  ``dr`` [B,K,hidden] -> [B,K,Dh]."""
    return dr @ Wv.T


def delivered_energy(dV: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of a delivered value edit over the (K, Dh) window, per example [B]."""
    return dV.flatten(1).norm(dim=1)


def head_output(base: torch.Tensor, a_win: torch.Tensor, dV_win: torch.Tensor) -> torch.Tensor:
    """``base + Σ_{j in window} a_win[j] · dV_win[j]`` -> [B, Dh].  ``base`` = A+_d · V- ([B,Dh]),
    ``a_win`` the donor attention weights to the window [B,K], ``dV_win`` the value edits [B,K,Dh]."""
    return base + torch.einsum("bk,bkd->bd", a_win, dV_win)


def inject_margin_donor(ad, task, corr, dn, dtok, recon):
    """Inject ``recon`` [B,Dh] as L3H1's donor head output on the corrupt run; return the margin.

    The head-cache source is zeros except at ``(H3, donor)`` -- ``patch_heads`` with a donor
    ``token_mask`` and head ``(L3,H3)`` reads only that slot.
    """
    B, Dh = recon.shape
    S = corr["input_ids"].shape[1]
    src = torch.zeros(B, ad.num_heads, S, Dh, device=recon.device, dtype=recon.dtype)
    b = torch.arange(B, device=recon.device)
    src[b, H3, dtok] = recon
    mask = torch.zeros(B, S, dtype=torch.bool, device=recon.device)
    mask[b, dtok] = True
    with patch_heads(ad, {L3: src}, [(L3, H3)], token_mask=mask):
        return task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
