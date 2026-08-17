"""Gate for the address/payload experiment: can a head's output be rebuilt from A and V?

A head's context vector is  ctx[i] = sum_j A[i,j] * V[j].  Separating "address" (the
attention row A[i,:] -- WHERE the head reads) from "payload" (the values V -- WHAT is
there) is only meaningful if that product reproduces the head output the patcher operates
on, to numerical tolerance. If it does not, every swap built on it is measuring an
approximation error rather than a mechanism.

Nothing here is a result. It is the precondition.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "src")

from gfmcd.circuits.hf_bert import capture_head_outputs  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CouplingTask, _dataset, eligible, load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    MATCHED_OPERATOR, TREATMENT_MOTIF, assert_seal_intact, train_chromosomes,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402

GTF = "/Users/cberg/Research/data/genomes/gencode.v50.annotation.gtf.gz"
FASTA = "/Users/cberg/Research/data/genomes/hg38.fa"
MODEL = ("/Users/cberg/Research/projects/gfm-circuit-discovery/data/models/"
         "splicebert_official_archive/extracted/models/SpliceBERT.1024nt")

ad = load_adapter(MODEL, pick_device(None), eager=True)  # eager: sdpa hides attention probs
task = CouplingTask(ad, batch=4)
H, Dh = ad.num_heads, ad.head_dim

D = _dataset(GTF, FASTA, allowed_chromosomes=train_chromosomes())
assert_seal_intact(D)
rows = [x for x in eligible(D, "train", donor_motifs=(TREATMENT_MOTIF,))
        if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)][:4]
pos = [acceptor_readout(x) for x in rows]
enc = task.encode([x.sequence for x in rows], pos)


def capture_av(layer: int):
    """Attention probs [B,H,S,S] and per-head values [B,H,S,D] for one layer."""
    store = {}

    def vhook(_m, _i, out):
        b, s, _ = out.shape
        store["V"] = out.detach().view(b, s, H, Dh).permute(0, 2, 1, 3).clone()

    handle = ad.attention_self_modules[layer].value.register_forward_hook(vhook)
    try:
        with torch.no_grad():
            out = ad.model(**enc, output_attentions=True)
        store["A"] = out.attentions[layer].detach().clone()
    finally:
        handle.remove()
    return store["A"], store["V"]


print("layer   max|recon - true|   relative")
ok = True
for layer in range(ad.num_layers):
    A, V = capture_av(layer)

    def fwd():
        with torch.no_grad():
            ad.model(**enc)

    true = capture_head_outputs(ad, fwd)[layer]          # [B,H,S,Dh]
    recon = torch.einsum("bhij,bhjd->bhid", A.float(), V.float())
    err = (recon - true.float()).abs().max().item()
    scale = true.float().abs().max().item()
    rel = err / max(scale, 1e-12)
    ok &= rel < 1e-4
    print(f"  L{layer}   {err:.3e}          {rel:.3e}")

print()
print(f"RECONSTRUCTION FAITHFUL: {ok}")
if not ok:
    print("  -> the A/V decomposition does not reproduce the head output; the")
    print("     address/payload swap would measure approximation error. STOP.")
    sys.exit(1)
print("  -> address (A) and payload (V) can be swapped independently and injected")
print("     through the same patch_heads path used by every other assay here.")
