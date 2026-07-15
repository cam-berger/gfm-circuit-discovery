import torch

from gfmcd.splicing.unembed import margin_hard, margin_smooth, read_gradient

BID = {"A": 0, "C": 1, "G": 2, "T": 3}


def test_margin_hard_matches_reference():
    logits = torch.tensor([[0.5, 1.0, 3.0, -2.0, 9.9]])  # extra vocab col ignored
    assert torch.allclose(margin_hard(logits, BID), torch.tensor([3.0 - 1.0]))


def test_margin_smooth_is_logit_minus_logsumexp():
    logits = torch.randn(7, 5)
    g = logits[:, BID["G"]]
    lse = torch.logsumexp(torch.stack([logits[:, BID[b]] for b in "ACT"], -1), -1)
    assert torch.allclose(margin_smooth(logits, BID), g - lse)


def test_read_gradient_of_linear_head_is_analytic():
    torch.manual_seed(0)
    cls = torch.nn.Linear(6, 5, bias=False)
    h = torch.randn(3, 6)
    g = read_gradient(cls, h, BID)  # [3,6]
    # analytic: d/dh [W_G - logsumexp(W_A,W_C,W_T) . h] = W_G - sum_b softmax_b W_b
    W = cls.weight.detach()
    logits = h @ W.T
    p = torch.softmax(torch.stack([logits[:, BID[b]] for b in "ACT"], -1), -1)  # [3,3]
    want = W[BID["G"]][None, :] - (
        p[:, 0:1] * W[BID["A"]] + p[:, 1:2] * W[BID["C"]] + p[:, 2:3] * W[BID["T"]]
    )
    assert torch.allclose(g, want, atol=1e-5)
