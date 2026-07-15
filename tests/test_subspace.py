import numpy as np

from gfmcd.splicing.subspace import (
    energy_spectrum,
    patch_component,
    project,
    project_out,
    random_subspace,
    subspace_overlap,
    top_k_subspace,
)


def test_top_k_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((10, 2)))[0].T  # [2,10]
    rows = rng.standard_normal((200, 2)) @ basis  # lie in that plane
    U = top_k_subspace(rows, 2)
    assert subspace_overlap(U, basis) > 0.999


def test_project_is_identity_in_span_and_zero_orthogonal():
    rng = np.random.default_rng(1)
    U = random_subspace(8, 2, rng)
    x_in = (rng.standard_normal((5, 2))) @ U  # in span
    assert np.allclose(project(x_in, U), x_in, atol=1e-10)
    assert np.allclose(project_out(x_in, U), 0, atol=1e-10)


def test_patch_component_transfers_only_the_span():
    rng = np.random.default_rng(2)
    U = random_subspace(8, 3, rng)
    a, b = rng.standard_normal((4, 8)), rng.standard_normal((4, 8))
    p = patch_component(a, b, U)
    assert np.allclose(project(p, U), project(b, U), atol=1e-10)  # span comes from b
    assert np.allclose(project_out(p, U), project_out(a, U), atol=1e-10)  # rest stays a


def test_overlap_bounds():
    rng = np.random.default_rng(3)
    U = random_subspace(64, 4, rng)
    assert abs(subspace_overlap(U, U) - 1.0) < 1e-9
    V = random_subspace(64, 4, rng)
    assert subspace_overlap(U, V) < 0.30  # ~4/64, well below 1


def test_energy_spectrum_monotone_to_one():
    rng = np.random.default_rng(4)
    e = energy_spectrum(rng.standard_normal((50, 10)))
    assert e[-1] == 1.0 and np.all(np.diff(e) >= -1e-12)
