import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from btp.core import encode, make_basis, normalize, replace_outliers


@pytest.mark.parametrize("pipeline", ["A", "B", "C"])
def test_permutation_invariance(pipeline):
    x = np.random.default_rng(1).normal(size=(200, 3)).astype(np.float32)
    basis = make_basis(32)
    a = encode(normalize(x, pipeline)[0], basis)
    b = encode(normalize(x[::-1], pipeline)[0], basis)
    np.testing.assert_allclose(a, b, atol=2e-6)


@pytest.mark.parametrize("pipeline", ["A", "B", "C"])
def test_positive_scale_and_translation_invariance(pipeline):
    x = np.random.default_rng(2).normal(size=(200, 3)).astype(np.float32)
    a = normalize(x, pipeline)[0]
    b = normalize(x*3 + np.array([2, -1, 4], dtype=np.float32), pipeline)[0]
    np.testing.assert_allclose(a, b, atol=2e-6)


def test_shared_basis_and_exact_distance_definition():
    basis = make_basis(24)
    np.testing.assert_array_equal(basis, make_basis(24))
    x = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    expected = np.linalg.norm(basis[:, None]-x[None], axis=2).min(axis=1)
    np.testing.assert_allclose(encode(x, basis), expected, atol=1e-7)


def test_matched_corruptions_and_no_input_mutation():
    x = np.random.default_rng(3).normal(size=(2048, 3)).astype(np.float32)
    original = x.copy()
    a, mask_a = replace_outliers(x, 0.02, 3, 101, 7)
    b, mask_b = replace_outliers(x, 0.02, 10, 101, 7)
    assert a.shape == x.shape and mask_a.sum() == 41
    np.testing.assert_array_equal(mask_a, mask_b)
    np.testing.assert_array_equal(x, original)
    centre = x.mean(0)
    np.testing.assert_allclose((a[mask_a]-centre)/3, (b[mask_b]-centre)/10, atol=2e-6)


def test_sparse_far_outliers_expose_maximum_scale_failure():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2048, 3)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    noisy, _ = replace_outliers(x, 0.01, 10, 101, 0)
    a_ratio = normalize(noisy, "A")[2]/normalize(x, "A")[2]
    c_ratio = normalize(noisy, "C")[2]/normalize(x, "C")[2]
    assert a_ratio > 9
    assert 0.9 < c_ratio < 1.1


def test_percentile_scale_is_not_immune_to_heavy_contamination():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2048, 3)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    noisy, _ = replace_outliers(x, 0.1, 10, 101, 0)
    assert normalize(noisy, "C")[2]/normalize(x, "C")[2] > 8


@pytest.mark.parametrize("pipeline", ["A", "B", "C"])
def test_degenerate_cloud_stays_finite(pipeline):
    y, _, scale = normalize(np.ones((20, 3)), pipeline)
    assert np.isfinite(y).all() and scale > 0
