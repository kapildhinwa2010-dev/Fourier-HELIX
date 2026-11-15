"""Tests for FourierHybridStore: 2-D FFT compression variant."""

import numpy as np
import pytest

from fourier_helix import FourierHybridStore, HybridStore, RecallMiss, _fft2_compress, _fft2_decompress


def _smooth_matrix(rows=16, cols=16):
    t = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    s = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)
    return np.outer(np.sin(2 * s), np.cos(3 * t)) + 0.5 * np.outer(np.ones_like(s), np.sin(t))


def test_fft_roundtrip_near_exact_at_keep_one():
    W = _smooth_matrix()
    genes, ratio = _fft2_compress(W, keep=1.0, bits=16)
    recon = _fft2_decompress(genes)
    assert recon.shape == W.shape
    assert np.max(np.abs(recon - W)) < 1e-3


def test_smooth_matrix_reconstructs_well_with_small_window():
    W = _smooth_matrix()
    genes, ratio = _fft2_compress(W, keep=0.25, bits=12)
    recon = _fft2_decompress(genes)
    rmse = float(np.sqrt(np.mean((recon - W) ** 2)))
    assert rmse < 0.05 * (np.max(np.abs(W)) or 1.0)
    assert ratio > 1.0


def test_store_put_get_with_2d_encoding():
    store = FourierHybridStore(keep=0.5)
    W = _smooth_matrix()
    ratio = store.put("m1", W)
    assert ratio > 1.0
    recon = store.get("m1")
    assert recon.shape == W.shape
    assert len(store) == 1
    assert store.list_models() == ["m1"]


def test_inherited_recall_behaviour():
    store = FourierHybridStore(keep=0.5)
    with pytest.raises(RecallMiss):
        store.get("nothing")
    store.put("m1", _smooth_matrix())
    with pytest.raises(RecallMiss):
        store.get("other")


def test_validation():
    with pytest.raises(ValueError):
        FourierHybridStore(keep=0.0)
    with pytest.raises(ValueError):
        FourierHybridStore(storage_bits=0)
    store = FourierHybridStore()
    with pytest.raises(ValueError):
        store.put("bad", np.zeros(4))


def test_noisy_recall_still_works():
    store = FourierHybridStore(keep=0.5)
    W = _smooth_matrix()
    store.put("m1", W)
    recon = store.get_noisy("m1", noise=0.05)
    assert recon.shape == W.shape
