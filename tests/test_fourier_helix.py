import numpy as np
import pytest

from fourier_helix import HybridStore, RecallMiss


def _smooth(rows=8, cols=48, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    return rng.standard_normal((rows, 1)) + 0.5 * rng.standard_normal((rows, 2)) @ np.stack([np.sin(t), np.cos(2 * t)])


def test_put_and_get_roundtrip():
    store = HybridStore(num_coeffs=4, bits=12)
    store.put("net-a", _smooth(seed=1))
    rebuilt = store.get("net-a")
    original = _smooth(seed=1)
    rel = np.linalg.norm(original - rebuilt) / np.linalg.norm(original)
    assert rel < 0.05


def test_put_returns_ratio_above_one():
    store = HybridStore(num_coeffs=4)
    ratio = store.put("net", _smooth(rows=16, cols=128))
    assert ratio > 1.0


def test_unknown_key_raises():
    store = HybridStore()
    store.put("known", _smooth())
    with pytest.raises(RecallMiss):
        store.get("unknown")


def test_empty_store_raises():
    with pytest.raises(RecallMiss):
        HybridStore().get("anything")


def test_multiple_models_distinct():
    store = HybridStore(num_coeffs=4, bits=16)
    store.put("smooth", _smooth(seed=1))
    store.put("other", _smooth(seed=2))
    a = store.get("smooth")
    b = store.get("other")
    assert not np.array_equal(a, b)


def test_noisy_recall_works():
    store = HybridStore(num_coeffs=4, bits=16)
    store.put("alpha", _smooth(seed=3))
    store.put("beta", _smooth(seed=4))
    rebuilt = store.get_noisy("alpha", noise=0.05, seed=7)
    original = _smooth(seed=3)
    rel = np.linalg.norm(original - rebuilt) / np.linalg.norm(original)
    assert rel < 0.05


def test_noisy_rejects_bad_noise():
    store = HybridStore()
    store.put("a", _smooth())
    with pytest.raises(ValueError):
        store.get_noisy("a", noise=1.0)


def test_list_models_sorted():
    store = HybridStore()
    store.put("b", _smooth())
    store.put("a", _smooth())
    assert store.list_models() == ["a", "b"]


def test_len():
    store = HybridStore()
    store.put("a", _smooth())
    store.put("b", _smooth())
    assert len(store) == 2


def test_put_rejects_empty_name():
    with pytest.raises(ValueError):
        HybridStore().put("", _smooth())


def test_put_rejects_1d():
    with pytest.raises(ValueError):
        HybridStore().put("x", np.zeros(10))


def test_bad_constructor_args():
    with pytest.raises(ValueError):
        HybridStore(num_coeffs=0)
    with pytest.raises(ValueError):
        HybridStore(bits=0)
    with pytest.raises(ValueError):
        HybridStore(dim=2)
    with pytest.raises(ValueError):
        HybridStore(threshold=0.0)
