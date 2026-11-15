# Fourier-HELIX - 2-D Fourier compression fused with wave-lattice recall
# Part of the Kapildhinwa2010-dev Toolkit.
"""HybridDNA - DNA compression plus wave-lattice memory in one store.

Lineage: DNA + HELIX. Weight matrices are encoded into quantized Fourier
genes (DNA) and the genes are stored in a wave lattice, so whole models are
recalled by interference instead of a filename - and stored at a fraction
of their raw size.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["HybridStore", "RecallMiss"]


class RecallMiss(Exception):
    """No stored model resonates strongly enough with the probe."""


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _quantize(values: np.ndarray, bits: int):
    lo = float(values.min())
    hi = float(values.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((values - lo) / scale), 0, levels)
    return codes.astype(np.uint16), lo, scale


def _dequantize(codes: np.ndarray, lo: float, scale: float) -> np.ndarray:
    return codes.astype(np.float64) * scale + lo


class HybridStore:
    """Store DNA-encoded matrices; recall them by key interference."""

    def __init__(self, num_coeffs: int = 8, bits: int = 8, dim: int = 32, threshold: float = 0.5) -> None:
        if num_coeffs < 1:
            raise ValueError("num_coeffs must be >= 1")
        if not 1 <= bits <= 16:
            raise ValueError("bits must be in [1, 16]")
        if dim < 4:
            raise ValueError("dim must be >= 4")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.num_coeffs = num_coeffs
        self.bits = bits
        self.dim = dim
        self.threshold = threshold
        self._names = []
        self._keys = []
        self._genes = []

    def _embed(self, name: str) -> np.ndarray:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vec = rng.standard_normal(self.dim)
        return vec / np.linalg.norm(vec)

    def _encode(self, weights: np.ndarray):
        W = np.asarray(weights, dtype=np.float64)
        if W.ndim != 2 or W.size == 0:
            raise ValueError("weights must be a non-empty 2-D matrix")
        basis = _fourier_basis(W.shape[1], self.num_coeffs)
        coeffs, _, _, _ = np.linalg.lstsq(basis, W.T, rcond=None)
        codes, lo, scale = _quantize(coeffs.T, self.bits)
        original = W.size * 8
        stored = codes.size * 2 + 16
        return {"shape": W.shape, "codes": codes, "lo": lo, "scale": scale}, original / max(stored, 1)

    def _decode(self, genes: dict) -> np.ndarray:
        rows, cols = genes["shape"]
        coeffs = _dequantize(genes["codes"], genes["lo"], genes["scale"])
        basis = _fourier_basis(cols, self.num_coeffs)
        return coeffs @ basis.T

    def put(self, name: str, weights: np.ndarray) -> float:
        """Encode and store a matrix; returns its compression ratio."""
        if not name:
            raise ValueError("name must be non-empty")
        genes, ratio = self._encode(weights)
        self._names.append(name)
        self._keys.append(self._embed(name))
        self._genes.append(genes)
        return ratio

    def get(self, name: str) -> np.ndarray:
        """Recall and decode a stored matrix by exact key."""
        return self._recall(self._embed(name), name)

    def get_noisy(self, name: str, noise: float = 0.05, seed: int = 1) -> np.ndarray:
        """Recall from a perturbed key wave."""
        if not 0.0 <= noise < 1.0:
            raise ValueError("noise must be in [0, 1)")
        rng = np.random.default_rng(seed + len(name))
        probe = self._embed(name) + noise * rng.standard_normal(self.dim)
        probe = probe / np.linalg.norm(probe)
        return self._recall(probe, name)

    def _recall(self, probe: np.ndarray, name: str) -> np.ndarray:
        if not self._keys:
            raise RecallMiss("store is empty")
        sims = np.stack(self._keys) @ probe
        best = int(np.argmax(sims))
        if float(sims[best]) < self.threshold:
            raise RecallMiss("no model resonates above threshold")
        return self._decode(self._genes[best])

    def list_models(self) -> list:
        return sorted(self._names)

    def __len__(self) -> int:
        return len(self._names)

__all__.append("FourierHybridStore")


def _fft2_compress(W: np.ndarray, keep: float, bits: int):
    """Keep a centered low-frequency window of the 2-D FFT, quantized."""
    F = np.fft.fftshift(np.fft.fft2(W))
    rows, cols = F.shape
    kr = max(1, int(round(rows * keep)))
    kc = max(1, int(round(cols * keep)))
    r0 = (rows - kr) // 2
    c0 = (cols - kc) // 2
    window = F[r0:r0 + kr, c0:c0 + kc]
    re_codes, re_lo, re_scale = _quantize(window.real, bits)
    im_codes, im_lo, im_scale = _quantize(window.imag, bits)
    genes = {
        "shape": (rows, cols), "kr": kr, "kc": kc, "r0": r0, "c0": c0,
        "re_codes": re_codes, "re_lo": re_lo, "re_scale": re_scale,
        "im_codes": im_codes, "im_lo": im_lo, "im_scale": im_scale,
    }
    original = W.size * 8
    stored = re_codes.size * 2 + im_codes.size * 2 + 40
    return genes, original / max(stored, 1)


def _fft2_decompress(genes: dict) -> np.ndarray:
    rows, cols = genes["shape"]
    F = np.zeros((rows, cols), dtype=np.complex128)
    re = _dequantize(genes["re_codes"], genes["re_lo"], genes["re_scale"])
    im = _dequantize(genes["im_codes"], genes["im_lo"], genes["im_scale"])
    kr, kc = genes["kr"], genes["kc"]
    r0, c0 = genes["r0"], genes["c0"]
    F[r0:r0 + kr, c0:c0 + kc] = re + 1j * im
    return np.fft.ifft2(np.fft.ifftshift(F)).real


class FourierHybridStore(HybridStore):
    """HybridStore variant that compresses whole 2-D spectra.

    Instead of fitting a 1-D Fourier basis per column, this store keeps a
    quantized low-frequency window of the full 2-D FFT. On smooth matrices
    (low-rank or slowly varying) it reconstructs far better than the 1-D
    fit at the same storage; on noise-like matrices it is worse. Choose
    per matrix, not by default.
    """

    def __init__(self, keep: float = 0.25, storage_bits: int = 12,
                 num_coeffs: int = 8, bits: int = 8, dim: int = 32,
                 threshold: float = 0.5) -> None:
        super().__init__(num_coeffs=num_coeffs, bits=bits, dim=dim, threshold=threshold)
        if not 0.0 < keep <= 1.0:
            raise ValueError("keep must be in (0, 1]")
        if not 1 <= storage_bits <= 16:
            raise ValueError("storage_bits must be in [1, 16]")
        self.keep = keep
        self.storage_bits = storage_bits

    def _encode(self, weights: np.ndarray):
        W = np.asarray(weights, dtype=np.float64)
        if W.ndim != 2 or W.size == 0:
            raise ValueError("weights must be a non-empty 2-D matrix")
        return _fft2_compress(W, self.keep, self.storage_bits)

    def _decode(self, genes: dict) -> np.ndarray:
        return _fft2_decompress(genes)

