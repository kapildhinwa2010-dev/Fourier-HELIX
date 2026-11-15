# Fourier-HELIX

[![CI](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/codeql.yml)

Wave-lattice model store with a 2-D Fourier (FFT) compression variant for smooth matrices.

## What it does

The base HybridStore encodes weight matrices with a 1-D Fourier basis per
column and recalls them by key interference. FourierHybridStore instead
keeps a quantized low-frequency window of the full 2-D FFT of the matrix,
which reconstructs smooth (low-rank, slowly varying) matrices much better
at the same storage.

    from fourier_helix import FourierHybridStore

    store = FourierHybridStore(keep=0.25, storage_bits=12)
    ratio = store.put("layer0", W)
    W2 = store.get("layer0")

## API

- HybridStore - the base 1-D per-column Fourier codec (unchanged from DNA-HELIX)
- FourierHybridStore(keep, storage_bits) - 2-D FFT codec, same put/get/get_noisy API
- _fft2_compress / _fft2_decompress - the codec primitives

## Honest results and limitations

- 2-D FFT compression wins on smooth matrices and loses on noise-like ones:
  for a random matrix the kept window is most of the spectrum and the ratio
  approaches 1 with visible quantization error. Pick the codec per matrix.
- Quantization is uniform per window; outliers dominate the error.
- Recall is by cosine similarity of hash-derived keys: similar names can
  collide, and the threshold trades misses against wrong recalls.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
