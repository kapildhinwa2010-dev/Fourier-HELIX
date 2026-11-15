# Fourier-HELIX

[![CI](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Fourier-HELIX/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

DNA compression plus wave-lattice memory in one store: matrices are encoded
into quantized Fourier genes and recalled by key interference. DNA + HELIX
lineage of the
[Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).

## Features

- `put(name, W)` - encode into quantized genes, store as a wave
- `get(name)` - interference recall + gene decoding
- `get_noisy(name, noise)` - recall from perturbed keys
- Compression ratio returned on every write
- `RecallMiss` instead of a wrong answer

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```python
import numpy as np
from fourier_helix import HybridStore

store = HybridStore(num_coeffs=8, bits=8)
ratio = store.put("my-model", np.random.default_rng(0).standard_normal((32, 128)))
W = store.get("my-model")
print("ratio:", round(ratio, 2), "| models:", store.list_models())
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
