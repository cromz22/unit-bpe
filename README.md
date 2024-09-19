# unit-bpe

![CI](https://github.com/cromz22/unit-bpe/actions/workflows/CI.yml/badge.svg)

BPE tokenizer that operates on integer sequences.
The implementation is in Rust and Python bindings are provided utilizing [pyo3](https://github.com/PyO3/pyo3) and [Maturin](https://github.com/PyO3/maturin).

## Installation

```
pip install unit-bpe
```

## Example usage from Python

```
from unit_bpe import fit_concurrent_py, encode_concurrent_py, decode_concurrent_py

units_list = [
    [0, 1, 0, 1, 2, 0, 1, 2, 3],
    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
]
vocab_size = 10
# Since there are 6 units in the training data, 10 - 6 = 4 merge operations are performed

encoded_units, merges = fit_concurrent_py(units_list, vocab_size)
print(encoded_units)  # [[6, 7, 8], [9, 9, 5]]
print(merges)  # [((0, 1), 6), ((8, 4), 9), ((7, 3), 8), ((6, 2), 7)]

units_list_to_encode = [[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]]
encoded = encode_concurrent_py(units_list_to_encode, merges)
print(encoded)  # [[6, 9, 5], [7, 8]]

decoded = decode_concurrent_py(encoded, merges)
print(decoded)  # [[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]]
```

## Development Guide

### Installation

- Rust environment
    - See https://www.rust-lang.org/learn/get-started if you don't have Rust development environment
    - Run `cargo install` to install dependencies

- Python environment
    - [uv](https://docs.astral.sh/uv/) is used as the package manager
    - Run `uv sync` to install dependencies

### Running tests

- Rust

    ```
    cargo test --lib
    ```

- Python

    ```
    uv run pytest
    ```

### Directory structure

```
unit-bpe
├── src
│   ├── lib.rs                # Rust library entry point
│   ├── core.rs               # Core logic of BPE
│   ├── concurrent.rs         # Extension of core.rs for concurrent processing
│   ├── python_bindings.rs    # Bindings to expose Rust functions to Python
│   └── test.rs               # Rust unit tests
├── tests
│   └── test_unit_bpe.py      # Python unit tests
├── .gitignore
├── Cargo.toml                # Rust dependency definitions
├── Cargo.lock                # Rust dependency lock file
├── README.md
├── pyproject.toml            # Python dependency definitions
└── uv.lock                   # Python dependency lock file

```
