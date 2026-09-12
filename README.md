[English|[中文版](./README_cn.md)]

## Introduction

FlagBLAS is part of [FlagOS](https://flagos.io/).
FlagBLAS is a computing library that follows the BLAS standard interface and is oriented towards multiple chip backends. It defines core operations for numerical calculations such as vectors and matrices, supporting high-performance computing in fields such as scientific computing, engineering simulation, machine learning, and artificial intelligence.

FlagBLAS is a high-performance general-purpose operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism

## Quick Installation

### Install Dependencies

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### Install FlagBLAS

```shell
git clone https://github.com/flagos-ai/FlagBLAS.git
cd FlagBLAS
pip install  .
```

For Moore Threads GPUs, prepare the vendor PyTorch/Triton (TorchMUSA) stack
first. Then install without dependency resolution so the vendor runtime is not
replaced by a generic CUDA build:

```shell
source tools/set-env.sh mthreads
pip install -e . --no-deps --no-build-isolation
```

Moore Threads Level 2 correctness tests use the vendor muBLAS library on the
same MUSA GPU by default. A high-precision CPU/SciPy reference remains
available explicitly:

```shell
pytest -q --ref musa tests/test_gemv.py
pytest -q --ref cpu tests/test_gemv.py
```

This project is licensed under the [Apache (Version 2.0) License](./LICENSE).
