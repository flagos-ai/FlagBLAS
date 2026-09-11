<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

[中文版|[English](./README.md)]

## 介绍

FlagBLAS 是 [FlagOS](https://flagos.io/) 的一部分。
FlagBLAS是一个遵循BLAS标准的接口的面向多种芯片后端的计算库，它定义了向量、矩阵等数值计算的核心操作，支持科学计算、工程仿真、机器学习和人工智能等领域高性能计算。

FlagBLAS 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能通用算子库，

## 特性

- 算子已经过深度性能调优
- Triton kernel 调用优化
- 灵活的多后端支持机制

## 快速安装

### 安装依赖

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### 安装FlagBLAS

```shell
git clone https://github.com/flagos-ai/FlagBLAS.git
cd FlagBLAS
pip install  .
```

在摩尔线程 GPU 上，请先准备厂商提供的 PyTorch/Triton（TorchMUSA）运行时，
再执行以下命令，避免安装过程替换厂商版 PyTorch：

```shell
source tools/set-env.sh mthreads
pip install -e . --no-deps --no-build-isolation
```

摩尔线程 Level 2 正确性测试默认使用同一张 MUSA GPU 上的官方 muBLAS
作为参考后端，也可显式选择 CPU/SciPy 高精度参考：

```shell
pytest -q --ref musa tests/test_gemv.py
pytest -q --ref cpu tests/test_gemv.py
```

本项目采用 [Apache (Version 2.0) License](./LICENSE) 授权许可。
