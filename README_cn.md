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




本项目采用 [Apache License (version 2.0)](./LICENSE) 授权许可。
