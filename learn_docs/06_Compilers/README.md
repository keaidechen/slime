# 编译器与执行运行时

说明从程序和计算图到 IR、布局、设备代码与运行时提交的分工。

## 主题地图

| 条目 | 定位 |
|---|---|
| [编译器分层与执行图](编译器分层与执行图.md) | 编译器分层：从模型表达式到设备执行 |
| [PyTorch编译栈](PyTorch编译栈.md) | PyTorch 编译栈：从 Python 模型到高性能 Kernel |
| [IR_PTX_SASS与编译模式](IR_PTX_SASS与编译模式.md) | IR、PTX、SASS 与编译模式 |
| [Triton_TileLang_CUDA-Tile与Helion](Triton_TileLang_CUDA-Tile与Helion.md) | Triton、TileLang、CUDA Tile、CuTe DSL 与 Helion |

## 问题与演进

高层优化获得跨算子信息，低层控制适应硬件；动态形状、编译成本与可移植性决定边界。

历史与方法的跨领域关系见[技术演进索引](../技术演进索引.md)。

## 查阅与关联

[百科总览](../README.md) · [概念关系](../概念关系与正文归属.md) · [术语索引](../术语索引.md) · [问题索引](../问题索引.md)

