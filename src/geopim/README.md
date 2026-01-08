# GeoPIM v3.0: HBM-PIM 几何引导采样加速器

> 基于 HBM 存内计算的 3DGS 前馈生成 Encoder 加速模拟器

## 概述

GeoPIM 是一个用于模拟 HBM-PIM（高带宽内存存内计算）架构的 Python 模块，专门针对 TransPlat/PixelSplat 等 3D Gaussian Splatting 模型中的几何引导采样操作进行加速。

### 设计目标

| 目标 | 指标 | 说明 |
|------|------|------|
| 几何引导采样加速 | **4-8×** (vs GPU) | 保守可信区间 |
| 带宽利用率 | 38% → 70%+ | 利用 row buffer 局部性 |
| 中间数据消除 | ~2GB → ~4MB | 流式聚合消除物化 |

### 设计约束

| 约束 | 规格 | 来源 |
|------|------|------|
| 工艺节点 | 28nm | Samsung HBM-PIM |
| 门级预算 | <10K gates/bank | HBM-PIM 实际约束 |
| 功耗限制 | <1mW/bank | HBM 热设计 |

## 模块结构

```
src/geopim/
├── __init__.py              # 模块入口
├── interface/               # PyTorch 集成接口
│   ├── geopim_sampler.py    # 自定义采样算子
│   └── pixelsplat_adapter.py
├── simulator/               # HBM-PIM 模拟器
│   ├── hbm_model.py         # HBM3 结构模型
│   ├── pim_unit.py          # Per-Bank PIM Unit
│   └── controller.py        # PIM 控制器
├── timing/                  # 时序分析
│   ├── cycle_model.py       # 周期精确模型
│   └── power_model.py       # 功耗模型
├── utils/                   # 工具函数
│   ├── bank_scheduler.py    # Bank 调度器
│   └── fp16_ops.py          # FP16 运算
└── benchmark/               # 性能评估
    └── transplat_bench.py   # TransPlat benchmark
```

## 快速开始

```python
from src.geopim import GeoPIMSimulator, GeoPIMSampler

# 创建模拟器
simulator = GeoPIMSimulator()

# 使用 GeoPIM 采样
output = GeoPIMSampler.apply(
    feature_map,   # [B, C, H, W] FP16
    geo_params,    # [B, Q, param_size]
    weights,       # [B, Q, S] 预计算权重
    offsets        # [B, Q, S, 2] 可选偏移
)

# 获取性能统计
stats = simulator.get_stats()
print(f"Row hit rate: {stats['row_hit_rate']:.1%}")
print(f"Estimated speedup: {stats['speedup']:.2f}x")
```

## 参考文档

- 架构设计: [`engine/GeoPIM_architecture_design.md`](../../engine/GeoPIM_architecture_design.md)
- Samsung HBM-PIM: Hot Chips 2021
- TransPlat: ECCV 2024

