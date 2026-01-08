# GeoPIM 架构设计文档

> **版本**: 3.0 (可实现性收敛版)  
> **日期**: 2026-01-08  
> **状态**: 设计阶段  
> **目标**: 面向3DGS前馈生成Encoder的存内计算加速器

---

## 1. 设计目标与约束

### 1.1 设计目标

| 目标 | 指标 | 说明 |
|------|------|------|
| 几何引导采样加速 | **4-8×**（vs GPU原始） | 保守可信区间 |
| 带宽利用率 | 38% → 70%+ | 利用 row buffer 局部性 |
| 中间数据消除 | ~2GB → ~4MB | 流式聚合消除物化 |
| 端到端Encoder加速 | 1.2-1.4× | 几何引导采样占比 22-25% |

### 1.2 设计约束（严格遵守）

| 约束 | 规格 | 来源 | 说明 |
|------|------|------|------|
| 工艺节点 | 28nm | Samsung HBM-PIM | 匹配 HBM 逻辑层工艺 |
| **门级预算** | **<10K gates/bank** | HBM-PIM 实际约束 | 最关键约束 |
| **功耗限制** | **<1mW/bank** | HBM 热设计 | 更保守估计 |
| 接口兼容 | 标准 HBM3 接口 | 系统集成 | 不修改 PHY |

### 1.3 核心设计原则

**PIM bank 只做"轻计算 + 大带宽"擅长的事**

1. **统一计算路径**：所有方法（包括 PixelSplat）都走"预计算权重"路径
2. **权重在 GPU 算**：Q·K + softmax 由 GPU 完成，PIM 只做加权累加
3. **时分复用**：少量 MAC 单元处理高维向量
4. **利用 DRAM 局部性**：Row buffer aware 而非 per-bank cache

### 1.4 工作负载特征

```
GeoPIM 统一工作负载模型：

输入：
├── Feature map     : [B, C, H, W], C=128, FP16
├── 几何参数        : [B, Q, param_size], ~4KB total
├── 预计算权重      : [B, Q, S], S=采样点数, FP16
└── 偏移量（可选）  : [B, Q, S, 2], FP16

输出：
└── 聚合特征        : [B, Q, C], ~4MB total

典型配置：
├── TransPlat : B=2, Q=1024 (32×32), D=128, P=4, S=512
├── PixelSplat: B=2, Q=4096 (64×64), S=32
└── 采样点总数: 1M ~ 4M
```

---

## 2. 系统架构

### 2.1 总体架构

```
┌────────────────────────────────────────────────────────────────────────┐
│                    GeoPIM System Architecture (v3.0)                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌──────────────────────┐         ┌─────────────────────────────┐    │
│   │      Host GPU        │         │        HBM-PIM Stack        │    │
│   │                      │         │                             │    │
│   │  ┌────────────────┐  │         │  ┌─────────────────────┐   │    │
│   │  │   Backbone     │  │         │  │    PIM Controller    │   │    │
│   │  │  (ViT/CNN)     │  │         │  │  ┌───────────────┐  │   │    │
│   │  └───────┬────────┘  │         │  │  │ Cmd Decoder   │  │   │    │
│   │          │           │         │  │  │ Bank Scheduler│  │   │    │
│   │  ┌───────▼────────┐  │         │  │  └───────────────┘  │   │    │
│   │  │ Geometry Param │  │  Cmd    │  └──────────┬──────────┘   │    │
│   │  │   Generator    │──┼────────►│             │              │    │
│   │  └───────┬────────┘  │         │  ┌──────────▼──────────┐   │    │
│   │          │           │         │  │   Per-Bank PIM Unit  │   │    │
│   │  ┌───────▼────────┐  │         │  │  (极简设计 <10K gates)│   │    │
│   │  │Weight Compute  │  │  Wgt    │  │  ┌───────────────┐  │   │    │
│   │  │ (Q·K→softmax)  │──┼────────►│  │  │ Addr Generator │  │   │    │
│   │  │  ↑PixelSplat   │  │         │  │  │ (几何参数驱动) │  │   │    │
│   │  └───────┬────────┘  │         │  │  └───────┬───────┘  │   │    │
│   │          │           │         │  │          │          │   │    │
│   │  ┌───────▼────────┐  │         │  │  ┌───────▼───────┐  │   │    │
│   │  │  Post Process  │◄─┼─────────┤  │  │ 8-lane FP16   │  │   │    │
│   │  │ (UNet, Decode) │  │  Result │  │  │ MAC + Accum   │  │   │    │
│   │  └────────────────┘  │         │  │  └───────────────┘  │   │    │
│   │                      │         │  └─────────────────────┘   │    │
│   └──────────────────────┘         └─────────────────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Host-PIM 工作负载划分

| 阶段 | 执行位置 | 说明 |
|------|----------|------|
| Backbone 特征提取 | Host GPU | 计算密集 |
| 几何参数计算 | Host GPU | 数据量小 |
| **权重计算（含 PixelSplat Q·K→softmax）** | **Host GPU** | **统一在 GPU 侧** |
| **几何引导采样** | **HBM-PIM** | 内存受限 |
| **加权累加** | **HBM-PIM** | 流式消费 |
| 后续处理 | Host GPU | 计算密集 |

### 2.3 数据流时序

```
时间轴 →
────────────────────────────────────────────────────────────────────────

Host GPU:
  [Backbone]──►[几何参数]──►[权重计算]──────────────────►[后处理]
                  │         (含PixelSplat               ▲
                  │          Q·K→softmax)               │
                  ▼              │                      │
              传输参数       传输权重                传输结果
               (4KB)        (4-16MB)                (4MB)
                  │              │                      │
────────────────────┼──────────────┼───────────────────────┼────────
                    ▼              ▼                       │
HBM-PIM:
              [参数加载]──►[采样]──►[加权累加]────────────►[输出]
                              ▲
                         统一路径：预计算权重 + MAC
```

---

## 3. Per-Bank PIM Unit 硬件设计

### 3.1 整体架构（满足 <10K gates）

```
┌─────────────────────────────────────────────────────────────────────┐
│            Per-Bank PIM Unit (GeoPIM Core, <10K gates)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                  Geometry Address Generator                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐ │ │
│  │  │  Parameter  │  │   Coord     │  │   Address             │ │ │
│  │  │  Registers  │─►│  Calculator │─►│   Translator          │ │ │
│  │  │  (32B×4)    │  │  (FP16 ALU) │  │   (HBM addr gen)      │ │ │
│  │  └─────────────┘  └─────────────┘  └───────────┬───────────┘ │ │
│  │                                                │              │ │
│  │  门级估算: ~2K gates                                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     Tile Buffer (256B)                         │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │  Ping Buffer (128B): 4 neighbors × 8 channels × FP16    │ │ │
│  │  │  Pong Buffer (128B): 4 neighbors × 8 channels × FP16    │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  门级估算: ~1K gates (寄存器 + 控制)                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              Compute Unit (8-lane FP16 MAC)                    │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │                                                         │ │ │
│  │  │   ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐                 │ │ │
│  │  │   │MAC_0│ │MAC_1│ │MAC_2│ ... │MAC_7│  (8× FP16 FMA)  │ │ │
│  │  │   └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘                 │ │ │
│  │  │      └───────┴───────┴───────────┘                     │ │ │
│  │  │                      │                                 │ │ │
│  │  │               Bilinear Interpolation                   │ │ │
│  │  │               + Weighted Accumulation                  │ │ │
│  │  │                      │                                 │ │ │
│  │  │               ┌──────▼──────┐                          │ │ │
│  │  │               │ Accumulator │  (16 × FP32)             │ │ │
│  │  │               │ Registers   │  (C/tile_C = 128/8)      │ │ │
│  │  │               └─────────────┘                          │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  门级估算: ~5K gates (8×FP16 MAC + 累加器 + 控制)            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Control Logic (~1.5K gates)                 │ │
│  │  - State machine (IDLE → LOAD → COMPUTE → OUTPUT)             │ │
│  │  - Tile counter (C 维度分块)                                   │ │
│  │  - Sample counter                                              │ │
│  │  - DRAM request sequencer                                      │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  总门级估算: 2K + 1K + 5K + 1.5K ≈ 9.5K gates  ✓ <10K             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Geometry Address Generator

#### 3.2.1 参数寄存器（128B total）

```
┌─────────────────────────────────────────────────────────────────┐
│              Parameter Registers (4 entries × 32B)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Entry Format (32 Bytes):                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ query_id[16b] │ num_samples[16b] │ current_sample[16b]  │   │
│  │ ref_x[16b]    │ ref_y[16b]       │ stride_x[16b]        │   │
│  │ stride_y[16b] │ weight_base[48b] │ offset_base[48b]     │   │
│  │ valid[1b]     │ has_offset[1b]   │ reserved[46b]        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  坐标计算公式:                                                  │
│    (x, y) = (ref_x, ref_y) + sample_idx × (stride_x, stride_y)│
│           + offset[sample_idx]  (if has_offset)                │
│                                                                 │
│  容量: 4 entries，支持 4 个 query 的参数预加载                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 坐标计算单元

```
输入: ref_x, ref_y, stride_x, stride_y, sample_idx, offset (optional)
输出: (x, y) 浮点坐标

硬件: 2× FP16 乘法器 + 2× FP16 加法器
延迟: 2 cycles

// 伪代码
x = ref_x + sample_idx * stride_x + (has_offset ? offset_x : 0)
y = ref_y + sample_idx * stride_y + (has_offset ? offset_y : 0)
```

#### 3.2.3 地址转换单元

```
输入: (x, y) 浮点坐标, base_addr, W, C
输出: 4 个邻域的 HBM 地址 (48-bit each)

计算:
  x0 = floor(x), y0 = floor(y)
  x1 = x0 + 1,   y1 = y0 + 1
  
  addr[0] = base + (y0 * W + x0) * C * 2  // (x0, y0)
  addr[1] = base + (y0 * W + x1) * C * 2  // (x1, y0)
  addr[2] = base + (y1 * W + x0) * C * 2  // (x0, y1)
  addr[3] = base + (y1 * W + x1) * C * 2  // (x1, y1)

同时计算双线性插值权重:
  wx = x - x0, wy = y - y0
  w[0] = (1-wx)*(1-wy), w[1] = wx*(1-wy)
  w[2] = (1-wx)*wy,     w[3] = wx*wy
```

### 3.3 Tile Buffer (Ping-Pong)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tile Buffer Design (256B)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  tile_C = 8 channels (可配置为 8/16)                            │
│  tiles_per_sample = C / tile_C = 128 / 8 = 16                   │
│                                                                 │
│  Ping Buffer (128B):                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ neighbor[0][0:7] │ neighbor[1][0:7] │ ... │ neighbor[3] │   │
│  │     (16B)        │     (16B)        │     │   (16B)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│  = 4 neighbors × 8 channels × 2B(FP16) = 64B per tile          │
│  × 2 tiles (ping-pong) = 128B                                   │
│                                                                 │
│  Pong Buffer (128B): 同上                                       │
│                                                                 │
│  操作模式:                                                      │
│  - 当 Compute 处理 Ping 中的 tile_i 时                          │
│  - DRAM 预取 tile_{i+1} 到 Pong                                │
│  - 隐藏 DRAM 访问延迟                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Compute Unit (8-lane FP16 MAC)

```
┌─────────────────────────────────────────────────────────────────┐
│               Compute Unit: 8-lane FP16 MAC                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入:                                                          │
│  - 4 neighbors × 8 channels (from Tile Buffer)                  │
│  - 4 bilinear weights (from Addr Generator)                     │
│  - 1 attention weight (from Weight Buffer)                      │
│                                                                 │
│  计算流程 (per cycle):                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 1: Bilinear Interpolation (8 channels parallel)    │   │
│  │   sample[c] = w0*f0[c] + w1*f1[c] + w2*f2[c] + w3*f3[c] │   │
│  │   使用 8-lane MAC: 4 cycles (4 neighbors sequential)    │   │
│  │                                                         │   │
│  │ Step 2: Weighted Accumulation                           │   │
│  │   acc[c] += attn_weight * sample[c]                     │   │
│  │   使用 8-lane MAC: 1 cycle                              │   │
│  │                                                         │   │
│  │ 每 tile 计算: 5 cycles                                  │   │
│  │ 每 sample 计算: 16 tiles × 5 = 80 cycles                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  硬件:                                                          │
│  - 8× FP16 FMA units                                            │
│  - 16× FP32 Accumulator registers (for C=128)                   │
│  - Bilinear weight registers (4× FP16)                          │
│  - Attention weight register (1× FP16)                          │
│                                                                 │
│  门级估算:                                                      │
│  - 8× FP16 FMA: ~3K gates (简化实现)                           │
│  - 16× FP32 Acc: ~1K gates                                     │
│  - 控制逻辑: ~1K gates                                         │
│  - 总计: ~5K gates                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```


---

## 4. 数据流与时序模型

### 4.1 C-Tiling 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                    C-Tiling Data Flow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  参数:                                                          │
│  - C = 128 channels                                             │
│  - tile_C = 8 channels                                          │
│  - tiles_per_sample = 16                                        │
│  - 4 neighbors per sample                                       │
│                                                                 │
│  每 tile 数据量:                                                │
│  - 4 neighbors × 8 channels × 2B = 64B = 1 burst (HBM3)        │
│                                                                 │
│  每 sample 数据量:                                              │
│  - 16 tiles × 64B = 1024B                                       │
│  - 对应 16 次 64B burst                                         │
│                                                                 │
│  时序 (每 tile):                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase        │ Cycles │ 说明                             │  │
│  ├──────────────┼────────┼──────────────────────────────────┤  │
│  │ DRAM Fetch   │ 4-20   │ 64B burst (row hit: 4, miss: 20) │  │
│  │ Bilinear     │ 4      │ 4 neighbors × 1 cycle each       │  │
│  │ Accumulate   │ 1      │ weighted add                     │  │
│  │ Total/tile   │ 9-25   │ 取决于 row buffer hit rate       │  │
│  └──────────────┴────────┴──────────────────────────────────┘  │
│                                                                 │
│  Ping-Pong 隐藏延迟后:                                          │
│  - 理想 (100% row hit): ~5 cycles/tile → 80 cycles/sample      │
│  - 典型 (70% row hit): ~10 cycles/tile → 160 cycles/sample     │
│  - 最差 (30% row hit): ~18 cycles/tile → 288 cycles/sample     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 完整执行流程

### 5.1 伪代码

```python
def geopim_execute_v3(feature_map, geo_params, weights, offsets=None):
    """
    GeoPIM v3.0: 统一预计算权重路径
    
    Args:
        feature_map: [B, C, H, W] - FP16, 存储在 HBM
        geo_params: [B, Q, param_size] - 几何参数
        weights: [B, Q, S] - 预计算的注意力权重 (GPU 已算好)
        offsets: [B, Q, S, 2] - 采样偏移 (optional)
    
    Returns:
        output: [B, Q, C] - FP16, 聚合结果
    """
    B, Q, S = weights.shape
    C = 128
    tile_C = 8
    num_tiles = C // tile_C  # 16
    
    # Step 1: GPU 准备数据并传输到 HBM
    # (feature_map 已在 HBM，只需传输 geo_params 和 weights)
    pim_controller.load_params(geo_params)      # ~4KB
    pim_controller.load_weights(weights)        # ~4-16MB
    if offsets is not None:
        pim_controller.load_offsets(offsets)    # ~8-32MB
    
    # Step 2: 分配 queries 到 banks
    bank_assignments = distribute_queries_to_banks(B, Q, num_banks=512)
    
    # Step 3: 每个 bank 并行执行
    for bank_id in parallel(range(512)):
        queries = bank_assignments[bank_id]
        
        for q in queries:
            # 初始化累加器
            accumulator = zeros(C, dtype=FP32)
            
            for s in range(S):
                # === Geometry Address Generation ===
                coord = compute_coord(geo_params[q], s, offsets)
                addrs, bilinear_weights = generate_addrs(coord)
                attn_weight = weights[q, s]
                
                # === C-Tiling Loop ===
                for tile in range(num_tiles):
                    tile_start = tile * tile_C
                    tile_end = tile_start + tile_C
                    
                    # Fetch tile data (64B per tile)
                    # 使用 ping-pong buffer 隐藏延迟
                    neighbors = fetch_tile(addrs, tile_start, tile_end)
                    
                    # Bilinear interpolation (8 channels)
                    sample_tile = bilinear_interp(neighbors, bilinear_weights)
                    
                    # Weighted accumulation
                    accumulator[tile_start:tile_end] += attn_weight * sample_tile
            
            # 输出当前 query 的结果
            output[q] = accumulator.to(FP16)
    
    # Step 4: 传输结果回 GPU
    return output  # ~4MB
```

### 5.2 状态机

```
┌─────────────────────────────────────────────────────────────────┐
│               PIM Bank State Machine (Simplified)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              ┌─────────┐                                        │
│              │  IDLE   │                                        │
│              └────┬────┘                                        │
│                   │ start_cmd                                   │
│                   ▼                                             │
│          ┌─────────────────┐                                    │
│          │  LOAD_PARAMS    │  ← 加载几何参数 + 权重基址        │
│          └────────┬────────┘                                    │
│                   │                                             │
│                   ▼                                             │
│  ┌───────────────────────────────────────────────┐             │
│  │              PROCESS_QUERY                     │             │
│  │  ┌─────────────────────────────────────────┐  │             │
│  │  │           PROCESS_SAMPLE                 │  │             │
│  │  │  ┌───────────────────────────────────┐  │  │             │
│  │  │  │         PROCESS_TILE              │  │  │             │
│  │  │  │  ┌─────────────────────────────┐  │  │  │             │
│  │  │  │  │ FETCH → COMPUTE → ACCUM    │  │  │  │             │
│  │  │  │  └─────────────────────────────┘  │  │  │             │
│  │  │  │  loop: tile < 16                  │  │  │             │
│  │  │  └───────────────────────────────────┘  │  │             │
│  │  │  loop: sample < S                       │  │             │
│  │  └─────────────────────────────────────────┘  │             │
│  │  loop: query in assigned_queries              │             │
│  └───────────────────────────────────────────────┘             │
│                   │                                             │
│                   ▼                                             │
│          ┌─────────────────┐                                    │
│          │  OUTPUT_RESULT  │  ← 写回聚合结果                   │
│          └────────┬────────┘                                    │
│                   │                                             │
│                   ▼                                             │
│              ┌─────────┐                                        │
│              │  DONE   │──────► IDLE                            │
│              └─────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Row Buffer Aware 调度

### 6.1 几何感知 Bank 分配

```
┌─────────────────────────────────────────────────────────────────┐
│              Geometry-Aware Bank Assignment                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  目标: 最大化 row buffer hit rate                               │
│                                                                 │
│  策略:                                                          │
│  1. 将特征图按空间位置划分到不同 bank                           │
│  2. 将采样区域相近的 queries 分配到同一 bank                    │
│  3. 同一 bank 内的 queries 按采样区域排序                       │
│                                                                 │
│  特征图 Bank 映射 (假设 32×32 feature map, 512 banks):          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  特征图被划分为 16×16 = 256 个 tile                     │    │
│  │  每个 tile (2×2 pixels) 映射到 2 个 bank               │    │
│  │  相邻 tile 映射到相邻 bank group                        │    │
│  │                                                        │    │
│  │  bank_id = hash(tile_x, tile_y) % 512                  │    │
│  │                                                        │    │
│  │  这样设计使得:                                         │    │
│  │  - 同一 query 的采样点可能访问相邻 tiles               │    │
│  │  - 相邻 tiles 在相邻 banks，减少 bank conflict         │    │
│  │  - 单个 bank 内访问有空间局部性，提高 row hit rate     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Query 分配:                                                    │
│  - 计算每个 query 的采样区域中心                               │
│  - 将采样区域中心映射到 bank                                   │
│  - 同一 bank 内的 queries 按中心坐标排序                       │
│                                                                 │
│  预期 row hit rate: 60-70% (vs 随机访问的 ~30%)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Per-Vault Scratchpad (可选优化)

```
┌─────────────────────────────────────────────────────────────────┐
│              Per-Vault Scratchpad (Optional)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HBM3 结构: 6 stacks × 8 channels × 32 banks/channel            │
│  Vault = 1 stack × 1 channel = 32 banks                         │
│  总共 48 vaults                                                 │
│                                                                 │
│  Per-Vault Scratchpad:                                          │
│  - 容量: 2-4KB per vault                                        │
│  - 功能: 缓存跨 bank 共享的 feature tiles                       │
│  - 位置: Vault 级别的共享 SRAM                                  │
│                                                                 │
│  使用场景:                                                      │
│  - 当多个 banks 需要同一 feature tile                          │
│  - 第一个 bank 加载后写入 scratchpad                            │
│  - 后续 banks 从 scratchpad 读取                                │
│                                                                 │
│  收益:                                                          │
│  - 减少重复 DRAM 访问                                           │
│  - 可选实现，不影响基础功能                                     │
│                                                                 │
│  门级开销:                                                      │
│  - 不计入 per-bank 预算 (vault 级别)                            │
│  - 约 10-20K gates per vault (可接受)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 接口设计

### 7.1 GPU-PIM 命令接口

```c
// GeoPIM v3.0 命令格式
typedef struct {
    uint8_t  cmd_type;        // 0: LOAD, 1: EXECUTE, 2: READ
    uint8_t  reserved;
    uint16_t num_queries;     // Query 数量
    uint32_t num_samples;     // 每个 query 的采样点数
    uint64_t param_addr;      // 参数在 HBM 中的地址 (48-bit)
    uint64_t weight_addr;     // 权重在 HBM 中的地址 (48-bit)
    uint64_t offset_addr;     // 偏移在 HBM 中的地址 (48-bit, 可选)
    uint64_t output_addr;     // 输出在 HBM 中的地址 (48-bit)
    uint32_t flags;           // bit0: has_offset
} GeoPIMCommand;

// GPU 端驱动接口
void geopim_load_params(void* params, size_t size);
void geopim_load_weights(void* weights, size_t size);
void geopim_load_offsets(void* offsets, size_t size);  // optional
void geopim_execute(int num_queries, int num_samples, int flags);
void geopim_read_output(void* output, size_t size);
```

### 7.2 PyTorch 集成

```python
class GeoPIMSampler(torch.autograd.Function):
    """
    GeoPIM v3.0: 统一预计算权重路径
    
    对于所有方法（包括 PixelSplat），权重都由 GPU 预计算
    """
    
    @staticmethod
    def forward(ctx, feature_map, geo_params, weights, offsets=None):
        """
        Args:
            feature_map: [B, C, H, W] - FP16
            geo_params: [B, Q, param_size] - 几何参数
            weights: [B, Q, S] - 预计算权重 (GPU 已算好 Q·K→softmax)
            offsets: [B, Q, S, 2] - 采样偏移 (optional)
        
        Returns:
            output: [B, Q, C] - FP16
        """
        output = torch.ops.geopim.fused_sample_aggregate(
            feature_map.contiguous(),
            geo_params.contiguous(),
            weights.contiguous(),
            offsets.contiguous() if offsets is not None else None
        )
        
        ctx.save_for_backward(feature_map, geo_params, weights, offsets)
        return output


def pixelsplat_with_geopim(query, key_features, geo_params, feature_map):
    """
    PixelSplat 使用 GeoPIM 的完整流程
    
    1. GPU: 采样 key features (用于计算 Q·K)
    2. GPU: 计算 Q·K → softmax → weights
    3. PIM: 使用预计算 weights 做加权采样
    """
    # Step 1: GPU 采样 key features (轻量级，只取投影后的低维特征)
    # 这里可以用一个简化的采样获取 key
    key_samples = sample_keys_gpu(key_features, geo_params)  # [B, Q, S, D_key]
    
    # Step 2: GPU 计算 attention weights
    scores = torch.einsum('bqc,bqsc->bqs', query, key_samples)  # [B, Q, S]
    weights = F.softmax(scores, dim=-1)  # [B, Q, S]
    
    # Step 3: PIM 执行加权采样 (这是重的操作)
    output = GeoPIMSampler.apply(feature_map, geo_params, weights)
    
    return output


def transplat_with_geopim(query, geo_params, feature_map, offsets):
    """
    TransPlat 使用 GeoPIM 的完整流程
    
    权重直接由 query 通过 MLP 预测，不依赖采样值
    """
    # Step 1: GPU 预测权重和偏移
    weights = weight_predictor(query)   # [B, Q, D*P]
    offsets = offset_predictor(query)   # [B, Q, D*P, 2]
    
    # Step 2: PIM 执行加权采样
    output = GeoPIMSampler.apply(feature_map, geo_params, weights, offsets)
    
    return output
```

---
