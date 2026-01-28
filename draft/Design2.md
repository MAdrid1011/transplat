# SGM：流式高斯合并设计文档

## 1. 设计目标

设计一个流式高斯合并单元（Streaming Gaussian Merging, SGM），满足以下要求：

1. **数据流友好**：利用光栅扫描顺序，仅需片上行缓存，无需全局搜索
2. **硬件可实现**：固定位置邻域比较，简单的差异计算与阈值判断
3. **自适应合并**：根据局部离散度执行分类化合并策略
4. **写入拦截**：在 Encoder 到 HBM 的数据通路中完成合并，消除冗余往返

## 2. 高斯基元数据结构

每个高斯基元包含以下属性：

| 属性 | 维度 | 位宽（FP16） | 说明 |
|-----|------|-------------|------|
| `mean` | 3 | 6 B | 3D 位置 $(x, y, z)$ |
| `cov` | 6 | 12 B | 协方差矩阵（上三角） |
| `opacity` | 1 | 2 B | 不透明度 $\alpha$ |
| `sh` | 48 | 96 B | 球谐系数（3阶，16×3通道） |
| **总计** | 58 | **116 B** | 每高斯约 120 字节（含对齐） |

## 3. 数据流模型

### 3.1 光栅扫描顺序

高斯按像素的光栅扫描顺序（row-major）逐个生成：

```
像素顺序：(0,0) → (1,0) → ... → (W-1,0) → (0,1) → (1,1) → ...

高斯流：G[0,0] → G[1,0] → ... → G[W-1,0] → G[0,1] → G[1,1] → ...
```

**关键性质**：当处理像素 $(u, v)$ 时，其邻居 $(u-1, v)$、$(u, v-1)$、$(u-1, v-1)$ 已经被处理过。

### 3.2 固定邻域定义

对于当前高斯 $G_{cur}$ 位于像素 $(u, v)$，其邻域为：

```
        (u-1, v-1)  (u, v-1)   [已处理，在行缓存中]
        (u-1, v)    (u, v)     [已处理]  [当前]
```

邻居列表（按优先级）：
- **左邻居** $G_L$：像素 $(u-1, v)$，刚刚处理
- **上邻居** $G_U$：像素 $(u, v-1)$，在行缓存中
- **左上邻居** $G_{LU}$：像素 $(u-1, v-1)$，在行缓存中

## 4. 行缓存设计

### 4.1 缓存结构

```
┌─────────────────────────────────────────────────────────┐
│                    Line Buffer (W entries)               │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────────┤
│ G_0 │ G_1 │ G_2 │ ... │G_u-1│ G_u │G_u+1│ ... │ G_{W-1} │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────────┘
                          ↑     ↑
                      上一行对应位置   当前写入位置
```

### 4.2 存储需求

| 配置 | 行宽 W | 缓存大小 | 说明 |
|-----|--------|---------|------|
| TransPlat | 256 | 256 × 120 B = **30 KB** | 双视图 |
| MVSplat | 128 | 128 × 120 B = **15 KB** | 单视图特征图 |
| DepthSplat | 256 | 256 × 120 B = **30 KB** | 高分辨率 |

**优化**：可只缓存合并所需的关键属性（mean, cov, opacity），减少至 **~6 KB**。

### 4.3 缓存操作

```python
class LineBuffer:
    def __init__(self, width: int):
        self.buffer = [None] * width  # 上一行的高斯
        self.prev_gaussian = None      # 左邻居（刚处理的）
        self.width = width
    
    def get_neighbors(self, u: int) -> dict:
        """获取当前像素的邻居高斯"""
        return {
            'left': self.prev_gaussian,           # G_L
            'up': self.buffer[u],                 # G_U
            'up_left': self.buffer[u-1] if u > 0 else None  # G_LU
        }
    
    def update(self, u: int, gaussian):
        """更新缓存"""
        self.buffer[u] = gaussian  # 写入当前位置供下一行使用
        self.prev_gaussian = gaussian  # 更新左邻居
```

## 5. 局部离散度计算

### 5.1 差异度量

对于当前高斯 $G_{cur}$ 与邻居 $G_{nb}$，计算归一化差异：

```python
def compute_dispersion(G_cur, G_nb, scene_scale):
    """计算两个高斯之间的局部离散度"""
    
    # 1. 位置差异（归一化到场景尺度）
    pos_diff = norm(G_cur.mean - G_nb.mean) / scene_scale
    
    # 2. 协方差差异（Frobenius 范数，归一化到均值）
    cov_mean = (norm(G_cur.cov) + norm(G_nb.cov)) / 2
    cov_diff = norm(G_cur.cov - G_nb.cov) / (cov_mean + eps)
    
    # 3. 不透明度差异（绝对差）
    opacity_diff = abs(G_cur.opacity - G_nb.opacity)
    
    # 4. 颜色差异（球谐 DC 分量，即 SH[0]）
    color_diff = norm(G_cur.sh[0:3] - G_nb.sh[0:3]) / sqrt(3)
    
    # 加权综合
    dispersion = (
        0.4 * pos_diff +      # 位置权重最大
        0.3 * cov_diff +      # 形状次之
        0.15 * opacity_diff + # 不透明度
        0.15 * color_diff     # 颜色
    )
    
    return dispersion
```

### 5.2 硬件简化

为减少硬件复杂度，可采用以下简化：

```python
def compute_dispersion_simplified(G_cur, G_nb, scene_scale):
    """简化版：仅用位置和不透明度"""
    
    # L1 距离代替 L2（避免开方）
    pos_diff = sum(abs(G_cur.mean - G_nb.mean)) / scene_scale
    opacity_diff = abs(G_cur.opacity - G_nb.opacity)
    
    # 二元权重
    dispersion = 0.7 * pos_diff + 0.3 * opacity_diff
    
    return dispersion
```

**硬件实现**：3 次减法 + 3 次绝对值 + 1 次加法 + 2 次乘法 = **单周期**

### 5.3 场景尺度估计

场景尺度 `scene_scale` 可在线估计：

```python
# 方法1：使用深度范围（来自 DFCC）
scene_scale = depth_max - depth_min

# 方法2：滑动窗口统计
scene_scale = running_std(recent_positions)

# 方法3：固定值（针对特定数据集预设）
scene_scale = 1.0  # 假设场景已归一化
```

## 6. 分类化合并策略

### 6.1 离散度阈值分类

| 离散度范围 | 分类 | 占比 | 合并策略 |
|-----------|------|------|---------|
| $d < 0.05$ | 低离散度 | ~73.6% | 大窗口激进合并（4×4） |
| $0.05 \leq d < 0.15$ | 中等离散度 | ~19.2% | 小窗口保守合并（2×2） |
| $d \geq 0.15$ | 高离散度 | ~7.2% | 保持独立 |

### 6.2 合并窗口机制

```python
class MergeWindow:
    def __init__(self, max_size=16):
        self.buffer = []      # 待合并的高斯
        self.anchor = None    # 锚点高斯（第一个进入的）
        self.max_size = max_size
    
    def try_add(self, gaussian, dispersion):
        """尝试将高斯加入合并窗口"""
        
        if self.anchor is None:
            # 窗口为空，设为锚点
            self.anchor = gaussian
            self.buffer = [gaussian]
            return True
        
        # 与锚点比较
        if dispersion < 0.05 and len(self.buffer) < self.max_size:
            # 低离散度，加入大窗口
            self.buffer.append(gaussian)
            return True
        elif dispersion < 0.15 and len(self.buffer) < 4:
            # 中等离散度，加入小窗口
            self.buffer.append(gaussian)
            return True
        else:
            # 高离散度或窗口已满，触发输出
            return False
    
    def flush(self):
        """输出合并结果并清空窗口"""
        if len(self.buffer) == 0:
            return None
        
        merged = self._weighted_merge(self.buffer)
        self.buffer = []
        self.anchor = None
        return merged
```

### 6.3 加权平均合并

```python
def weighted_merge(gaussians: list) -> Gaussian:
    """按不透明度加权合并多个高斯"""
    
    # 权重：不透明度
    weights = [g.opacity for g in gaussians]
    total_weight = sum(weights) + eps
    norm_weights = [w / total_weight for w in weights]
    
    merged = Gaussian()
    
    # 位置：加权平均
    merged.mean = sum(w * g.mean for w, g in zip(norm_weights, gaussians))
    
    # 协方差：加权平均（简化处理）
    merged.cov = sum(w * g.cov for w, g in zip(norm_weights, gaussians))
    
    # 不透明度：取最大值（保守策略）或平均
    merged.opacity = max(g.opacity for g in gaussians)
    
    # 球谐系数：加权平均
    merged.sh = sum(w * g.sh for w, g in zip(norm_weights, gaussians))
    
    return merged
```

## 7. 完整处理流程

```
输入：Encoder 输出的高斯流 G[0], G[1], ..., G[N-1]

初始化：
    line_buffer = LineBuffer(width=W)
    merge_window = MergeWindow(max_size=16)
    output_queue = []

For each 高斯 G[i] at 像素 (u, v):
    
    Step 1: 获取邻居 [1 cycle]
        neighbors = line_buffer.get_neighbors(u)
    
    Step 2: 计算局部离散度 [1 cycle]
        if neighbors['left'] is not None:
            disp_L = compute_dispersion(G[i], neighbors['left'])
        if neighbors['up'] is not None:
            disp_U = compute_dispersion(G[i], neighbors['up'])
        
        dispersion = min(disp_L, disp_U)  # 取最小（最相似的邻居）
    
    Step 3: 分类合并决策 [1 cycle]
        
        IF dispersion >= 0.15:  // 高离散度
            // 输出当前窗口，保持独立
            merged = merge_window.flush()
            if merged: output_queue.append(merged)
            output_queue.append(G[i])  // 独立输出
            
        ELIF merge_window.try_add(G[i], dispersion):
            // 成功加入窗口，继续
            pass
            
        ELSE:
            // 窗口已满或不满足条件，输出并开始新窗口
            merged = merge_window.flush()
            if merged: output_queue.append(merged)
            merge_window.try_add(G[i], dispersion)
    
    Step 4: 更新行缓存 [1 cycle]
        line_buffer.update(u, G[i])
    
    Step 5: 行尾处理
        IF u == W - 1:  // 行尾
            // 强制输出当前窗口
            merged = merge_window.flush()
            if merged: output_queue.append(merged)

输出：合并后的高斯流写入 HBM
```

## 8. 硬件架构

### 8.1 模块框图

```
                    ┌─────────────────────────────────────┐
                    │          SGM Pipeline               │
                    │                                     │
    Encoder ──────► │  ┌──────────┐    ┌──────────────┐  │ ──────► HBM
    高斯流          │  │  Line    │    │   Merge      │  │  合并后
                    │  │  Buffer  │───►│   Decision   │  │  高斯流
                    │  │  (30KB)  │    │   Unit       │  │
                    │  └──────────┘    └──────┬───────┘  │
                    │        ▲                │          │
                    │        │          ┌─────▼─────┐    │
                    │        │          │  Merge    │    │
                    │        └──────────│  Window   │    │
                    │                   │  Buffer   │    │
                    │                   └───────────┘    │
                    └─────────────────────────────────────┘
```

### 8.2 关键组件

| 组件 | 功能 | 资源 |
|-----|------|------|
| **Line Buffer** | 存储上一行高斯 | 30 KB SRAM |
| **Dispersion Unit** | 计算局部离散度 | 6 减法 + 6 绝对值 + 2 乘加 |
| **Merge Decision** | 阈值比较，决策 | 2 比较器 + 状态机 |
| **Merge Window** | 暂存待合并高斯 | 2 KB SRAM（16 × 120B） |
| **Weighted Avg** | 加权平均合并 | 4 MAC + 1 除法 |

### 8.3 流水线时序

| Cycle | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-------|---------|---------|---------|---------|
| 1 | Fetch G[i] | | | |
| 2 | Fetch G[i+1] | Get neighbors | | |
| 3 | Fetch G[i+2] | Get neighbors | Dispersion | |
| 4 | Fetch G[i+3] | Get neighbors | Dispersion | Decision/Merge |
| ... | ... | ... | ... | Output |

**吞吐量**：1 高斯 / cycle（与 Encoder 输出速率匹配）

## 9. 硬件资源估算

| 组件 | 资源 | 说明 |
|-----|------|-----|
| Line Buffer | 30 KB SRAM | 可优化至 6 KB |
| Merge Window | 2 KB SRAM | 16 条目 |
| Dispersion Unit | ~200 LUT | 简化版 |
| Merge Logic | ~500 LUT | 加权平均 |
| 状态机 | ~100 LUT | 控制逻辑 |
| **总面积** | **~0.1 mm² (28nm)** | 含 SRAM |
| **功耗** | **~15 mW** | 典型工作负载 |

## 10. 预期效果

### 10.1 压缩率

| 场景类型 | 原始高斯数 | 合并后高斯数 | 压缩率 |
|---------|-----------|-------------|-------|
| 室内平坦（墙壁） | 131K | ~35K | **3.7×** |
| 室内复杂（家具） | 131K | ~50K | **2.6×** |
| 室外场景 | 131K | ~45K | **2.9×** |
| **平均** | **131K** | **~45K** | **~2.9×** |

### 10.2 HBM 流量节省

| 指标 | 原始 | 合并后 | 节省 |
|-----|------|--------|------|
| 高斯数据量 | 44 MB | ~15 MB | **66%** |
| Decoder 读取 | 44 MB/帧 | ~15 MB/帧 | **66%** |
| 额外写入开销 | 0 | 0 | — |

### 10.3 渲染质量

| 指标 | 原始 | 合并后 | 变化 |
|-----|------|--------|------|
| PSNR | 28.5 dB | 28.2 dB | -0.3 dB |
| SSIM | 0.92 | 0.91 | -0.01 |

## 11. 与 DFCC 的协同

SGM 与 DFCC 在数据流中串联工作：

```
像素特征 ─► DFCC ─► 深度结果 ─► 高斯生成 ─► SGM ─► HBM ─► Decoder
           │                               │
           └─ 压缩计算流 ──────────────────┴─ 压缩数据流
```

**协同点**：
1. DFCC 的语义签名可为 SGM 提供"语义区域"分组信息
2. 若 DFCC 命中，表明相邻像素语义相似，SGM 合并概率更高
3. 两者共享场景尺度估计

## 12. 总结

本设计通过以下关键决策实现了可落地的 SGM：

1. **数据流友好**：利用光栅扫描顺序，仅需 O(W) 的行缓存
2. **固定邻域比较**：无需全局搜索，访问延迟确定
3. **分类化合并**：根据离散度自适应调整合并粒度
4. **流水线设计**：吞吐量匹配 Encoder 输出速率
5. **写入拦截**：在 HBM 写入前完成合并，消除冗余往返
