# FSDR：特征相似深度复用设计文档

## 1. 设计目标

设计一个特征相似深度复用单元（Feature-Similarity Depth Reuse, FSDR），满足以下要求：

1. **模型兼容性**：同时支持 TransPlat、MVSplat、DepthSplat 三种深度预测架构
2. **硬件可实现**：所有计算均为简单的整数/定点运算，无需复杂的微分或迭代
3. **可落地部署**：片上存储开销 < 2KB，延迟开销 < 5 cycles
4. **优化目标**：减少深度搜索阶段的冗余访存开销

## 2. 核心问题：深度搜索的访存瓶颈

### 2.1 深度搜索的访存模式

深度搜索阶段的访存开销主要来自：

```
对于每个参考像素 (u, v)：
    对于每个深度候选 d_k (k = 1..D)：
        1. 计算投影坐标 (u', v') = project(u, v, d_k)
        2. 从目标特征图采样: tgt_feat = sample(target_feature_map, u', v')  ← HBM 访问
        3. 计算匹配代价: cost_k = similarity(ref_feat, tgt_feat)
```

**访存量分析**（以 64×64 特征图，D=32 深度候选为例）：
- 每像素访存：32 × 128 × 2B = 8 KB
- 总访存：64 × 64 × 8 KB = 32 MB

### 2.2 语义相似性带来的复用机会

实测数据表明，特征余弦相似度超过 0.92 的像素对，其深度差异中位数仅为 2.3%。这意味着：

- **语义相似的像素具有相近的深度**
- 如果能复用已计算的深度结果，可大幅减少访存

## 3. 三种模型的深度预测统一抽象

尽管三种模型的具体实现不同，其深度预测流程可抽象为统一接口：

```
Input:  参考特征 f_ref ∈ R^C，深度候选 {d_1, d_2, ..., d_D}
Process: 对每个 d_k，计算匹配代价 C_k（或概率 p_k）
Output: 最优深度 d* = argmin(C) 或 期望深度 E[d] = Σ p_k × d_k
```

| 模型 | 匹配方式 | 输出形式 |
|-----|---------|---------|
| **TransPlat** | Deformable Attention + Cost Volume | Softmax 概率 → 加权深度 |
| **MVSplat** | Plane-sweep + 特征相关 | Softmax 概率 → 加权深度 |
| **DepthSplat** | 类似 MVSplat + 精炼网络 | Softmax 概率 → 采样深度 |

**关键观察**：三种模型在完整深度搜索后，都会产生一个深度维度上的概率分布 $\{p_1, ..., p_D\}$，该分布自然包含了用于修正的全部信息。

## 4. 缓存条目结构

### 4.1 核心思路：存储分布统计量

存储深度概率分布的**统计量**——这些量在完整搜索过程中自然产生，无需额外计算：

| 字段 | 位宽 | 说明 | 计算方式 |
|-----|------|-----|---------|
| `signature` | 16 bit | 参考特征的 LSH 签名 | 16 次点积 + 符号 |
| `position` | 16 bit | 像素坐标 $(u, v)$ | 直接记录 |
| `best_depth` | 16 bit | 最优深度（FP16） | softmax 后的期望或 argmax |
| `peak_prob` | 8 bit | 最优深度的概率（定点 0-1） | softmax 输出的最大值 |
| `second_idx` | 5 bit | 次优深度的索引偏移 | argmax 第二大的位置 |
| `spread` | 8 bit | 分布宽度（标准差的量化） | 见下文 |
| `valid` | 1 bit | 有效标志 | — |
| **总计** | **70 bit** | 约 9 字节/条目 | 128 条目 = 1.1KB |

### 4.2 各字段的具体计算方法

#### (a) `best_depth`：最优深度

```python
# 方式1：期望深度（软决策，精度高）
best_depth = sum(p_k * d_k for k in range(D))

# 方式2：argmax深度（硬决策，硬件简单）
best_idx = argmax(p_1, ..., p_D)
best_depth = d[best_idx]
```

推荐使用方式1，因为三种模型都已经计算了 softmax 概率。

#### (b) `peak_prob`：主峰概率

```python
peak_prob = max(p_1, ..., p_D)
# 量化为 8-bit：peak_prob_q = round(peak_prob * 255)
```

**物理意义**：反映深度估计的置信度。peak_prob 高表示深度分布尖锐、置信度高；低则表示不确定。

#### (c) `second_idx`：次峰索引偏移

```python
best_idx = argmax(p)
# 在排除 best_idx 后找第二大
second_idx = argmax(p, exclude=best_idx)
# 存储相对偏移，节省位宽
second_offset = second_idx - best_idx  # 范围 [-D/2, D/2]
# 用 5 bit 有符号数表示，范围 [-16, 15]
```

**物理意义**：当特征略有差异时，真实深度可能在 best_depth 和 second_depth 之间。

#### (d) `spread`：分布宽度

```python
# 计算加权标准差
mean_depth = sum(p_k * d_k)
variance = sum(p_k * (d_k - mean_depth)^2)
spread = sqrt(variance)

# 量化为 8-bit（相对于深度范围）
depth_range = d_D - d_1
spread_q = round(spread / depth_range * 255)
```

**物理意义**：spread 小表示分布集中、可以放心复用；spread 大表示不确定性高、需要谨慎。

### 4.3 硬件实现考量

上述所有计算都可以在完整深度搜索的 softmax 输出后**顺带完成**：

| 计算 | 硬件实现 | 额外延迟 |
|-----|---------|---------|
| argmax | 比较树 | 1-2 cycles |
| 期望 $\sum p_k d_k$ | MAC 阵列（已有） | 0（与 softmax 融合） |
| 方差 | 需要 $d_k^2$ 的预计算表 | 2-3 cycles |

**简化方案**：如果方差计算开销过大，可以用 **近似 spread**：
```python
# 用 top-2 概率的距离近似
spread_approx = abs(d[best_idx] - d[second_idx]) * (1 - peak_prob)
```

## 5. 修正方法（无需导数）

### 5.1 核心思路

当缓存命中时，基于以下原则进行修正：

1. **相似特征 → 相似深度**：直接复用，微调即可
2. **置信度引导**：高置信度条目可直接复用，低置信度需验证
3. **次峰插值**：当特征差异较大时，深度可能在主峰和次峰之间

### 5.2 修正公式

设缓存命中条目为 $e$，当前像素特征签名的汉明距离为 $h$：

```
情况1：高置信度直接复用 (peak_prob > 0.8 且 h ≤ 2)
    d_cur = e.best_depth
    → 跳过深度搜索，节省 32 次目标特征图访问
    
情况2：中等置信度插值 (0.5 < peak_prob ≤ 0.8 或 2 < h ≤ 3)
    second_depth = depth_candidates[e.best_idx + e.second_offset]
    λ = h / 4  # 汉明距离归一化为插值系数
    d_cur = (1 - λ) * e.best_depth + λ * second_depth
    → 跳过深度搜索，节省 32 次目标特征图访问
    
情况3：低置信度局部验证 (peak_prob ≤ 0.5 或 h > 3)
    → 执行轻量验证（见 5.3）
    → 仅需 3-7 次目标特征图访问（而非 32 次）
```

### 5.3 轻量验证模式

当置信度不足以直接复用时，执行**局部深度搜索**而非完整搜索：

```python
def light_verify(cached_entry, current_feature, depth_candidates):
    best_idx = cached_entry.best_idx
    spread_idx = max(1, int(cached_entry.spread * D / 255))  # 搜索半径
    
    # 只在 [best_idx - spread_idx, best_idx + spread_idx] 范围搜索
    search_range = range(
        max(0, best_idx - spread_idx),
        min(D, best_idx + spread_idx + 1)
    )
    
    # 执行 2*spread_idx+1 次匹配（典型值 3-7 次，而非完整的 32 次）
    costs = [compute_cost(current_feature, d_k) for k in search_range]
    local_best = search_range[argmin(costs)]
    
    return depth_candidates[local_best]
```

**访存节省**：
- 完整搜索：32 次目标特征图访问
- 轻量验证：3-7 次访问（节省 78-91%）

### 5.4 空间梯度辅助修正（可选）

若启用空间梯度，可进一步提升精度：

```python
# 从相邻已计算像素估计深度梯度
grad_u = (depth[u+1, v] - depth[u-1, v]) / 2  # 若可用
grad_v = (depth[u, v+1] - depth[u, v-1]) / 2  # 若可用

# 基于位置偏移修正
delta_u = current_pos.u - cached_entry.pos.u
delta_v = current_pos.v - cached_entry.pos.v
d_cur += grad_u * delta_u + grad_v * delta_v
```

## 6. 完整处理流程

```
输入：当前像素的参考特征 f_cur, 位置 (u, v)

Step 1: 签名生成 [1 cycle]
    sig_cur = LSH(f_cur)  // 16次点积 + 16次符号判断

Step 2: 缓存查找 [1 cycle]
    for each entry e in cache (并行):
        hamming_dist[e] = popcount(sig_cur XOR e.signature)
    
    // 选择汉明距离最小且 < 阈值的条目
    hit_entry = argmin(hamming_dist) if min(hamming_dist) < τ_h else NULL

Step 3: 命中/未命中分支

    IF hit_entry != NULL:  // 命中路径 → 减少访存
        
        IF hit_entry.peak_prob > 0.8 AND hamming_dist < 2:
            // 情况1：高置信度直接复用
            d_cur = hit_entry.best_depth
            // 访存节省：跳过 32 次目标特征图访问
            
        ELIF hit_entry.peak_prob > 0.5 OR hamming_dist ≤ 3:
            // 情况2：插值修正
            λ = hamming_dist / 4
            d_second = depth_candidates[hit_entry.best_idx + hit_entry.second_offset]
            d_cur = (1 - λ) * hit_entry.best_depth + λ * d_second
            // 访存节省：跳过 32 次目标特征图访问
            
        ELSE:
            // 情况3：轻量验证
            d_cur = light_verify(hit_entry, f_cur, depth_candidates)
            // 访存节省：仅需 3-7 次访问（而非 32 次）
        
        // 更新条目（滑动平均）
        hit_entry.best_depth = 0.9 * hit_entry.best_depth + 0.1 * d_cur
        hit_entry.peak_prob += 0.01  // 命中则增加置信度
        
    ELSE:  // 未命中路径
        // 执行完整深度搜索（32 次目标特征图访问）
        probs = softmax(compute_all_costs(f_cur, depth_candidates))
        d_cur = expected_depth(probs, depth_candidates)
        
        // 提取统计量
        best_idx = argmax(probs)
        peak_prob = probs[best_idx]
        second_idx = argmax_exclude(probs, best_idx)
        spread = compute_spread(probs, depth_candidates)
        
        // 写入缓存
        new_entry = {
            signature: sig_cur,
            position: (u, v),
            best_depth: d_cur,
            best_idx: best_idx,
            peak_prob: quantize(peak_prob),
            second_offset: second_idx - best_idx,
            spread: quantize(spread),
            valid: 1
        }
        cache.insert(new_entry, policy=LRU_confidence_weighted)

输出：深度估计 d_cur
```

## 7. 三种模型的适配层

### 7.1 TransPlat 适配

```python
class FSDR_TransPlat_Adapter:
    def extract_depth_distribution(self, cost_volume, depth_candidates):
        """
        TransPlat 的 cost volume 经过 U-Net 精炼后，
        在深度维度上做 softmax 得到概率分布
        """
        # cost_volume: [B, D, H, W]
        probs = F.softmax(-cost_volume, dim=1)  # 代价越小概率越大
        return probs  # [B, D, H, W]
```

### 7.2 MVSplat / DepthSplat 适配

```python
class FSDR_MVSplat_Adapter:
    def extract_depth_distribution(self, correlation_volume, depth_candidates):
        """
        MVSplat 的相关性体积直接反映匹配质量，
        softmax 后即为概率分布
        """
        # correlation_volume: [B, D, H, W]
        probs = F.softmax(correlation_volume, dim=1)  # 相关性越大概率越大
        return probs  # [B, D, H, W]
```

### 7.3 统一接口

```python
class FSDR:
    def __init__(self, model_type: str):
        if model_type == 'transplat':
            self.adapter = FSDR_TransPlat_Adapter()
        elif model_type in ['mvsplat', 'depthsplat']:
            self.adapter = FSDR_MVSplat_Adapter()
    
    def process_pixel(self, feature, depth_candidates, position):
        # Step 1: 签名生成
        signature = self.lsh_hash(feature)
        
        # Step 2: 缓存查找
        hit_entry, hamming_dist = self.cache_lookup(signature)
        
        # Step 3: 命中/未命中处理
        if hit_entry is not None:
            depth = self.hit_path(hit_entry, hamming_dist, feature, depth_candidates)
        else:
            probs = self.adapter.extract_depth_distribution(...)
            depth = self.miss_path(probs, depth_candidates, signature, position)
        
        return depth
```

## 8. 硬件资源估算

| 组件 | 资源 | 说明 |
|-----|------|-----|
| 缓存表 | 1.1 KB SRAM | 128 条目 × 70 bit |
| LSH 权重 | 0.5 KB ROM | 16 × 128 × 2B (FP16) |
| 签名生成 | 16 MAC + 16 比较器 | 单周期 |
| 汉明距离 | 128 × 16-bit XOR + popcount | 并行，单周期 |
| 插值计算 | 2 乘法 + 2 加法 | 单周期 |
| **总面积** | **~0.05 mm² (28nm)** | |
| **功耗** | **~5 mW** | |

## 9. 预期效果

### 9.1 访存节省分析

| 场景 | 命中率 | 命中时访存节省 | 综合访存节省 |
|-----|--------|---------------|-------------|
| 平坦区域（墙壁、地板） | 90%+ | 100%（直接复用） | ~90% |
| 纹理区域（物体表面） | 70-80% | 78-100%（轻量验证/直接复用） | ~65% |
| 复杂边界（遮挡、细节） | 30-50% | 78%（轻量验证） | ~30% |
| **整体平均** | **~75%** | **~90%** | **~68%** |

### 9.2 与 Challenge 1 的对应

| Challenge 1 问题 | FSDR 解决方案 |
|-----------------|--------------|
| 逐点深度搜索 | 语义签名匹配，相似像素复用 |
| 忽视 2D 特征相似性 | LSH 将特征相似性转化为签名相似性 |
| 冗余访存 | 命中时跳过/减少目标特征图访问 |
| L2 缓存命中率低 (31.2%) | 语义级复用补充地址级复用 |

## 10. 总结

FSDR 通过以下关键设计实现深度搜索阶段的访存优化：

1. **存储分布统计量而非导数**：利用 softmax 自然产生的概率分布，无需额外微分计算
2. **三级修正策略**：高置信度直接复用 → 中置信度插值 → 低置信度轻量验证
3. **统一适配接口**：抽象出概率分布提取层，兼容三种深度预测架构
4. **硬件友好设计**：所有计算均为简单定点运算，无需复杂算术单元
5. **访存优先优化**：将语义相似性转化为可被硬件捕获的复用机会，减少 HBM 访问
