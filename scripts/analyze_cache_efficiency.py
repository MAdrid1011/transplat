#!/usr/bin/env python3
"""分析 Cache 效率"""

# depth_predictor 的数据量估算
features_size = 2 * 128 * 64 * 64 * 4  # 2 views, 128 ch, 64x64, float32
cost_volume_size = 2 * 32 * 128 * 64 * 64 * 4  # 2 views, 32 depths
attention_qkv = 3 * 2 * 128 * 64 * 64 * 4  # Q, K, V
attention_scores = 2 * 64 * 64 * 128 * 4  # attention weights

total_data = features_size + cost_volume_size + attention_qkv + attention_scores
print(f"=== depth_predictor 数据量分析 ===")
print(f"特征图: {features_size / 1e6:.1f} MB")
print(f"Cost Volume: {cost_volume_size / 1e6:.1f} MB")
print(f"Attention Q/K/V: {attention_qkv / 1e6:.1f} MB")
print(f"Attention Scores: {attention_scores / 1e6:.1f} MB")
print(f"总数据量: {total_data / 1e6:.1f} MB")
print()

# L2 Cache 容量
l2_cache = 50 * 1e6  # H100: 50MB
print(f"H100 L2 Cache: {l2_cache / 1e6:.0f} MB")
print(f"数据量 / L2 Cache = {total_data / l2_cache:.1f}x (无法全部缓存)")
print()

# 实际 DRAM 流量 vs 理论最小
measured_dram_traffic = 19.5e9  # 19.5 GB
ideal_traffic = total_data * 2  # 读一次 + 写一次

print(f"=== Cache 效率分析 ===")
print(f"理论最小流量 (完美复用): {ideal_traffic / 1e6:.1f} MB")
print(f"实际测量流量: {measured_dram_traffic / 1e9:.1f} GB")
print(f"流量放大倍数: {measured_dram_traffic / ideal_traffic:.0f}x")
print()

# 估算 Cache 命中率
# 假设: 每次访问如果 miss，从 HBM 读 128B (cache line)
# 如果理想情况命中率 100%，流量 = ideal_traffic
# 实际命中率 = 1 - (actual - ideal) / (total_accesses * 128)
cache_miss_rate = (measured_dram_traffic - ideal_traffic) / measured_dram_traffic
cache_hit_rate = 1 - cache_miss_rate
print(f"估算 L2 Cache 命中率: {cache_hit_rate * 100:.1f}%")
print(f"估算 L2 Cache Miss 率: {cache_miss_rate * 100:.1f}%")
print()

# 不规则访问的影响
cache_line_size = 128  # bytes
useful_bytes_per_access = 4  # float32
cache_line_utilization = useful_bytes_per_access / cache_line_size
print(f"=== 不规则访问影响 ===")
print(f"Cache Line 大小: {cache_line_size} bytes")
print(f"每次采样实际使用: {useful_bytes_per_access} bytes (1 float)")
print(f"Cache Line 利用率 (不规则访问): {cache_line_utilization * 100:.1f}%")
print(f"流量浪费: {(1 - cache_line_utilization) * 100:.0f}%")
