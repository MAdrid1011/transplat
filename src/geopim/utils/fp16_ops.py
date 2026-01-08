"""
FP16 运算模拟

模拟 PIM Unit 中的 FP16 运算行为。
"""

import numpy as np
from typing import Union

# 类型别名
FP16Array = Union[np.ndarray, float]


def fp16_add(a: FP16Array, b: FP16Array) -> np.ndarray:
    """
    FP16 加法
    
    Args:
        a, b: FP16 操作数
        
    Returns:
        FP16 结果
    """
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    return (a + b).astype(np.float16)


def fp16_mul(a: FP16Array, b: FP16Array) -> np.ndarray:
    """
    FP16 乘法
    
    Args:
        a, b: FP16 操作数
        
    Returns:
        FP16 结果
    """
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    return (a * b).astype(np.float16)


def fp16_fma(a: FP16Array, b: FP16Array, c: FP16Array) -> np.ndarray:
    """
    FP16 融合乘加 (Fused Multiply-Add)
    
    result = a * b + c
    
    Args:
        a, b, c: FP16 操作数
        
    Returns:
        FP16 结果
    """
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    c = np.asarray(c, dtype=np.float16)
    
    # 在 FP32 中计算以模拟 FMA 的精度优势
    result = np.float32(a) * np.float32(b) + np.float32(c)
    return result.astype(np.float16)


def fp16_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    FP16 向量点积
    
    Args:
        a, b: [N] FP16 向量
        
    Returns:
        FP16 标量结果
    """
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    
    # 使用 FP32 累加以避免精度损失
    result = np.sum(a.astype(np.float32) * b.astype(np.float32))
    return np.float16(result)


def fp16_mac_8lane(
    features: np.ndarray,  # [8] FP16
    weight: float,          # FP16 scalar
    accumulator: np.ndarray  # [8] FP32
) -> np.ndarray:
    """
    8-lane FP16 MAC 操作
    
    模拟 PIM Unit 的 8-lane 并行 MAC。
    
    acc[0:8] += weight * features[0:8]
    
    Args:
        features: [8] FP16 特征
        weight: FP16 权重
        accumulator: [8] FP32 累加器
        
    Returns:
        更新后的 FP32 累加器
    """
    features = np.asarray(features, dtype=np.float16)[:8]
    weight = np.float16(weight)
    accumulator = np.asarray(accumulator, dtype=np.float32)[:8]
    
    # FP16 乘法，FP32 累加
    product = features.astype(np.float32) * np.float32(weight)
    return accumulator + product


def check_fp16_overflow(value: FP16Array) -> bool:
    """
    检查 FP16 溢出
    
    FP16 范围: [-65504, 65504]
    
    Args:
        value: 要检查的值
        
    Returns:
        True 如果溢出
    """
    value = np.asarray(value, dtype=np.float32)
    max_fp16 = 65504.0
    return np.any(np.abs(value) > max_fp16)


def fp32_to_fp16_safe(value: np.ndarray) -> np.ndarray:
    """
    安全的 FP32 到 FP16 转换 (带饱和)
    
    Args:
        value: FP32 数组
        
    Returns:
        FP16 数组 (溢出值被饱和)
    """
    value = np.asarray(value, dtype=np.float32)
    max_fp16 = 65504.0
    
    # 饱和处理
    value = np.clip(value, -max_fp16, max_fp16)
    return value.astype(np.float16)

