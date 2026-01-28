"""
优化的 SH 旋转实现

原始实现的问题:
- e3nn 的 wigner_D 函数对每个批次元素单独计算
- 当输入包含广播的旋转矩阵时（如 [b, v, 1, 1, 1, 1, 3, 3]），
  它会被展开为 [b, v, h*w, srf, spp, 3, 3] 导致重复计算

优化方案:
1. 预计算 Wigner-D 矩阵（只对 [b, v] 个不同的旋转）
2. 使用简单的 einsum 应用到 SH 系数

使用方式:
1. 在 Encoder 开始时调用 precompute_wigner_d_matrices()
2. 在 GaussianAdapter 中使用 rotate_sh_with_precomputed()
"""

from math import isqrt
from typing import List, Tuple

import torch
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def precompute_wigner_d_matrices(
    rotations: Float[Tensor, "batch views 3 3"],
    max_degree: int = 3,
) -> Tuple[List[Float[Tensor, "batch views size size"]], int]:
    """
    预计算所有 degree 的 Wigner-D 矩阵
    
    Args:
        rotations: 旋转矩阵 [b, v, 3, 3]
        max_degree: 最大 SH degree
    
    Returns:
        wigner_matrices: 每个 degree 的 Wigner-D 矩阵列表
        d_sh: SH 系数维度 ((max_degree+1)^2)
    """
    device = rotations.device
    dtype = rotations.dtype
    
    # 计算欧拉角
    alpha, beta, gamma = matrix_to_angles(rotations)
    
    # 计算每个 degree 的 Wigner-D 矩阵
    wigner_matrices = []
    for degree in range(max_degree + 1):
        with torch.device(device):
            D = wigner_D(degree, alpha, beta, gamma).type(dtype)
        wigner_matrices.append(D)  # [b, v, 2*degree+1, 2*degree+1]
    
    d_sh = (max_degree + 1) ** 2
    return wigner_matrices, d_sh


def rotate_sh_with_precomputed(
    sh_coefficients: Float[Tensor, "*batch 3 n"],
    wigner_matrices: List[Float[Tensor, "... size size"]],
    rotation_batch_dims: int = 2,
) -> Float[Tensor, "*batch 3 n"]:
    """
    使用预计算的 Wigner-D 矩阵旋转 SH 系数
    
    Args:
        sh_coefficients: SH 系数 [b, v, h*w, srf, spp, 3, d_sh]
        wigner_matrices: 预计算的 Wigner-D 矩阵列表，每个形状为 [b, v, size, size]
        rotation_batch_dims: rotations 的批次维度数量（默认为 2，即 [b, v]）
    
    Returns:
        旋转后的 SH 系数
    """
    n = sh_coefficients.shape[-1]
    max_degree = isqrt(n) - 1
    
    results = []
    for degree in range(max_degree + 1):
        D = wigner_matrices[degree]  # [b, v, 2l+1, 2l+1]
        
        # 获取对应 degree 的 SH 系数切片
        start_idx = degree ** 2
        end_idx = (degree + 1) ** 2
        sh_slice = sh_coefficients[..., start_idx:end_idx]  # [..., 3, 2l+1]
        
        # 扩展 D 以匹配 sh_slice 的维度
        # D: [b, v, size, size]
        # sh_slice: [b, v, h*w, srf, spp, 3, size]
        # 需要在中间添加维度
        num_extra_dims = sh_slice.dim() - rotation_batch_dims - 2  # -2 for (3, size)
        for _ in range(num_extra_dims + 1):  # +1 for the 3 (xyz) dimension
            D = D.unsqueeze(-3)
        # D 现在是 [b, v, 1, 1, 1, 1, size, size]
        
        # 应用旋转
        rotated = einsum(D, sh_slice, "... i j, ... j -> ... i")
        results.append(rotated)
    
    return torch.cat(results, dim=-1)


def rotate_sh_optimized(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    """
    优化的 SH 旋转函数
    
    自动检测旋转矩阵是否被广播，如果是，则使用预计算优化
    
    Args:
        sh_coefficients: SH 系数
        rotations: 旋转矩阵（可能包含广播维度）
    
    Returns:
        旋转后的 SH 系数
    """
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype
    
    *_, n = sh_coefficients.shape
    max_degree = isqrt(n) - 1
    
    # 检查旋转矩阵是否有广播维度
    # 找到旋转矩阵的有效批次维度（非 1 的维度）
    rot_shape = rotations.shape[:-2]  # 去掉最后两个 (3, 3)
    sh_shape = sh_coefficients.shape[:-1]  # 去掉最后一个 (n)
    
    # 找到最小的非广播维度
    effective_rot_dims = []
    for i, (r, s) in enumerate(zip(rot_shape, sh_shape)):
        if r != 1:
            effective_rot_dims.append(i)
    
    # 如果旋转矩阵没有太多广播，使用原始实现
    if len(effective_rot_dims) == len(rot_shape):
        # 没有广播，使用原始实现
        from src.misc.sh_rotation import rotate_sh
        return rotate_sh(sh_coefficients, rotations)
    
    # 提取有效的旋转矩阵
    # 假设广播维度在后面（如 [b, v, 1, 1, 1, 1, 3, 3]）
    # 我们需要 squeeze 这些维度
    rot_squeezed = rotations
    for i in range(len(rot_shape) - 1, -1, -1):
        if rot_shape[i] == 1 and i > 1:  # 保留前两个维度 (b, v)
            rot_squeezed = rot_squeezed.squeeze(i)
    
    # 预计算 Wigner-D 矩阵
    wigner_matrices, _ = precompute_wigner_d_matrices(rot_squeezed, max_degree)
    
    # 应用旋转
    return rotate_sh_with_precomputed(sh_coefficients, wigner_matrices, rotation_batch_dims=2)


# 测试函数
if __name__ == "__main__":
    import time
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试参数
    b, v = 1, 2
    h, w = 256, 256
    num_points = h * w
    d_sh = 16
    srf, spp = 1, 1
    num_iterations = 10
    
    # 创建测试数据
    c2w_rot_np = R.random(b * v).as_matrix().reshape(b, v, 3, 3)
    c2w_rot = torch.from_numpy(c2w_rot_np).float().to(device)
    
    rotations_for_sh = c2w_rot[:, :, None, None, None, None, :, :]
    sh_coeffs = torch.randn(b, v, num_points, srf, spp, 3, d_sh, device=device)
    
    # 测试原始实现
    from src.misc.sh_rotation import rotate_sh
    
    # Warmup
    for _ in range(3):
        _ = rotate_sh(sh_coeffs, rotations_for_sh)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        result_original = rotate_sh(sh_coeffs, rotations_for_sh)
    torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) / num_iterations * 1000
    
    # 测试优化实现
    wigner_matrices, _ = precompute_wigner_d_matrices(c2w_rot, max_degree=3)
    
    # Warmup
    for _ in range(3):
        _ = rotate_sh_with_precomputed(sh_coeffs, wigner_matrices)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        result_optimized = rotate_sh_with_precomputed(sh_coeffs, wigner_matrices)
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / num_iterations * 1000
    
    # 验证结果
    max_diff = (result_original - result_optimized).abs().max().item()
    
    print(f"原始实现时间: {original_time:.3f} ms")
    print(f"优化实现时间: {optimized_time:.3f} ms")
    print(f"加速比: {original_time / optimized_time:.2f}x")
    print(f"最大误差: {max_diff:.2e}")

