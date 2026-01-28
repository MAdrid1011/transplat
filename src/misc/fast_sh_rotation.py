"""
快速 SH 旋转实现

关键发现：
- e3nn 的 wigner_D 函数非常慢（即使对于小批量也需要 ~80ms）
- 应用预计算的 Wigner-D 矩阵只需要 ~2ms

优化方向：
1. 使用 PyTorch 原生操作实现 Wigner-D 矩阵计算
2. 对于低阶 SH (degree 0-3)，可以使用解析形式

SH 旋转数学背景：
- 旋转矩阵 R 作用于 SH 系数时，不同 degree 的系数独立变换
- degree l 的 SH 系数通过 (2l+1)x(2l+1) 的 Wigner-D 矩阵变换
- degree 0: 1x1 (标量，不变)
- degree 1: 3x3 (与旋转矩阵 R 直接相关)
- degree 2: 5x5
- degree 3: 7x7

关键洞察：
- degree 1 的 Wigner-D 矩阵与旋转矩阵 R 直接相关
- 可以避免 euler angle 转换，直接从 R 构建 Wigner-D
"""

import torch
from jaxtyping import Float
from torch import Tensor
from typing import List, Tuple
from math import sqrt


def rotation_matrix_to_wigner_d1(
    R: Float[Tensor, "*batch 3 3"]
) -> Float[Tensor, "*batch 3 3"]:
    """
    从旋转矩阵直接计算 degree 1 的 Wigner-D 矩阵
    
    对于 degree 1，Wigner-D 矩阵与旋转矩阵 R 相关：
    D^1 = U^† R U
    其中 U 是将实 SH 基转换到复 SH 基的矩阵
    
    对于实 SH 基 (Y_{-1}, Y_0, Y_1)，这简化为对 R 的坐标变换
    """
    # 实 SH 基下的 degree 1 Wigner-D 矩阵
    # 与旋转矩阵 R 的关系：D^1_{mm'} = R_{m,m'} (在某个坐标系下)
    # 这是因为 Y^1_m 与 (x, y, z) 成正比
    
    # 对于标准的实 SH 基序 (Y_{1,-1}, Y_{1,0}, Y_{1,1}) ~ (y, z, x)
    # 我们需要一个排列和符号调整
    # D1[i,j] 变换从基 (y, z, x) 到旋转后的坐标
    
    # R 作用于 (x, y, z) -> (x', y', z')
    # Y_1^{-1} ~ y, Y_1^0 ~ z, Y_1^1 ~ x
    # 所以 D1 是 R 在 (y, z, x) 基下的表示
    
    # 重新排列 R 的行列: (x,y,z) -> (y,z,x) 即 (1, 2, 0)
    perm = [1, 2, 0]
    D1 = R[..., perm, :]
    D1 = D1[..., :, perm]
    
    return D1


def rotation_matrix_to_wigner_d2(
    R: Float[Tensor, "*batch 3 3"]
) -> Float[Tensor, "*batch 5 5"]:
    """
    从旋转矩阵计算 degree 2 的 Wigner-D 矩阵
    
    使用 Wigner-D 矩阵的递推关系或直接公式
    """
    device = R.device
    dtype = R.dtype
    batch_shape = R.shape[:-2]
    
    # 提取旋转矩阵元素
    R = R.reshape(-1, 3, 3)
    n = R.shape[0]
    
    # D2 是 5x5 矩阵，对应 m, m' = -2, -1, 0, 1, 2
    D2 = torch.zeros(n, 5, 5, device=device, dtype=dtype)
    
    # 使用预计算的公式
    # 这些公式来自于球谐函数的 Clebsch-Gordan 系数
    # D2[m+2, m'+2] = ...
    
    r = R  # [n, 3, 3]
    
    # (0,0) element: (3 r_33^2 - 1) / 2
    D2[:, 2, 2] = (3 * r[:, 2, 2]**2 - 1) / 2
    
    # (0, 1) element: sqrt(3) * r_33 * r_31
    D2[:, 2, 3] = sqrt(3) * r[:, 2, 2] * r[:, 2, 0]
    
    # (0, -1) element: sqrt(3) * r_33 * r_32
    D2[:, 2, 1] = sqrt(3) * r[:, 2, 2] * r[:, 2, 1]
    
    # (0, 2) element: sqrt(3)/2 * (r_31^2 - r_32^2)
    D2[:, 2, 4] = sqrt(3) / 2 * (r[:, 2, 0]**2 - r[:, 2, 1]**2)
    
    # (0, -2) element: sqrt(3) * r_31 * r_32
    D2[:, 2, 0] = sqrt(3) * r[:, 2, 0] * r[:, 2, 1]
    
    # 其他元素类似...（完整实现需要更多公式）
    # 为简化，这里使用一个近似
    
    # 完整的 D2 矩阵计算需要所有 25 个元素的公式
    # 这里只是一个示例框架
    
    D2 = D2.reshape(*batch_shape, 5, 5)
    return D2


def fast_precompute_wigner_d(
    rotations: Float[Tensor, "*batch 3 3"],
    max_degree: int = 3,
) -> Tuple[List[Float[Tensor, "*batch size size"]], int]:
    """
    快速预计算 Wigner-D 矩阵
    
    对于低阶使用解析公式，高阶使用 e3nn
    """
    from e3nn.o3 import matrix_to_angles, wigner_D
    
    device = rotations.device
    dtype = rotations.dtype
    batch_shape = rotations.shape[:-2]
    
    wigner_matrices = []
    
    # Degree 0: 1x1 单位矩阵
    D0 = torch.ones(*batch_shape, 1, 1, device=device, dtype=dtype)
    wigner_matrices.append(D0)
    
    if max_degree >= 1:
        # Degree 1: 直接从旋转矩阵计算
        D1 = rotation_matrix_to_wigner_d1(rotations)
        wigner_matrices.append(D1)
    
    if max_degree >= 2:
        # Degree 2+: 使用 e3nn（可以后续优化）
        alpha, beta, gamma = matrix_to_angles(rotations)
        for degree in range(2, max_degree + 1):
            with torch.device(device):
                D = wigner_D(degree, alpha, beta, gamma).type(dtype)
            wigner_matrices.append(D)
    
    d_sh = (max_degree + 1) ** 2
    return wigner_matrices, d_sh


def rotate_sh_fast(
    sh_coefficients: Float[Tensor, "*#batch 3 n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 3 n"]:
    """
    快速 SH 旋转
    
    使用混合策略：
    - 低阶 (degree 0-1): 使用解析公式
    - 高阶 (degree 2+): 使用 e3nn
    """
    from math import isqrt
    
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype
    
    *_, n = sh_coefficients.shape
    max_degree = isqrt(n) - 1
    
    # 检查是否需要广播
    rot_shape = rotations.shape[:-2]
    sh_shape = sh_coefficients.shape[:-1]
    
    # 找到实际的旋转批次维度
    effective_rot_dims = sum(1 for r in rot_shape if r != 1)
    
    # 如果旋转没有广播维度，直接使用
    if effective_rot_dims == len(rot_shape):
        # 展平处理
        flat_sh = sh_coefficients.reshape(-1, n)
        flat_rot = rotations.reshape(-1, 3, 3)
        
        # 使用 e3nn 原始实现（对于非广播情况是最优的）
        from src.misc.sh_rotation import rotate_sh
        return rotate_sh(sh_coefficients, rotations)
    
    # 有广播维度，提取有效旋转
    rot_squeezed = rotations
    for i in range(len(rot_shape) - 1, -1, -1):
        if rot_shape[i] == 1 and i >= 2:
            rot_squeezed = rot_squeezed.squeeze(i)
    
    # 预计算 Wigner-D
    wigner_matrices, _ = fast_precompute_wigner_d(rot_squeezed, max_degree)
    
    # 应用旋转
    from src.misc.sh_rotation_optimized import rotate_sh_with_precomputed
    return rotate_sh_with_precomputed(sh_coefficients, wigner_matrices, rotation_batch_dims=2)


# 测试
if __name__ == "__main__":
    import time
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试参数
    b, v = 1, 2
    
    # 创建旋转矩阵
    rot_np = R.random(b * v).as_matrix().reshape(b, v, 3, 3)
    rotations = torch.from_numpy(rot_np).float().to(device)
    
    # 测试 D1 计算
    print("Testing D1 calculation...")
    D1_fast = rotation_matrix_to_wigner_d1(rotations)
    
    from e3nn.o3 import matrix_to_angles, wigner_D
    alpha, beta, gamma = matrix_to_angles(rotations)
    with torch.device(device):
        D1_e3nn = wigner_D(1, alpha, beta, gamma).type(rotations.dtype)
    
    print(f"D1 fast:\n{D1_fast[0, 0]}")
    print(f"D1 e3nn:\n{D1_e3nn[0, 0]}")
    print(f"Max diff: {(D1_fast - D1_e3nn).abs().max().item():.6f}")
    
    # 测量时间
    print("\nTiming comparison...")
    
    # e3nn
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        alpha, beta, gamma = matrix_to_angles(rotations)
        for degree in range(4):
            with torch.device(device):
                D = wigner_D(degree, alpha, beta, gamma)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    print(f"e3nn all degrees: {sum(times)/len(times):.2f} ms")
    
    # Fast (D0 + D1 fast + D2-3 e3nn)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        wigner_matrices, _ = fast_precompute_wigner_d(rotations, max_degree=3)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    print(f"Fast precompute: {sum(times)/len(times):.2f} ms")

