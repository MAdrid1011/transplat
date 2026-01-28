#!/usr/bin/env python3
"""
高斯基元贡献分析脚本

用于 Transplat 和 pixelSplat 的真模型、真数据推理中统计高斯基元贡献比例

功能:
- 统计特定视角下真正有效参与（贡献比较高）的高斯基元占总高斯基元的百分比
- 支持多种贡献度阈值
- 输出详细的统计报告

使用方法:
    # Transplat
    python scripts/analyze_gaussian_contribution.py --model transplat --checkpoint checkpoints/re10k.ckpt

    # pixelSplat
    python scripts/analyze_gaussian_contribution.py --model pixelsplat --checkpoint pixelsplat/checkpoints/re10k.ckpt
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def analyze_adjacent_gaussians(gaussians):
    """分析相邻像素生成的高斯基元的相似性"""
    means = gaussians.means[0]  # [N, 3]
    covariances = gaussians.covariances[0]  # [N, 3, 3]
    opacities = gaussians.opacities[0]  # [N]
    harmonics = gaussians.harmonics[0]  # [N, 3, K]
    
    N = means.shape[0]
    # 假设是 256x256 双视图
    H, W = 256, 256
    n_views = N // (H * W)
    
    print(f"\n  ========== 相邻像素高斯相似性分析 ==========")
    print(f"  总高斯数: {N}, 视图数: {n_views}, 每视图: {H}x{W}")
    
    # 重塑为 [n_views, H, W, ...]
    means_2d = means.view(n_views, H, W, 3)
    cov_2d = covariances.view(n_views, H, W, 3, 3)
    op_2d = opacities.view(n_views, H, W)
    
    # === 位置差异 ===
    h_pos_diff = (means_2d[:, :, 1:, :] - means_2d[:, :, :-1, :]).norm(dim=-1)
    v_pos_diff = (means_2d[:, 1:, :, :] - means_2d[:, :-1, :, :]).norm(dim=-1)
    all_pos_diff = torch.cat([h_pos_diff.flatten(), v_pos_diff.flatten()])
    
    scene_scale = means.std()
    rel_pos_diff = all_pos_diff / scene_scale
    
    print(f"\n  【位置差异】")
    print(f"  场景尺度 (std): {scene_scale:.4f}")
    print(f"  相邻高斯位置差异: mean={all_pos_diff.mean():.4f}, median={all_pos_diff.median():.4f}")
    print(f"  相对位置差异分布:")
    for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        pct = (rel_pos_diff < t).float().mean() * 100
        print(f"    < {t*100:.0f}% 场景尺度: {pct:.1f}%")
    
    # === 协方差差异 ===
    h_cov_diff = (cov_2d[:, :, 1:, :, :] - cov_2d[:, :, :-1, :, :]).norm(dim=(-2,-1))
    v_cov_diff = (cov_2d[:, 1:, :, :, :] - cov_2d[:, :-1, :, :, :]).norm(dim=(-2,-1))
    all_cov_diff = torch.cat([h_cov_diff.flatten(), v_cov_diff.flatten()])
    
    cov_scale = covariances.norm(dim=(-2,-1)).mean()
    rel_cov_diff = all_cov_diff / cov_scale
    
    print(f"\n  【协方差(形状)差异】")
    print(f"  协方差尺度 (mean norm): {cov_scale:.6f}")
    print(f"  相邻高斯协方差差异: mean={all_cov_diff.mean():.6f}, median={all_cov_diff.median():.6f}")
    print(f"  相对协方差差异分布:")
    for t in [0.01, 0.05, 0.1, 0.2, 0.5]:
        pct = (rel_cov_diff < t).float().mean() * 100
        print(f"    < {t*100:.0f}%: {pct:.1f}%")
    
    # === Opacity 差异 ===
    h_op_diff = (op_2d[:, :, 1:] - op_2d[:, :, :-1]).abs()
    v_op_diff = (op_2d[:, 1:, :] - op_2d[:, :-1, :]).abs()
    all_op_diff = torch.cat([h_op_diff.flatten(), v_op_diff.flatten()])
    
    print(f"\n  【Opacity 差异】")
    print(f"  Opacity 范围: [{opacities.min():.3f}, {opacities.max():.3f}]")
    print(f"  相邻高斯 Opacity 差异: mean={all_op_diff.mean():.4f}, median={all_op_diff.median():.4f}")
    for t in [0.01, 0.05, 0.1, 0.2]:
        pct = (all_op_diff < t).float().mean() * 100
        print(f"    < {t}: {pct:.1f}%")
    
    # === 球谐系数差异 ===
    sh_2d = harmonics.view(n_views, H, W, harmonics.shape[1], harmonics.shape[2])
    h_sh_diff = (sh_2d[:, :, 1:, :, :] - sh_2d[:, :, :-1, :, :]).norm(dim=(-2,-1))
    v_sh_diff = (sh_2d[:, 1:, :, :, :] - sh_2d[:, :-1, :, :, :]).norm(dim=(-2,-1))
    all_sh_diff = torch.cat([h_sh_diff.flatten(), v_sh_diff.flatten()])
    
    sh_scale = harmonics.norm(dim=(-2,-1)).mean()
    rel_sh_diff = all_sh_diff / sh_scale
    
    print(f"\n  【球谐系数(颜色)差异】")
    print(f"  球谐尺度 (mean norm): {sh_scale:.4f}")
    print(f"  相邻高斯球谐差异: mean={all_sh_diff.mean():.4f}, median={all_sh_diff.median():.4f}")
    for t in [0.01, 0.05, 0.1, 0.2, 0.5]:
        pct = (rel_sh_diff < t).float().mean() * 100
        print(f"    < {t*100:.0f}%: {pct:.1f}%")
    
    # === 综合可合并估算 ===
    print(f"\n  【可合并高斯估算】")
    print(f"  (相邻像素对总数: {all_pos_diff.shape[0]:,})")
    
    # 不同阈值组合
    criteria = [
        ("宽松", 0.1, 0.2, 0.1, 0.2),
        ("中等", 0.05, 0.1, 0.05, 0.1),
        ("严格", 0.02, 0.05, 0.02, 0.05),
    ]
    
    for name, pos_t, cov_t, op_t, sh_t in criteria:
        mergeable = (
            (rel_pos_diff < pos_t) & 
            (rel_cov_diff < cov_t) & 
            (all_op_diff < op_t) &
            (rel_sh_diff < sh_t)
        ).float().mean() * 100
        print(f"  {name} (位置<{pos_t*100:.0f}%, 协方差<{cov_t*100:.0f}%, opacity<{op_t}, SH<{sh_t*100:.0f}%): {mergeable:.1f}%")
    
    # 估算合并后的高斯数量
    strict_mergeable_ratio = (
        (rel_pos_diff < 0.05) & 
        (rel_cov_diff < 0.1) & 
        (all_op_diff < 0.05) &
        (rel_sh_diff < 0.1)
    ).float().mean().item()
    
    # 假设相邻可合并的高斯可以 2:1 合并
    estimated_reduction = strict_mergeable_ratio * 0.5  # 每对可合并的贡献 50% 减少
    new_gaussian_count = int(N * (1 - estimated_reduction))
    
    print(f"\n  【HBM 节省估算】")
    print(f"  原始高斯数: {N:,}")
    print(f"  严格可合并相邻对比例: {strict_mergeable_ratio*100:.1f}%")
    print(f"  估算合并后高斯数: {new_gaussian_count:,} (减少 {(1-new_gaussian_count/N)*100:.1f}%)")
    print(f"  估算 HBM 节省: {(1-new_gaussian_count/N)*44:.1f} MB (原始 44 MB)")
    print(f"  ================================================\n")


def load_transplat_model(checkpoint_path: str, device: str = 'cuda'):
    """加载 TransPlat 模型"""
    from src.config import load_typed_root_config
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.global_cfg import set_cfg
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config_path = str(Path(__file__).parent.parent / "config")
    GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg_dict = compose(config_name="main", overrides=["+experiment=re10k"])
    
    cfg_dict.mode = 'test'
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    losses = get_losses(cfg.loss)
    step_tracker = StepTracker()
    
    model = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train,
        encoder, encoder_visualizer, decoder, losses, step_tracker
    )
    
    state_dict = ckpt.get('state_dict', ckpt)
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() 
                      if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_pixelsplat_model(checkpoint_path: str, device: str = 'cuda'):
    """加载 pixelSplat 模型"""
    # 切换到 pixelsplat 目录
    pixelsplat_path = Path(__file__).parent.parent / "pixelsplat"
    sys.path.insert(0, str(pixelsplat_path))
    
    from src.config import load_typed_root_config
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.global_cfg import set_cfg
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config_path = str(pixelsplat_path / "config")
    GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg_dict = compose(config_name="main", overrides=["+experiment=re10k"])
    
    cfg_dict.mode = 'test'
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    losses = get_losses(cfg.loss)
    step_tracker = StepTracker()
    
    model = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train,
        encoder, encoder_visualizer, decoder, losses, step_tracker
    )
    
    state_dict = ckpt.get('state_dict', ckpt)
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() 
                      if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_dataset(model_type: str, device: str = 'cuda'):
    """加载数据集"""
    if model_type == 'transplat':
        from src.config import load_typed_root_config
        from src.dataset.data_module import DataModule, get_data_shim
        from src.global_cfg import set_cfg
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        config_path = str(Path(__file__).parent.parent / "config")
        GlobalHydra.instance().clear()
        
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg_dict = compose(config_name="main", overrides=["+experiment=re10k"])
        
        cfg_dict.mode = 'test'
        set_cfg(cfg_dict)
        cfg = load_typed_root_config(cfg_dict)
        
        data_module = DataModule(cfg.dataset, cfg.data_loader)
        data_module.setup("test")
        test_loader = data_module.test_dataloader()
        
    else:  # pixelsplat
        pixelsplat_path = Path(__file__).parent.parent / "pixelsplat"
        sys.path.insert(0, str(pixelsplat_path))
        
        from src.config import load_typed_root_config
        from src.dataset.data_module import DataModule, get_data_shim
        from src.global_cfg import set_cfg
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        config_path = str(pixelsplat_path / "config")
        GlobalHydra.instance().clear()
        
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg_dict = compose(config_name="main", overrides=["+experiment=re10k"])
        
        cfg_dict.mode = 'test'
        set_cfg(cfg_dict)
        cfg = load_typed_root_config(cfg_dict)
        
        data_module = DataModule(cfg.dataset, cfg.data_loader)
        data_module.setup("test")
        test_loader = data_module.test_dataloader()
    
    return test_loader


def analyze_gaussian_contribution(
    model,
    model_type: str,
    device: str = 'cuda',
    num_samples: int = 5,
    opacity_thresholds: list = [0.001, 0.01, 0.05, 0.1, 0.2],
):
    """
    分析高斯基元贡献比例
    
    Args:
        model: 加载的模型
        model_type: 'transplat' 或 'pixelsplat'
        device: 设备
        num_samples: 测试场景数量
        opacity_thresholds: 不同的透明度阈值列表
    """
    from einops import rearrange
    
    print(f"\n加载 {model_type.upper()} 数据集...")
    test_loader = load_dataset(model_type, device)
    
    # 获取 data shim
    if model_type == 'transplat':
        from src.dataset.data_module import get_data_shim
    else:
        pixelsplat_path = Path(__file__).parent.parent / "pixelsplat"
        sys.path.insert(0, str(pixelsplat_path))
        from src.dataset.data_module import get_data_shim
    
    data_shim = get_data_shim(model.encoder)
    
    print(f"处理 {num_samples} 个场景...\n")
    
    # 用于存储所有场景的统计数据
    all_scene_stats = []
    
    model.eval()
    
    # Warmup
    print("Warmup...")
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv 
                    for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in batch.items()
            }
            batch = data_shim(batch)
            context = batch["context"]
            _ = model.encoder(context, global_step=0)
            break
    torch.cuda.synchronize()
    print("Warmup 完成\n")
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_samples:
                break
            
            # 移动数据到 GPU
            batch = {
                k: {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv 
                    for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in batch.items()
            }
            
            batch = data_shim(batch)
            
            scene_name = batch['scene'][0] if 'scene' in batch else f"scene_{idx:04d}"
            
            print(f"处理场景 [{idx+1}/{num_samples}]: {scene_name}")
            
            context = batch["context"]
            
            # Encoder 推理
            gaussians = model.encoder(context, global_step=0)
            
            total_gaussians = gaussians.means.shape[1]
            print(f"  总高斯基元数: {total_gaussians:,}")
            
            # ========== 相邻高斯相似性分析 ==========
            if idx == 0:  # 只对第一个场景详细分析
                analyze_adjacent_gaussians(gaussians)
            
            # 渲染并统计
            target_h, target_w = batch["target"]["image"].shape[-2:]
            num_target_views = batch["target"]["extrinsics"].shape[1]
            
            # 只使用部分视角进行统计（避免时间过长）
            max_views_for_stats = min(20, num_target_views)
            
            scene_stats = {
                'scene': scene_name,
                'total_gaussians': total_gaussians,
                'num_views': max_views_for_stats,
                'threshold_stats': {}
            }
            
            for threshold in opacity_thresholds:
                # 使用带统计的渲染方法
                if hasattr(model.decoder, 'forward_with_gaussian_stats'):
                    _, stats_list = model.decoder.forward_with_gaussian_stats(
                        gaussians,
                        batch["target"]["extrinsics"][:, :max_views_for_stats],
                        batch["target"]["intrinsics"][:, :max_views_for_stats],
                        batch["target"]["near"][:, :max_views_for_stats],
                        batch["target"]["far"][:, :max_views_for_stats],
                        (target_h, target_w),
                        depth_mode=None,
                        opacity_threshold=threshold,
                    )
                    
                    visible_ratios = [s.visible_ratio for s in stats_list]
                    high_contrib_ratios = [s.high_contribution_ratio for s in stats_list]
                    
                    scene_stats['threshold_stats'][threshold] = {
                        'avg_visible_ratio': np.mean(visible_ratios),
                        'min_visible_ratio': np.min(visible_ratios),
                        'max_visible_ratio': np.max(visible_ratios),
                        'avg_high_contrib_ratio': np.mean(high_contrib_ratios),
                        'min_high_contrib_ratio': np.min(high_contrib_ratios),
                        'max_high_contrib_ratio': np.max(high_contrib_ratios),
                    }
                else:
                    print(f"  警告: decoder 不支持统计方法")
                    break
            
            all_scene_stats.append(scene_stats)
            
            # 打印当前场景统计
            print(f"\n  【场景 {scene_name} 高斯贡献统计】")
            print(f"  ───────────────────────────────────────────────────────────────")
            print(f"  {'阈值':<10} {'可见比例 (平均)':<18} {'高贡献比例 (平均)':<20} {'高贡献比例范围':<25}")
            print(f"  ───────────────────────────────────────────────────────────────")
            
            for threshold, stats in scene_stats['threshold_stats'].items():
                print(f"  {threshold:<10.3f} "
                      f"{stats['avg_visible_ratio']*100:<18.2f}% "
                      f"{stats['avg_high_contrib_ratio']*100:<20.2f}% "
                      f"[{stats['min_high_contrib_ratio']*100:.2f}% - {stats['max_high_contrib_ratio']*100:.2f}%]")
            
            print()
            
            # 清理 GPU 内存
            del gaussians
            torch.cuda.empty_cache()
    
    # ========== 打印总结 ==========
    print("\n" + "=" * 100)
    print(f"  ███ {model_type.upper()} 高斯基元贡献分析 (Challenge 2) ███")
    print("=" * 100)
    
    # 计算所有场景的平均值
    print(f"\n  统计场景数: {len(all_scene_stats)}")
    avg_total_gaussians = np.mean([s['total_gaussians'] for s in all_scene_stats])
    print(f"  平均总高斯数: {avg_total_gaussians:,.0f}")
    
    # 估算每高斯字节数（假设 sh_degree=4）
    bytes_per_gaussian = 352  # 12 + 36 + 300 + 4
    total_mb = avg_total_gaussians * bytes_per_gaussian / 1024 / 1024
    print(f"  每高斯字节: {bytes_per_gaussian} B")
    print(f"  总数据量: {total_mb:.2f} MB")
    
    print(f"\n  【按 Opacity 阈值分析高斯贡献】")
    print(f"  ┌{'─'*12}┬{'─'*15}┬{'─'*15}┬{'─'*18}┬{'─'*20}┐")
    print(f"  │{'Opacity阈值':^12}│{'高贡献比例':^15}│{'可节省HBM':^15}│{'节省流量(MB)':^18}│{'贡献范围':^20}│")
    print(f"  ├{'─'*12}┼{'─'*15}┼{'─'*15}┼{'─'*18}┼{'─'*20}┤")
    
    for threshold in opacity_thresholds:
        all_high_contrib = []
        all_min_high = []
        all_max_high = []
        
        for scene_stats in all_scene_stats:
            if threshold in scene_stats['threshold_stats']:
                stats = scene_stats['threshold_stats'][threshold]
                all_high_contrib.append(stats['avg_high_contrib_ratio'])
                all_min_high.append(stats['min_high_contrib_ratio'])
                all_max_high.append(stats['max_high_contrib_ratio'])
        
        if all_high_contrib:
            contrib_pct = np.mean(all_high_contrib) * 100
            save_pct = 100 - contrib_pct
            save_mb = total_mb * save_pct / 100
            range_str = f"{np.min(all_min_high)*100:.1f}-{np.max(all_max_high)*100:.1f}%"
            print(f"  │  >= {threshold:<6.2f}│{contrib_pct:>12.1f}%  │{save_pct:>12.1f}%  │{save_mb:>15.2f}  │{range_str:^20}│")
    
    print(f"  └{'─'*12}┴{'─'*15}┴{'─'*15}┴{'─'*18}┴{'─'*20}┘")
    
    print("=" * 100)
    
    # 关键发现
    print("\n  【关键发现】")
    
    # 找到不同阈值的数据
    threshold_data = {}
    for threshold in opacity_thresholds:
        all_high_contrib = []
        for scene_stats in all_scene_stats:
            if threshold in scene_stats['threshold_stats']:
                all_high_contrib.append(
                    scene_stats['threshold_stats'][threshold]['avg_high_contrib_ratio']
                )
        if all_high_contrib:
            threshold_data[threshold] = np.mean(all_high_contrib)
    
    # 找出最有意义的阈值点
    if 0.5 in threshold_data:
        ratio_05 = threshold_data[0.5] * 100
        save_05 = 100 - ratio_05
        print(f"  • Opacity >= 0.5: 高贡献高斯 {ratio_05:.1f}%, 可节省 {save_05:.1f}% HBM ({total_mb * save_05 / 100:.2f} MB)")
    
    if 0.7 in threshold_data:
        ratio_07 = threshold_data[0.7] * 100
        save_07 = 100 - ratio_07
        print(f"  • Opacity >= 0.7: 核心高斯 {ratio_07:.1f}%, 可节省 {save_07:.1f}% HBM ({total_mb * save_07 / 100:.2f} MB)")
    
    print(f"\n  【优化建议】")
    print(f"  • 核心高斯 (opacity >= 0.7): 完整存储, 优先加载")
    print(f"  • 辅助高斯 (opacity < 0.7): 可使用紧凑表示或按需加载")
    print(f"  • Encoder 已知每个高斯的 opacity, 可在存储阶段进行分层")
    
    print()
    
    return all_scene_stats


def main():
    parser = argparse.ArgumentParser(description='高斯基元贡献分析')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['transplat', 'mvsplat', 'depthsplat', 'pixelsplat'],
                        help='模型类型')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='测试场景数量')
    parser.add_argument('--thresholds', type=float, nargs='+', 
                        default=[0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                        help='透明度阈值列表')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print(f"高斯基元贡献分析 - {args.model.upper()}")
    print("=" * 100)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  场景数量:   {args.num_samples}")
    print(f"  阈值列表:   {args.thresholds}")
    print()
    
    # 加载模型
    print(f"加载 {args.model} 模型...")
    if args.model == 'transplat':
        model = load_transplat_model(args.checkpoint, args.device)
    elif args.model == 'mvsplat':
        # mvsplat 使用与 transplat 相同的加载方式，但从 mvsplat-main 目录
        import os
        os.chdir(str(Path(__file__).parent.parent / "mvsplat-main"))
        model = load_transplat_model(args.checkpoint, args.device)
    elif args.model == 'depthsplat':
        # depthsplat 使用与 transplat 相同的加载方式，但从 depthsplat-main 目录
        import os
        os.chdir(str(Path(__file__).parent.parent / "depthsplat-main"))
        model = load_transplat_model(args.checkpoint, args.device)
    else:
        model = load_pixelsplat_model(args.checkpoint, args.device)
    
    # 运行分析
    analyze_gaussian_contribution(
        model,
        args.model,
        args.device,
        args.num_samples,
        args.thresholds,
    )


if __name__ == '__main__':
    main()
