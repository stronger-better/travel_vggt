#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT 深度图生成测试脚本 - MapLilary 数据集版本

功能：
1. 加载 VGGT 模型（从 checkpoint 或 pretrained）
2. 在 maplilary_test 数据集上生成深度图
3. 与 ground truth 深度图进行对比
4. 可视化误差和统计信息

数据格式：
- 图像文件：.jpg 格式
- 深度图文件：.png 格式（16位，与图像文件名相同，扩展名不同）
- 数据位于指定文件夹中，图像和深度图成对出现
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import logging
import glob
import json
from datetime import datetime

# 设置 OpenCV 支持 EXR（虽然这个数据集用 PNG，但保持兼容）
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# 添加路径
sys.path.append("vggt/")
sys.path.append("training/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from training.data.dataset_util import read_depth, read_image_cv2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_depth_metrics(pred_depth, gt_depth, mask):
    """
    计算深度估计的标准评估指标
    
    Args:
        pred_depth: 预测深度图 (H, W)
        gt_depth: 真实深度图 (H, W)
        mask: 有效像素 mask (H, W)
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    # 提取有效像素
    pred_valid = pred_depth[mask]
    gt_valid = gt_depth[mask]
    
    if len(pred_valid) == 0:
        return None
    
    # 避免除以零
    eps = 1e-8
    
    # 1. RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    
    # 2. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred_valid - gt_valid))
    
    # 3. AbsRel (Absolute Relative Error)
    absrel = np.mean(np.abs(pred_valid - gt_valid) / (gt_valid + eps))
    
    # 4. SqRel (Squared Relative Error)
    sqrel = np.mean(((pred_valid - gt_valid) ** 2) / (gt_valid + eps))
    
    # 5. RMSE log
    log_diff = np.log(gt_valid + eps) - np.log(pred_valid + eps)
    rmse_log = np.sqrt(np.mean(log_diff ** 2))
    
    # 6. log10 error
    log10_error = np.mean(np.abs(np.log10(gt_valid + eps) - np.log10(pred_valid + eps)))
    
    # 7. Accuracy metrics: δ < threshold
    # δ = max(pred/gt, gt/pred)
    ratio = np.maximum(pred_valid / (gt_valid + eps), gt_valid / (pred_valid + eps))
    delta1 = np.mean(ratio < 1.25) * 100.0      # δ < 1.25
    delta2 = np.mean(ratio < 1.25 ** 2) * 100.0  # δ < 1.25²
    delta3 = np.mean(ratio < 1.25 ** 3) * 100.0  # δ < 1.25³
    
    # 8. Median absolute error
    median_ae = np.median(np.abs(pred_valid - gt_valid))
    
    # 9. Relative error percentage
    rel_error_percent = np.mean(np.abs(pred_valid - gt_valid) / (gt_valid + eps)) * 100.0
    
    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "AbsRel": float(absrel),
        "SqRel": float(sqrel),
        "RMSE_log": float(rmse_log),
        "log10_error": float(log10_error),
        "delta1": float(delta1),      # δ < 1.25 (%)
        "delta2": float(delta2),      # δ < 1.25² (%)
        "delta3": float(delta3),      # δ < 1.25³ (%)
        "median_AE": float(median_ae),
        "rel_error_percent": float(rel_error_percent),
        "valid_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
        "valid_ratio": float(mask.sum() / mask.size * 100.0)
    }
    
    return metrics


def save_metrics_to_file(metrics_dict, output_path, format="json"):
    """
    将评估指标保存到文件
    
    Args:
        metrics_dict: 包含所有指标的字典
        output_path: 输出文件路径
        format: 文件格式 ("json" 或 "txt")
    """
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    elif format == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("深度估计评估指标\n")
            f.write("=" * 60 + "\n\n")
            
            if "overall" in metrics_dict:
                f.write("总体统计:\n")
                f.write("-" * 60 + "\n")
                overall = metrics_dict["overall"]
                f.write(f"测试样本数: {metrics_dict.get('num_samples', 'N/A')}\n")
                f.write(f"总有效像素数: {overall.get('valid_pixels', 'N/A'):,}\n")
                f.write(f"\n")
                
                f.write("误差指标:\n")
                f.write(f"  RMSE (m):              {overall.get('RMSE', 0):.6f}\n")
                f.write(f"  MAE (m):               {overall.get('MAE', 0):.6f}\n")
                f.write(f"  Median AE (m):         {overall.get('median_AE', 0):.6f}\n")
                f.write(f"  AbsRel:                {overall.get('AbsRel', 0):.6f}\n")
                f.write(f"  SqRel:                 {overall.get('SqRel', 0):.6f}\n")
                f.write(f"  RMSE_log:              {overall.get('RMSE_log', 0):.6f}\n")
                f.write(f"  log10_error:           {overall.get('log10_error', 0):.6f}\n")
                f.write(f"  Rel Error (%):         {overall.get('rel_error_percent', 0):.4f}\n")
                f.write(f"\n")
                
                f.write("准确度指标 (越高越好):\n")
                f.write(f"  δ < 1.25 (%):          {overall.get('delta1', 0):.4f}\n")
                f.write(f"  δ < 1.25² (%):         {overall.get('delta2', 0):.4f}\n")
                f.write(f"  δ < 1.25³ (%):         {overall.get('delta3', 0):.4f}\n")
                f.write(f"\n")
            
            if "per_sample" in metrics_dict and len(metrics_dict["per_sample"]) > 0:
                f.write("\n" + "=" * 60 + "\n")
                f.write("每个样本的详细指标:\n")
                f.write("=" * 60 + "\n\n")
                for idx, sample_metrics in enumerate(metrics_dict["per_sample"]):
                    f.write(f"样本 {idx + 1}: {sample_metrics.get('image_name', 'N/A')}\n")
                    f.write(f"  RMSE: {sample_metrics.get('RMSE', 0):.6f} m, "
                           f"MAE: {sample_metrics.get('MAE', 0):.6f} m, "
                           f"AbsRel: {sample_metrics.get('AbsRel', 0):.6f}, "
                           f"δ1: {sample_metrics.get('delta1', 0):.2f}%\n")
            f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def load_model(checkpoint_path=None, device="cuda"):
    """
    加载 VGGT 模型
    
    Args:
        checkpoint_path: 模型 checkpoint 路径，如果为 None 则加载 pretrained
        device: 设备
    
    Returns:
        model: 加载的模型
    """
    logger.info("Loading VGGT model...")
    model = VGGT()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # 处理不同的 checkpoint 格式
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.info("Loading pretrained model from HuggingFace...")
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        logger.info("Pretrained model loaded successfully")
    
    model.eval()
    model = model.to(device)
    return model


def load_maplilary_data(data_dir, image_ext=".jpg", depth_ext=".npy", depth_scale=1.0, scene_name=None):
    """
    加载 MapLilary 测试数据集
    
    支持两种数据格式：
    1. 新格式：data_dir/scene_name/images/ 和 data_dir/scene_name/depths/
    2. 旧格式：data_dir/ 中直接包含图像和深度图文件对
    
    Args:
        data_dir: 数据目录（例如 processed_maplilary 或 maplilary_test/val）
        image_ext: 图像文件扩展名（默认 ".jpg"）
        depth_ext: 深度图文件扩展名（默认 ".npy"）
        depth_scale: 深度图缩放因子（默认 1.0，.npy 格式通常已经是米单位）
        scene_name: 场景名称（例如 "scene_001"），如果为 None 则自动检测或使用旧格式
    
    Returns:
        list: 包含 (image_path, depth_path, image, depth_map) 的列表
    """
    logger.info(f"Loading MapLilary test data from: {data_dir}")
    
    # 检测数据格式
    images_dir = None
    depths_dir = None
    
    if scene_name:
        # 指定了场景名称，使用新格式
        images_dir = os.path.join(data_dir, scene_name, "images")
        depths_dir = os.path.join(data_dir, scene_name, "depths")
        if not os.path.exists(images_dir) or not os.path.exists(depths_dir):
            logger.warning(f"Scene directories not found: {images_dir} or {depths_dir}")
            logger.warning("Trying to auto-detect scene or use old format...")
            scene_name = None
    
    if scene_name is None:
        # 自动检测场景或使用旧格式
        # 首先尝试检测是否有 scene_xxx 格式的文件夹
        potential_scenes = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("scene_")]
        
        if potential_scenes:
            # 使用第一个找到的场景（如果只有一个）
            if len(potential_scenes) == 1:
                scene_name = potential_scenes[0]
                images_dir = os.path.join(data_dir, scene_name, "images")
                depths_dir = os.path.join(data_dir, scene_name, "depths")
                logger.info(f"Auto-detected scene: {scene_name}")
            else:
                # 多个场景，使用第一个或让用户指定
                scene_name = sorted(potential_scenes)[0]
                images_dir = os.path.join(data_dir, scene_name, "images")
                depths_dir = os.path.join(data_dir, scene_name, "depths")
                logger.info(f"Multiple scenes found, using: {scene_name}")
                logger.info(f"  Available scenes: {potential_scenes}")
        
        # 如果还是没有找到，检查是否直接存在 images/ 和 depths/ 目录
        if not images_dir or not os.path.exists(images_dir):
            if os.path.exists(os.path.join(data_dir, "images")) and os.path.exists(os.path.join(data_dir, "depths")):
                images_dir = os.path.join(data_dir, "images")
                depths_dir = os.path.join(data_dir, "depths")
                logger.info("Using direct images/ and depths/ directories")
    
    # 确定图像和深度图文件位置
    if images_dir and depths_dir and os.path.exists(images_dir) and os.path.exists(depths_dir):
        # 新格式：分别的 images/ 和 depths/ 目录
        logger.info(f"Using new format: images={images_dir}, depths={depths_dir}")
        image_pattern = os.path.join(images_dir, f"*{image_ext}")
        image_files = sorted(glob.glob(image_pattern))
    else:
        # 旧格式：同一目录下的图像和深度图
        logger.info(f"Using old format: single directory with image-depth pairs")
        image_pattern = os.path.join(data_dir, f"*{image_ext}")
        image_files = sorted(glob.glob(image_pattern))
        images_dir = None
        depths_dir = None
    
    logger.info(f"Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        logger.error(f"No image files found with extension {image_ext}")
        logger.error(f"  Searched in: {images_dir if images_dir else data_dir}")
        return []
    
    test_data = []
    
    for image_path in tqdm(image_files, desc="Loading images and depth maps"):
        # 获取图像文件名（不含扩展名）
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        
        # 构建对应的深度图路径
        if images_dir and depths_dir:
            # 新格式：在 depths/ 目录中查找
            depth_path = os.path.join(depths_dir, f"{image_name}{depth_ext}")
        else:
            # 旧格式：在同一目录中查找（替换扩展名）
            depth_path = os.path.splitext(image_path)[0] + depth_ext
        
        # 检查深度图是否存在
        if not os.path.exists(depth_path):
            logger.warning(f"Depth map not found for: {image_basename}")
            logger.warning(f"  Expected: {depth_path}")
            continue
        
        # 加载图像
        image = read_image_cv2(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue
        
        # 加载深度图
        try:
            if depth_ext.lower() == ".npy":
                # .npy 格式：直接加载 numpy 数组
                depth_map = np.load(depth_path).astype(np.float32)
                # 应用深度缩放
                if depth_scale != 1.0:
                    depth_map = depth_map * depth_scale
            else:
                # 其他格式（.png, .exr等）：使用 read_depth 函数
                depth_map = read_depth(depth_path, scale_adjustment=depth_scale)
        except Exception as e:
            logger.warning(f"Failed to load depth map {depth_path}: {e}")
            continue
        
        if depth_map is None:
            logger.warning(f"Failed to load depth map: {depth_path}")
            continue
        
        # 确保深度图是 2D
        if len(depth_map.shape) > 2:
            depth_map = depth_map[..., 0] if depth_map.shape[2] > 0 else depth_map.squeeze()
        elif len(depth_map.shape) < 2:
            logger.warning(f"Invalid depth map shape: {depth_map.shape}")
            continue
        
        # 清理无效值
        depth_map[~np.isfinite(depth_map)] = 0.0
        depth_map[depth_map < 0] = 0.0
        
        test_data.append({
            "image_path": image_path,
            "depth_path": depth_path,
            "image": image,
            "depth_map": depth_map,
        })
    
    logger.info(f"Successfully loaded {len(test_data)} test samples")
    
    if len(test_data) == 0:
        logger.warning("WARNING: No test samples were loaded!")
        logger.warning("This could be due to:")
        logger.warning("  1. Image files not found in the directory")
        logger.warning("  2. Depth map files not found (different extension or naming)")
        logger.warning("  3. Files are missing or corrupted")
        logger.warning("  4. Depth scale mismatch")
        logger.warning("  5. Directory structure mismatch")
    
    return test_data


def predict_depth(model, images, device="cuda", dtype=torch.bfloat16):
    """
    使用 VGGT 模型预测深度图（按照 README 推荐的方式）
    
    Args:
        model: VGGT 模型
        images: 输入图像 tensor，形状 (S, 3, H, W) 或 (B, S, 3, H, W)
        device: 设备
        dtype: 计算数据类型
    
    Returns:
        depth_map: 预测的深度图，形状 (S, H, W, 1) 或 (H, W, 1)
        depth_conf: 深度置信度，形状 (S, H, W) 或 (H, W)
    """
    # 确保有 batch 维度
    if len(images.shape) == 4:
        images = images.unsqueeze(0)  # (S, 3, H, W) -> (1, S, 3, H, W)
    
    images = images.to(device)
    
    with torch.no_grad():
        # 按照 README 的方式：aggregator 在 autocast 内，depth_head 在 autocast 外
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        
        # depth_head 在 autocast 外调用（model.forward 中也是 enabled=False）
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images=images, patch_start_idx=ps_idx
        )
    
    # 调试信息
    logger.debug(f"Model output depth_map shape: {depth_map.shape}")
    logger.debug(f"Model output depth_map range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    
    # depth_head 输出形状是 (B, S, H, W, 1) 和 (B, S, H, W)
    # 移除 batch 维度
    if depth_map.shape[0] == 1:
        depth_map = depth_map.squeeze(0)  # (S, H, W, 1)
        depth_conf = depth_conf.squeeze(0)  # (S, H, W)
    
    # 如果只有一个序列，移除序列维度
    if depth_map.shape[0] == 1:
        depth_map = depth_map.squeeze(0)  # (H, W, 1)
        depth_conf = depth_conf.squeeze(0)  # (H, W)
    
    logger.debug(f"After squeezing depth_map shape: {depth_map.shape}")
    logger.debug(f"After squeezing depth_map range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    
    return depth_map, depth_conf


def visualize_depth_comparison(image, pred_depth, gt_depth, save_path=None):
    """
    可视化深度图对比（只显示 GT、预测深度图和误差图，使用真实深度值）
    
    Args:
        image: 原始图像 (H, W, 3) - 不再使用，但保留参数以兼容
        pred_depth: 预测深度图 (H, W)，单位：米
        gt_depth: 真实深度图 (H, W)，单位：米
        save_path: 保存路径
    """
    # 创建 mask
    mask = (gt_depth > 1e-8) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    
    # 计算误差图（真实误差值，单位：米）
    error_map = np.abs(pred_depth - gt_depth)
    error_map[~mask] = 0
    
    # 准备深度图用于可视化（使用真实值，不归一化）
    pred_depth_vis = pred_depth.copy()
    pred_depth_vis[~mask] = np.nan
    
    gt_depth_vis = gt_depth.copy()
    gt_depth_vis[~mask] = np.nan
    
    # 计算深度范围（用于设置 colormap）
    if mask.sum() > 0:
        depth_min = min(gt_depth[mask].min(), pred_depth[mask].min())
        depth_max = max(gt_depth[mask].max(), pred_depth[mask].max())
    else:
        depth_min = 0.0
        depth_max = 1.0
    
    # 误差图使用真实误差值
    error_map_vis = error_map.copy()
    error_map_vis[~mask] = np.nan
    
    # 计算误差范围（用于设置 colormap）
    if mask.sum() > 0:
        error_max = error_map[mask].max()
    else:
        error_max = 1.0
    
    # 创建可视化 - 只显示三个图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # GT 深度图（真实值）
    im1 = axes[0].imshow(gt_depth_vis, cmap='jet', vmin=depth_min, vmax=depth_max)
    axes[0].set_title("Ground Truth Depth (m)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Depth (m)', rotation=270, labelpad=15)
    
    # 预测深度图（真实值）
    im2 = axes[1].imshow(pred_depth_vis, cmap='jet', vmin=depth_min, vmax=depth_max)
    axes[1].set_title("Predicted Depth (m)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Depth (m)', rotation=270, labelpad=15)
    
    # 误差图（真实误差值）
    im3 = axes[2].imshow(error_map_vis, cmap='hot', vmin=0, vmax=error_max)
    axes[2].set_title("Absolute Error (m)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('Error (m)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test VGGT depth generation on MapLilary dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory containing images and depth maps. "
                             "Supports two formats:\n"
                             "  1. New format: data_dir/scene_001/images/ and data_dir/scene_001/depths/\n"
                             "  2. Old format: data_dir/ with image-depth pairs directly")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Scene name (e.g., 'scene_001'). If None, auto-detects first scene")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (if None, use pretrained)")
    parser.add_argument("--output_dir", type=str, default="maplilary_test_results",
                        help="Output directory for visualizations")
    parser.add_argument("--image_ext", type=str, default=".jpg",
                        help="Image file extension (.jpg, .png, etc.)")
    parser.add_argument("--depth_ext", type=str, default=".npy",
                        help="Depth map file extension (.npy, .png, .exr, etc.)")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Depth scale factor (default: 1.0 for .npy format in meters)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to test (None for all)")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save individual visualizations for each sample")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.checkpoint, device=args.device)
    
    # 加载测试数据
    test_data = load_maplilary_data(
        args.data_dir, 
        args.image_ext, 
        args.depth_ext, 
        args.depth_scale,
        args.scene_name
    )
    
    if len(test_data) == 0:
        logger.error("No test data loaded! Please check:")
        logger.error(f"  1. Data directory exists: {args.data_dir}")
        logger.error(f"  2. Image files with extension {args.image_ext} exist")
        logger.error(f"  3. Depth map files with extension {args.depth_ext} exist")
        logger.error(f"  4. Depth scale is correct: {args.depth_scale}")
        raise ValueError("No test data loaded. Please check the paths and file structure.")
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    logger.info(f"Testing on {len(test_data)} samples")
    
    # 处理每个样本
    processed_count = 0
    if args.device == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    logger.info(f"Using dtype: {dtype}, device: {args.device}")
    
    # 存储所有误差用于统计
    all_errors = []
    all_relative_errors = []
    
    # 存储每个样本的指标
    per_sample_metrics = []
    # 累积总体指标
    all_rmse = []
    all_mae = []
    all_absrel = []
    all_sqrel = []
    all_rmse_log = []
    all_log10_error = []
    all_delta1 = []
    all_delta2 = []
    all_delta3 = []
    all_median_ae = []
    all_rel_error_percent = []
    total_valid_pixels = 0
    total_pixels = 0
    
    for idx, sample in enumerate(tqdm(test_data, desc="Processing samples")):
        image = sample["image"]
        gt_depth = sample["depth_map"]
        image_path = sample["image_path"]
        
        # 预处理图像
        try:
            image_tensor = load_and_preprocess_images([image_path]).to(args.device)
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            continue
        
        # 预测深度
        try:
            pred_depth, pred_conf = predict_depth(model, image_tensor, args.device, dtype)
        except Exception as e:
            logger.error(f"Failed to predict depth for {image_path}: {e}")
            continue
        
        # 转换为 numpy
        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.cpu().numpy()
        if isinstance(pred_conf, torch.Tensor):
            pred_conf = pred_conf.cpu().numpy()
        
        # 调试信息：打印原始形状
        logger.debug(f"Original pred_depth shape: {pred_depth.shape}")
        logger.debug(f"Original pred_depth range: [{pred_depth.min():.4f}, {pred_depth.max():.4f}]")
        
        # 处理预测深度图的形状
        # predict_depth 已经处理了 batch 和 sequence 维度，现在应该是 (H, W, 1) 或 (H, W)
        if len(pred_depth.shape) == 3:
            # (H, W, 1) -> (H, W)
            if pred_depth.shape[-1] == 1:
                pred_depth = pred_depth[..., 0]
            else:
                # 可能是 (1, H, W)，取第一个
                if pred_depth.shape[0] == 1:
                    pred_depth = pred_depth[0]
        elif len(pred_depth.shape) == 2:
            # 已经是 (H, W)，不需要处理
            pass
        else:
            logger.warning(f"Unexpected pred_depth shape: {pred_depth.shape}, attempting to fix...")
            # 尝试修复：取最后一个维度如果是1，则移除
            while len(pred_depth.shape) > 2 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth[..., 0]
            # 如果还有多余的维度，取第一个
            while len(pred_depth.shape) > 2:
                pred_depth = pred_depth[0]
        
        logger.debug(f"Final pred_depth shape: {pred_depth.shape}")
        logger.debug(f"Final pred_depth range: [{pred_depth.min():.4f}, {pred_depth.max():.4f}]")
        
        # 调整预测深度图大小以匹配 GT
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        
        # 创建有效 mask
        mask = (gt_depth > 1e-8) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
        
        # 计算统计信息
        if mask.sum() > 0:
            pred_valid = pred_depth[mask]
            gt_valid = gt_depth[mask]
            
            # 打印预测值和GT值的统计信息
            logger.info(f"\n{'='*60}")
            logger.info(f"Sample {idx+1}/{len(test_data)}: {os.path.basename(image_path)}")
            logger.info(f"{'='*60}")
            logger.info(f"Predicted Depth (m):")
            logger.info(f"  Min:    {pred_valid.min():.4f}")
            logger.info(f"  Max:    {pred_valid.max():.4f}")
            logger.info(f"  Mean:   {pred_valid.mean():.4f}")
            logger.info(f"  Median: {np.median(pred_valid):.4f}")
            logger.info(f"  Std:    {pred_valid.std():.4f}")
            logger.info(f"  Valid pixels: {mask.sum()}/{mask.size} ({mask.sum()/mask.size*100:.1f}%)")
            logger.info(f"")
            logger.info(f"Ground Truth Depth (m):")
            logger.info(f"  Min:    {gt_valid.min():.4f}")
            logger.info(f"  Max:    {gt_valid.max():.4f}")
            logger.info(f"  Mean:   {gt_valid.mean():.4f}")
            logger.info(f"  Median: {np.median(gt_valid):.4f}")
            logger.info(f"  Std:    {gt_valid.std():.4f}")
            logger.info(f"")
            
            # 计算标准深度评估指标
            metrics = compute_depth_metrics(pred_depth, gt_depth, mask)
            if metrics is not None:
                # 保存每个样本的指标
                sample_metrics = metrics.copy()
                sample_metrics["image_name"] = os.path.basename(image_path)
                sample_metrics["sample_idx"] = idx
                per_sample_metrics.append(sample_metrics)
                
                # 累积总体指标
                all_rmse.append(metrics["RMSE"])
                all_mae.append(metrics["MAE"])
                all_absrel.append(metrics["AbsRel"])
                all_sqrel.append(metrics["SqRel"])
                all_rmse_log.append(metrics["RMSE_log"])
                all_log10_error.append(metrics["log10_error"])
                all_delta1.append(metrics["delta1"])
                all_delta2.append(metrics["delta2"])
                all_delta3.append(metrics["delta3"])
                all_median_ae.append(metrics["median_AE"])
                all_rel_error_percent.append(metrics["rel_error_percent"])
                total_valid_pixels += metrics["valid_pixels"]
                total_pixels += metrics["total_pixels"]
            
            # 计算误差统计
            error = np.abs(pred_valid - gt_valid)
            relative_error = error / (gt_valid + 1e-8) * 100  # 相对误差百分比
            
            all_errors.extend(error.tolist())
            all_relative_errors.extend(relative_error.tolist())
            
            # 打印标准评估指标
            if metrics is not None:
                logger.info(f"Depth Estimation Metrics:")
                logger.info(f"  RMSE:        {metrics['RMSE']:.6f} m")
                logger.info(f"  MAE:         {metrics['MAE']:.6f} m")
                logger.info(f"  AbsRel:      {metrics['AbsRel']:.6f}")
                logger.info(f"  SqRel:       {metrics['SqRel']:.6f}")
                logger.info(f"  RMSE_log:    {metrics['RMSE_log']:.6f}")
                logger.info(f"  log10_error: {metrics['log10_error']:.6f}")
                logger.info(f"  δ < 1.25:    {metrics['delta1']:.2f}%")
                logger.info(f"  δ < 1.25²:   {metrics['delta2']:.2f}%")
                logger.info(f"  δ < 1.25³:   {metrics['delta3']:.2f}%")
                logger.info(f"")
            
            logger.info(f"Absolute Error (m):")
            logger.info(f"  Min:    {error.min():.4f}")
            logger.info(f"  Max:    {error.max():.4f}")
            logger.info(f"  Mean:   {error.mean():.4f}")
            logger.info(f"  Median: {np.median(error):.4f}")
            logger.info(f"  Std:    {error.std():.4f}")
            logger.info(f"")
            logger.info(f"Relative Error (%):")
            logger.info(f"  Mean:   {relative_error.mean():.2f}%")
            logger.info(f"  Median: {np.median(relative_error):.2f}%")
            logger.info(f"  Max:    {relative_error.max():.2f}%")
            
            # 检查是否存在单位问题
            if pred_valid.mean() > 0 and gt_valid.mean() > 0:
                ratio = pred_valid.mean() / gt_valid.mean()
                if ratio > 10 or ratio < 0.1:
                    logger.warning(f"⚠️  WARNING: Large scale difference detected!")
                    logger.warning(f"   Pred/GT ratio: {ratio:.4f}")
                    logger.warning(f"   This might indicate a unit mismatch or scale issue.")
                    logger.warning(f"   Try adjusting --depth_scale parameter")
            
            logger.info(f"{'='*60}\n")
        else:
            logger.warning(f"Sample {idx+1}: No valid depth pixels found")
        
        # 保存可视化
        if args.save_individual:
            save_path = os.path.join(args.output_dir, f"sample_{idx:04d}.png")
            visualize_depth_comparison(image, pred_depth, gt_depth, save_path)
        
        processed_count += 1
    
    # 检查是否有处理过的样本
    if processed_count == 0:
        logger.error("No samples were processed! Please check:")
        logger.error("  1. Model loaded correctly")
        logger.error("  2. Images can be loaded and preprocessed")
        logger.error("  3. Model inference works")
        raise ValueError("No samples were processed successfully.")
    
    # 计算总体统计指标
    if len(all_rmse) > 0:
        overall_metrics = {
            "RMSE": float(np.mean(all_rmse)),
            "MAE": float(np.mean(all_mae)),
            "AbsRel": float(np.mean(all_absrel)),
            "SqRel": float(np.mean(all_sqrel)),
            "RMSE_log": float(np.mean(all_rmse_log)),
            "log10_error": float(np.mean(all_log10_error)),
            "delta1": float(np.mean(all_delta1)),
            "delta2": float(np.mean(all_delta2)),
            "delta3": float(np.mean(all_delta3)),
            "median_AE": float(np.median(all_median_ae)),
            "rel_error_percent": float(np.mean(all_rel_error_percent)),
            "valid_pixels": int(total_valid_pixels),
            "total_pixels": int(total_pixels),
            "valid_ratio": float(total_valid_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
        }
        
        # 打印总体统计
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL STATISTICS - Depth Estimation Metrics")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples processed: {processed_count}")
        logger.info(f"Total valid pixels: {total_valid_pixels:,} / {total_pixels:,} ({overall_metrics['valid_ratio']:.2f}%)")
        logger.info(f"")
        logger.info(f"Error Metrics (lower is better):")
        logger.info(f"  RMSE:        {overall_metrics['RMSE']:.6f} m")
        logger.info(f"  MAE:         {overall_metrics['MAE']:.6f} m")
        logger.info(f"  Median AE:   {overall_metrics['median_AE']:.6f} m")
        logger.info(f"  AbsRel:      {overall_metrics['AbsRel']:.6f}")
        logger.info(f"  SqRel:       {overall_metrics['SqRel']:.6f}")
        logger.info(f"  RMSE_log:    {overall_metrics['RMSE_log']:.6f}")
        logger.info(f"  log10_error: {overall_metrics['log10_error']:.6f}")
        logger.info(f"  Rel Error:   {overall_metrics['rel_error_percent']:.4f}%")
        logger.info(f"")
        logger.info(f"Accuracy Metrics (higher is better):")
        logger.info(f"  δ < 1.25:    {overall_metrics['delta1']:.4f}%")
        logger.info(f"  δ < 1.25²:   {overall_metrics['delta2']:.4f}%")
        logger.info(f"  δ < 1.25³:   {overall_metrics['delta3']:.4f}%")
        logger.info(f"{'='*60}\n")
        
        # 保存指标到文件
        metrics_dict = {
            "num_samples": processed_count,
            "overall": overall_metrics,
            "per_sample": per_sample_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存为 JSON
        json_path = os.path.join(args.output_dir, "metrics.json")
        save_metrics_to_file(metrics_dict, json_path, format="json")
        logger.info(f"Metrics saved to JSON: {json_path}")
        
        # 保存为文本
        txt_path = os.path.join(args.output_dir, "metrics.txt")
        save_metrics_to_file(metrics_dict, txt_path, format="txt")
        logger.info(f"Metrics saved to TXT: {txt_path}")
    
    # 打印额外的基本统计
    if len(all_errors) > 0:
        logger.info(f"\nAdditional Statistics:")
        logger.info(f"Absolute Error (m) - Overall:")
        logger.info(f"  Mean:   {np.mean(all_errors):.4f}")
        logger.info(f"  Median: {np.median(all_errors):.4f}")
        logger.info(f"  Std:    {np.std(all_errors):.4f}")
        logger.info(f"")
        logger.info(f"Relative Error (%) - Overall:")
        logger.info(f"  Mean:   {np.mean(all_relative_errors):.2f}%")
        logger.info(f"  Median: {np.median(all_relative_errors):.2f}%")
        logger.info(f"  Std:    {np.std(all_relative_errors):.2f}%")
        logger.info(f"{'='*60}\n")
    
    logger.info(f"\nSuccessfully processed {processed_count} samples")
    
    # 创建汇总可视化（随机选择几个样本）
    if len(test_data) > 0:
        num_show = min(4, len(test_data))
        indices = np.linspace(0, len(test_data) - 1, num_show, dtype=int)
        
        # 只显示三个图：GT、预测、误差
        fig, axes = plt.subplots(num_show, 3, figsize=(18, 6 * num_show))
        if num_show == 1:
            axes = axes.reshape(1, -1)
        
        for row, idx in enumerate(indices):
            sample = test_data[idx]
            gt_depth = sample["depth_map"]
            image_path = sample["image_path"]
            
            # 重新预测（为了可视化）
            image_tensor = load_and_preprocess_images([image_path]).to(args.device)
            pred_depth, _ = predict_depth(model, image_tensor, args.device, dtype)
            
            if isinstance(pred_depth, torch.Tensor):
                pred_depth = pred_depth.cpu().numpy()
            if len(pred_depth.shape) == 4:
                pred_depth = pred_depth[0, ..., 0]
            elif len(pred_depth.shape) == 3:
                pred_depth = pred_depth[0, ..., 0] if pred_depth.shape[-1] == 1 else pred_depth[0]
            
            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
            
            mask = (gt_depth > 1e-8) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
            
            # 准备深度图用于可视化（使用真实值，不归一化）
            pred_depth_vis = pred_depth.copy()
            pred_depth_vis[~mask] = np.nan
            
            gt_depth_vis = gt_depth.copy()
            gt_depth_vis[~mask] = np.nan
            
            # 计算深度范围（用于设置 colormap）
            if mask.sum() > 0:
                depth_min = min(gt_depth[mask].min(), pred_depth[mask].min())
                depth_max = max(gt_depth[mask].max(), pred_depth[mask].max())
            else:
                depth_min = 0.0
                depth_max = 1.0
            
            # 误差图（真实误差值，单位：米）
            error_map = np.abs(pred_depth - gt_depth)
            error_map[~mask] = np.nan
            
            # 计算误差范围
            if mask.sum() > 0:
                error_max = error_map[mask].max()
            else:
                error_max = 1.0
            
            # GT 深度图（真实值）
            im1 = axes[row, 0].imshow(gt_depth_vis, cmap='jet', vmin=depth_min, vmax=depth_max)
            axes[row, 0].set_title("GT Depth (m)", fontsize=12, fontweight='bold')
            axes[row, 0].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[row, 0], fraction=0.046, pad=0.04)
            cbar1.set_label('Depth (m)', rotation=270, labelpad=10, fontsize=9)
            
            # 预测深度图（真实值）
            im2 = axes[row, 1].imshow(pred_depth_vis, cmap='jet', vmin=depth_min, vmax=depth_max)
            axes[row, 1].set_title("Predicted Depth (m)", fontsize=12, fontweight='bold')
            axes[row, 1].axis('off')
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], fraction=0.046, pad=0.04)
            cbar2.set_label('Depth (m)', rotation=270, labelpad=10, fontsize=9)
            
            # 误差图（真实误差值）
            im3 = axes[row, 2].imshow(error_map, cmap='hot', vmin=0, vmax=error_max)
            axes[row, 2].set_title("Absolute Error (m)", fontsize=12, fontweight='bold')
            axes[row, 2].axis('off')
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)
            cbar3.set_label('Error (m)', rotation=270, labelpad=10, fontsize=9)
        
        plt.tight_layout()
        summary_path = os.path.join(args.output_dir, "summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved summary visualization to: {summary_path}")
        plt.close()
    
    logger.info("\n" + "="*60)
    logger.info("Testing completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

