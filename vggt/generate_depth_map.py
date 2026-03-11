# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
只生成深度图的脚本
用于从输入图像生成深度图并保存
"""

import argparse
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT深度图生成工具")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含输入图像的目录路径（目录中应包含images文件夹，或直接指定图像目录）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录路径，如果不指定则使用input_dir/depth_maps"
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        default=True,
        help="保存numpy格式的深度图（.npy文件）"
    )
    parser.add_argument(
        "--save_gray",
        action="store_true",
        default=False,
        help="保存灰度格式的深度图（.png文件）"
    )
    parser.add_argument(
        "--save_confidence",
        action="store_true",
        default=False,
        help="同时保存深度置信度图"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备，'cuda'或'cpu'，默认自动选择"
    )
    return parser.parse_args()


def depth_to_grayscale(depth_map):
    """
    将深度图转换为灰度图
    
    Args:
        depth_map: 深度图数组，形状为 (H, W) 或 (H, W, 1)
    
    Returns:
        numpy数组，形状为 (H, W)，值域 [0, 255]，dtype为uint8
    """
    # 确保是2D数组
    if len(depth_map.shape) == 3:
        depth_map = depth_map.squeeze(-1)
    
    # 归一化到 [0, 1]
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max > depth_min:
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_map)
    
    # 转换为 [0, 255] 范围的uint8
    depth_gray = (depth_normalized * 255).astype(np.uint8)
    
    return depth_gray


def generate_depth_maps(model, images, device, dtype, resolution=518):
    """
    使用VGGT模型生成深度图
    
    Args:
        model: VGGT模型
        images: 输入图像张量，形状 [B, 3, H, W]
        device: 计算设备
        dtype: 数据类型
        resolution: 目标分辨率，默认518
    
    Returns:
        tuple: (depth_maps, depth_confs)
            - depth_maps: 深度图，形状 [S, H, W, 1]
            - depth_confs: 深度置信度，形状 [S, H, W]
    """
    assert len(images.shape) == 4
    assert images.shape[1] == 3
    
    # 调整到VGGT固定分辨率518x518
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # 添加batch维度 [1, S, 3, H, W]
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        
        # 预测深度图
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    
    # 移除batch维度并转换为numpy
    depth_map = depth_map.squeeze(0).cpu().numpy()  # [S, H, W, 1]
    depth_conf = depth_conf.squeeze(0).cpu().numpy()  # [S, H, W]
    
    return depth_map, depth_conf


def main():
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 设置数据类型
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    print(f"使用设备: {device}")
    print(f"使用数据类型: {dtype}")
    
    # 加载模型
    print("正在加载VGGT模型...")
    model = VGGT(enable_depth=True)  # 确保启用深度输出
    _URL = "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("模型加载完成")
    
    # 查找图像文件
    input_dir = Path(args.input_dir)
    if (input_dir / "images").exists():
        image_dir = input_dir / "images"
    else:
        image_dir = input_dir
    
    # 支持的图像格式
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(str(image_dir / ext)))
    
    if len(image_path_list) == 0:
        raise ValueError(f"在 {image_dir} 中未找到图像文件")
    
    image_path_list = sorted(image_path_list)
    print(f"找到 {len(image_path_list)} 张图像")
    
    # 加载和预处理图像
    print("正在加载和预处理图像...")
    images, _ = load_and_preprocess_images_square(image_path_list, target_size=1024)
    images = images.to(device)
    print(f"图像形状: {images.shape}")
    
    # 生成深度图
    print("正在生成深度图...")
    depth_maps, depth_confs = generate_depth_maps(model, images, device, dtype, resolution=518)
    print(f"深度图形状: {depth_maps.shape}")
    print(f"深度置信度形状: {depth_confs.shape}")
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = input_dir / "depth_maps"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 保存深度图
    num_images = depth_maps.shape[0]
    for i in range(num_images):
        image_name = Path(image_path_list[i]).stem
        
        # 提取单张深度图
        depth_map = depth_maps[i]  # [H, W, 1]
        depth_conf = depth_confs[i]  # [H, W]
        
        # 保存numpy格式
        if args.save_npy:
            npy_path = output_dir / f"{image_name}_depth.npy"
            np.save(npy_path, depth_map)
            print(f"已保存: {npy_path}")
            
            if args.save_confidence:
                conf_path = output_dir / f"{image_name}_depth_conf.npy"
                np.save(conf_path, depth_conf)
                print(f"已保存: {conf_path}")
        
        # 保存灰度图格式
        if args.save_gray:
            depth_gray = depth_to_grayscale(depth_map)
            gray_path = output_dir / f"{image_name}_depth.png"
            Image.fromarray(depth_gray, mode='L').save(gray_path)
            print(f"已保存: {gray_path}")
            
            if args.save_confidence:
                conf_gray = depth_to_grayscale(depth_conf)
                conf_gray_path = output_dir / f"{image_name}_depth_conf.png"
                Image.fromarray(conf_gray, mode='L').save(conf_gray_path)
                print(f"已保存: {conf_gray_path}")
    
    print(f"\n完成！共处理 {num_images} 张图像")
    print(f"所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()