import argparse
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def preprocess_image_depth_pairs(in_path, out_path, image_ext=".jpg", depth_ext=".png", depth_scale=100.0):
    """
    处理图像和深度图对，转换为VGGT训练所需的格式。
    
    Args:
        in_path: 输入目录路径，支持两种结构：
            - 结构1: in_path/images/ 和 in_path/depths/
            - 结构2: in_path/ 中直接包含 image.jpg 和 image.png 文件
        out_path: 输出目录路径
        image_ext: 图像文件扩展名，默认 ".jpg"
        depth_ext: 深度图文件扩展名，默认 ".png"
        depth_scale: 深度图缩放因子（将深度值除以此值转换为米），默认 100.0（厘米转米）
    """
    # 检测输入目录结构
    images_dir = os.path.join(in_path, "images")
    depths_dir = os.path.join(in_path, "depths")
    
    if os.path.exists(images_dir) and os.path.exists(depths_dir):
        # 结构1: 分别的 images/ 和 depths/ 目录
        image_files = sorted(glob.glob(os.path.join(images_dir, f"*{image_ext}")))
        print(f"Found {len(image_files)} images in {images_dir}")
    else:
        # 结构2: 同一目录下的图像和深度图
        all_files = glob.glob(os.path.join(in_path, f"*{image_ext}"))
        image_files = sorted(all_files)
        print(f"Found {len(image_files)} images in {in_path}")
    
    if len(image_files) == 0:
        print(f"Error: No images found with extension {image_ext} in {in_path}")
        return
    
    # 创建输出目录结构（按场景组织，这里使用单个场景）
    scene_name = "scene_001"
    out_scene_dir = os.path.join(out_path, scene_name)
    output_img_dir = os.path.join(out_scene_dir, "images")
    output_depth_dir = os.path.join(out_scene_dir, "depths")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    
    num_processed = 0
    num_skipped = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # 获取图像文件名（不含扩展名）
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        
        # 查找对应的深度图
        if os.path.exists(images_dir) and os.path.exists(depths_dir):
            # 结构1: 在 depths/ 目录中查找
            depth_path = os.path.join(depths_dir, f"{image_name}{depth_ext}")
        else:
            # 结构2: 在同一目录中查找
            depth_path = os.path.join(in_path, f"{image_name}{depth_ext}")
        
        if not os.path.exists(depth_path):
            print(f"Warning: Depth map not found for {image_basename}, skipping...")
            num_skipped += 1
            continue
        
        try:
            # 加载图像和深度图
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            depth = np.array(Image.open(depth_path))
            
            # 如果深度图是3通道，转换为单通道（取第一个通道或转换为灰度）
            if len(depth.shape) == 3:
                depth = depth[:, :, 0] if depth.shape[2] >= 1 else depth.mean(axis=2)
            
            # 深度单位转换（除以缩放因子转换为米）
            depth = depth.astype(np.float32) / depth_scale
            
            # 将图像resize到深度图尺寸
            depth_height, depth_width = depth.shape
            image = image.resize((depth_width, depth_height), Image.Resampling.LANCZOS)
            
            # 保存处理后的图像和深度图
            output_image_path = os.path.join(output_img_dir, f"{image_name}.jpg")
            output_depth_path = os.path.join(output_depth_dir, f"{image_name}.npy")
            
            image.save(output_image_path, quality=95)
            np.save(output_depth_path, depth)
            
            num_processed += 1
            
        except Exception as e:
            print(f"Error processing {image_basename}: {e}")
            num_skipped += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {num_processed} image-depth pairs")
    print(f"  Skipped: {num_skipped} files")
    print(f"  Output directory: {out_scene_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="处理图像和深度图数据集，转换为VGGT训练格式"
    )
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="输入目录路径。支持两种结构：\n"
             "  1. in_path/images/ 和 in_path/depths/ 分别存放图像和深度图\n"
             "  2. in_path/ 中直接包含 image.jpg 和 image.png 文件对",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="输出目录路径",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="图像文件扩展名，默认 '.jpg'",
    )
    parser.add_argument(
        "--depth_ext",
        type=str,
        default=".png",
        help="深度图文件扩展名，默认 '.png'",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=100.0,
        help="深度图缩放因子（将深度值除以此值转换为米），默认 100.0（厘米转米）。"
             "如果深度图已经是米为单位，设置为 1.0",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.in_path):
        print(f"Error: Input path does not exist: {args.in_path}")
        exit(1)
    
    os.makedirs(args.out_path, exist_ok=True)
    
    preprocess_image_depth_pairs(
        args.in_path,
        args.out_path,
        image_ext=args.image_ext,
        depth_ext=args.depth_ext,
        depth_scale=args.depth_scale,
    )
    
    print("Processing done.")