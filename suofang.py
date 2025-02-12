import cv2
import os
from PIL import Image
import numpy as np
from scipy.ndimage import zoom

# 方法1：使用OpenCV
def resize_half_opencv(input_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误：无法读取图像 {input_path}")
        return

    # 获取新尺寸（原尺寸的一半）
    h, w = img.shape[:2]
    new_size = (w//2, h//2)

    # 缩放图像
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # 保存结果
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    cv2.imwrite(output_path, resized)
    print(f"已保存：{output_path}")
    
    
def resize_depth_npy(input_path, output_dir, method='nearest'):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        depth = np.load(input_path)
        
        # 新增维度处理逻辑
        if depth.ndim == 3:
            # 去除单通道维度（假设是形状为 [H, W, 1] 的情况）
            depth = depth.squeeze(axis=-1)
            print(f"调整维度：{input_path} 原始形状 {depth.shape} -> 压缩后 {depth.shape}")
            
        if depth.ndim != 2:
            # 提供更详细的错误信息
            raise ValueError(
                f"无效的深度图维度 {depth.ndim},"
                f"期望二维数组。文件路径：{input_path}\n"
                "可能原因：\n"
                "1. 文件损坏\n"
                "2. 数据保存格式错误\n"
                "3. 多通道深度图（非常规情况）"
            )
            
        # 计算新尺寸（长宽各减半）
        new_shape = (depth.shape[0]//2, depth.shape[1]//2)
        
        # 方法1：OpenCV最近邻插值（推荐用于深度图）
        if method == 'nearest':
            resized = cv2.resize(
                depth.astype(np.float32), 
                (new_shape[1], new_shape[0]),  # OpenCV需要 (width, height)
                interpolation=cv2.INTER_NEAREST
            )
            
        # 方法2：SciPy缩放（更灵活的控制）
        elif method == 'zoom':
            zoom_factor = (0.5, 0.5)
            resized = zoom(depth, zoom_factor, order=0)  # order=0 表示最近邻
            
        else:
            raise ValueError("不支持的缩放方法")
        
        # 保持数据类型一致
        resized = resized.astype(depth.dtype)
        
        # 保存结果
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, resized)
        print(f"深度图已保存：{output_path}")
        
    except Exception as e:
        print(f"处理失败：{str(e)}")
        # 可以添加更多调试信息
        if 'depth' in locals():
            print(f"数据形状：{depth.shape}")
            print(f"数据类型：{depth.dtype}")
            print(f"数据范围：[{depth.min()}, {depth.max()}]")
            
def suofang(input_image,depth_file,output_folder):
    if not depth_file == None:
        resize_depth_npy(depth_file, output_folder, method='nearest')
    resize_half_opencv(input_image, output_folder)



if __name__ == "__main__":
    input_image = "datasets/diode/val/indoor/scene_00020/scan_00186/00020_00186_indoors_110_000.png"
    depth_file = "datasets/diode/val/indoor/scene_00020/scan_00186/00020_00186_indoors_110_000_depth.npy"
    output_folder = "datasets/diode/val/indoor/scene_00020/scan_00186/suofang"
    
    resize_depth_npy(depth_file, output_folder, method='nearest')
    resize_half_opencv(input_image, output_folder)
    
