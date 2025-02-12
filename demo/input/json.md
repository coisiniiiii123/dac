{
    "dataset_name": "scannetpp",  // 数据集名称，可以自定义
    "image_filename": "demo/input/scannetpp_rgb.jpg",  // RGB图像路径
    "annotation_filename_depth": "demo/input/scannetpp_depth.png",  // 深度图路径（如果有）
    "depth_scale": 1000.0,  // 深度图的缩放因子，用于将深度图的值转换为实际距离（米）
    "fishey_grid": "demo/input/scannetpp_grid_fisheye.npy",  // 鱼眼矫正网格文件路径
    "crop_wFoV": 180,  // 鱼眼镜头的视场角（度）
    "fwd_sz": [500, 750],  // 输出图像的尺寸 [高度, 宽度]
    "erp": false,  // 是否使用等距圆柱投影(Equirectangular Projection)
    
    "cam_params": 
        "dataset": "scannetpp",  // 数据集名称
        "fl_x": 789.9080967683176,  // x方向焦距
        "fl_y": 791.5566599926353,  // y方向焦距
        "cx": 879.203786509326,  // 图像中心x坐标
        "cy": 584.7893145555763,  // 主点y坐标
        "k1": -0.029473047856246333,  // 畸变系数k1
        "k2": -0.005769803970428537,  // 畸变系数k2
        "k3": -0.002148236771485755,  // 畸变系数k3
        "k4": 0.00014840568362061509,  // 畸变系数k4
        "camera_model": "OPENCV_FISHEYE"  // 相机模型类型
    
}