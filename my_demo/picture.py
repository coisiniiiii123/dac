#!/usr/bin/env python
"""
Depth-Any-Camera demo script for inference different types of camera data on a single perspective trained model.
Model: DAC-Indoor
Test data source: Scannet++, Matterport3D, NYU
"""
#python demo/demo_dac_single.py --config-file checkpoints/dac_swinl_outdoor.json --model-file checkpoints/dac_swinl_outdoor.pt --sample-file demo/input/kitti_sample.json --out-dir demo/output

import sys
sys.path.append('/home/jack/YOLO_World/depth_any_camera')

import argparse
import json
import os
from typing import Any, Dict
from suofang import suofang

import numpy as np
import cv2
import torch
import torch.cuda as tcuda
from PIL import Image

from dac.models.idisc_erp import IDiscERP
from dac.models.idisc import IDisc
from dac.models.idisc_equi import IDiscEqui
from dac.models.cnn_depth import CNNDepth
from dac.utils.visualization import save_file_ply, save_val_imgs_v2
from dac.utils.unproj_pcd import reconstruct_pcd, reconstruct_pcd_erp
from dac.utils.erp_geometry import erp_patch_to_cam_fast, cam_to_erp_patch_fast, fisheye_mei_to_erp
from dac.dataloders.dataset import resize_for_input
import torchvision.transforms.functional as TF
from ultralytics import YOLOWorld

# 可视化缩放因子
DISPLAY_SCALE = 1
            
            
def bound_callback(x, y, param):
    pred_depth_map = param['pred_depth_map']  # 模型预测的深度图
    gt_depth_map = param['gt_depth_map']      # 真实深度图
    
    x_original = x
    y_original = y

    # 检查坐标是否在有效范围内
    if 0 <= x_original < pred_depth_map.shape[1] and 0 <= y_original < pred_depth_map.shape[0]:
        pred_depth_value = pred_depth_map[y_original, x_original]   # 获取预测深度值
        gt_depth_value = gt_depth_map[y_original, x_original]       # 获取真实深度值
        depth = [pred_depth_value,gt_depth_value]
        print(depth)
        return depth
    else:
        return 0
            

def demo_one_sample(model, model_name, device, sample, cano_sz, args: argparse.Namespace, if_suofang=False):
    #######################################################################
    ############# data prepare (A simple version dataloader) ##############
    #######################################################################
    if if_suofang:
        image,depth = suofang(input_image=sample["image_filename"],depth_file=sample["annotation_filename_depth"])
    else:
        image = np.asarray(
            Image.open(sample["image_filename"])
        )
        if sample["annotation_filename_depth"] is not None:
            depth = np.load(sample["annotation_filename_depth"])
        
        
    org_img_h, org_img_w = image.shape[:2]
    if sample["annotation_filename_depth"] is None:
        depth = np.zeros((org_img_h, org_img_w), dtype=np.float32)
    else:
        # depth = (
        #     np.asarray(
        #         cv2.imread(sample["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
        #     ).astype(np.float32)
        #     / sample["depth_scale"]
        # )
        # depth = np.load(sample["annotation_filename_depth"])
        if depth.ndim == 2:
            depth = depth[..., np.newaxis]  # 添加通道维度
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / sample["depth_scale"]
        elif depth.dtype == np.float32:
            depth = depth  # 无需缩放
        elif depth.dtype == np.uint8:
            depth_max = 10.0  # 根据数据集调整
            depth = depth.astype(np.float32) * (depth_max / 255.0)
        else:
            raise ValueError(f"不支持的深度数据类型: {depth.dtype}")

    
    dataset_name = sample["dataset_name"]
    fwd_sz=sample["fwd_sz"]
    
    if not sample["erp"]:
        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0
        
        image = image.astype(np.float32) / 255.0
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > 0.01
                
        # Automatically calculate the erp crop size
        crop_width = int(cano_sz[0] * sample["crop_wFoV"] / 180)
        crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])

        # convert to ERP
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            crop_height, crop_width, cano_sz[0], cano_sz[0]*2, sample["cam_params"], roll, scale_fac=None
        )
        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])
                
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask = resize_for_input((image * 255.).astype(np.uint8), depth, fwd_sz, None, [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask)
    else:
        attn_mask = np.ones_like(depth)
        lat_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
        long_range = torch.tensor([-np.pi, np.pi], dtype=torch.float32)
        
        # resizing process to fwd_sz.
        to_cano_ratio = cano_sz[0] / image.shape[0]
        image, depth, pad, pred_scale_factor = resize_for_input(image, depth, fwd_sz, None, cano_sz, to_cano_ratio)


    # convert to tensor batch
    normalization_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    image = TF.normalize(TF.to_tensor(image), **normalization_stats)
    gt = TF.to_tensor(depth)
    mask = TF.to_tensor((depth > 0.01).astype(np.uint8))
    attn_mask = TF.to_tensor((attn_mask>0).astype(np.float32)) # the non-empty region after ERP conversion
    batch = {
        "image": image.unsqueeze(0),
        "gt": gt.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "attn_mask": attn_mask.unsqueeze(0),
        "lat_range": lat_range.unsqueeze(0),
        "long_range": long_range.unsqueeze(0),
        "info": {
            "pred_scale_factor": pred_scale_factor,
        },
    }
    
    #######################################################################
    ########################### model inference ###########################
    #######################################################################

    gt, mask, attn_mask, lat_range, long_range = batch["gt"].to(device), batch["mask"].to(device), batch['attn_mask'].to(device), batch["lat_range"].to(device), batch["long_range"].to(device)
    with torch.no_grad():
        if model_name == "IDiscERP":
            preds, _, _ = model(batch["image"].to(device), lat_range, long_range)
        else:
            preds, _, _ = model(batch["image"].to(device))
    preds *= pred_scale_factor
    
    # 获取深度图数据
    pred_depth = preds[0, 0].detach().cpu().numpy()  # 转换为numpy数组
    pred_depth_full = cv2.resize(pred_depth, (org_img_w, org_img_h), interpolation=cv2.INTER_LINEAR) #恢复大小

    # # 直接获取特定坐标点的深度值
    # x, y = 1000, 850
    # depth_value = pred_depth[y, x]  # numpy数组是[行,列]顺序
    depth_normalized = (pred_depth_full - pred_depth_full.min()) / (pred_depth_full.max() - pred_depth_full.min())
    depth_vis = (depth_normalized * 255).astype(np.uint8)
    # depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # 创建交互窗口

    
    
    
    #######################################################################
    ##################  Visualization and Output results  #################
    #######################################################################
    save_img_dir = os.path.join(args.out_dir)
    os.makedirs(save_img_dir, exist_ok=True)
    if 'attn_mask' in batch.keys():
        attn_mask = batch['attn_mask'][0]
    else:
        attn_mask = None

    # adjust vis_depth_max for outdoor datasets
    if dataset_name == 'kitti360':
        vis_depth_max = 40.0
        vis_arel_max = 0.3
    elif dataset_name == 'kitti':
        vis_depth_max = 80.0
        vis_arel_max = 0.8
    else:
        # default indoor visulization parameters
        vis_depth_max = 10.0
        vis_arel_max = 0.5

    rgb = save_val_imgs_v2(
        0,
        preds[0],
        batch["gt"][0],
        batch["image"][0],
        f'{dataset_name}_output.jpg',
        save_img_dir,
        active_mask=attn_mask,
        valid_depth_mask=batch["mask"][0],
        depth_max=vis_depth_max,
        arel_max=vis_arel_max
    )
    
    pred_depth = preds[0, 0].detach().cpu().numpy()
    # if args.save_pcd:
    pcd = reconstruct_pcd_erp(pred_depth, mask=(batch['attn_mask'][0][0]).numpy(), lat_range=batch['lat_range'][0], long_range=batch['long_range'][0])
    save_pcd_dir = os.path.join(args.out_dir)
    os.makedirs(os.path.join(save_pcd_dir), exist_ok=True)
    pc_file = os.path.join(save_pcd_dir, f'{dataset_name}_pcd.ply')
    pcd = pcd.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    save_file_ply(pcd, rgb, pc_file)

    ##########  Convert the ERP result back to camera space for visualization (No need for original ERP image)  ##########
    if not sample['erp']:                    
        if dataset_name == 'kitti360':
            out_h = int(org_img_h/2)
            out_w = int(org_img_w/2)
            grid_fisheye = np.load(sample["fishey_grid"])
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'kitti360'}
        elif dataset_name == 'scannetpp':
            """
                Currently work perfect with phi = 0. For larger phi, corners may have artifacts.
            """
            grid_fisheye = np.load(sample["fishey_grid"])
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            out_h = int(org_img_h/2)
            out_w = int(org_img_w/2)
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'scannetpp'} # when grid table is available, no need for intrinsic parameters
        else:
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            out_h = org_img_h
            out_w = org_img_w
            grid_fisheye = None
            cam_params = sample["cam_params"]
            
        # scale the full erp_size depth scaling factor is equivalent to resizing data (given same aspect ratio)
        erp_h = cano_sz[0]
        erp_h = erp_h * batch['info']['pred_scale_factor']
        if 'f_align_factor' in batch['info']:
            erp_h = erp_h / batch['info']['f_align_factor'][0].detach().cpu().numpy()
        img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
            batch["image"][0], preds[0].detach().cpu(), attn_mask, 0., 0., out_h=out_h, out_w=out_w, erp_h=erp_h, erp_w=erp_h*2, cam_params=cam_params, 
            fisheye_grid2ray=grid_fisheye, depth_erp_gt=batch["gt"][0].detach().cpu())
        rgb = save_val_imgs_v2(
            0,
            depth_out,
            depth_out_gt,
            img_out,
            f'{dataset_name}_output_remap.jpg',
            save_img_dir,
            active_mask=active_mask,
            depth_max=vis_depth_max,
            arel_max=vis_arel_max
            )        
        # print(depth_out.shape)
        # print(depth_out_gt.shape)
        # print(depth_vis.shape)
    depth_out_full = cv2.resize(depth_out[0, 0].cpu().numpy(), (org_img_w, org_img_h), interpolation=cv2.INTER_LINEAR)
    depth_out_gt_full = cv2.resize(depth_out_gt[0, 0].cpu().numpy(), (org_img_w, org_img_h), interpolation=cv2.INTER_LINEAR)
    
    cv2.namedWindow('Depth Map')
    callback_params = {
        'pred_depth_map': depth_out_full,  # 模型预测的深度图
        'gt_depth_map': depth_out_gt_full,              # 真实深度图
        'last_click': None,
        'display_size': (org_img_w, org_img_h),
        'original_size': (org_img_w, org_img_h),  # 原始深度图尺寸
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #交互
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    #yolo
    yolo_model = YOLOWorld(args.yolo_model)
    results = yolo_model.predict(sample["image_filename"])
    boxes = results[0].boxes  # 包含边界框的对象
    display_img = results[0].orig_img.copy()
    for box in boxes:
        coordinates = box.xywh.tolist()[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(display_img, (x1,y1), (x2,y2), (0,255,0), 2)
        x = int(coordinates[0])
        y = int(coordinates[1])
        if if_suofang:
            x = x
            y = y
        print(f"坐标 (x,y,w,h): {coordinates}")
        class_id = int(box.cls)
        class_name = yolo_model.names[class_id]  # 获取类名
    
        #dac
        if if_suofang:
            depth_value = bound_callback(x//2, y//2 ,callback_params)
        else:
            depth_value = bound_callback(x,y,callback_params)
        pred_depth_value = depth_value[0]
        gt_depth_value = depth_value[1]
        text_pred = f"depth:{pred_depth_value:.3f}"
        cv2.putText(display_img, text_pred, (x1, y1-8), font, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, class_name, (x1, y1-20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(display_img, text_gt, (x+5, y+8), font, 0.5, (255, 255, 255), 1)
        cv2.circle(display_img, (x, y), 2, (0, 255, 255), -1)
    
    while True:
        cv2.imshow('Depth Map', display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()
    # print(rgb.shape,depth_out_full.shape,depth_out_gt_full.shape)
        

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--config-file", type=str, default="checkpoints/dac_swinl_indoor.json")
    parser.add_argument("--model-file", type=str, default="checkpoints/dac_swinl_indoor.pt")
    parser.add_argument("--sample-file", type=str, default='data/laptop.json')
    parser.add_argument("--out-dir", type=str, default='output')
    parser.add_argument("--yolo-model", type=str, default="/home/jack/YOLO_World/yolov8/yolov8s-world.pt")

    args = parser.parse_args()  #这行代码会将这些命令行参数解析并存储在 args 对象中
    with open(args.config_file, "r") as f:
        config = json.load(f)

    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = eval(config["model_name"]).build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()
    cano_sz=config["cano_sz"] # the ERP size model was trained on

    with open(args.sample_file, "r") as f:
        sample = json.load(f)
    print(f"demo for sample from {sample['dataset_name']}")
    demo_one_sample(model, config["model_name"], device, sample, cano_sz, args, if_suofang=False)
    print("Demo finished")
