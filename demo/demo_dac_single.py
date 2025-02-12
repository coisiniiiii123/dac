#!/usr/bin/env python
"""
Depth-Any-Camera demo script for inference different types of camera data on a single perspective trained model.
Model: DAC-Indoor
Test data source: Scannet++, Matterport3D, NYU
"""
#python demo/demo_dac_single.py --config-file checkpoints/dac_swinl_outdoor.json --model-file checkpoints/dac_swinl_outdoor.pt --sample-file demo/input/kitti_sample.json --out-dir demo/output

import sys
sys.path.append('/home/jack/YOLO_World/depth_any_camera')
from suofang import suofang

import argparse
import json
import os
from typing import Any, Dict

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

DISPLAY_SCALE = 1

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pred_depth_map = param['pred_depth_map']  # 模型预测的深度图
        gt_depth_map = param['gt_depth_map']      # 真实深度图
        
        # 由于窗口大小与图像一致，直接使用点击坐标
        x_original = int(x/DISPLAY_SCALE)
        y_original = int(y/DISPLAY_SCALE)
        
        # 检查坐标是否在有效范围内
        if 0 <= x_original < pred_depth_map.shape[1] and 0 <= y_original < pred_depth_map.shape[0]:
            pred_depth_value = pred_depth_map[y_original, x_original]   # 获取预测深度值
            gt_depth_value = gt_depth_map[y_original, x_original]       # 获取真实深度值
            print(f"prediction: {pred_depth_value:.3f}m, ground truth: {gt_depth_value:.3f}m")
            param['last_click'] = (x_original, y_original)
            param['click_history'].append({
                    'position': (x_original*DISPLAY_SCALE, y_original*DISPLAY_SCALE),
                    'pred': pred_depth_value,
                    'gt': gt_depth_value
            })
        else:
            print("警告：点击位置超出深度图范围")


def demo_one_sample(model, model_name, device, sample, cano_sz, args: argparse.Namespace, if_suofang=False):
    #######################################################################
    ############# data prepare (A simple version dataloader) ##############
    #######################################################################
    
    if if_suofang:
        suofang(input_image=sample["oringinal_img"],depth_file=sample["oringinal_depth"],output_folder=sample["output_folder"])
    
    image = np.asarray(
        Image.open(sample["image_filename"])
    )
    org_img_h, org_img_w = image.shape[:2]
    if sample["annotation_filename_depth"] is None:
        depth = np.zeros((org_img_h, org_img_w), dtype=np.float32)
    else:
        depth = (
            np.asarray(
                cv2.imread(sample["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
            ).astype(np.float32)
            / sample["depth_scale"]
        )
    
    dataset_name = sample["dataset_name"]
    fwd_sz=sample["fwd_sz"]
    
    if not sample["erp"]:
        # convert depth from zbuffer to euclid 
        if dataset_name in ['nyu', 'kitti']:
            x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
            depth = depth * np.sqrt((x - sample["cam_params"]['cx'])**2 + (y - sample["cam_params"]['cy'])**2 + sample["cam_params"]['fx']**2) / sample["cam_params"]['fx']
            depth = depth.astype(np.float32)
        elif dataset_name == 'scannetpp': # Very critical for scannet++ fisheye. Skip kitti360 because we prepared the depth already in euclid.
            # For fisheye, converting back to euclid with undistorted ray direction via the ray lookup table for efficiency
            fisheye_grid = np.load(sample["fishey_grid"])
            fisheye_grid_z = cv2.resize(fisheye_grid[:, :, 2], (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
            depth = depth / fisheye_grid_z
            depth = depth.astype(np.float32)
            
        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0

        image = image.astype(np.float32) / 255.0
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
        
        depth_out_full = cv2.resize(depth_out[0, 0].cpu().numpy(), (org_img_w, org_img_h), interpolation=cv2.INTER_LINEAR)
        depth_out_gt_full = cv2.resize(depth_out_gt[0, 0].cpu().numpy(), (org_img_w, org_img_h), interpolation=cv2.INTER_LINEAR)
        rgb_full = cv2.resize(rgb, (org_img_w, org_img_h), interpolation=cv2.INTER_LINEAR)
        
        cv2.namedWindow('Depth Map')
        callback_params = {
            'pred_depth_map': depth_out_full,  # 模型预测的深度图
            'gt_depth_map': depth_out_gt_full,              # 真实深度图
            'last_click': None,
            'display_size': (org_img_w, org_img_h),
            'original_size': (org_img_w, org_img_h),  # 原始深度图尺寸
            'click_history': []
        }
        cv2.setMouseCallback('Depth Map', mouse_callback, callback_params)
        font = cv2.FONT_HERSHEY_SIMPLEX
        print("点击深度图上的任意位置显示深度值，按'q'退出")
        
        # print(org_img_w, org_img_h)
        
    #交互
    while True:
        display_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        resize_img = cv2.resize(display_img, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        # display_img = depth_vis.copy()
        # 获取鼠标回调中保存的最后点击位置
        last_click = callback_params['last_click']
        for click in callback_params['click_history']:
            x, y = click['position']
            pred_depth_value = click['pred']
            gt_depth_value = click['gt']
            text_pred = f"pred:{pred_depth_value:.3f}"
            text_gt = f"GT:{gt_depth_value:.3f}"
            cv2.putText(resize_img, text_pred, (x+5, y-8), font, 0.5, (255, 255, 255), 1)
            cv2.putText(resize_img, text_gt, (x+5, y+8), font, 0.5, (255, 255, 255), 1)
            cv2.circle(resize_img, (x, y), 2, (0, 255, 255), -1)
            # cv2.putText(display_img, text, (x+10, y-10), font, 0.6, (255, 255, 255), 2)

        cv2.imshow('Depth Map', resize_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):  # 按下o键清除点击记录
            callback_params['last_click'] = None
            callback_params['click_history'] = []
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--config-file", type=str, default="checkpoints/dac_swinl_indoor.json")
    parser.add_argument("--model-file", type=str, default="checkpoints/dac_swinl_indoor.pt")
    parser.add_argument("--sample-file", type=str, default='data/phone.json')
    parser.add_argument("--out-dir", type=str, default='demo/output')
    # parser.add_argument("--save-pcd", action="store_true")

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
    # ... existing code ...
    # args = parser.parse_args()
    # print("args类型:", type(args))
    # print("args内容:", args)  # vars() 将Namespace对象转换为字典
    
    # with open(args.config_file, "r") as f:
    #     config = json.load(f)