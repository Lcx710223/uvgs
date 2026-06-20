# DISCLAIMER:
# 
# This code is part of the research paper titled "UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping."
# Use it freely, modify it, and build upon it in accordance with the terms of the CC BY-NC 4.0 License. 
# If you use or reference this work, please cite our paper.
# 
# For more information, please refer to the paper: https://arxiv.org/abs/2502.01846
# For updates, please visit the project website: https://aashishrai3799.github.io/uvgs
# 
# Copyright (C) 2025 Aashish Rai.
# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# https://creativecommons.org/licenses/by-nc/4.0
#


import os
import cv2
import torch
import numpy as np
from natsort import natsorted

from uvgs_scripts.gs_dataloader import *
from uvgs_scripts.mapping_functions import *
from uvgs_scripts.helper_functions import prune_points_counter, spherical_unwrap_count_K_prune_mask, spherical_unwrap_valid_mask


def GetUVGSmap(
    gs_path,
    K=32,
    return_shs=True,
    uvgs_size=1024,
    mapping_type='S',
    device='cpu'
):
    """
    Generate UV mappings from a 3DGS scene and save generated UV maps as numpy files.

    Args:
        gs_path (str):
            3DGS 场景的路径（通常是一个包含 .ply 或类似文件的目录）。
        K (int):
            UVGS 的层数（top-K 映射层数），即每个像素最多保留 K 个高斯索引。
        return_shs (bool):
            是否在加载时返回 SH 系数（这里实际没用到，传给 loader）。
        uvgs_size (int):
            UV 图的分辨率，高和宽（正方形）。
        mapping_type (str):
            映射类型：'R'（equirectangular）、'S'（spherical）、'RS'（两者拼接）。
            这里只处理 'S'。
        device (str):
            Torch 设备（'cpu' 或 'cuda'），本函数里其实没用到。
    """

    # 从 gs_path 加载 3DGS 数据（通常是一个 .ply 文件）
    # 返回一个 torch.Tensor，形状为 (N, C)，N 为高斯数量，C 为属性维度
    input_data = load_gs_from_file(gs_path, return_shs=False)

    # 提取各通道数据
    # 所有属性（位置、尺度、颜色、opacity、rotation 等）
    gs_points = input_data[:, :].detach().cpu().numpy()    # shape: (N, C)
    # 仅 3D 坐标 (x, y, z)，用于球面映射
    points = input_data[:, :3].detach().cpu().numpy()      # shape: (N, 3)
    # 不透明度通道，用于某些基于 opacity 的 top-K 策略
    opacity = input_data[:, 10:11].detach().cpu().numpy()  # shape: (N, 1)

    # 任意的赋值：把第 0 个高斯的所有属性置为 0
    # 这样在后续索引中，如果某个像素没有有效高斯，可以用索引 0 作为“空”占位
    gs_points[0, :] = 0

    # 准备做 top-K 展开（unwrap）
    if mapping_type == 'S':
        # 这里使用的是“填充所有通道”的快速球面展开函数，
        # 返回一个整数索引张量 UV_mapping_S，形状为 (H, W, K)，
        # 每个像素位置 (h, w) 有 K 个索引，对应 K 个高斯点。
        # fast_spherical_unwrap_fill_channels 不用 opacity 做排序，而是填满 K 层。
        UV_mapping_S = fast_spherical_unwrap_fill_channels(
            points,
            height=uvgs_size,
            width=uvgs_size,
            K=K
        )

        # 对于每一层 k，取出该层的索引 UV_mapping_S[:, :, k]，
        # 用这些索引在 gs_points 中取出对应的高斯属性，
        # 得到一个 (H, W, C) 的属性图；然后对 K 层进行堆叠。
        UV_gs_points_topK_S = np.stack(
            [gs_points[UV_mapping_S[:, :, k]] for k in range(K)],
            axis=-1  # 最后得到形状 (H, W, C, K)
        )

        # 这里只做 spherical 映射，所以直接返回该结果
        UV_gs_points_topK = UV_gs_points_topK_S

        return UV_gs_points_topK


def GetUVGSmap_Fast(
    gs_path,
    K=32,
    return_shs=True,
    uvgs_size=1024,
    mapping_type='S',
    device='cpu'
):
    """
    更“工程化”的 UVGS 生成函数，使用 fast_spherical_unwrap_topK_opacity，
    并且显式地做索引展开和 reshape。

    Args:
        gs_path (str): 3DGS 场景路径。
        K (int): UVGS 层数（top-K）。
        return_shs (bool): 是否加载 SH 系数（这里传给 loader）。
        uvgs_size (int): UV 图的分辨率（宽高）。
        mapping_type (str): 映射类型，这里只处理 'S'（球面）。
        device (str): Torch 设备。
    """

    # 从文件加载 3DGS 数据，这里多传了一个 iter='45000'，
    # 说明 loader 可能支持按迭代号加载某个 checkpoint。
    input_data = load_gs_from_file(gs_path, iter='45000', return_shs=return_shs)

    # 把数据放到指定设备上（CPU 或 GPU）
    input_data = input_data.to(device)
    # 再转回 CPU 并变成 numpy，方便后续用纯 numpy 做索引和展开
    data_np = input_data.cpu().numpy()  # shape: (N, C)

    # 提取各通道
    gs_points = data_np            # 所有属性，shape: (N, C)
    points    = data_np[:, :3]     # 3D 坐标，shape: (N, 3)
    opacity   = data_np[:, 10:11]  # 不透明度，shape: (N, 1)

    # 同样把第 0 个高斯置为 0，作为“空占位”
    gs_points[0, :] = 0

    # 这里只展示 spherical 映射的情况
    if mapping_type == 'S':
        # fast_spherical_unwrap_topK_opacity:
        # 根据点的球面方向和 opacity，在每个像素位置选出 top-K 高斯索引，
        # 返回一个整数数组 UV_mapping_S，形状为 (H, W, K)。
        UV_mapping_S = fast_spherical_unwrap_topK_opacity(
            points,
            opacity,
            height=uvgs_size,
            width=uvgs_size,
            K=K
        )

        H, W, K_ = UV_mapping_S.shape  # K_ 应该等于 K

        # 把 (H, W, K) 展平为一维索引数组，形状 (H*W*K,)
        flat_idx = UV_mapping_S.reshape(-1)

        # 用这些索引在 gs_points 中取出对应的高斯属性，
        # 得到形状 (H*W*K, C) 的数组
        mapped_points = gs_points[flat_idx]

        # 再把它 reshape 回 (H, W, K, C)
        mapped_points = mapped_points.reshape(H, W, K_, -1)

        # 原始代码期望的排列是 (H, W, C, K)，
        # 所以做一个维度交换：从 (H, W, K, C) → (H, W, C, K)
        mapped_points = np.transpose(mapped_points, (0, 1, 3, 2))

        # 最终返回的 UVGS：形状 (H, W, C, K)，
        # 即每个像素有 C 维属性，并且有 K 层（top-K 高斯）。
        return mapped_points


if __name__ == "__main__":

    device = 'cpu'

    # 拟合好的 3DGS 场景所在目录，每个子文件夹对应一个场景
    gs_path = "./fillted_3DGS_scene"
    # 输出 UVGS 的目录
    out_path = "./test_UV_Maps"

    # 创建输出目录（若不存在）
    os.makedirs(out_path, exist_ok=True)

    # UVGS 层数（top-K）
    K = 8

    # 是否保存 UVGS 的 .npy 文件
    SAVE_UVGS = True
    # 是否把 UVGS 转成 .ply 点云（方便可视化）
    SAVE_PLY = True

    # 列出所有场景子目录，并按自然顺序排序（避免 '10' 排在 '2' 前面）
    gs_folders = os.listdir(gs_path)
    gs_folders = natsorted(gs_folders)

    # 遍历每一个拟合好的 3DGS 场景
    for i, folder in enumerate(gs_folders):

        print(f'Processing K: {K} || Index K: {folder} || {i}')

        # 当前场景的根路径
        root_gs_path = os.path.join(gs_path, folder)

        # 生成该场景的 UVGS 映射，得到形状 (H, W, C, K) 的 numpy 数组
        uvgs_map = GetUVGSmap_Fast(
            root_gs_path,
            K=K,
            return_shs=False,
            uvgs_size=1024,
            mapping_type='S',
            device=device
        )

        # 保存为 .npy 文件，文件名为场景文件夹名
        if SAVE_UVGS:
            np.save(os.path.join(out_path, folder + '.npy'), uvgs_map)

        # 保存为 .ply 点云，方便在 MeshLab / CloudCompare 等工具中查看
        if SAVE_PLY:
            # uvgs_map: (H, W, C, K)
            # 这里 permute 成 (H, W, K, C)，再交给 save_uv_gs_2_ply，
            # 说明该函数内部可能按 (H, W, K, C) 来解释为点云。
            save_uv_gs_2_ply(
                torch.from_numpy(uvgs_map).permute(0, 1, 3, 2),
                ply_path=f'{out_path}/{folder}.ply'
            )
     

    
 
