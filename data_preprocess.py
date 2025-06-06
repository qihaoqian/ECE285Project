#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import trimesh

def preprocess_obj(src_path: str, dst_path: str):
    """
    加载单个 .obj 文件，创建 ProximityQuery，
    然后根据包围盒中心和平移到原点，按 0.6 * sqrt(3)/(diag/2) 的比例缩放，
    并将处理后网格导出到 dst_path (.obj 格式)。
    """
    # 1) 加载 OBJ 网格
    mesh = trimesh.load(src_path, force='mesh')

    # 2) 创建 ProximityQuery（按照要求保留此步骤）
    _ = trimesh.proximity.ProximityQuery(mesh)

    # normalize to [-1, 1] (different from instant-sdf where it is [0, 1])
    vs = mesh.vertices
    vmin = vs.min(0)
    vmax = vs.max(0)
    v_center = (vmin + vmax) / 2
    v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.8
    vs = (vs - v_center[None, :]) * v_scale
    mesh.vertices = vs

    # 4) 导出为 OBJ
    mesh.export(dst_path)


def batch_preprocess(input_root: str, output_root: str):
    """
    遍历 input_root 下所有子目录，找到 .obj 文件，
    对每个 .obj 调用 preprocess_obj()，并将结果保存到 output_root。
    保持与 input_root 相同的目录层级结构，只替换文件内容。
    """
    os.makedirs(output_root, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(input_root):
        # 仅处理后缀为 .obj 的文件
        obj_files = [f for f in filenames if f.lower().endswith(".obj")]
        if not obj_files:
            continue

        # 计算 dirpath 相对于 input_root 的相对路径
        rel_dir = os.path.relpath(dirpath, input_root)
        dst_dir = os.path.join(output_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        print(f"[INFO] 处理目录：{dirpath} → 输出到：{dst_dir}（共 {len(obj_files)} 个 .obj 文件）")
        for fname in obj_files:
            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dst_dir, fname)

            try:
                preprocess_obj(src_path, dst_path)
                print(f"  [OK] {src_path} → {dst_path}")
            except Exception as e:
                print(f"  [ERROR] 处理 {src_path} 失败：{e}")

        print("")

    print(f"[DONE] 所有 OBJ 文件已预处理完毕，结果保存在：{output_root}")


if __name__ == "__main__":
    # ——————— 请根据实际情况修改下面两个路径 ———————
    INPUT_ROOT = "data/ModelNet10_mini_repaired"        # 原始 .obj 所在根目录
    OUTPUT_ROOT = "data/ModelNet10_preprocessed" # 预处理后保存的目标根目录
    # ——————————————————————————————————————————————

    batch_preprocess(INPUT_ROOT, OUTPUT_ROOT)
