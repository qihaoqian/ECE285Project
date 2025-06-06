#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import trimesh

def convert_off_to_obj(input_root: str):
    """
    遍历 input_root 下所有子目录，找到 .off 文件并将其转换成 .obj 文件，
    输出路径与 .off 文件相同，仅替换后缀为 .obj。
    
    Args:
        input_root (str): 修补后数据集的根目录，例如 "data/ModelNet10_mini_repaired"
    """
    for dirpath, dirnames, filenames in os.walk(input_root):
        # 筛选出当前目录下所有 .off 文件
        off_files = [f for f in filenames if f.lower().endswith(".off")]
        if not off_files:
            continue

        print(f"[INFO] 处理目录：{dirpath}，共找到 {len(off_files)} 个 .off 文件")
        for fname in off_files:
            off_path = os.path.join(dirpath, fname)
            obj_fname = os.path.splitext(fname)[0] + ".obj"
            obj_path = os.path.join(dirpath, obj_fname)

            try:
                # 读取 OFF 网格
                mesh = trimesh.load(off_path, force="mesh")
                # 导出为 OBJ
                mesh.export(obj_path)
                print(f"  [OK] {off_path} → {obj_path}")
            except Exception as e:
                print(f"  [ERROR] 无法转换 {off_path}：{e}")

        print("")

if __name__ == "__main__":
    # 请根据实际路径修改
    INPUT_ROOT = "data/ModelNet10_mini_repaired"

    convert_off_to_obj(INPUT_ROOT)
