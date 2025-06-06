#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import trimesh

def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    对输入的 Trimesh 对象进行一次自动修补，步骤如下：
      1. 删除退化面、重复面和未被引用的顶点
      2. 修正面法线方向一致性
      3. 自动填补边界洞
      4. 再次删除可能新生成的退化面、重复面和未引用顶点
      5. 修正 winding（面顶点索引顺序），并再次修正法线

    返回修补后的 mesh。修补后如果仍不是 watertight，会在控制台打印警告。
    """
    # 1. 删除退化面、重复面和未引用顶点
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    # 2. 修正法线方向（让所有法线保持一致，朝外）
    trimesh.repair.fix_normals(mesh)

    # 3. 自动填补边界洞
    holes_filled = mesh.fill_holes()
    if holes_filled > 0:
        print(f"      [INFO] fill_holes(): 成功填补 {holes_filled} 个洞。")
    else:
        print(f"      [INFO] fill_holes(): 没有检测到可自动补的洞。")

    # 4. 再次删除退化面、重复面、未引用顶点
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    # 5. 修正 winding，使所有面的索引顺序保证法线朝外
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    return mesh


def select_and_repair(input_root: str, output_root: str, num_per_class: int = 5):
    """
    从原始数据集（input_root）中每个类别的 train 子目录随机或顺序挑选 .off 文件，
    只要某个文件修补后 water­tight，就保存修补结果到 output_root/<category>/ 下，
    每个类别最多保存 num_per_class 个。修补失败或补完后仍不是 watertight 的会跳过。

    Args:
        input_root (str): 原始数据集根目录，如 "data/ModelNet10"
        output_root (str): 输出的 mini 修补后数据集根目录，如 "data/ModelNet10_mini_repaired"
        num_per_class (int): 每个类别最多挑选的修补成功的样本数
    """
    os.makedirs(output_root, exist_ok=True)

    # 遍历 input_root 下的每个一级子目录（每个子目录即一个类别名称）
    for category in os.listdir(input_root):
        category_dir = os.path.join(input_root, category)
        if not os.path.isdir(category_dir):
            # 只处理文件夹，跳过非目录项
            continue

        train_dir = os.path.join(category_dir, "train")
        if not os.path.isdir(train_dir):
            print(f"[WARN] 类别 '{category}' 下没有 train 子目录，跳过该类别。")
            continue

        # 列出 train 目录下所有 .off 文件
        off_files = [
            f for f in os.listdir(train_dir)
            if os.path.isfile(os.path.join(train_dir, f)) and f.lower().endswith(".off")
        ]
        if not off_files:
            print(f"[WARN] 类别 '{category}' 的 train 目录下没有 .off，跳过。")
            continue

        # 为该类别创建输出文件夹：output_root/<category>/
        dst_category_dir = os.path.join(output_root, category)
        os.makedirs(dst_category_dir, exist_ok=True)

        print(f"[INFO] 正在处理类别 '{category}'，共 {len(off_files)} 个 train 文件 → 目标最多 {num_per_class} 个修补成功")
        count_selected = 0

        # 如果希望“随机抽取”而不是按文件名顺序，可以先打乱列表：
        # import random; random.shuffle(off_files)

        for fname in off_files:
            if count_selected >= num_per_class:
                break

            src_path = os.path.join(train_dir, fname)
            print(f"    - 尝试加载并修补：{src_path}")
            try:
                mesh = trimesh.load(src_path, force='mesh')
            except Exception as e:
                print(f"      [ERROR] 加载失败: {e}，跳过该文件")
                continue

            # 检查原始是否水密（可选打印）
            if mesh.is_watertight:
                print("      [INFO] 原始网格已是 watertight")
            else:
                print("      [INFO] 原始网格不是 watertight，进行自动修补...")

            # 调用修补函数
            repaired = repair_mesh(mesh)

            # 检查修补后是否真的变成 watertight
            if not repaired.is_watertight:
                print("      [WARN] 修补后仍然不是 watertight，跳过该文件")
                continue

            # 如果修补成功，把修补后的网格导出到目标目录
            dst_path = os.path.join(dst_category_dir, fname)
            try:
                repaired.export(dst_path)
                print(f"      [OK] 修补成功并保存到：{dst_path}")
                count_selected += 1
            except Exception as e:
                print(f"      [ERROR] 保存修补结果失败: {e}")

        if count_selected < num_per_class:
            print(f"  [WARN] 类别 '{category}' 只找到 {count_selected} 个可修补样本（目标 {num_per_class}）。")
        else:
            print(f"  [INFO] 类别 '{category}' 已成功选出 {num_per_class} 个可修补样本。")

        print("")

    print(f"[DONE] 全部类别处理完毕，修补后的 mini 数据集保存在：{output_root}")


if __name__ == "__main__":
    # 1. 指定原始数据集根目录（请根据实际路径修改）
    INPUT_ROOT = "data/ModelNet10"

    # 2. 指定修补后 mini 数据集根目录（会自动创建）
    OUTPUT_ROOT = "data/ModelNet10_mini_repaired"
    # 3. 每个类别选 5 个可修补的 water­tight 样本
    NUM_PER_CLASS = 5

    select_and_repair(INPUT_ROOT, OUTPUT_ROOT, num_per_class=NUM_PER_CLASS)