#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil

def create_mini_modelnet10(src_root, dst_root, num_per_class=5, seed=None):
    """
    从 src_root（比如 data/ModelNet10）中每个类别的 train 子目录随机抽取 num_per_class 个 .off 文件，
    并将它们复制到 dst_root（比如 data/ModelNet10_mini）中的对应目录下。
    
    Args:
        src_root (str): 原始 ModelNet10 数据集根目录，例如 "data/ModelNet10"
        dst_root (str): 要输出的 mini 数据集根目录，例如 "data/ModelNet10_mini"
        num_per_class (int): 每个类别抽取的样本数量
        seed (int or None): 随机种子，若不需要固定结果可设为 None
    """
    if seed is not None:
        random.seed(seed)

    # 确保输出目录存在
    os.makedirs(dst_root, exist_ok=True)

    # 遍历 src_root 下的所有一级子目录（即各个类别名：bathtub, bed, chair, ...）
    for category in os.listdir(src_root):
        category_path = os.path.join(src_root, category)
        if not os.path.isdir(category_path):
            # 如果不是文件夹（比如有 README.txt），就跳过
            continue

        train_dir = os.path.join(category_path, "train")
        if not os.path.isdir(train_dir):
            # 如果某个类别下没有 train 子目录，也跳过
            print(f"[WARN] {category} 下没有 train 文件夹，跳过。")
            continue

        # 列出 train 目录下所有后缀为 .off 的文件
        all_off = [
            f for f in os.listdir(train_dir)
            if os.path.isfile(os.path.join(train_dir, f)) and f.lower().endswith(".off")
        ]

        if len(all_off) < num_per_class:
            print(f"[WARN] 类别 {category} 下的 train 文件少于 {num_per_class} 个 (.off)；仅发现 {len(all_off)} 个，全部拷贝。")
            selected = all_off[:]  # 全部拷贝
        else:
            # 随机选取 num_per_class 个
            selected = random.sample(all_off, num_per_class)

        # 准备目标目录：dst_root/<category>
        dst_category = os.path.join(dst_root, category)
        os.makedirs(dst_category, exist_ok=True)

        # 复制文件
        for fname in selected:
            src_file = os.path.join(train_dir, fname)
            dst_file = os.path.join(dst_category, fname)
            shutil.copy2(src_file, dst_file)

        print(f"[INFO] 类别 {category}: 从 {len(all_off)} 个文件中随机选取 {len(selected)} 个 → 已复制到 {dst_category}")

    print(f"[INFO] mini 数据集已创建完毕，路径在：{dst_root}")


if __name__ == "__main__":
    # 例子：假设当前工作目录下有 data/ModelNet10
    SRC_ROOT = "data/ModelNet10"
    DST_ROOT = "data/ModelNet10_mini"
    NUM_PER_CLASS = 5
    SEED = 42  # 如果想要每次都“一样”的随机结果，可以设置一个固定种子。否则设为 None

    create_mini_modelnet10(SRC_ROOT, DST_ROOT, num_per_class=NUM_PER_CLASS, seed=SEED)
