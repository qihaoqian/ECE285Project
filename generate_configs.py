#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ====================== 配置部分 ======================
# 请确认以下路径在你本地工程中是存在的：
DATA_ROOT     = "data/ModelNet10_preprocessed"
TEMPLATE_YAML = "config/ngp/template.yaml"   # 你提前复制好的模板
OUTPUT_DIR    = "config/ngp"                 # 脚本会把所有生成的 yaml 都放在这里
# ====================================================

def main():
    # 1. 先检查 template.yaml 是否存在
    if not os.path.isfile(TEMPLATE_YAML):
        print(f"[Error] 找不到模板文件: {TEMPLATE_YAML}")
        print("请先把已有的某个 config（如 bathtub_0025.yaml）复制为 template.yaml，并填好正确的 workspace & dataset_path")
        return

    # 2. 读入 template.yaml 的内容（以行列表保存）
    with open(TEMPLATE_YAML, 'r', encoding='utf-8') as f:
        template_lines = f.readlines()

    # 3. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. 递归遍历 data/ModelNet10_preprocessed 目录，查找所有 .obj
    for root, dirs, files in os.walk(DATA_ROOT):
        # 只处理后缀为 .obj 的文件
        for fname in files:
            if not fname.lower().endswith(".obj"):
                continue

            # 当前 mesh 文件的绝对（或相对）路径，比如 "data/ModelNet10_preprocessed/bathtub/bathtub_0025.obj"
            obj_rel_path = os.path.join(root, fname).replace("\\", "/")  # 统一用 "/" 分隔

            # base_name = "bathtub_0025"
            base_name = os.path.splitext(fname)[0]

            # category = "bathtub"（即 obj 所在的上层目录名）
            category = os.path.basename(root)

            # 计算新的 workspace 路径 和 dataset_path
            new_workspace   = f"workspace/ngp/{base_name}"
            new_dataset_path= f"data/ModelNet10_preprocessed/{category}/{fname}"

            # 拼出输出的 yaml 文件名，比如 "config/ngp/bathtub_0025.yaml"
            out_yaml = os.path.join(OUTPUT_DIR, f"{base_name}.yaml")

            # 5. 对 template_lines 做替换
            new_lines = []
            for line in template_lines:
                stripped = line.lstrip()  # 不含缩进的行内容
                indent = line[:len(line) - len(stripped)]  # 原始行前面的缩进

                # 如果检测到包含 “workspace:”，则整行替换
                if stripped.startswith("workspace:"):
                    new_lines.append(f"{indent}workspace: {new_workspace}\n")
                # 如果检测到 “dataset_path:”，则整行替换
                elif stripped.startswith("dataset_path:"):
                    new_lines.append(f"{indent}dataset_path: {new_dataset_path}\n")
                else:
                    # 其他行保持不变
                    new_lines.append(line)

            # 6. 把 new_lines 写入到新的 yaml 文件
            with open(out_yaml, 'w', encoding='utf-8') as outf:
                outf.writelines(new_lines)

            print(f"--> 已生成: {out_yaml}")

    print("\n全部完成。请检查 config/ngp 目录下是否都生成了对应 .yaml 文件。")


if __name__ == "__main__":
    main()
