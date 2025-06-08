import os
import subprocess
from pathlib import Path

def run_comparison(gt_path, pred_path, config_path):
    cmd = [
        "python",
        "comparative_experiment/compare_mesh_accuracy_deepsdf.py",
        "--gt", gt_path,
        "--pred", pred_path,
        "--config", config_path
    ]
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    # 基础路径
    base_dir = Path("data/ModelNet10_preprocessed")
    workspace_dir = Path("workspace/deepsdf")
    config_dir = Path("config/deepsdf")
    
    # 遍历所有类别目录
    for category_dir in base_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        print(f"\n处理类别: {category_name}")
        
        # 遍历该类别下的所有obj文件
        for obj_file in category_dir.glob("*.obj"):
            # 获取文件名（不含扩展名）
            file_stem = obj_file.stem
            
            # 构建预测结果路径
            pred_path = workspace_dir / file_stem / "results" / "output.ply"
            
            # 构建配置文件路径
            config_path = config_dir / f"{file_stem}.yaml"
            
            # 检查预测结果和配置文件是否存在
            if not pred_path.exists():
                print(f"警告: 预测结果不存在: {pred_path}")
                continue
            if not config_path.exists():
                print(f"警告: 配置文件不存在: {config_path}")
                continue
                
            # 执行比较
            run_comparison(
                str(obj_file),
                str(pred_path),
                str(config_path)
            )

if __name__ == "__main__":
    main() 