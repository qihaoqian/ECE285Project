#!/usr/bin/env bash

# 遍历 config/deepsdf 目录下所有的 .yaml，并逐个执行 mainsdf.py
for cfg in config/deepsdf/*.yaml; do
    echo ">>> 运行：python main_deepsdf.py --config $cfg"
    python main_deepsdf.py --config "$cfg"
done