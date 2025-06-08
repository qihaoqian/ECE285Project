#!/usr/bin/env bash

# 遍历 config/deepsdf 目录下所有的 .yaml，并逐个执行 mainsdf.py
for cfg in config/ngp/*.yaml; do
    echo ">>> 运行：python main.py --config $cfg"
    python main.py --config "$cfg"
done