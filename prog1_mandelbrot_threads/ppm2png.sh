#!/bin/bash

# 遍历当前文件夹下的所有 .ppm 文件
for ppm_file in *.ppm; do
    if [[ -f "$ppm_file" ]]; then
        # 获取文件名（不包括扩展名）
        base_name="${ppm_file%.*}"

        # 转换为 .png 文件
        ffmpeg -i "$ppm_file" "${base_name}.png"

        echo "Converted $ppm_file to ${base_name}.png"
    fi
done
