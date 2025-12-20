#!/bin/bash

# DivPrune 可视化对比示例脚本
# 用于对比指定图片在prune前后的区别

# 设置参数
IMAGE_PATH="prune_example.PNG"  # 修改为你的图片路径
MODEL_PATH="liuhaotian/llava-v1.5-13b"  # 或者使用 llava-v1.5-13b, llava-v1.6-vicuna-7b
PRUNE_RATIO=0.098  # 剪枝比例，论文中默认为0.098
OUTPUT_DIR="./prune_comparison_results"  # 输出目录
PROMPT="Describe this image in detail."  # 可选：自定义提示词

# 运行可视化脚本
python visualize_prune_comparison.py \
    --image_path "$IMAGE_PATH" \
    --model_path "$MODEL_PATH" \
    --prune_ratio $PRUNE_RATIO \
    --output_dir "$OUTPUT_DIR" \
    --prompt "$PROMPT" \
    --device cuda

echo ""
echo "=========================================="
echo "可视化完成！"
echo "=========================================="
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "生成的文件："
echo "  1. original_image.jpg - 原始图片"
echo "  2. token_selection.png - token选择可视化（绿色=保留，红色=剪枝）"
echo "  3. attention_comparison.png - 注意力热力图对比"
echo "  4. feature_similarity.png - token间相似度矩阵"
echo "  5. statistics.json - 详细统计报告"
echo ""
echo "你可以尝试不同的剪枝比例："
echo "  - 0.05  (保留95%的tokens)"
echo "  - 0.098 (保留90.2%的tokens, 论文默认)"
echo "  - 0.2   (保留80%的tokens)"
echo "  - 0.5   (保留50%的tokens)"
echo "=========================================="
