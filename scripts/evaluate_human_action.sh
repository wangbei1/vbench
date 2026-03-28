#!/bin/bash
# =============================================================================
# evaluate_human_action.sh
#
# 适配 "XXXX_seedY.mp4" 命名格式的视频，评测 VBench human_action 维度。
#
# 原理：human_action.py 从文件名解析动作标签，期望文件名格式为
#       "A person is <action>-<idx>.mp4"
#       而你的视频命名为 "XXXX_seedY.mp4"（XXXX 是 all_dimension.txt 的行号）。
#       本脚本通过创建符号链接目录来适配。
#
# 用法:
#   bash scripts/evaluate_human_action.sh <video_dir> [output_path] [prompt_file]
#
# 参数:
#   video_dir    - 包含 XXXX_seedY.mp4 视频的目录
#   output_path  - 评测结果保存路径（默认: ./evaluation_results/）
#   prompt_file  - prompt 文件路径（默认: prompts/all_dimension.txt）
#
# 示例:
#   bash scripts/evaluate_human_action.sh /path/to/videos_human_action
#   bash scripts/evaluate_human_action.sh /path/to/videos_human_action ./results
# =============================================================================

set -e

VIDEO_DIR="${1:?请指定视频目录，用法: bash $0 <video_dir> [output_path] [prompt_file]}"
OUTPUT_PATH="${2:-./evaluation_results/}"
PROMPT_FILE="${3:-prompts/all_dimension.txt}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VBENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到 VBench 根目录
cd "$VBENCH_ROOT"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "错误: prompt 文件不存在: $PROMPT_FILE"
    exit 1
fi

if [ ! -d "$VIDEO_DIR" ]; then
    echo "错误: 视频目录不存在: $VIDEO_DIR"
    exit 1
fi

# 用绝对路径
VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"

# 创建临时符号链接目录
LINK_DIR=$(mktemp -d "/tmp/vbench_human_action_XXXXXX")
echo "创建符号链接目录: $LINK_DIR"

# 清理函数
cleanup() {
    echo "清理临时目录: $LINK_DIR"
    rm -rf "$LINK_DIR"
}
trap cleanup EXIT

# 用 Python 创建符号链接：XXXX_seedY.mp4 -> "A person is <action>-seedY.mp4"
python3 -c "
import os
import sys

prompt_file = '$PROMPT_FILE'
video_dir = '$VIDEO_DIR'
link_dir = '$LINK_DIR'

# 读取所有 prompt（0-indexed）
with open(prompt_file, 'r') as f:
    prompts = [line.strip() for line in f.readlines()]

# 遍历视频文件
count = 0
for fname in sorted(os.listdir(video_dir)):
    if not fname.endswith('.mp4') and not fname.endswith('.gif'):
        continue

    # 解析文件名: XXXX_seedY.mp4
    stem = os.path.splitext(fname)[0]  # XXXX_seedY
    ext = os.path.splitext(fname)[1]   # .mp4
    parts = stem.split('_seed')
    if len(parts) != 2:
        print(f'警告: 跳过无法解析的文件名: {fname}')
        continue

    idx_str, seed_str = parts
    try:
        idx = int(idx_str)
    except ValueError:
        print(f'警告: 跳过无法解析的文件名: {fname}')
        continue

    if idx >= len(prompts):
        print(f'警告: 索引 {idx} 超出 prompt 范围 ({len(prompts)} 行)，跳过: {fname}')
        continue

    prompt = prompts[idx]

    # 只处理 human_action 的 prompt
    if 'person is' not in prompt.lower():
        print(f'警告: 索引 {idx} 不是 human_action prompt (\"{prompt}\")，跳过: {fname}')
        continue

    # 创建符号链接: 'A person is riding a bike-seed1.mp4'
    link_name = f'{prompt}-seed{seed_str}{ext}'
    src = os.path.join(video_dir, fname)
    dst = os.path.join(link_dir, link_name)
    os.symlink(src, dst)
    count += 1

print(f'成功创建 {count} 个符号链接')
if count == 0:
    print('错误: 没有找到任何 human_action 视频！请检查视频目录和 prompt 文件。')
    sys.exit(1)
"

echo ""
echo "符号链接目录内容示例:"
ls "$LINK_DIR" | head -5
echo "..."
echo "共 $(ls "$LINK_DIR" | wc -l) 个文件"
echo ""

# 运行 VBench 评测
echo "=========================================="
echo "开始 human_action 评测"
echo "视频目录: $LINK_DIR"
echo "输出路径: $OUTPUT_PATH"
echo "=========================================="

python evaluate.py \
    --videos_path "$LINK_DIR" \
    --dimension human_action \
    --output_path "$OUTPUT_PATH" \
    --mode custom_input

echo "=========================================="
echo "评测完成！结果保存在: $OUTPUT_PATH"
echo "=========================================="
