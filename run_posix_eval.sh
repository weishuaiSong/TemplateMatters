#!/bin/bash

# 用法: ./run_posix_eval.sh "model1 model2" "dataset1 dataset2" /path/to/templates /path/to/output

# 参数
MODELS=($1)
DATASETS=($2)
TEMPLATE_PATH=$3
OUTPUT_DIR=$4

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 循环模型和数据集组合
for MODEL in ${MODELS[@]}; do
  for DATASET in ${DATASETS[@]}; do
    echo "Running evaluation for model=$MODEL, dataset=$DATASET"



    # 执行 Python 脚本
    python posix.py \
      --model_name "$MODEL" \
      --dataset_name "$DATASET" \
      --template_path "$TEMPLATE_PATH" \
      --output_path "$OUTPUT_DIR"
    echo "Evaluation completed for model=$MODEL, dataset=$DATASET"
    echo "--------------------------------------"
  done
done
