#!/bin/bash

# --- 1. 配置参数 ---
MODEL_NAME="qwen3-30b"
EP=$1
TP=$2
DP=$3
PP=$4

# 自动生成文件名，避免手动硬编码路径
LOG_DIR = "${MODEL_NAME}_DP${DP}_PP${PP}_EP${EP}_TP${TP}"
WORKLOAD_FILE="./${MODEL_NAME}_EP${EP}_TP${TP}_DP${DP}_PP${PP}.json"
LOG_FILE="${MODEL_NAME}_DP${DP}_PP${PP}_EP${EP}_TP${TP}.txt"

# --- 2. 生成 Workload 配置 ---
echo "Generating workload: $WORKLOAD_FILE ..."

# 注意：变量引用需要加 $ 符号
python3 /home/code/npu-sim/llm/test/tool_script/moe_workload_gen.py \
    --preset "$MODEL_NAME" \
    --dp "$DP" \
    --pp "$PP" \
    --ep "$EP" \
    --tp "$TP" \
    --file_name "$WORKLOAD_FILE"

# 检查上一步是否成功
if [ $? -ne 0 ]; then
    echo "Error: Workload generation failed!"
    exit 1
fi

# --- 3. 运行 NPU 仿真 ---
echo "Starting simulation, logging to $LOG_FILE ..."

../build/npusim \
    --workload-config "$WORKLOAD_FILE" \
    --simulation-config ../llm/test/simulation_config/default_spec.json \
    --hardware-config ../llm/test/hardware_config/default/4x4.json \
    --mapping-config ../llm/test/mapping_config/default_mapping.spec \
    > "$LOG_FILE" 2>&1  # 将标准输出和错误输出都重定向到日志

echo "Simulation completed successfully."

# --- 4. 移动文件逻辑 (新增) ---
DEST_BASE_DIR="../run_logs" # 归档的总目录
FINAL_DEST="${DEST_BASE_DIR}/${MODEL_NAME}_DP${DP}_PP${PP}_EP${EP}_TP${TP}"

echo "Moving files to ${FINAL_DEST} ..."

# 创建目标文件夹（-p 确保父目录存在且重复运行不报错）
mkdir -p "$FINAL_DEST"

# # 1. 开启扩展通配符功能
# shopt -s extglob
# mv !(run.sh) "$FINAL_DEST/"
# shopt -u extglob
mv * "$FINAL_DEST/"

echo "Simulation completed successfully. Files are moved to $FINAL_DEST"
