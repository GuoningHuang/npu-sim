# Expert-wise mode: each core holds E_N/ep complete experts, All-to-All dispatch/combine
python3 /home/code/npu-sim/llm/test/tool_script/moe_workload_gen_entwine.py \
 --preset qwen3-30b \
 --B 4 \
 --tp 2_2 --dp 4   \
 --avg_output 3 \
 --phase both \
 --ep_mode expert_wise \
 --file_name /home/code/npu-sim/entewine/30B-A3B.json

# MoEntwine FTD mode: FTD-local All-to-All (dp cores per FTD), each core holds E_N/dp experts
python3 /home/code/npu-sim/llm/test/tool_script/moe_workload_gen_moentwine.py \
 --preset qwen3-30b \
 --B 4 \
 --tp 2_2 --dp 4   \
 --avg_output 3 \
 --phase both \
 --file_name /home/code/npu-sim/entewine/30B-A3B-ftd.json
