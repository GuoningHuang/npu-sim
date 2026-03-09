# Separate attention/MoE cores mode:
#   Attention cores = dp*tp = 2*(2x2)=8 (IDs 0..7)
#   MoE cores      = moe_ep*moe_tp = 8*1=8 (IDs 8..15)
#   Total cores    = tp*dp + moe_tp*moe_ep = 16
#   Experts per EP group = experts/moe_ep, IS per TP core = IS/moe_tp
# python3 /home/code/npu-sim/idea1/moe_workload_gen_separate.py \
#  --preset qwen3-30b \
#  --B 4 \
#  --micro_batch 2 \
#  --tp 2_2 --dp 2 \
#  --moe_ep 8 \
#  --avg_output 1 \
#  --T 256 \
#  --phase both \
#  --file_name ./30B-A3B.json
python3 /home/code/npu-sim/idea1/moe_workload_gen_without_pp.py \
 --preset qwen3-30b \
 --B 4 \
 --tp 2_2 --dp 2 \
 --avg_output 1 \
 --T 256 \
 --moe_ep 8 \
 --moe_tp 1 \
 --file_name ./30B-A3B-afd.json

python3 /home/code/npu-sim/idea1/moe_workload_gen_entwine-ftd.py \
 --preset qwen3-30b \
 --B 4 \
 --tp 2_2 --dp 4 \
 --avg_output 1 \
 --T 256 \
 --file_name ./30B-A3B-ftd.json

python3 /home/code/npu-sim/idea1/moe_workload_gen_entwine-baseline.py \
 --preset qwen3-30b \
 --B 4 \
 --tp 2_2 --dp 4 \
 --avg_output 1 \
 --T 256 \
 --file_name ./30B-A3B-baseline.json
