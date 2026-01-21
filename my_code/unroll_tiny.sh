python3 unrolled_cascade_workload_gen.py \
  --output_name tiny_16l_cascade \
  --B 1 \
  --T 8 \
  --model1_output_tokens 2 \
  --model2_output_tokens 2 \
  --pp 2 \
  --tp 1_2 \
  \
  --model1_type qwen \
  --model1_HS 256 \
  --model1_NH 4 \
  --model1_DH 64 \
  --model1_KVH 4 \
  --model1_L 2 \
  --model1_IS 1024 \
  \
  --model2_type qwen \
  --model2_HS 256 \
  --model2_NH 4 \
  --model2_DH 64 \
  --model2_KVH 4 \
  --model2_L 2 \
  --model2_IS 1024

