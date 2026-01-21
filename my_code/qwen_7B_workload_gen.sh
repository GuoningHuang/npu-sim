python3 ../llm/test/tool_script/workload_autogen.py \
  --B 2 \
  --T 256 \
  --DH 128 \
  --NH 32 \
  --KVH 32 \
  --HS 4096 \
  --IS 11008 \
  --L 32 \
  --pp 1 \
  --dp 2 \
  --tp 1_1 \
  --avg_output 50 \
  --model gpt \
  --output_name config_7b

