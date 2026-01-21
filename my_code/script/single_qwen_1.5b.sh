#!/bin/bash
set -e

# Generate workload config
python3 ../../llm/test/tool_script/workload_autogen.py \
  --B 1 \
  --T 1024 \
  --HS 2048 \
  --NH 16 \
  --DH 128 \
  --KVH 16 \
  --L 24 \
  --IS 5504 \
  --pp 16 \
  --dp 1 \
  --tp 2_2 \
  --avg_output 2 \
  --model qwen \
  --output_name config_1p5b

# Run simulation
../../build/npusim \
    --workload-config /workspace/my_code/test/qwen_config_1p5b_2_2.json \
    --simulation-config ../../llm/test/simulation_config/default_spec.json \
    --hardware-config ../../llm/test/hardware_config/default/8x8.json \
    --mapping-config ../../llm/test/mapping_config/default_mapping.spec