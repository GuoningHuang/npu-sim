#!/bin/bash
set -e

# Generate workload config
python3 ../unrolled_cascade_workload_gen.py \
    --model1_preset qwen_1.5B \
    --model2_preset qwen_1.5B \
    --pp 16 \
    --tp 2_2 \
    --B 1 \
    --T 1024 \
    --model1_output_tokens 2 \
    --model2_output_tokens 1 \
    --output_name test_unrolled_cascade \
    --output_dir ../output

# Run simulation
../../build/npusim \
    --workload-config ../output/test_unrolled_cascade.json \
    --simulation-config ../../llm/test/simulation_config/default_spec.json \
    --hardware-config ../../llm/test/hardware_config/default/8x8.json \
    --mapping-config ../../llm/test/mapping_config/default_mapping.spec
