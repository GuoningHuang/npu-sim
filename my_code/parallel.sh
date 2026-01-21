python3 parallel_workload_gen.py \
    --model_preset qwen_1.5B \
    --model_b_preset qwen_1.5B \
    --pp 8 --tp 2_2 \
    --B 1 --T 1024 \
    --avg_output 2 \
    --avg_output_b 2 \
    --output_name parallel_1p5B