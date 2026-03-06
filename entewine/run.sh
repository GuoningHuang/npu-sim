#!/bin/bash

../build/npusim \
    --workload-config /home/code/npu-sim/entewine/30B-A3B-ftd.json \
    --simulation-config ../llm/test/simulation_config/default_spec.json \
    --hardware-config ../llm/test/hardware_config/default/4x4.json \
    --mapping-config ../llm/test/mapping_config/MOE-baseline.spec \
    > "log-baseline-ftd.txt" 2>&1  # 将标准输出和错误输出都重定向到日志
