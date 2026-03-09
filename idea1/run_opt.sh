#!/bin/bash

../build/npusim \
    --workload-config /home/code/npu-sim/idea1/30B-A3B-afd.json \
    --simulation-config ../llm/test/simulation_config/default_spec.json \
    --hardware-config ../llm/test/hardware_config/default/4x4.json \
    --mapping-config ../llm/test/mapping_config/default_mapping.spec \
    > "log-afd.txt" 2>&1  # 将标准输出和错误输出都重定向到日志
