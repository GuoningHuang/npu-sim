我现在想实现多个大模型的workload的生成，多个大模型指的是很多个大模型一起运行，代表性的连接特征有级联和并行，级联指的是一个模型的输出作为第二个模型的输入。
现在这个项目的输入是一个workload.json，如/workspace/llm/test/workload_config/playground.json中所示。
第一步，我现在想实现级联形式的负载书写，比如模型a先执行prefill和decode，我可以自定义tokens数，然后第二个模型使用第一个模型的所有输出作为prefill输入，执行decode。
我的要求是第一个模型使用了所有核，第二个模型也使用了所有核，具体实现方法参考下面，你先假设模型a是1.5B的，模型b是3B的：
  Because the simulator lacks native support for multi-phase execution with distinct loops, you must unroll the execution into a single, long linear sequence of
  primitives.

  Recommended Solution: "Unrolled" Workload Generation

  You should modify your generation script (or use a new one) to produce a workload.json with the following structure:

   1. `loop`: Set to 1.
   2. `worklist`: A single list containing all iterations of Model 1 followed by all iterations of Model 2.

  Conceptual Structure of the generated JSON:

    1 {
    2     "vars": { "B": 1, "T_M1": 1024, "T_M2": 1034, ... },
    3     "chips": [{
    4         "cores": [{
    5             "id": 0,
    6             "loop": 1,
    7             "worklist": [
    8                 { "comment": "--- Model 1 Iteration 1 ---", "prims": [...] },
    9                 { "comment": "--- Model 1 Iteration 2 ---", "prims": [...] },
   10                 ...
   11                 { "comment": "--- Model 1 Iteration N ---", "prims": [...] },
   12                 { "comment": "--- Model 2 Iteration 1 ---", "prims": [...] },
   13                 ...
   14                 { "comment": "--- Model 2 Iteration M ---", "prims": [...] }
   15             ]
   16         }]
   17     }]
   18 }

  How to modify `true_cascade_workload_gen.py`:

  Instead of creating a phases list, simply extend the worklist for each core.

   1. Remove the phases structure construction.
   2. Flatten the logic:
       * Generate m1_worklist (unrolled for m1_loop iterations).
       * Generate m2_worklist (unrolled for m2_loop iterations).
       * Combine them: final_worklist = m1_worklist + m2_worklist.
       * Set loop in the JSON to 1.

  This will force the simulator to execute Model 1 entirely before moving on to Model 2, respecting the "output -> input" data flow you define (e.g., using parse_output
  from M1 to parse_input of M2).

