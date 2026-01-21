#!/usr/bin/env python3
"""
Unrolled Cascade Model Workload Generator for WaferAI-SIM

This script generates workload configurations for cascaded LLM inference with
UNROLLED execution - all iterations are expanded into a single linear worklist.

Key Features:
- Model 1 (e.g., 1.5B) runs prefill + decode iterations
- Model 1's output becomes Model 2's (e.g., 3B) prefill input
- Model 2 then executes its decode iterations
- Both models use ALL cores (temporal serialization, not spatial)
- Loop is set to 1, all iterations unrolled into worklist

Execution flow:
  Model 1 Iteration 1 (prefill with T tokens)
  Model 1 Iteration 2 (decode, generate 1 token)
  ...
  Model 1 Iteration N (decode, generate 1 token)
  Model 2 Iteration 1 (prefill with M1's output tokens)
  Model 2 Iteration 2 (decode)
  ...
  Model 2 Iteration M (decode)

Example:
    python unrolled_cascade_workload_gen.py \
        --model1_preset qwen_1.5B \
        --model2_preset qwen_3B \
        --pp 4 --tp 2_2 \
        --B 1 --T 1024 \
        --model1_output_tokens 10 \
        --output_name unrolled_cascade
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
from copy import deepcopy


# Model architecture definitions
MODEL_ARCHITECTURES = {
    "gpt": ["layernorm", "matmul", "attention", "matmul", "residual",
            "layernorm", "matmul", "gelu", "matmul", "residual"],
    "qwen": ["rmsnorm", "matmul_rope", "attention", "matmul", "residual",
             "rmsnorm", "matmul×2", "swiglu", "matmul", "residual"],
    "llama": ["rmsnorm", "matmul_rope", "attention", "matmul", "residual",
              "rmsnorm", "matmul×2", "swiglu", "matmul", "residual"]
}


class UnrolledCascadeWorkloadGenerator:
    """
    Generator for unrolled cascaded model workload configurations.

    Instead of using phases and loops, this generator expands all iterations
    into a single linear worklist with loop=1.
    """

    def __init__(self):
        self.vars = {}
        self.base_vars = set()

    def find_const(self, word: str) -> Tuple[List[int], str]:
        """Parse a variable expression like '3C/4' into components."""
        word_all = word.split("/")
        num = ""
        word_type = ""
        for index, char in enumerate(word_all[0]):
            if char.isdigit():
                num += char
            else:
                word_type = word_all[0][index:]
                break
        word_all[0] = num if num else "1"
        if len(word_all) == 1:
            word_all.append("1")
        return [int(word_all[0]), int(word_all[1])], word_type

    def add_var(self, key: str) -> None:
        """Add a derived variable to vars dict."""
        if key not in self.vars:
            [num, den], var_type = self.find_const(key)
            if var_type in self.vars:
                value = max(1, int(num * self.vars[var_type] / den))
                self.vars[key] = value
            else:
                if "/" in key:
                    keys_first, keys_num = key.split("/")
                else:
                    keys_first = key
                    keys_num = 1
                
                # Sort known base variables by length descending to match longest first
                known_vars = sorted(list(self.base_vars), key=len, reverse=True)
                
                value = 1
                temp_key = keys_first
                while temp_key:
                    matched = False
                    for v in known_vars:
                        if temp_key.startswith(v):
                            value *= self.vars[v]
                            temp_key = temp_key[len(v):]
                            matched = True
                            break
                    if not matched:
                        if temp_key[0].isdigit():
                            # Handle embedded constants
                            num_str = ""
                            for char in temp_key:
                                if char.isdigit():
                                    num_str += char
                                else:
                                    break
                            value *= int(num_str)
                            temp_key = temp_key[len(num_str):]
                        else:
                            # Skip non-alpha characters or unknown variables
                            temp_key = temp_key[1:]
                
                value = max(1, int(value / int(keys_num)))
                self.vars[key] = value

    def cal_size(self, word1: str, word2: str = None, word3: str = None,
                 mul_num: int = None, div_num: int = None) -> str:
        """Calculate combined size expression."""
        [w1_num, w1_den], w1_type = self.find_const(word1)

        if word2:
            [w2_num, w2_den], w2_type = self.find_const(word2)
        else:
            [w2_num, w2_den], w2_type = [1, 1], ""

        if word3:
            [w3_num, w3_den], w3_type = self.find_const(word3)
        else:
            [w3_num, w3_den], w3_type = [1, 1], ""

        num = w1_num * w2_num * w3_num
        den = w1_den * w2_den * w3_den

        if mul_num:
            if mul_num % den == 0:
                num = num * mul_num / den
                den = 1
            elif den % mul_num == 0:
                den = int(den / mul_num)
            else:
                num = num * mul_num

        if div_num:
            if num % div_num == 0:
                num = num / div_num
            elif div_num % num == 0:
                den = int(div_num / num) * den
                num = 1
            else:
                den = den * div_num

        combined_type = f"{w1_type}{w2_type}{w3_type}"
        if num == den:
            return combined_type
        elif num % den == 0:
            return f"{int(num/den)}{combined_type}"
        elif den % num == 0:
            return f"{combined_type}/{int(den/num)}"
        else:
            return f"{int(num)}{combined_type}/{int(den)}"

    def init_model_vars(self, model_config: Dict, pp: int, mn: int, k: int,
                        seq_var: str = "T") -> None:
        """Initialize variables for a specific model with parallelism."""
        self.vars["HS"] = model_config["HS"]
        self.vars["NH"] = model_config["NH"]
        self.vars["DH"] = model_config.get("DH", 128)
        self.vars["KVH"] = model_config.get("KVH", model_config["NH"] // 4)
        self.vars["L"] = model_config["L"]
        self.vars["IS"] = model_config.get("IS", model_config["HS"] * 4)

        self.vars["C"] = self.vars["DH"] * self.vars["NH"]
        self.vars["R"] = self.vars["NH"] // self.vars["KVH"]
        self.vars["P"] = self.vars["HS"]
        self.vars["J"] = self.vars["IS"]
        self.vars["G"] = self.vars["C"] + 2 * self.vars["C"] // self.vars["R"]

        self.vars["pp"] = pp
        self.vars["mn"] = mn
        self.vars["k"] = k
        self.vars["dp"] = 1
        
        for k_var in ["HS", "NH", "DH", "KVH", "L", "IS", "C", "R", "P", "J", "G", "pp", "mn", "k", "dp"]:
            self.base_vars.add(k_var)

    def layer_adapt_pp(self, num_layers: int, pp: int) -> List[int]:
        """Distribute layers across pipeline stages."""
        if pp >= num_layers:
            return [1] * num_layers + [0] * (pp - num_layers)

        cores_num = num_layers // pp
        cores_list = [cores_num] * pp
        additional = num_layers - cores_num * pp
        for i in range(additional):
            cores_list[i] += 1
        return cores_list

    def produce_recv_cast_tag(self, recv_id: int, core_id: int,
                               cast_id: int, base_tag: int = 64) -> Tuple[int, int]:
        """Produce receive and cast tags for communication."""
        if cast_id is not None:
            recv_tag = core_id + base_tag
            cast_tag = cast_id + base_tag
            return recv_tag, cast_tag
        return None, None

    def add_rope(self, B: str, T: str, C: str, NH: str, R: str,
                 sram_indata: str, rp_num: int, prims_list: List,
                 prefix: str = "", weight_prefix: str = "", job_type: int = 0) -> Tuple[List, str, int]:
        """Add RoPE primitive."""
        rp_prim = {
            "type": "rope_forward_pd",
            "B": B,
            "T": T,
            "C": C,
            "NH": NH,
            "R": R,
            "job_type": job_type,
            "sram_address": {
                "indata": sram_indata,
                "outdata": f"{prefix}rope{rp_num}_out"
            },
            "dram_address": {
                "data": f"{weight_prefix}rope{rp_num}_data"
            }
        }
        prims_list.append(rp_prim)
        return prims_list, f"{prefix}rope{rp_num}_out", rp_num + 1

    def process_single_model(self, operation: List[str], core_layer: int,
                              core_id: int, mn_cast_id: int, mn_recv_id: int,
                              k_cast_id: int, k_recv_id: int,
                              is_last_stage: bool,
                              prefix: str = "",
                              weight_prefix: str = "",
                              is_first_stage: bool = True,
                              input_from_host: bool = False,
                              seq_var: str = "T",
                              next_stage_dest: int = None,
                              loop_back_dest: int = None,
                              is_final_iteration: bool = False,
                              is_cascade_output: bool = False,
                              iteration_idx: int = 0,
                              is_prefill: bool = True,
                              prev_output_label: str = None) -> List[Dict]:
        """
        Process a single model's computation for one pipeline stage.

        Args:
            operation: Model architecture operations
            core_layer: Number of layers for this stage
            core_id: This core's ID
            mn_cast_id, mn_recv_id: For tensor parallelism (mn dimension)
            k_cast_id, k_recv_id: For tensor parallelism (k dimension)
            is_last_stage: Whether this is the last pipeline stage
            prefix: Prefix for variable names (e.g., "m1_i0_" for model1 iteration 0)
            is_first_stage: Whether this is the first pipeline stage
            seq_var: Sequence length variable name
            next_stage_dest: Core ID of next pipeline stage
            loop_back_dest: Core ID to loop back for next iteration
            is_final_iteration: Whether this is the final iteration of this model
            is_cascade_output: Whether this iteration's output goes to next model
            iteration_idx: Current iteration index
            is_prefill: Whether this is prefill (True) or decode (False)
        """
        if core_layer == 0:
            return []

        # job_type: 0 = prefill, 1 = decode
        job_type = 0 if is_prefill else 1

        prims_list = []

        ln_num = 1
        mm_num = 1
        att_num = 1
        res_num = 1
        gelu_num = 1
        rp_num = 1
        swiglu_num = 1

        mn = self.vars['mn']
        k = self.vars['k']
        pp = self.vars['pp']

        B = self.cal_size("B", div_num=self.vars['dp'])
        self.add_var(B)

        if k != 1:
            NH = f"NH/{k}"
        elif mn != 1:
            NH = f"NH/{mn}"
        else:
            NH = "NH"
        NH = self.cal_size(NH)
        self.add_var(NH)

        mn_recv_tag, mn_cast_tag = self.produce_recv_cast_tag(mn_recv_id, core_id, mn_cast_id, base_tag=1000)
        k_recv_tag, k_cast_tag = self.produce_recv_cast_tag(k_recv_id, core_id, k_cast_id, base_tag=2000)

        id_den = mn if mn != 1 and k != 1 else 1

        ic = "P"
        pipeline_tag_base = 3000
        loopback_tag_base = 4000
        host_tag = core_id
        stage_size = k * mn
        stage_leader = (core_id % stage_size == 0)

        # Determine if we should use SRAM loopback (no network recv needed)
        # This happens when: first stage, not from host, same-core loopback (loop_back_dest == core_id)
        use_sram_loopback = (is_first_stage and not input_from_host and
                            prev_output_label is not None and
                            loop_back_dest == core_id)

        if use_sram_loopback:
            # Use previous iteration's output directly from SRAM
            sram_indata = f"_{prev_output_label}"
            res_start = prev_output_label
            need_recv = False
            parse_input_added = True  # No parse_input needed
        else:
            # Normal case: need to recv data
            sram_indata = f"_{prefix}input_label"
            res_start = f"{prefix}input_label"
            need_recv = True
            parse_input_added = False

        res_end = res_start

        # Initial parse_input for TP (only for first stage, and only if we need recv)
        if is_first_stage and (mn != 1 or k != 1) and not use_sram_loopback:
            size = f"B{seq_var}{ic}/{mn}"
            size = self.cal_size(size)
            self.add_var(size)
            if input_from_host:
                recv_tag = host_tag
            else:
                recv_tag = loopback_tag_base + core_id
            prim = {
                "type": "parse_input",
                "size": size,
                "sram_address": {
                    "indata": f"{prefix}layernorm1_in",
                    "outdata": f"{prefix}layernorm1_in"
                },
                "recv_cnt": 1,
                "recv_tag": recv_tag
            }
            prims_list.append(prim)
            sram_indata = f"_{prefix}layernorm1_in"
            res_start = f"{prefix}layernorm1_in"
            parse_input_added = True

        for layer_index in range(core_layer):
            for num, operates in enumerate(operation):
                self.add_var(ic)

                if "norm" in operates:
                    # In unrolled mode, bind the incoming INPUT_LABEL to a
                    # per-iteration label before any compute.
                    if need_recv and not parse_input_added and layer_index == 0 and num == 0:
                        size = f"B{seq_var}{ic}/{mn}"
                        size = self.cal_size(size)
                        self.add_var(size)
                        if is_first_stage:
                            recv_tag = host_tag if input_from_host else loopback_tag_base + core_id
                        else:
                            recv_tag = pipeline_tag_base + core_id
                        prims_list.append({
                            "type": "parse_input",
                            "size": size,
                            "sram_address": {
                                "indata": f"{prefix}input_label",
                                "outdata": f"{prefix}input_label"
                            },
                            "recv_cnt": 1,
                            "recv_tag": recv_tag
                        })
                        sram_indata = f"_{prefix}input_label"
                        res_start = f"{prefix}input_label"
                        parse_input_added = True

                    T = f"{seq_var}/{mn}" if mn != 1 else seq_var
                    self.add_var(T)

                    ln_type = "rmsnorm_forward" if operates == "rmsnorm" else "Layernorm_f"
                    ln_prim = {
                        "type": ln_type,
                        "B": B,
                        "T": T,
                        "C": "P",
                        "sram_address": {
                            "indata": sram_indata,
                            "outdata": f"{prefix}layernorm{ln_num}_out"
                        },
                        "dram_address": {
                            "data": f"{prefix}layernorm{ln_num}_data"
                        }
                    }
                    sram_indata = f"{prefix}layernorm{ln_num}_out"
                    prims_list.append(ln_prim)

                    # Switch data for k parallelism
                    if k != 1:
                        sd_indata = self.cal_size(B, T, "P")
                        sd_outdata = self.cal_size(B, T, f"P/{k}")
                        self.add_var(sd_indata)
                        self.add_var(sd_outdata)

                        sd_prim = {
                            "type": "switch_data",
                            "IN": sd_indata,
                            "OUT": sd_outdata,
                            "sram_address": {
                                "indata": f"{prefix}layernorm{ln_num}_out",
                                "outdata": f"{prefix}switch_layernorm{ln_num}_out"
                            }
                        }
                        sram_indata = f"{prefix}switch_layernorm{ln_num}_out"
                        prims_list.append(sd_prim)

                    oc = "P"
                    ln_num += 1

                elif "matmul" in operates:
                    if num == 1:
                        oc = "G" if "rope" in operates else "3C"
                        mm_type = "matmul_forward_pd"
                    elif num == 3:
                        oc = "P"
                        mm_type = "Matmul_f"
                    elif num == 6:
                        oc = "J"
                        mm_type = "Matmul_f"
                    elif num == 8:
                        oc = "P"
                        mm_type = "Matmul_f"
                    else:
                        oc = "P"
                        mm_type = "Matmul_f"

                    mm_time = 1
                    if "×" in operates:
                        mm_time = int(operates.split("×")[1])

                    T = self.cal_size(f"{seq_var}/{mn}")
                    self.add_var(T)

                    mm_ic = self.cal_size(f"{ic}/{k}")
                    self.add_var(mm_ic)

                    mm_oc = self.cal_size(f"{oc}/{mn}")
                    self.add_var(mm_oc)

                    swiglu_indata = ""

                    for mm_index in range(mm_time):
                        mm_split_num = 1
                        mm_indata = sram_indata
                        if mn != 1:
                            mm_indata = f"_{sram_indata}" if not sram_indata.startswith("_") else sram_indata
                        # For matmul×N (N > 1), keep input alive until the last matmul uses it.
                        elif mm_time > 1 and mm_index < mm_time - 1:
                            mm_indata = f"_{sram_indata}" if not sram_indata.startswith("_") else sram_indata

                        mm_outdata = f"{prefix}matmul{mm_num}_{mm_split_num}_out" if mn != 1 else f"{prefix}matmul{mm_num}_out"

                        mm_prim = {
                            "type": mm_type,
                            "B": B,
                            "T": T,
                            "C": mm_ic,
                            "OC": mm_oc,
                            "sram_address": {
                                "indata": mm_indata,
                                "outdata": mm_outdata
                            },
                            "dram_address": {
                                "data": f"{prefix}matmul{mm_num}_data"
                            }
                        }
                        if mm_type == "matmul_forward_pd":
                            mm_prim["R"] = "R"
                            mm_prim["chunk"] = "chunk"
                            mm_prim["job_type"] = job_type
                        prims_list.append(mm_prim)

                        res_end = mm_outdata
                        mn_merge_indata = f"{prefix}matmul{mm_num}_{mm_split_num}_out"

                        if "rope" in operates:
                            prims_list, rope_outdata, rp_num = self.add_rope(
                                B, T, mm_oc, NH, "R", mm_outdata, rp_num, prims_list, prefix, job_type
                            )
                            mn_merge_indata = rope_outdata

                        # MN communication
                        for split_mn_index in range(mn - 1):
                            inout_size = self.cal_size(mm_ic, mm_oc)
                            self.add_var(inout_size)

                            out_prim = {
                                "type": "parse_output",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": f"{prefix}eternal_matmul{mm_num}_{mm_split_num}_w",
                                    "outdata": f"{prefix}DEL_eternal_matmul{mm_num}_{mm_split_num}_w"
                                },
                                "cast": [{"dest": mn_cast_id, "tag": mn_cast_tag}]
                            }
                            in_prim = {
                                "type": "parse_input",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": f"{prefix}eternal_matmul{mm_num}_{mm_split_num+1}_w",
                                    "outdata": f"{prefix}eternal_matmul{mm_num}_{mm_split_num+1}_w"
                                },
                                "recv_cnt": 1,
                                "recv_tag": mn_recv_tag,
                            }

                            mm_indata_next = mm_indata if mm_split_num < mn - 1 else (
                                mm_indata if mm_index < mm_time - 1 else mm_indata.lstrip("_")
                            )

                            mm_prim_next = {
                                "type": mm_type,
                                "B": B,
                                "T": T,
                                "C": mm_ic,
                                "OC": mm_oc,
                                "R": "R",
                                "chunk": "chunk",
                                "job_type": job_type,
                                "sram_address": {
                                    "indata": mm_indata_next,
                                    "outdata": f"{prefix}matmul{mm_num}_{mm_split_num + 1}_out"
                                },
                                "dram_address": {
                                    "data": f"{prefix}matmul{mm_num}_data"
                                }
                            }

                            if core_id % 2 == 0:
                                prims_list.extend([out_prim, in_prim, mm_prim_next])
                            else:
                                prims_list.extend([in_prim, out_prim, mm_prim_next])

                            if "rope" in operates:
                                rope_indata = f"{prefix}matmul{mm_num}_{mm_split_num + 1}_out"
                                prims_list, rope_outdata, rp_num = self.add_rope(
                                    B, T, mm_oc, NH, "R", rope_indata, rp_num, prims_list, prefix, job_type
                                )
                                mn_merge_indata += " " + rope_outdata
                            else:
                                mn_merge_indata += f" {prefix}matmul{mm_num}_{mm_split_num + 1}_out"

                            mm_split_num += 1

                        # MN merge
                        mmm_ic = mm_oc
                        mn_mmm_outdata = f"{prefix}matmul{mm_num}_out"
                        if mn != 1:
                            mmm_prim = {
                                "type": "Merge_matmul",
                                "B": B,
                                "T": T,
                                "C": mmm_ic,
                                "dim": 1,
                                "slice": mn,
                                "sram_address": {
                                    "indata": mn_merge_indata,
                                    "outdata": f"{prefix}matmul{mm_num}_out"
                                }
                            }
                            prims_list.append(mmm_prim)
                            res_end = mn_mmm_outdata

                        # K communication and merge
                        inout_size = self.cal_size(B, T, mmm_ic)
                        self.add_var(inout_size)
                        k_merge_indata = f"{prefix}matmul{mm_num}_out"
                        k_mmm_outdata = f"{prefix}matmul{mm_num}_out"

                        if k != 1:
                            if mn != 1:
                                sd_in = self.cal_size(B, f"{seq_var}/{mn}", mmm_ic, mul_num=mn)
                                sd_out = self.cal_size(B, f"{seq_var}/{mn}", mmm_ic, mul_num=mn, div_num=k)
                                sd_indata = mn_mmm_outdata
                            else:
                                sd_in = self.cal_size(B, seq_var, mm_oc)
                                sd_out = self.cal_size(B, seq_var, mm_oc, div_num=k)
                                sd_indata = mm_outdata if num in [1, 6] else "_" + mm_outdata

                            if num in [1, 6]:
                                sd_prim = {
                                    "type": "switch_data",
                                    "IN": sd_in,
                                    "OUT": sd_out,
                                    "sram_address": {
                                        "indata": sd_indata,
                                        "outdata": f"{prefix}k_matmul{mm_num}_1_out"
                                    }
                                }
                                prims_list.append(sd_prim)
                                inout_size = sd_out
                                k_merge_indata = f"{prefix}k_matmul{mm_num}_1_out"
                                mmm_oc = mmm_ic
                            else:
                                k_merge_indata = f"{prefix}matmul{mm_num}_out"
                                mmm_oc = self.cal_size(mmm_ic, mul_num=mn) if mn != 1 else mmm_ic
                                inout_size = sd_in

                            self.add_var(sd_in)
                            self.add_var(sd_out)

                        if k != 1:
                            k_merge_num = 1
                            for k_index in range(k - 1):
                                if mn == 1:
                                    inout_indata = f"{prefix}k_matmul{mm_num}_{k_merge_num}_out"
                                    inout_outdata = f"{prefix}k_matmul{mm_num}_{k_merge_num}_out"
                                else:
                                    if num in [1, 6]:
                                        inout_indata = "_" + f"{prefix}k_matmul{mm_num}_{k_merge_num}_out"
                                        inout_outdata = f"{prefix}k_matmul{mm_num}_{k_merge_num}_out"
                                    else:
                                        inout_indata = f"{prefix}matmul{mm_num}_out"
                                        inout_outdata = inout_indata

                                out_prim = {
                                    "type": "parse_output",
                                    "size": inout_size,
                                    "sram_address": {
                                        "indata": inout_indata,
                                        "outdata": inout_outdata
                                    },
                                    "cast": [{"dest": k_cast_id, "tag": k_cast_tag}],
                                }
                                in_prim = {
                                    "type": "parse_input",
                                    "size": inout_size,
                                    "sram_address": {
                                        "indata": f"{prefix}k_matmul{mm_num}_{k_merge_num+1}_out",
                                        "outdata": f"{prefix}k_matmul{mm_num}_{k_merge_num+1}_out"
                                    },
                                    "recv_cnt": 1,
                                    "recv_tag": k_recv_tag,
                                }
                                k_merge_indata += f" {prefix}k_matmul{mm_num}_{k_merge_num+1}_out"

                                if int(core_id / id_den) % 2 == 0:
                                    prims_list.extend([out_prim, in_prim])
                                else:
                                    prims_list.extend([in_prim, out_prim])

                                if mn != 1:
                                    mmm_oc = mm_oc if num in [1, 6] else self.cal_size(mm_oc, mul_num=mn)
                                else:
                                    mmm_oc = self.cal_size(mm_oc, div_num=k)

                                # Update res_end for k-merge output (applies when k != 1)
                                # This must be outside the mn check so it works for both mn==1 and mn!=1
                                if num not in [1, 6]:
                                    res_end = f"{prefix}k_matmul{mm_num}_out"

                                self.add_var(mmm_oc)

                                if k_index == k - 2:
                                    T_merge = self.cal_size(f"{seq_var}/{mn}")
                                    mmm_prim = {
                                        "type": "Merge_matmul",
                                        "B": B,
                                        "T": T_merge,
                                        "C": mmm_oc,
                                        "dim": 2,
                                        "slice": k,
                                        "sram_address": {
                                            "indata": k_merge_indata,
                                            "outdata": f"{prefix}k_matmul{mm_num}_out"
                                        }
                                    }
                                    prims_list.append(mmm_prim)
                                    k_mmm_outdata = f"{prefix}k_matmul{mm_num}_out"
                                k_merge_num += 1

                        # Update sram_indata
                        if mm_time != 1:
                            pass
                        elif mn == 1 and k == 1:
                            sram_indata = rope_outdata if "rope" in operates else mm_outdata
                        elif mn != 1 and k == 1:
                            sram_indata = mn_mmm_outdata
                        elif k != 1:
                            sram_indata = k_mmm_outdata

                        if mm_time != 1:
                            if mn == 1 and k == 1:
                                swiglu_indata += mm_outdata + " "
                            elif mn != 1 and k == 1:
                                swiglu_indata += mn_mmm_outdata + " "
                            elif k != 1:
                                swiglu_indata += k_mmm_outdata + " "

                        mm_num += 1

                elif "attention" in operates:
                    ic_num = ic[:-1] if ic.endswith("C") or ic.endswith("P") else ""

                    if mn != 1:
                        inout_size = f"{ic_num}B{seq_var}C/{mn * k}" if ic_num else f"B{seq_var}C/{mn * k}"
                        self.add_var(inout_size)

                        for mn_idx in range(mn - 1):
                            out_prim = {
                                "type": "parse_output",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": sram_indata,
                                    "outdata": sram_indata
                                },
                                "cast": [{"dest": mn_cast_id, "tag": mn_cast_tag}],
                            }
                            in_prim = {
                                "type": "parse_input",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": sram_indata,
                                    "outdata": sram_indata
                                },
                                "recv_cnt": 1,
                                "recv_tag": mn_recv_tag,
                            }
                            if core_id % 2 == 0:
                                prims_list.extend([out_prim, in_prim])
                            else:
                                prims_list.extend([in_prim, out_prim])

                    C = self.cal_size(f"C/{mn * k}")
                    self.add_var(C)

                    att_prim = {
                        "type": "Attention_f_pd",
                        "B": B,
                        "T": seq_var,
                        "C": C,
                        "NH": NH,
                        "DH": "DH",
                        "R": "R",
                        "job_type": job_type,
                        "sram_address": {
                            "indata": sram_indata,
                            "outdata": f"{prefix}attention{att_num}_out"
                        },
                        "dram_address": {
                            "data": f"{prefix}attention{att_num}_data",
                            "out": "TODO"
                        }
                    }
                    prims_list.append(att_prim)
                    sram_indata = f"{prefix}attention{att_num}_out"

                    if mn != 1:
                        inout_size = f"B{seq_var}C/{mn * k}"
                        for mn_idx in range(mn - 1):
                            out_prim = {
                                "type": "parse_output",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": sram_indata,
                                    "outdata": sram_indata
                                },
                                "cast": [{"dest": mn_cast_id, "tag": mn_cast_tag}],
                            }
                            in_prim = {
                                "type": "parse_input",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": sram_indata,
                                    "outdata": sram_indata
                                },
                                "recv_cnt": 1,
                                "recv_tag": mn_recv_tag,
                            }
                            if core_id % 2 == 0:
                                prims_list.extend([out_prim, in_prim])
                            else:
                                prims_list.extend([in_prim, out_prim])

                    oc = "C"
                    att_num += 1

                elif "residual" in operates:
                    N = self.cal_size(B, seq_var, f"{ic}/{mn}")
                    self.add_var(N)

                    res_prim = {
                        "type": "Residual_f",
                        "N": N,
                        "sram_address": {
                            "indata": f"{res_start} {res_end}",
                            "outdata": f"{prefix}residual{res_num}_out"
                        }
                    }

                    # Handle cast for last residual of this stage
                    if num == 9 and layer_index == core_layer - 1:
                        if is_last_stage:
                            casts = []
                            if is_final_iteration and stage_leader:
                                # Emit a single DONE per stage leader on final iteration.
                                casts.append({"dest": -1, "loopout": "true"})
                            if not is_final_iteration and loop_back_dest is not None:
                                # Skip loopback cast if sending to same core (data stays in SRAM)
                                # Only add network cast if sending to different core
                                if loop_back_dest != core_id:
                                    casts.append({
                                        "dest": loop_back_dest,
                                        "tag": loopback_tag_base + loop_back_dest
                                    })
                                # If loop_back_dest == core_id, data stays in SRAM for next iteration
                            if casts:
                                res_prim["cast"] = casts
                        else:
                            # Not last stage: send to next stage.
                            casts = [{
                                "dest": next_stage_dest,
                                "tag": pipeline_tag_base + next_stage_dest
                            }]
                            # Note: Do NOT add loopout here - it will be added as a separate work item
                            # after all primitives are processed. This is because the simulator cannot
                            # correctly handle a work item that has both a regular cast and a loopout.
                            res_prim["cast"] = casts

                    prims_list.append(res_prim)
                    sram_indata = f"_{prefix}residual{res_num}_out"
                    res_start = f"{prefix}residual{res_num}_out"
                    res_num += 1

                elif "gelu" in operates:
                    N = self.cal_size(B, f"{seq_var}/{mn}", f"{ic}/{k}")
                    self.add_var(N)

                    gelu_prim = {
                        "type": "Gelu_f",
                        "N": N,
                        "sram_address": {
                            "indata": sram_indata,
                            "outdata": f"{prefix}gelu{gelu_num}_out"
                        },
                        "dram_address": {
                            "data": f"{prefix}gelu{gelu_num}_data",
                            "out": "TODO"
                        }
                    }
                    prims_list.append(gelu_prim)
                    sram_indata = f"{prefix}gelu{gelu_num}_out"
                    gelu_num += 1

                elif "swiglu" in operates:
                    N = self.cal_size(B, f"{seq_var}/{mn}", f"{ic}/{k}")
                    self.add_var(N)

                    swiglu_prim = {
                        "type": "swiglu_forward",
                        "N": N,
                        "sram_address": {
                            "indata": swiglu_indata[:-1],
                            "outdata": f"{prefix}swiglu{swiglu_num}_out"
                        },
                        "dram_address": {
                            "input": 0,
                            "data": -1
                        }
                    }
                    prims_list.append(swiglu_prim)
                    sram_indata = f"{prefix}swiglu{swiglu_num}_out"
                    swiglu_num += 1

                ic = oc

        # Add loopout marker for non-last stage, final iteration, stage leader
        # This needs to be a separate work item because the simulator cannot
        # handle a work item that has both a regular cast and a loopout
        stage_size = k * mn
        stage_leader = (core_id % stage_size == 0)
        if not is_last_stage and is_final_iteration and stage_leader:
            loopout_marker = {
                "type": "_loopout_marker",
                "cast": [{"dest": -1, "loopout": "true"}]
            }
            prims_list.append(loopout_marker)

        return prims_list

    def split_prims(self, prims: List[Dict], core_id: int) -> List[Dict]:
        """Split primitives into worklist format."""
        if not prims:
            return []

        primslist = []
        current = []
        has_recv = False

        for prim in prims:
            prim_type = prim.get("type")

            # Handle loopout marker - it becomes a separate work item
            if prim_type == "_loopout_marker":
                if current:
                    primslist.append(current)
                    current = []
                    has_recv = False
                # Add the loopout marker as its own work item
                primslist.append([prim])
                continue

            if prim_type == "parse_input":
                if current:
                    primslist.append(current)
                    current = []
                    has_recv = False
                current.append(prim)
                has_recv = True
                continue

            if prim_type == "parse_output":
                if has_recv and current:
                    # If this worklist only received data and immediately forwards it,
                    # keep recv->send in the same work item to avoid extra trailing recvs.
                    if len(current) == 1 and current[0].get("type") == "parse_input":
                        current.append(prim)
                        primslist.append(current)
                        current = []
                        has_recv = False
                        continue
                    primslist.append(current)
                    current = []
                    has_recv = False
                current.append(prim)
                primslist.append(current)
                current = []
                has_recv = False
                continue

            current.append(prim)

        if current:
            primslist.append(current)

        done_worklist = []
        for prim_list in primslist:
            work_prim = []
            one_work = {'recv_cnt': 0}

            for prim in prim_list:
                if "recv_cnt" in prim:
                    one_work["recv_cnt"] = prim.pop("recv_cnt")
                    if "recv_tag" in prim:
                        one_work["recv_tag"] = prim.pop("recv_tag")
                if "cast" in prim:
                    casts = prim.pop("cast")
                    if not isinstance(casts, list):
                        casts = [casts]
                    if "cast" in one_work:
                        one_work["cast"].extend(casts)
                    else:
                        one_work["cast"] = list(casts)
                # Skip the loopout marker itself - it's not a real primitive
                if prim.get("type") == "_loopout_marker":
                    continue
                work_prim.append(prim)

            if "cast" not in one_work:
                one_work["cast"] = []
            one_work["prims"] = work_prim
            done_worklist.append(one_work)

        return done_worklist

    def trim_trailing_recvs(self, worklist: List[Dict]) -> List[Dict]:
        """Drop trailing recv-only work items that would wait forever."""
        if not worklist:
            return worklist
        end = len(worklist)
        while end > 0:
            w = worklist[end - 1]
            if w.get("recv_cnt", 0) <= 0:
                break
            prims = w.get("prims", [])
            if not prims:
                end -= 1
                continue
            if all(p.get("type") == "parse_input" for p in prims):
                end -= 1
                continue
            break
        return worklist[:end]

    def process_source(self, mn: int, k: int, pp: int, dp: int, hs: int) -> List[Dict]:
        """Generate source configuration."""
        source = []
        for dp_index in range(dp):
            for k_index in range(k):
                for mn_index in range(mn):
                    size = "BTP" if mn == 1 else f"BTP/{mn}"
                    if mn == 1:
                        dest_id = dp_index * k * mn * pp + mn_index * mn + k_index
                    else:
                        dest_id = dp_index * k * mn * pp + k_index * mn + mn_index
                    source.append({"dest": dest_id, "size": size})
                    self.add_var(size)
        return source

    def calculate_core_ids(self, stage_idx: int, k_index: int, mn_index: int,
                           mn: int, k: int, pp: int) -> Dict:
        """Calculate core IDs and communication partners."""
        if k != 1 and mn != 1:
            core_id = stage_idx * k * mn + k_index * mn + mn_index
            mn_cast_id = (stage_idx * k * mn + k_index * mn + mn_index + 1
                         if mn_index < mn - 1
                         else stage_idx * k * mn + (k_index - 1) * mn + mn_index + 1)
            mn_recv_id = (stage_idx * k * mn + k_index * mn + mn_index - 1
                         if mn_index > 0
                         else stage_idx * k * mn + (k_index + 1) * mn + mn_index - 1)
            k_cast_id = (stage_idx * k * mn + (k_index + 1) * mn + mn_index
                        if k_index < k - 1
                        else (stage_idx - 1) * k * mn + (k_index + 1) * mn + mn_index)
            k_recv_id = (stage_idx * k * mn + (k_index - 1) * mn + mn_index
                        if k_index > 0
                        else (stage_idx + 1) * k * mn + (k_index - 1) * mn + mn_index)
        elif k == 1 and mn != 1:
            core_id = stage_idx * mn + mn_index
            mn_cast_id = stage_idx * mn + mn_index + 1 if mn_index < mn - 1 else stage_idx * mn
            mn_recv_id = stage_idx * mn + mn_index - 1 if mn_index > 0 else stage_idx * mn + mn - 1
            k_cast_id = None
            k_recv_id = None
        elif k != 1 and mn == 1:
            core_id = stage_idx * k + k_index
            mn_cast_id = None
            mn_recv_id = None
            k_cast_id = stage_idx * k + k_index + 1 if k_index < k - 1 else stage_idx * k
            k_recv_id = stage_idx * k + k_index - 1 if k_index > 0 else stage_idx * k + k - 1
        else:
            core_id = stage_idx
            mn_recv_id = None
            mn_cast_id = None
            k_recv_id = None
            k_cast_id = None

        return {
            "core_id": core_id,
            "mn_cast_id": mn_cast_id,
            "mn_recv_id": mn_recv_id,
            "k_cast_id": k_cast_id,
            "k_recv_id": k_recv_id
        }

    def generate_unrolled_cascade_workload(self,
                                            model1_config: Dict, model2_config: Dict,
                                            model1_type: str, model2_type: str,
                                            pp: int, mn: int, k: int,
                                            batch_size: int, seq_length: int,
                                            model1_output_tokens: int = 10,
                                            model2_output_tokens: int = 1) -> Dict:
        """
        Generate UNROLLED cascade workload.

        All iterations are expanded into a single worklist with loop=1.

        Structure:
        - M1 iteration 0 (prefill): T tokens input
        - M1 iteration 1..N (decode): 1 token input each
        - M2 iteration 0 (prefill): model1_output_tokens input
        - M2 iteration 1..M (decode): 1 token input each
        """

        total_cores = pp * mn * k
        m1_iterations = model1_output_tokens + 1  # 1 prefill + N decode
        m2_iterations = model2_output_tokens + 1  # 1 prefill + M decode

        # Initialize common variables
        self.vars = {
            "B": batch_size,
            "T": seq_length,                    # M1 prefill sequence length
            "T_M1_decode": 1,                   # M1 decode sequence length (1 token)
            "T_M2_prefill": model1_output_tokens,  # M2 prefill = M1's output
            "T_M2_decode": 1,                   # M2 decode sequence length
            "DH": 128,
            "chunk": 1,
            "pp": pp,
            "mn": mn,
            "k": k,
            "dp": 1
        }
        for k_var in ["B", "T", "T_M1_decode", "T_M2_prefill", "T_M2_decode", "DH", "chunk", "pp", "mn", "k", "dp"]:
            self.base_vars.add(k_var)

        # Distribute layers for each model
        m1_layers_per_stage = self.layer_adapt_pp(model1_config["L"], pp)
        m2_layers_per_stage = self.layer_adapt_pp(model2_config["L"], pp)

        print(f"Model 1: {model1_config['L']} layers -> stages: {m1_layers_per_stage}")
        print(f"Model 2: {model2_config['L']} layers -> stages: {m2_layers_per_stage}")
        print(f"All models use the same {total_cores} cores")
        print(f"M1 iterations: {m1_iterations} (1 prefill + {model1_output_tokens} decode)")
        print(f"M2 iterations: {m2_iterations} (1 prefill + {model2_output_tokens} decode)")

        decoder1 = MODEL_ARCHITECTURES.get(model1_type, MODEL_ARCHITECTURES["llama"])
        decoder2 = MODEL_ARCHITECTURES.get(model2_type, MODEL_ARCHITECTURES["llama"])

        cores = []

        for stage_idx in range(pp):
            for k_index in range(k):
                for mn_index in range(mn):
                    # Calculate core IDs
                    ids = self.calculate_core_ids(stage_idx, k_index, mn_index, mn, k, pp)
                    core_id = ids["core_id"]
                    mn_cast_id = ids["mn_cast_id"]
                    mn_recv_id = ids["mn_recv_id"]
                    k_cast_id = ids["k_cast_id"]
                    k_recv_id = ids["k_recv_id"]

                    is_first_stage = (stage_idx == 0)
                    is_last_stage = (stage_idx == pp - 1)

                    # Calculate destinations
                    next_stage_dest = core_id + k * mn if not is_last_stage else None
                    loop_back_dest = core_id - k * mn * (pp - 1) if is_last_stage else None

                    final_worklist = []
                    prev_output_label = None  # Track last output for SRAM loopback

                    # ===== Model 1 iterations (unrolled) =====
                    for iter_idx in range(m1_iterations):
                        # First iteration is prefill (T tokens), rest are decode (1 token)
                        is_prefill = (iter_idx == 0)
                        if is_prefill:
                            seq_var = "T"  # Prefill with T tokens
                        else:
                            seq_var = "T_M1_decode"  # Decode with 1 token

                        self.init_model_vars(model1_config, pp, mn, k)
                        m1_layers = m1_layers_per_stage[stage_idx]

                        is_last_m1_iter = (iter_idx == m1_iterations - 1)
                        prefix = f"m1_i{iter_idx}_"

                        m1_prims = self.process_single_model(
                            operation=decoder1,
                            core_layer=m1_layers,
                            core_id=core_id,
                            mn_cast_id=mn_cast_id,
                            mn_recv_id=mn_recv_id,
                            k_cast_id=k_cast_id,
                            k_recv_id=k_recv_id,
                            is_last_stage=is_last_stage,
                            prefix=prefix,
                            is_first_stage=is_first_stage,
                            input_from_host=is_first_stage and iter_idx == 0,
                            seq_var=seq_var,
                            next_stage_dest=next_stage_dest,
                            loop_back_dest=loop_back_dest,
                            is_final_iteration=False,
                            is_cascade_output=is_last_m1_iter,
                            iteration_idx=iter_idx,
                            is_prefill=is_prefill,
                            prev_output_label=prev_output_label
                        )

                        m1_worklist = self.split_prims(m1_prims, core_id)

                        # Add comment to identify iteration
                        if m1_worklist:
                            m1_worklist[0]["comment"] = f"--- Model 1 Iteration {iter_idx} ({'prefill' if iter_idx == 0 else 'decode'}) ---"

                        final_worklist.extend(m1_worklist)

                        # Track last output label for next iteration (2 residuals per layer)
                        num_residuals = m1_layers * 2
                        prev_output_label = f"{prefix}residual{num_residuals}_out"

                    # ===== Model 2 iterations (unrolled) =====
                    for iter_idx in range(m2_iterations):
                        # First iteration is prefill (M1's output tokens), rest are decode (1 token)
                        is_prefill = (iter_idx == 0)
                        if is_prefill:
                            seq_var = "T_M2_prefill"  # Prefill with M1's output
                        else:
                            seq_var = "T_M2_decode"  # Decode with 1 token

                        self.init_model_vars(model2_config, pp, mn, k)
                        m2_layers = m2_layers_per_stage[stage_idx]

                        is_last_m2_iter = (iter_idx == m2_iterations - 1)
                        prefix = f"m2_i{iter_idx}_"

                        m2_prims = self.process_single_model(
                            operation=decoder2,
                            core_layer=m2_layers,
                            core_id=core_id,
                            mn_cast_id=mn_cast_id,
                            mn_recv_id=mn_recv_id,
                            k_cast_id=k_cast_id,
                            k_recv_id=k_recv_id,
                            is_last_stage=is_last_stage,
                            prefix=prefix,
                            is_first_stage=is_first_stage,
                            input_from_host=False,
                            seq_var=seq_var,
                            next_stage_dest=next_stage_dest,
                            loop_back_dest=loop_back_dest,
                            is_final_iteration=is_last_m2_iter,
                            is_cascade_output=False,
                            iteration_idx=iter_idx,
                            is_prefill=is_prefill,
                            prev_output_label=prev_output_label
                        )

                        # Track last output label for next iteration (2 residuals per layer)
                        num_residuals = m2_layers * 2
                        prev_output_label = f"{prefix}residual{num_residuals}_out"

                        m2_worklist = self.split_prims(m2_prims, core_id)

                        # Add comment to identify iteration
                        if m2_worklist:
                            m2_worklist[0]["comment"] = f"--- Model 2 Iteration {iter_idx} ({'prefill' if iter_idx == 0 else 'decode'}) ---"

                        final_worklist.extend(m2_worklist)

                    # Create core with unrolled worklist (loop=1)
                    worklist = self.trim_trailing_recvs(final_worklist)
                    if not is_first_stage and worklist:
                        first = worklist[0]
                        if first.get("recv_cnt", 0) > 0:
                            worklist = [{
                                "recv_cnt": 0,
                                "cast": [],
                                "prims": []
                            }] + worklist
                    core = {
                        "id": core_id,
                        "loop": 1,
                        "worklist": worklist
                    }
                    cores.append(core)

        # Setup source for Model 1
        self.vars["P"] = model1_config["HS"]
        self.add_var("BTP")
        source = self.process_source(mn, k, pp, 1, model1_config["HS"])

        # Store cascade info
        cascade_info = {
            "cascade_type": "unrolled_cascade",
            "description": "All iterations unrolled into single worklist with loop=1",
            "execution_flow": [
                f"Model 1: {m1_iterations} iterations (1 prefill + {model1_output_tokens} decode)",
                f"Model 2: {m2_iterations} iterations (1 prefill + {model2_output_tokens} decode)",
                f"Total worklist items per core: {m1_iterations + m2_iterations} iteration blocks"
            ],
            "model1": {
                "type": model1_type,
                "HS": model1_config["HS"],
                "L": model1_config["L"],
                "layers_per_stage": m1_layers_per_stage,
                "prefill_seq_length": seq_length,
                "decode_seq_length": 1,
                "iterations": m1_iterations
            },
            "model2": {
                "type": model2_type,
                "HS": model2_config["HS"],
                "L": model2_config["L"],
                "layers_per_stage": m2_layers_per_stage,
                "prefill_seq_length": model1_output_tokens,
                "decode_seq_length": 1,
                "iterations": m2_iterations
            },
            "parallelism": {
                "pp": pp,
                "tp": f"{mn}_{k}",
                "total_cores": total_cores
            }
        }

        config = {
            "vars": self.vars.copy(),
            "cascade_info": cascade_info,
            "pipeline": 1,
            "source": source,
            "chips": [{
                "chip_id": 0,
                "cores": cores
            }]
        }

        return config


def get_model_config(size: str) -> Optional[Dict]:
    """Get predefined model configurations."""
    configs = {
        "qwen_0.5B": {"HS": 1024, "NH": 16, "DH": 64, "KVH": 16, "L": 24, "IS": 2816},
        "qwen_1.5B": {"HS": 2048, "NH": 16, "DH": 128, "KVH": 16, "L": 24, "IS": 5504},
        "qwen_3B":   {"HS": 2560, "NH": 20, "DH": 128, "KVH": 20, "L": 32, "IS": 6912},
        "qwen_7B":   {"HS": 4096, "NH": 32, "DH": 128, "KVH": 32, "L": 32, "IS": 11008},
        "llama_1B":  {"HS": 2048, "NH": 32, "DH": 64, "KVH": 8,  "L": 22, "IS": 8192},
        "llama_3B":  {"HS": 3200, "NH": 32, "DH": 100,"KVH": 8,  "L": 26, "IS": 8640},
        "llama_7B":  {"HS": 4096, "NH": 32, "DH": 128,"KVH": 32, "L": 32, "IS": 11008},
        "llama_13B": {"HS": 5120, "NH": 40, "DH": 128,"KVH": 40, "L": 40, "IS": 13824},
        "tiny":      {"HS": 512,  "NH": 8,  "DH": 64, "KVH": 8,  "L": 4,  "IS": 2048},
        "small":     {"HS": 1024, "NH": 16, "DH": 64, "KVH": 16, "L": 12, "IS": 4096},
    }

    return configs.get(size, None)


def main():
    parser = argparse.ArgumentParser(
        description="Generate UNROLLED cascade workload (all iterations in single worklist)"
    )

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--output_name", type=str, default="unrolled_cascade_workload")

    # Common settings
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--T", type=int, default=1024, help="Model 1 prefill sequence length")
    parser.add_argument("--model1_output_tokens", type=int, default=10,
                        help="Tokens Model 1 generates (becomes Model 2's prefill input)")
    parser.add_argument("--model2_output_tokens", type=int, default=1,
                        help="Tokens Model 2 generates")

    # Parallelism settings (shared by both models)
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    parser.add_argument("--tp", type=str, default="1_1", help="Tensor parallelism (mn_k)")

    # Model 1 settings
    parser.add_argument("--model1_preset", type=str, default=None)
    parser.add_argument("--model1_type", type=str, default="qwen", choices=["gpt", "qwen", "llama"])
    parser.add_argument("--model1_HS", type=int, default=2048)
    parser.add_argument("--model1_NH", type=int, default=16)
    parser.add_argument("--model1_DH", type=int, default=128)
    parser.add_argument("--model1_KVH", type=int, default=16)
    parser.add_argument("--model1_L", type=int, default=24)
    parser.add_argument("--model1_IS", type=int, default=5504)

    # Model 2 settings
    parser.add_argument("--model2_preset", type=str, default=None)
    parser.add_argument("--model2_type", type=str, default="qwen", choices=["gpt", "qwen", "llama"])
    parser.add_argument("--model2_HS", type=int, default=2048)
    parser.add_argument("--model2_NH", type=int, default=16)
    parser.add_argument("--model2_DH", type=int, default=128)
    parser.add_argument("--model2_KVH", type=int, default=16)
    parser.add_argument("--model2_L", type=int, default=24)
    parser.add_argument("--model2_IS", type=int, default=5504)

    args = parser.parse_args()

    # Parse tensor parallelism
    mn, k = [int(x) for x in args.tp.split("_")]

    # Build model configs
    if args.model1_preset:
        model1_config = get_model_config(args.model1_preset)
        if not model1_config:
            print(f"Unknown preset: {args.model1_preset}")
            return
        model1_type = "qwen" if "qwen" in args.model1_preset else "llama"
    else:
        model1_config = {
            "HS": args.model1_HS, "NH": args.model1_NH, "DH": args.model1_DH,
            "KVH": args.model1_KVH, "L": args.model1_L, "IS": args.model1_IS
        }
        model1_type = args.model1_type

    if args.model2_preset:
        model2_config = get_model_config(args.model2_preset)
        if not model2_config:
            print(f"Unknown preset: {args.model2_preset}")
            return
        model2_type = "qwen" if "qwen" in args.model2_preset else "llama"
    else:
        model2_config = {
            "HS": args.model2_HS, "NH": args.model2_NH, "DH": args.model2_DH,
            "KVH": args.model2_KVH, "L": args.model2_L, "IS": args.model2_IS
        }
        model2_type = args.model2_type

    total_cores = args.pp * mn * k
    m1_iterations = args.model1_output_tokens + 1
    m2_iterations = args.model2_output_tokens + 1

    print("=" * 70)
    print("UNROLLED Cascade Model Workload Generator")
    print("=" * 70)
    print(f"\n[All iterations unrolled into single worklist, loop=1]")
    print(f"\nModel 1 - {model1_type.upper()} ({m1_iterations} iterations):")
    print(f"  Prefill: {args.T} tokens")
    print(f"  Decode:  {args.model1_output_tokens} iterations (1 token each)")
    print(f"  Layers:  {model1_config['L']}, Hidden: {model1_config['HS']}")
    print(f"\nModel 2 - {model2_type.upper()} ({m2_iterations} iterations):")
    print(f"  Prefill: {args.model1_output_tokens} tokens (from Model 1)")
    print(f"  Decode:  {args.model2_output_tokens} iterations (1 token each)")
    print(f"  Layers:  {model2_config['L']}, Hidden: {model2_config['HS']}")
    print(f"\nParallelism:")
    print(f"  - Pipeline Parallelism (PP): {args.pp}")
    print(f"  - Tensor Parallelism (TP):   {mn} × {k}")
    print(f"  - Total cores:               {total_cores}")
    print(f"\nBatch size: {args.B}")
    print("=" * 70)

    # Generate workload
    generator = UnrolledCascadeWorkloadGenerator()
    config = generator.generate_unrolled_cascade_workload(
        model1_config=model1_config,
        model2_config=model2_config,
        model1_type=model1_type,
        model2_type=model2_type,
        pp=args.pp,
        mn=mn,
        k=k,
        batch_size=args.B,
        seq_length=args.T,
        model1_output_tokens=args.model1_output_tokens,
        model2_output_tokens=args.model2_output_tokens
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\nWorkload saved to: {output_path}")
    print(f"Total cores: {total_cores}")
    print(f"Total iterations: {m1_iterations + m2_iterations}")

    cascade_info = config.get("cascade_info", {})
    if cascade_info:
        print(f"\nExecution Summary:")
        for flow in cascade_info.get("execution_flow", []):
            print(f"  {flow}")


if __name__ == "__main__":
    main()
