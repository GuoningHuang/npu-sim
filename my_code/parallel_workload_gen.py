#!/usr/bin/env python3
"""
Dual-Model Parallel Workload Generator for WaferAI-SIM

This script generates workload configurations for two LLM models running
in PARALLEL on separate cores (spatial parallelism).

Key Features:
- Model A runs on the first half of cores: [0, pp*mn*k)
- Model B runs on the second half of cores: [pp*mn*k, 2*pp*mn*k)
- Both models share the same pp/tp configuration
- Each model has independent loop value (avg_output + 1)
- Communication tags are isolated to prevent conflicts

Core Layout:
    Total cores = 2 * pp * mn * k
    Model A: cores [0, pp*mn*k)         loop = avg_output_a + 1
    Model B: cores [pp*mn*k, 2*pp*mn*k) loop = avg_output_b + 1

Example:
    python parallel_workload_gen.py \
        --model qwen --L 4 --avg_output 5 \
        --model_b llama --L_b 4 --avg_output_b 8 \
        --pp 2 --tp 1_1 \
        --output_dir ./test_output \
        --output_name dual_test
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


class ParallelWorkloadGenerator:
    """
    Generator for dual-model parallel workload configurations.

    Two models run simultaneously on separate core sets:
    - Model A: cores [0, cores_per_model)
    - Model B: cores [cores_per_model, 2*cores_per_model)
    """

    def __init__(self):
        self.vars = {}
        self.base_vars = set()
        self.datatype = 0

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
                            num_str = ""
                            for char in temp_key:
                                if char.isdigit():
                                    num_str += char
                                else:
                                    break
                            value *= int(num_str)
                            temp_key = temp_key[len(num_str):]
                        else:
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
                        batch_size: int, seq_length: int, avg_output: int,
                        suffix: str = "") -> Dict:
        """
        Initialize variables for a specific model.

        Args:
            model_config: Model architecture parameters (HS, NH, DH, KVH, L, IS)
            pp, mn, k: Parallelism settings
            batch_size, seq_length: Input dimensions
            avg_output: Average output tokens (determines loop count)
            suffix: Variable suffix for model B (e.g., "_b")

        Returns:
            Dictionary of model variables
        """
        model_vars = {}

        # Basic dimensions
        model_vars[f"B{suffix}"] = batch_size
        model_vars[f"T{suffix}"] = seq_length
        model_vars[f"DH{suffix}"] = model_config.get("DH", 128)
        model_vars[f"NH{suffix}"] = model_config["NH"]
        model_vars[f"KVH{suffix}"] = model_config.get("KVH", model_config["NH"] // 4)
        model_vars[f"HS{suffix}"] = model_config["HS"]
        model_vars[f"L{suffix}"] = model_config["L"]
        model_vars[f"IS{suffix}"] = model_config.get("IS", model_config["HS"] * 4)

        # Derived dimensions
        model_vars[f"C{suffix}"] = model_vars[f"DH{suffix}"] * model_vars[f"NH{suffix}"]
        model_vars[f"R{suffix}"] = model_vars[f"NH{suffix}"] // model_vars[f"KVH{suffix}"]
        model_vars[f"P{suffix}"] = model_vars[f"HS{suffix}"]
        model_vars[f"J{suffix}"] = model_vars[f"IS{suffix}"]
        model_vars[f"G{suffix}"] = (model_vars[f"C{suffix}"] +
                                     2 * model_vars[f"C{suffix}"] // model_vars[f"R{suffix}"])

        # Loop count
        model_vars[f"loop{suffix}"] = avg_output + 1
        model_vars[f"chunk{suffix}"] = 1

        return model_vars

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
                 prefix: str = "", weight_prefix: str = "") -> Tuple[List, str, int]:
        """Add RoPE primitive."""
        rp_prim = {
            "type": "rope_forward_pd",
            "B": B,
            "T": T,
            "C": C,
            "NH": NH,
            "R": R,
            "job_type": 2,
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

    def process_single_model_core(self, operation: List[str], core_layer: int,
                                   core_id: int, mn_cast_id: int, mn_recv_id: int,
                                   k_cast_id: int, k_recv_id: int,
                                   is_last_stage: bool,
                                   pp: int, mn: int, k: int,
                                   core_id_offset: int = 0,
                                   mn_tag_base: int = 1000,
                                   k_tag_base: int = 2000,
                                   prefix: str = "") -> List[Dict]:
        """
        Process a single model's computation for one pipeline stage.

        Args:
            operation: Model architecture operations
            core_layer: Number of layers for this stage
            core_id: This core's ID (with offset applied)
            mn_cast_id, mn_recv_id: For tensor parallelism (mn dimension)
            k_cast_id, k_recv_id: For tensor parallelism (k dimension)
            is_last_stage: Whether this is the last pipeline stage
            pp, mn, k: Parallelism parameters
            core_id_offset: Core ID offset for this model
            mn_tag_base, k_tag_base: Communication tag bases (for isolation)
            prefix: Variable prefix for this model
        """
        if core_layer == 0:
            return []

        prims_list = []

        ln_num = 1
        mm_num = 1
        att_num = 1
        res_num = 1
        gelu_num = 1
        rp_num = 1
        swiglu_num = 1

        B = self.cal_size("B", div_num=self.vars.get('dp', 1))
        self.add_var(B)

        if k != 1:
            NH = f"NH/{k}"
        elif mn != 1:
            NH = f"NH/{mn}"
        else:
            NH = "NH"
        NH = self.cal_size(NH)
        self.add_var(NH)

        # Use model-specific tag bases for communication isolation
        mn_recv_tag, mn_cast_tag = self.produce_recv_cast_tag(mn_recv_id, core_id, mn_cast_id, base_tag=mn_tag_base)
        k_recv_tag, k_cast_tag = self.produce_recv_cast_tag(k_recv_id, core_id, k_cast_id, base_tag=k_tag_base)

        id_den = mn if mn != 1 and k != 1 else 1

        ic = "P"
        sram_indata = f"_{prefix}input_label"
        res_start = f"{prefix}input_label"
        res_end = res_start

        # Initial parse_input for TP
        if mn != 1 or k != 1:
            size = f"BT{ic}/{mn}"
            size = self.cal_size(size)
            self.add_var(size)
            prim = {
                "type": "parse_input",
                "size": size,
                "sram_address": {
                    "indata": f"{prefix}layernorm1_in",
                    "outdata": f"{prefix}layernorm1_in"
                },
                "recv_cnt": 1
            }
            prims_list.append(prim)
            sram_indata = f"_{prefix}layernorm1_in"
            res_start = f"{prefix}layernorm1_in"

        for layer_index in range(core_layer):
            for num, operates in enumerate(operation):
                self.add_var(ic)

                if "norm" in operates:
                    T = f"T/{mn}" if mn != 1 else "T"
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
                    if mn * k == 1:
                        ln_prim['recv_cnt'] = 1
                    sram_indata = f"{prefix}layernorm{ln_num}_out"
                    prims_list.append(ln_prim)

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

                    T = self.cal_size(f"T/{mn}")
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
                            mm_prim["job_type"] = 2
                        prims_list.append(mm_prim)

                        res_end = mm_outdata
                        mn_merge_indata = f"{prefix}matmul{mm_num}_{mm_split_num}_out"

                        if "rope" in operates:
                            prims_list, rope_outdata, rp_num = self.add_rope(
                                B, T, mm_oc, NH, "R", mm_outdata, rp_num, prims_list, prefix
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
                                    "indata": f"{prefix}DEL_eternal_matmul{mm_num}_{mm_split_num}_w",
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
                                "job_type": 2,
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
                                    B, T, mm_oc, NH, "R", rope_indata, rp_num, prims_list, prefix
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
                                sd_in = self.cal_size(B, f"T/{mn}", mmm_ic, mul_num=mn)
                                sd_out = self.cal_size(B, f"T/{mn}", mmm_ic, mul_num=mn, div_num=k)
                                sd_indata = mn_mmm_outdata
                            else:
                                sd_in = self.cal_size(B, "T", mm_oc)
                                sd_out = self.cal_size(B, "T", mm_oc, div_num=k)
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

                                if num not in [1, 6]:
                                    res_end = f"{prefix}k_matmul{mm_num}_out"

                                self.add_var(mmm_oc)

                                if k_index == k - 2:
                                    T_merge = self.cal_size(f"T/{mn}")
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
                        inout_size = f"{ic_num}BTC/{mn * k}" if ic_num else f"BTC/{mn * k}"
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
                        "T": "T",
                        "C": C,
                        "NH": NH,
                        "DH": "DH",
                        "R": "R",
                        "job_type": 2,
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
                        inout_size = f"BTC/{mn * k}"
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
                    N = self.cal_size(B, "T", f"{ic}/{mn}")
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
                            # Last stage: loop back to first stage
                            loop_id = core_id - k * mn * (pp - 1)
                            res_prim["cast"] = [
                                {"dest": -1, "loopout": "true"},
                                {"dest": loop_id, "loopout": "false"}
                            ]
                        else:
                            # Not last stage: send to next stage
                            res_prim["cast"] = [{"dest": core_id + k * mn}]

                    prims_list.append(res_prim)
                    sram_indata = f"_{prefix}residual{res_num}_out"
                    res_start = f"{prefix}residual{res_num}_out"
                    res_num += 1

                elif "gelu" in operates:
                    N = self.cal_size(B, f"T/{mn}", f"{ic}/{k}")
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
                    N = self.cal_size(B, f"T/{mn}", f"{ic}/{k}")
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

        return prims_list

    def split_prims(self, prims: List[Dict], core_id: int) -> List[Dict]:
        """Split primitives into worklist format."""
        if not prims:
            return []

        primslist = []
        first_index = 0

        for index, prim in enumerate(prims):
            if prim["type"] == "parse_input" and index - first_index != 0:
                primslist.append(prims[first_index:index])
                first_index = index
            if prim["type"] == "parse_output" and index != len(prims) - 1:
                primslist.append(prims[first_index:index + 1])
                first_index = index + 1

        primslist.append(prims[first_index:])

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
                    one_work["cast"] = prim.pop("cast")
                work_prim.append(prim)

            if "cast" not in one_work:
                one_work["cast"] = []
            one_work["prims"] = work_prim
            done_worklist.append(one_work)

        return done_worklist

    def calculate_core_ids(self, stage_idx: int, k_index: int, mn_index: int,
                           mn: int, k: int, pp: int, core_id_offset: int = 0) -> Dict:
        """
        Calculate core IDs and communication partners.

        Args:
            stage_idx, k_index, mn_index: Position indices
            mn, k, pp: Parallelism parameters
            core_id_offset: Offset for this model's core IDs
        """
        if k != 1 and mn != 1:
            base_id = stage_idx * k * mn + k_index * mn + mn_index
            core_id = base_id + core_id_offset

            mn_cast_id = (core_id_offset + stage_idx * k * mn + k_index * mn + mn_index + 1
                         if mn_index < mn - 1
                         else core_id_offset + stage_idx * k * mn + (k_index - 1) * mn + mn_index + 1)
            mn_recv_id = (core_id_offset + stage_idx * k * mn + k_index * mn + mn_index - 1
                         if mn_index > 0
                         else core_id_offset + stage_idx * k * mn + (k_index + 1) * mn + mn_index - 1)
            k_cast_id = (core_id_offset + stage_idx * k * mn + (k_index + 1) * mn + mn_index
                        if k_index < k - 1
                        else core_id_offset + (stage_idx - 1) * k * mn + (k_index + 1) * mn + mn_index)
            k_recv_id = (core_id_offset + stage_idx * k * mn + (k_index - 1) * mn + mn_index
                        if k_index > 0
                        else core_id_offset + (stage_idx + 1) * k * mn + (k_index - 1) * mn + mn_index)
        elif k == 1 and mn != 1:
            base_id = stage_idx * mn + mn_index
            core_id = base_id + core_id_offset
            mn_cast_id = core_id_offset + stage_idx * mn + mn_index + 1 if mn_index < mn - 1 else core_id_offset + stage_idx * mn
            mn_recv_id = core_id_offset + stage_idx * mn + mn_index - 1 if mn_index > 0 else core_id_offset + stage_idx * mn + mn - 1
            k_cast_id = None
            k_recv_id = None
        elif k != 1 and mn == 1:
            base_id = stage_idx * k + k_index
            core_id = base_id + core_id_offset
            mn_cast_id = None
            mn_recv_id = None
            k_cast_id = core_id_offset + stage_idx * k + k_index + 1 if k_index < k - 1 else core_id_offset + stage_idx * k
            k_recv_id = core_id_offset + stage_idx * k + k_index - 1 if k_index > 0 else core_id_offset + stage_idx * k + k - 1
        else:
            core_id = stage_idx + core_id_offset
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

    def process_source_single(self, mn: int, k: int, pp: int, dp: int,
                               core_id_offset: int = 0) -> List[Dict]:
        """Generate source configuration for a single model."""
        source = []
        for dp_index in range(dp):
            for k_index in range(k):
                for mn_index in range(mn):
                    size = "BTP" if mn == 1 else f"BTP/{mn}"
                    if mn == 1:
                        dest_id = dp_index * k * mn * pp + mn_index * mn + k_index
                    else:
                        dest_id = dp_index * k * mn * pp + k_index * mn + mn_index
                    dest_id += core_id_offset
                    source.append({"dest": dest_id, "size": size})
                    self.add_var(size)
        return source

    def process_source_dual(self, mn: int, k: int, pp: int, dp: int,
                            cores_per_model: int) -> List[Dict]:
        """
        Generate source configuration for dual models.

        Model A: dest [0, cores_per_model)
        Model B: dest [cores_per_model, 2*cores_per_model)
        """
        source_a = self.process_source_single(mn, k, pp, dp, core_id_offset=0)
        source_b = self.process_source_single(mn, k, pp, dp, core_id_offset=cores_per_model)
        return source_a + source_b

    def process_cores_single(self, model_config: Dict, model_type: str,
                              pp: int, mn: int, k: int, dp: int,
                              loop_value: int,
                              core_id_offset: int = 0,
                              mn_tag_base: int = 1000,
                              k_tag_base: int = 2000,
                              prefix: str = "") -> List[Dict]:
        """
        Generate cores for a single model.

        Args:
            model_config: Model architecture config
            model_type: "gpt", "qwen", or "llama"
            pp, mn, k, dp: Parallelism parameters
            loop_value: Loop count for this model
            core_id_offset: Core ID offset
            mn_tag_base, k_tag_base: Communication tag bases
            prefix: Variable prefix
        """
        layers_per_stage = self.layer_adapt_pp(model_config["L"], pp)
        decoder = MODEL_ARCHITECTURES.get(model_type, MODEL_ARCHITECTURES["llama"])

        cores = []
        for dp_index in range(dp):
            for stage_idx, core_layer in enumerate(layers_per_stage):
                for k_index in range(k):
                    for mn_index in range(mn):
                        ids = self.calculate_core_ids(
                            stage_idx, k_index, mn_index, mn, k, pp, core_id_offset
                        )
                        core_id = ids["core_id"]
                        mn_cast_id = ids["mn_cast_id"]
                        mn_recv_id = ids["mn_recv_id"]
                        k_cast_id = ids["k_cast_id"]
                        k_recv_id = ids["k_recv_id"]

                        is_last_stage = (stage_idx == pp - 1)

                        prims = self.process_single_model_core(
                            operation=decoder,
                            core_layer=core_layer,
                            core_id=core_id,
                            mn_cast_id=mn_cast_id,
                            mn_recv_id=mn_recv_id,
                            k_cast_id=k_cast_id,
                            k_recv_id=k_recv_id,
                            is_last_stage=is_last_stage,
                            pp=pp, mn=mn, k=k,
                            core_id_offset=core_id_offset,
                            mn_tag_base=mn_tag_base,
                            k_tag_base=k_tag_base,
                            prefix=prefix
                        )

                        worklist = self.split_prims(prims, core_id)

                        core = {
                            "id": core_id,
                            "loop": loop_value,
                            "worklist": worklist
                        }
                        cores.append(core)

        return cores

    def generate_parallel_workload(self,
                                    model_a_config: Dict, model_b_config: Dict,
                                    model_a_type: str, model_b_type: str,
                                    pp: int, mn: int, k: int, dp: int,
                                    batch_size: int, seq_length: int,
                                    avg_output_a: int, avg_output_b: int,
                                    datatype: int = 0) -> Dict:
        """
        Generate parallel dual-model workload.

        Two models run on separate core sets simultaneously.

        Args:
            model_a_config, model_b_config: Model architecture configs
            model_a_type, model_b_type: Model types
            pp, mn, k, dp: Parallelism parameters
            batch_size, seq_length: Input dimensions
            avg_output_a, avg_output_b: Output token counts
            datatype: 0 for INT8, 1 for FP16
        """
        self.datatype = datatype
        cores_per_model = pp * mn * k * dp
        total_cores = 2 * cores_per_model
        loop_a = avg_output_a + 1
        loop_b = avg_output_b + 1

        # Initialize variables for Model A
        vars_a = self.init_model_vars(
            model_a_config, pp, mn, k, batch_size, seq_length, avg_output_a, suffix=""
        )

        # Initialize variables for Model B (with _b suffix)
        vars_b = self.init_model_vars(
            model_b_config, pp, mn, k, batch_size, seq_length, avg_output_b, suffix="_b"
        )

        # Set up base vars for computation
        self.vars = {
            "B": batch_size,
            "T": seq_length,
            "DH": model_a_config.get("DH", 128),
            "NH": model_a_config["NH"],
            "KVH": model_a_config.get("KVH", model_a_config["NH"] // 4),
            "HS": model_a_config["HS"],
            "L": model_a_config["L"],
            "IS": model_a_config.get("IS", model_a_config["HS"] * 4),
            "pp": pp,
            "mn": mn,
            "k": k,
            "dp": dp,
            "chunk": 1
        }
        self.vars["C"] = self.vars["DH"] * self.vars["NH"]
        self.vars["R"] = self.vars["NH"] // self.vars["KVH"]
        self.vars["P"] = self.vars["HS"]
        self.vars["J"] = self.vars["IS"]
        self.vars["G"] = self.vars["C"] + 2 * self.vars["C"] // self.vars["R"]

        for k_var in ["B", "T", "DH", "NH", "KVH", "HS", "L", "IS", "C", "R", "P", "J", "G", "pp", "mn", "k", "dp", "chunk"]:
            self.base_vars.add(k_var)

        # Generate cores for Model A (offset=0, tag bases=1000/2000)
        cores_a = self.process_cores_single(
            model_config=model_a_config,
            model_type=model_a_type,
            pp=pp, mn=mn, k=k, dp=dp,
            loop_value=loop_a,
            core_id_offset=0,
            mn_tag_base=1000,
            k_tag_base=2000,
            prefix="a_"
        )

        # Update vars for Model B
        self.vars.update({
            "DH": model_b_config.get("DH", 128),
            "NH": model_b_config["NH"],
            "KVH": model_b_config.get("KVH", model_b_config["NH"] // 4),
            "HS": model_b_config["HS"],
            "L": model_b_config["L"],
            "IS": model_b_config.get("IS", model_b_config["HS"] * 4),
        })
        self.vars["C"] = self.vars["DH"] * self.vars["NH"]
        self.vars["R"] = self.vars["NH"] // self.vars["KVH"]
        self.vars["P"] = self.vars["HS"]
        self.vars["J"] = self.vars["IS"]
        self.vars["G"] = self.vars["C"] + 2 * self.vars["C"] // self.vars["R"]

        # Generate cores for Model B (offset=cores_per_model, tag bases=5000/6000)
        cores_b = self.process_cores_single(
            model_config=model_b_config,
            model_type=model_b_type,
            pp=pp, mn=mn, k=k, dp=dp,
            loop_value=loop_b,
            core_id_offset=cores_per_model,
            mn_tag_base=5000,
            k_tag_base=6000,
            prefix="b_"
        )

        # Restore vars for Model A (for source generation)
        self.vars["P"] = model_a_config["HS"]
        self.add_var("BTP")

        # Generate sources
        source = self.process_source_dual(mn, k, pp, dp, cores_per_model)

        # Combine all vars
        all_vars = {**vars_a, **vars_b}
        all_vars["pp"] = pp
        all_vars["mn"] = mn
        all_vars["k"] = k
        all_vars["dp"] = dp

        # Add computed size variables
        all_vars.update(self.vars)

        # Build dual_model_info
        model_a_core_ids = list(range(0, cores_per_model))
        model_b_core_ids = list(range(cores_per_model, total_cores))

        dual_model_info = {
            "enabled": True,
            "parallel_execution": True,
            "model_a": {
                "type": model_a_type,
                "HS": model_a_config["HS"],
                "L": model_a_config["L"],
                "loop": loop_a,
                "cores": model_a_core_ids,
                "core_range": f"[0, {cores_per_model})"
            },
            "model_b": {
                "type": model_b_type,
                "HS": model_b_config["HS"],
                "L": model_b_config["L"],
                "loop": loop_b,
                "cores": model_b_core_ids,
                "core_range": f"[{cores_per_model}, {total_cores})"
            },
            "parallelism": {
                "pp": pp,
                "tp": f"{mn}_{k}",
                "dp": dp,
                "cores_per_model": cores_per_model,
                "total_cores": total_cores
            },
            "tag_isolation": {
                "model_a": {"mn_tag_base": 1000, "k_tag_base": 2000},
                "model_b": {"mn_tag_base": 5000, "k_tag_base": 6000}
            }
        }

        config = {
            "vars": all_vars,
            "dual_model_info": dual_model_info,
            "pipeline": 1,
            "source": source,
            "chips": [{
                "chip_id": 0,
                "cores": cores_a + cores_b
            }]
        }

        return config


def get_model_config(preset: str) -> Optional[Dict]:
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
    return configs.get(preset, None)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dual-model PARALLEL workload (two models on separate cores)"
    )

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to store output")
    parser.add_argument("--output_name", type=str, default="parallel_dual_model",
                        help="Output file name")

    # Common settings
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--T", type=int, default=54, help="Sequence length")
    parser.add_argument("--datatype", type=str, default="INT8", choices=["INT8", "FP16"], help="Data type")

    # Parallelism settings (shared by both models)
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism")
    parser.add_argument("--tp", type=str, default="1_1", help="Tensor parallelism (mn_k)")

    # Model A settings
    parser.add_argument("--model", type=str, default="qwen",
                        choices=["gpt", "qwen", "llama"], help="Model A type")
    parser.add_argument("--model_preset", type=str, default=None,
                        help="Model A preset (e.g., qwen_3B)")
    parser.add_argument("--DH", type=int, default=128, help="Model A dimension of head")
    parser.add_argument("--NH", type=int, default=32, help="Model A number of heads")
    parser.add_argument("--KVH", type=int, default=8, help="Model A KV heads")
    parser.add_argument("--HS", type=int, default=2560, help="Model A hidden size")
    parser.add_argument("--L", type=int, default=1, help="Model A transformer layers")
    parser.add_argument("--IS", type=int, default=9728, help="Model A intermediate size")
    parser.add_argument("--avg_output", type=int, default=10,
                        help="Model A average output tokens")

    # Model B settings
    parser.add_argument("--model_b", type=str, default="qwen",
                        choices=["gpt", "qwen", "llama"], help="Model B type")
    parser.add_argument("--model_b_preset", type=str, default=None,
                        help="Model B preset (e.g., llama_3B)")
    parser.add_argument("--DH_b", type=int, default=128, help="Model B dimension of head")
    parser.add_argument("--NH_b", type=int, default=32, help="Model B number of heads")
    parser.add_argument("--KVH_b", type=int, default=8, help="Model B KV heads")
    parser.add_argument("--HS_b", type=int, default=2560, help="Model B hidden size")
    parser.add_argument("--L_b", type=int, default=1, help="Model B transformer layers")
    parser.add_argument("--IS_b", type=int, default=9728, help="Model B intermediate size")
    parser.add_argument("--avg_output_b", type=int, default=10,
                        help="Model B average output tokens")

    args = parser.parse_args()

    # Parse tensor parallelism
    mn, k = [int(x) for x in args.tp.split("_")]

    # Build Model A config
    if args.model_preset:
        model_a_config = get_model_config(args.model_preset)
        if not model_a_config:
            print(f"Unknown Model A preset: {args.model_preset}")
            return
        model_a_type = "qwen" if "qwen" in args.model_preset else ("llama" if "llama" in args.model_preset else "gpt")
    else:
        model_a_config = {
            "HS": args.HS, "NH": args.NH, "DH": args.DH,
            "KVH": args.KVH, "L": args.L, "IS": args.IS
        }
        model_a_type = args.model

    # Build Model B config
    if args.model_b_preset:
        model_b_config = get_model_config(args.model_b_preset)
        if not model_b_config:
            print(f"Unknown Model B preset: {args.model_b_preset}")
            return
        model_b_type = "qwen" if "qwen" in args.model_b_preset else ("llama" if "llama" in args.model_b_preset else "gpt")
    else:
        model_b_config = {
            "HS": args.HS_b, "NH": args.NH_b, "DH": args.DH_b,
            "KVH": args.KVH_b, "L": args.L_b, "IS": args.IS_b
        }
        model_b_type = args.model_b

    cores_per_model = args.pp * mn * k * args.dp
    total_cores = 2 * cores_per_model
    loop_a = args.avg_output + 1
    loop_b = args.avg_output_b + 1

    print("=" * 70)
    print("Dual-Model PARALLEL Workload Generator")
    print("=" * 70)
    print(f"\n[Two models running in PARALLEL on separate cores]")
    print(f"\nModel A - {model_a_type.upper()}:")
    print(f"  Cores:   [0, {cores_per_model})")
    print(f"  Layers:  {model_a_config['L']}, Hidden: {model_a_config['HS']}")
    print(f"  Loop:    {loop_a} (avg_output={args.avg_output})")
    print(f"\nModel B - {model_b_type.upper()}:")
    print(f"  Cores:   [{cores_per_model}, {total_cores})")
    print(f"  Layers:  {model_b_config['L']}, Hidden: {model_b_config['HS']}")
    print(f"  Loop:    {loop_b} (avg_output={args.avg_output_b})")
    print(f"\nParallelism (shared):")
    print(f"  - Pipeline Parallelism (PP): {args.pp}")
    print(f"  - Tensor Parallelism (TP):   {mn} x {k}")
    print(f"  - Data Parallelism (DP):     {args.dp}")
    print(f"  - Cores per model:           {cores_per_model}")
    print(f"  - Total cores:               {total_cores}")
    print(f"\nBatch size: {args.B}, Sequence length: {args.T}")
    print("=" * 70)

    # Generate workload
    generator = ParallelWorkloadGenerator()
    config = generator.generate_parallel_workload(
        model_a_config=model_a_config,
        model_b_config=model_b_config,
        model_a_type=model_a_type,
        model_b_type=model_b_type,
        pp=args.pp,
        mn=mn,
        k=k,
        dp=args.dp,
        batch_size=args.B,
        seq_length=args.T,
        avg_output_a=args.avg_output,
        avg_output_b=args.avg_output_b
    )

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    file_name = f"{args.output_name}_{model_a_type}_{model_b_type}_{args.tp}"
    output_path = os.path.join(args.output_dir, f"{file_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\nWorkload saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Model A cores: {list(range(0, cores_per_model))}")
    print(f"  Model B cores: {list(range(cores_per_model, total_cores))}")
    print(f"  Model A loop:  {loop_a}")
    print(f"  Model B loop:  {loop_b}")


if __name__ == "__main__":
    main()
