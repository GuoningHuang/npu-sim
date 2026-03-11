import os
import json
import argparse
import re
global input_vars
MOE_ROUTE_BASE_SEED = 12345


def _lcg_next(state):
    return (1103515245 * state + 12345) & 0x7fffffff


def _sample_unique_lcg(total, count, seed):
    if total <= 0 or count <= 0:
        return []
    count = min(count, total)
    chosen = []
    used = [False] * total
    state = seed & 0x7fffffff
    while len(chosen) < count:
        state = _lcg_next(state)
        cand = state % total
        if not used[cand]:
            used[cand] = True
            chosen.append(cand)
    return chosen


def _split_size_across_peers(total_size, peers, weights=None):
    """
    Split integer traffic size across peers and preserve the total sum.
    If weights are provided, split proportionally to weights[peer].
    """
    n = len(peers)
    if n == 0:
        return {}
    if weights is None:
        weights = {p: 1 for p in peers}

    w_sum = sum(max(0, int(weights.get(p, 0))) for p in peers)
    if w_sum <= 0:
        base = total_size // n
        rem = total_size % n
        out = {}
        for i, p in enumerate(peers):
            out[p] = base + (1 if i < rem else 0)
        return out

    out = {}
    assigned = 0
    fracs = []
    for p in peers:
        w = max(0, int(weights.get(p, 0)))
        raw_num = total_size * w
        q = raw_num // w_sum
        r = raw_num % w_sum
        out[p] = q
        assigned += q
        fracs.append((r, p))

    rem = total_size - assigned
    fracs.sort(key=lambda x: x[0], reverse=True)
    for i in range(rem):
        out[fracs[i % len(fracs)][1]] += 1
    return out


def _moe_core_local_k_map(input_vars, layer_idx, moe_ep, moe_tp, moe_base_id):
    """
    Build local_k per MoE core id for a given layer.
    All TP shards within the same EP group share the same local_k.
    """
    experts = input_vars['experts']
    global_k = min(input_vars.get('Kglobal', input_vars['topk']), experts)
    route_seed = (MOE_ROUTE_BASE_SEED + layer_idx * 1000003) & 0x7fffffff
    chosen_global = _sample_unique_lcg(experts, global_k, route_seed)

    experts_per_ep = experts // moe_ep if moe_ep > 1 else experts
    local_k_map = {}
    for moe_idx in range(moe_ep * moe_tp):
        ep_idx = moe_idx // moe_tp
        local_begin = ep_idx * experts_per_ep
        local_end = local_begin + experts_per_ep
        local_k = sum(1 for e in chosen_global if local_begin <= e < local_end)
        local_k_map[moe_base_id + moe_idx] = local_k
    return local_k_map


def _compact_json(obj, indent=4, level=0, compact=False):
    """Serialize JSON with prim dicts (containing 'type') on a single line."""
    if compact:
        if isinstance(obj, dict):
            items = ", ".join(f"{json.dumps(k)}: {_compact_json(v, compact=True)}" for k, v in obj.items())
            return "{" + items + "}"
        elif isinstance(obj, list):
            items = ", ".join(_compact_json(item, compact=True) for item in obj)
            return "[" + items + "]"
        else:
            return json.dumps(obj)

    pad = " " * (indent * level)
    pad_inner = " " * (indent * (level + 1))

    if isinstance(obj, dict):
        if "type" in obj:
            return _compact_json(obj, compact=True)
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            val = _compact_json(v, indent, level + 1)
            items.append(f"{pad_inner}{json.dumps(k)}: {val}")
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        items = []
        for item in obj:
            val = _compact_json(item, indent, level + 1)
            items.append(f"{pad_inner}{val}")
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    else:
        return json.dumps(obj)


def init_vars(input_vars):
    input_vars["C"] = int(input_vars['DH'] * input_vars['NH'])
    input_vars["R"] = int(input_vars['NH'] / input_vars['KVH'])
    input_vars["P"] = int(input_vars['HS'])
    input_vars["J"] = int(input_vars['IS']) # Intermediate Size per expert or total? Assuming per expert for MoE logic or as configured
    input_vars["chunk"] = 1
    input_vars["loop"] = int(input_vars["avg_output"] + 1)

    input_vars['G'] = int(input_vars["C"] + 2 * input_vars["C"] / input_vars["R"])
    add_vars(input_vars,  "BTC")
    dp = input_vars.get('dp', 1)
    input_vars["Bdp"] = input_vars["B"] // dp if dp > 1 else input_vars["B"]

    # EP mode variables
    moe_ep = input_vars.get('moe_ep', 1)
    moe_tp = input_vars.get('moe_tp', 1)
    input_vars['moe_ep'] = moe_ep
    input_vars['moe_tp'] = moe_tp
    moe_total = moe_ep * moe_tp
    input_vars['moe_total'] = moe_total

    if moe_total > 1:
        # Parse TP from tp string
        tp_parts = input_vars['tp'].split("_")
        tp = int(tp_parts[0])* int(tp_parts[1])

        # MoE-related variables
        input_vars['moeIS'] = input_vars['IS']
        input_vars[f'moeIS/{moe_tp}'] = input_vars['IS'] // moe_tp
        input_vars['K'] = input_vars['topk']
        input_vars['Kdp'] = min(input_vars["Bdp"] * input_vars['topk'], input_vars['experts'])
        input_vars['Kglobal'] = min(input_vars["B"] * input_vars['topk'], input_vars['experts'])
        input_vars['E_N'] = input_vars['experts']
        if moe_ep > 1:
            input_vars[f'E_N/{moe_ep}'] = input_vars['experts'] // moe_ep

        # QKV dimension (3C-R means Q + K + V for GQA)
        qkv_dim = int(input_vars["C"] + 2 * input_vars["C"] / input_vars["R"])
        input_vars['3C-R'] = qkv_dim
        input_vars[f'3C-R/{tp}'] = qkv_dim // tp

        # Additional derived variables
        input_vars['3C'] = 3 * input_vars['C']
        input_vars[f'T/{tp}'] = input_vars['T'] // tp
        input_vars[f'C/{tp}'] = input_vars['C'] // tp
        input_vars[f'C/2'] = input_vars['C'] // 2
        input_vars[f'NH/{tp}'] = input_vars['NH'] // tp
        input_vars[f'NH/2'] = input_vars['NH'] // 2
        input_vars[f'BTC/{tp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // tp
        input_vars['3BTC/4'] = 3 * input_vars['B'] * input_vars['T'] * input_vars['C'] // 4
        if dp > 1:
            input_vars[f'BTC/{dp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // dp


def add_vars(input_vars, keys):
    if keys not in input_vars.keys():
        [num, den], type = find_const(keys)
        if type in input_vars.keys():
            value = int(num * input_vars[type] / den)
            input_vars[keys] = value
        else:
            if "/" in keys:
                keys_first, keys_num = keys.split("/")
            else:
                keys_first = keys
                keys_num = 1
            value = 1
            for key in keys_first:
                if key.isdigit():
                    value *= int(key)
                elif key.isalpha():
                    value *=  input_vars[key]

            value =  int(value / int(keys_num))
            if value == 0:
                value = 1
            input_vars[keys] = value


def process_source(input_vars):
    # EP mode: sources only for Attention cores in each dp group (no pp)
    moe_total = input_vars.get('moe_total', 1)
    if moe_total > 1:
        tp = input_vars['mn']*input_vars['k']
        dp = input_vars['dp']
        source = []
        for d in range(dp):
            base = d * tp
            for i in range(tp):
                source.append({"dest": base + i, "size": "BTC"})
        add_vars(input_vars, "BTC")
        return source

    mn_num = input_vars['mn']
    k_num = input_vars['k']

    source = []
    for dp_index in range(input_vars['dp']):
        for k_index in range(k_num):
            for mn_index in range(mn_num):
                if mn_num == 1 :
                    size =f"BTP"
                else:
                    size =f"BTP/{mn_num}"
                if mn_num == 1:
                    dest_id = dp_index * k_num * mn_num + mn_index * input_vars['mn'] + k_index
                else:
                    dest_id = dp_index * k_num * mn_num + k_index * input_vars['mn'] + mn_index

                # print(dest_id)
                source.append({"dest": dest_id, "size": size})

                add_vars(input_vars, size)

    return source


def find_const(word):
    word_all = word.split("/")
    num = ""
    word_type = ""
    for index, strs in enumerate(word_all[0]):
        if strs.isdigit():
           num += strs
        else:
            word_type = word_all[0][index:]
            break
    word_all[0] = num
    if word_all[0] == "":
        word_all[0] = "1"
    if len(word_all) == 1:
        word_all.append("1")
    for i in range(2):
        word_all[i] = int(word_all[i])

    return word_all, word_type


def cal_size(word1, word2=None , word3=None, mul_num=None, div_num=None):
    [word1_nun, word1_den], word1_type = find_const(word1)
    if word2 is not None:
        [word2_num, word2_den], word2_type = find_const(word2)
    else:
        [word2_num, word2_den], word2_type = [1,1], ""
    if word3 is not None:
        [word3_num, word3_den], word3_type = find_const(word3)
    else:
        [word3_num, word3_den], word3_type = [1,1], ""

    num = word1_nun * word2_num * word3_num
    den = word1_den * word2_den * word3_den
    if mul_num is not None:
        if mul_num % den == 0:
            num = num * mul_num / den
            den = 1
        elif den % mul_num == 0:
            den = int(den / mul_num)
        else:
            num = num * mul_num

    if div_num is not None:
        if num % div_num == 0:
            num = num / div_num
        elif div_num % num == 0:
            den = int(div_num / num) * den
            num = 1
        else:
            den = den * div_num

    if num == den:
        return_word = f"{word1_type}{word2_type}{word3_type}"
    elif num % den == 0:
        return_word = f"{int(num/den)}{word1_type}{word2_type}{word3_type}"
    elif den % num == 0:
        return_word = f"{word1_type}{word2_type}{word3_type}/{int(den / num)}"
    else:
        return_word = f"{num}{word1_type}{word2_type}{word3_type}/{den}"

    return return_word


gpt = ["layernorm", "matmul", "attention", "matmul", "residual", "layernorm", "matmul", "gelu", "matmul", "residual"]
qwen = ["rmsnorm", "matmul_rope", "attention", "matmul", "residual", "rmsnorm", "matmul×2", "swiglu", "matmul", "residual"]
llama = ["rmsnorm", "matmul_rope", "attention", "matmul", "residual", "rmsnorm", "matmul×2", "swiglu", "matmul", "residual"]

# Defined MoE architecture
# Replacing the MLP block with MoE block: load -> up -> gelu -> down
moe_gpt = ["layernorm", "matmul", "attention", "matmul", "residual", "layernorm", "load_expert", "moe_up", "gelu", "moe_down", "residual"]
moe_qwen = ["rmsnorm", "matmul_rope", "attention", "matmul", "residual", "rmsnorm", "moe_up×2", "swiglu", "moe_down", "residual"]


def produce_recv_cast_tag(recv_id, core_id, cast_id, base_tag=64):
    if cast_id is not None:
        recv_tag = core_id + base_tag
        cast_tag = cast_id + base_tag
        return  recv_tag, cast_tag
    else:
        return None, None


def split_prims(prims, id):

    primslist = []
    first_index = 0
    for index, prim in enumerate(prims):
        # if id == 0:
        #     print(prim['type'], prim['sram_address'])
        if prim["type"] == "parse_input" and index - first_index !=0:
            primslist.append(prims[first_index:index])
            first_index = index
        if prim['type'] == 'parse_output' and index != len(prims) - 1:
            primslist.append(prims[first_index:index + 1])
            first_index = index + 1

    primslist.append(prims[first_index:])

    done_worklist = []
    for prim_list in primslist:
        work_prim = []
        one_work = {
            'recv_cnt':0,
        }
        for prim in prim_list:
            if "recv_cnt" in prim.keys():
                one_work["recv_cnt"] = prim["recv_cnt"]
                prim.pop("recv_cnt")
                if "recv_tag" in prim.keys():
                    one_work["recv_tag"] = prim["recv_tag"]
                    prim.pop("recv_tag")
            if "cast" in prim.keys():
                one_work["cast"] = prim["cast"]
                prim.pop("cast")
            work_prim.append(prim)
        if "cast" not in one_work:
            one_work["cast"] = []
        one_work["prims"] = work_prim

        done_worklist.append(one_work)
        # break

    return done_worklist


def add_rope(B, T, C, NH, R, sram_indata, rp_num, prims_list):
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
            "outdata": f"rope{rp_num}_out"
        },
        "dram_address": {
            "data": f"rope{rp_num}_data"
        }
    }
    prims_list.append(rp_prim)
    sram_indata = f"rope{rp_num}_out"
    rp_num += 1
    return prims_list, sram_indata, rp_num


def process_one_work_mnk(input_vars, operation, core_layer, core_id, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, last_layer):
    global oc, mm_type
    prims_list = []

    ln_num = 1
    mm_num = 1
    att_num = 1
    res_num = 1
    gelu_num = 1
    rp_num = 1
    swiglu_num = 1
    moe_up_num = 1
    moe_down_num = 1

    B = cal_size("B", div_num=input_vars['dp'])
    add_vars(input_vars, B)

    if input_vars['k'] != 1:
        NH = f"NH/{input_vars['k']}"
    elif input_vars['mn'] != 1:
        NH = f"NH/{input_vars['mn']}"
    else:
        NH = f"NH/{input_vars['mn']}"
    NH = cal_size(NH)
    add_vars(input_vars, NH)

    mn_recv_tag, mn_cast_tag = produce_recv_cast_tag(mn_recv_id, core_id, mn_cast_id, base_tag=1000)
    k_recv_tag, k_cast_tag = produce_recv_cast_tag(k_recv_id, core_id, k_cast_id, base_tag=2000)

    if input_vars['mn'] != 1 and input_vars['k'] != 1:
        id_den = input_vars['mn']
    else:
        id_den = 1

    ic = "P"
    sram_indata = "_input_label"
    res_start = sram_indata
    res_end = res_start

    if input_vars['mn'] != 1 or input_vars['k'] != 1:
        size = f"BT{ic}/{input_vars['mn']}"
        size = cal_size(size)
        add_vars(input_vars, size)
        prim = {
            "type": "parse_input",
            "size": size,
            "sram_address": {
                "indata": "layernorm1_in",
                "outdata": "layernorm1_in"
            },
            "recv_cnt":1
        }
        prims_list.append(prim)

        sram_indata = "_layernorm1_in"
        res_start = "layernorm1_in"


    for layer_index in range(core_layer):
        for num, operates in enumerate(operation):
            add_vars(input_vars, ic)
            if "norm" in operates:
                if input_vars['mn'] != 1:
                    T = f"T/{input_vars['mn']}"
                else:
                    T = "T"
                add_vars(input_vars, T)

                ln_type = "Layernorm_f"
                if operates == "rmsnorm":
                    ln_type = "rmsnorm_forward"
                ln_prim = {
                    "type": ln_type,
                    "B": B,
                    "T": T,
                    "C": "P",
                    "sram_address": {
                        "indata": sram_indata,
                        "outdata": f"layernorm{ln_num}_out"
                    },
                    "dram_address": {
                        "data": f"layernorm{ln_num}_data"
                    }
                }
                if input_vars['mn'] * input_vars['k'] == 1:
                    ln_prim['recv_cnt'] = 1
                sram_indata = f"layernorm{ln_num}_out"
                prims_list += [ln_prim]

                if input_vars['k'] != 1:
                    sd_indata = cal_size(B, T, f"P")

                    sd_outdata = cal_size(B, T, f"P/{input_vars['k']}")
                    add_vars(input_vars, sd_indata)
                    add_vars(input_vars, sd_outdata)

                    sd_prim = {
                        "type": "switch_data",
                        "IN": sd_indata,
                        "OUT": sd_outdata,
                        "sram_address": {
                            "indata": f"layernorm{ln_num}_out",
                            "outdata": f"switch_layernorm{ln_num}_out"
                        }
                    }
                    sram_indata = f"switch_layernorm{ln_num}_out"
                    prims_list += [sd_prim]

                oc = "P"
                ln_num += 1
            elif "load_expert" in operates:
                le_prim = {
                    "type": "load_expert",
                    "E_N": input_vars["experts"],
                    "K": input_vars["topk"],
                    "C": "P",
                    "OC": "J", # Expert Hidden Size
                    "strategy": 2,
                    "sram_address": {
                        "indata": "TODO",
                        "outdata": "TODO"
                    }
                }
                prims_list.append(le_prim)
            elif "moe_up" in operates:
                moe_time = operates.split("×")[1:]
                if len(moe_time) == 0:
                    moe_time = 1
                else:
                    moe_time = int(moe_time[0])

                if input_vars['mn'] != 1:
                    T = f"T/{input_vars['mn']}"
                else:
                    T = "T"
                add_vars(input_vars, T)

                original_indata = sram_indata
                swiglu_indata = ""
                for moe_idx in range(moe_time):
                    if moe_time > 1 and moe_idx < moe_time - 1:
                        moe_indata = f"_{original_indata}"
                    else:
                        moe_indata = original_indata

                    moe_prim = {
                        "type": "matmul_forward_moe",
                        "B": B,
                        "T": T,
                        "C": "P",
                        "OC": "J",
                        "K": input_vars["topk"],
                        "E_N": input_vars["experts"],
                        "is_merge": False,
                        "need_choose": (moe_idx == 0),
                        "sram_address": {
                            "indata": moe_indata,
                            "outdata": f"moe_up{moe_up_num}_out"
                        },
                        "dram_address": {
                            "data": f"moe_up{moe_up_num}_data"
                        }
                    }
                    prims_list.append(moe_prim)

                    if moe_time > 1:
                        swiglu_indata += f"moe_up{moe_up_num}_out "

                    sram_indata = f"moe_up{moe_up_num}_out"
                    moe_up_num += 1

                oc = "J" # Intermediate
            elif "moe_down" in operates:
                if input_vars['mn'] != 1:
                    T = f"T/{input_vars['mn']}"
                else:
                    T = "T"

                moe_prim = {
                    "type": "matmul_forward_moe",
                    "B": B,
                    "T": T,
                    "C": "J",
                    "OC": "P",
                    "K": input_vars["topk"],
                    "E_N": input_vars["experts"],
                    "is_merge": True, # Merge results
                    "need_choose": False, # Reuse selection
                    "sram_address": {
                        "indata": sram_indata,
                        "outdata": f"moe_down{moe_down_num}_out"
                    },
                    "dram_address": {
                        "data": f"moe_down{moe_down_num}_data"
                    }
                }
                prims_list.append(moe_prim)
                sram_indata = f"moe_down{moe_down_num}_out"
                res_end = f"moe_down{moe_down_num}_out"
                oc = "P" # Back to HS
                moe_down_num += 1
            elif "matmul" in operates:
                if num == 1:
                    oc = "G" if "rope" in operates else "3C"
                    mm_type = "matmul_forward_pd"
                elif num == 3:
                    oc = "P"
                    mm_type = "Matmul_f"
                elif num == 6:
                    oc = f"J"
                    mm_type = "Matmul_f"
                elif num == 8:
                    oc = "P"
                    mm_type = "Matmul_f"
                mm_time = operates.split("×")[1:]
                if len(mm_time) == 0:
                    mm_time = 1
                else:
                    mm_time = int(mm_time[0])

                T = cal_size(f"T/{input_vars['mn']}")
                add_vars(input_vars, T)

                mm_ic = cal_size(f"{ic}/{input_vars['k']}")
                add_vars(input_vars, mm_ic)

                mm_oc = cal_size(f"{oc}/{input_vars['mn']}")
                add_vars(input_vars, mm_oc)
                swiglu_indata = ""
                for mm_index in range(mm_time):
                    # mn
                    mm_split_num = 1
                    mm_indata = sram_indata
                    if input_vars['mn'] != 1:
                        mm_indata = f"_{sram_indata}"
                    elif mm_time > 1 and mm_index < mm_time - 1:
                        mm_indata = f"_{sram_indata}"

                    if input_vars['mn'] != 1:
                        mm_outdata = f"matmul{mm_num}_{mm_split_num}_out"
                    else:
                        mm_outdata = f"matmul{mm_num}_out"
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
                            "data": f"matmul{mm_num}_data"
                        }
                    }
                    if mm_type == "matmul_forward_pd":
                        mm_prim_pd = {
                            "R": "R",
                            "chunk": "chunk",
                            "job_type": 2
                        }
                        mm_prim = mm_prim | mm_prim_pd
                    prims_list.append(mm_prim)

                    res_end = mm_outdata
                    mn_merge_indata = f"matmul{mm_num}_{mm_split_num}_out"

                    if "rope" in operates:
                        prims_list, rope_outdata, rp_num = add_rope(B, T, mm_oc, NH, "R", mm_outdata, rp_num, prims_list)
                        mn_merge_indata = rope_outdata

                    for split_mn_index in range(input_vars['mn']-1):
                        inout_size = cal_size(mm_ic, mm_oc)
                        add_vars(input_vars, inout_size)

                        out_prim = {
                            "type": "parse_output",
                            "size": inout_size,
                            "sram_address": {
                                "indata": f"eternal_matmul{mm_num}_{mm_split_num}_w",
                                "outdata": f"DEL_eternal_matmul{mm_num}_{mm_split_num}_w"
                            },
                            "cast":[{"dest": mn_cast_id,
                                     "tag": mn_cast_tag}]
                        }
                        in_prim = {
                            "type": "parse_input",
                            "size": inout_size,
                            "sram_address": {
                                "indata": f"eternal_matmul{mm_num}_{mm_split_num+1}_w",
                                "outdata": f"eternal_matmul{mm_num}_{mm_split_num+1}_w"
                            },
                            "recv_cnt": 1,
                            "recv_tag": mn_recv_tag,
                        }
                        if mm_split_num < input_vars['mn'] - 1:
                            mm_indata = mm_indata
                        else:
                            if mm_index < mm_time - 1:
                                mm_indata = mm_indata
                            else:
                                mm_indata = mm_indata[1:]

                        mm_prim = {
                            "type": mm_type,
                            "B": B,
                            "T": T,
                            "C": mm_ic,
                            "OC": mm_oc,
                            "R": "R",
                            "chunk": "chunk",
                            "job_type": 2,
                            "sram_address": {
                                "indata": mm_indata,
                                "outdata": f"matmul{mm_num}_{mm_split_num + 1}_out"
                            },
                            "dram_address": {
                                "data": f"matmul{mm_num}_data"
                            }
                        }

                        prims_list += [out_prim, in_prim, mm_prim]  if core_id  % 2 == 0 else [in_prim, out_prim, mm_prim]

                        if "rope" in operates:
                            rope_indata = f"matmul{mm_num}_{mm_split_num + 1}_out"
                            prims_list, rope_outdata, rp_num = add_rope(B, T, mm_oc, NH, "R", rope_indata, rp_num, prims_list)
                            mn_merge_indata += " " + rope_outdata
                        else:
                            mn_merge_indata += f" matmul{mm_num}_{mm_split_num + 1}_out"

                        mm_split_num += 1

                    mmm_ic = mm_oc
                    if input_vars['mn'] != 1:
                        mmm_prim = {
                            "type": "Merge_matmul",
                            "B": B,
                            "T": T,
                            "C": mmm_ic,
                            "dim": 1,
                            "slice": input_vars['mn'],
                            "sram_address": {
                                "indata": mn_merge_indata,
                                "outdata": f"matmul{mm_num}_out"
                            }
                        }
                        prims_list += [mmm_prim]
                        mn_mmm_outdata = f"matmul{mm_num}_out"
                        res_end = mn_mmm_outdata


                    # k
                    inout_size = cal_size(B, T, mmm_ic)
                    add_vars(input_vars, inout_size)
                    k_merge_indata = f"matmul{mm_num}_out"
                    if input_vars["k"] != 1:
                        if input_vars["mn"] != 1:
                            sd_in = cal_size(B, f"T/{input_vars['mn']}", mmm_ic, mul_num=input_vars['mn'])
                            sd_out = cal_size(B, f"T/{input_vars['mn']}", mmm_ic, mul_num=input_vars['mn'], div_num=input_vars['k'])
                            sd_indata = mn_mmm_outdata
                            if num == 1 or num == 6:
                                sd_prim = {
                                    "type": "switch_data",
                                    "IN": sd_in,
                                    "OUT": sd_out,
                                    "sram_address": {
                                        "indata": sd_indata,
                                        "outdata": f"k_matmul{mm_num}_1_out"
                                    }
                                }
                                prims_list.append(sd_prim)

                                inout_size = sd_out
                                k_merge_indata = f"k_matmul{mm_num}_1_out"
                                mmm_oc = mmm_ic
                            else:
                                k_merge_indata = f"matmul{mm_num}_out"
                                mmm_oc = cal_size(mmm_ic,mul_num=input_vars['mn'])
                                inout_size = sd_in
                        else:
                            sd_in = cal_size(B, "T", mm_oc)
                            sd_out = cal_size(B, "T", mm_oc, div_num=input_vars['k'])
                            if num == 1 or num == 6:
                                sd_indata = mm_outdata
                            else:
                                sd_indata = "_" + mm_outdata
                            sd_prim = {
                                "type": "switch_data",
                                "IN": sd_in,
                                "OUT": sd_out,
                                "sram_address": {
                                    "indata": sd_indata,
                                    "outdata": f"k_matmul{mm_num}_1_out"
                                }
                            }
                            prims_list.append(sd_prim)
                            inout_size = sd_out
                            mmm_oc = mmm_ic
                            k_merge_indata = f"k_matmul{mm_num}_1_out"
                        add_vars(input_vars, sd_in)
                        add_vars(input_vars, sd_out)

                    if input_vars['k'] != 1:
                        k_merge_num = 1
                        for k_index in range(input_vars['k']-1):
                            if input_vars['mn'] == 1:
                                inout_indata = f"k_matmul{mm_num}_{k_merge_num}_out"
                                inout_outdata = f"k_matmul{mm_num}_{k_merge_num}_out"
                            else:
                                if num == 1 or num == 6:
                                    inout_indata = "_" + f"k_matmul{mm_num}_{k_merge_num}_out"
                                    inout_outdata = f"k_matmul{mm_num}_{k_merge_num}_out"
                                else:
                                    inout_indata = f"matmul{mm_num}_out"
                                    inout_outdata = inout_indata

                            out_prim = {
                                "type": "parse_output",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": inout_indata,
                                    "outdata": inout_outdata
                                },
                                "cast": [{
                                    "dest": k_cast_id,
                                    "tag": k_cast_tag
                                }],
                            }
                            in_prim = {
                                "type": "parse_input",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": f"k_matmul{mm_num}_{k_merge_num+1}_out",
                                    "outdata": f"k_matmul{mm_num}_{k_merge_num+1}_out"
                                },
                                "recv_cnt": 1,
                                "recv_tag": k_recv_tag,
                            }
                            k_merge_indata += f" k_matmul{mm_num}_{k_merge_num+1}_out"
                            prims_list += [out_prim, in_prim] if int(core_id / id_den) % 2 == 0 else [in_prim, out_prim]


                            if input_vars['mn'] != 1:
                                if num == 1 or num == 6:
                                    mmm_oc = mm_oc
                                else:
                                    mmm_oc = cal_size(mm_oc, mul_num=input_vars['mn'])
                                    res_end =  f"k_matmul{mm_num}_out"
                            else:
                                mmm_oc = cal_size(mm_oc, div_num=input_vars['k'])

                            add_vars(input_vars,mmm_oc)
                            if k_index == input_vars['k']-2:
                                T = cal_size(f"T/{input_vars['mn']}")
                                mmm_prim = {
                                    "type": "Merge_matmul",
                                    "B": B,
                                    "T": T,
                                    "C": mmm_oc,
                                    "dim": 2,
                                    "slice": input_vars['k'],
                                    "sram_address": {
                                        "indata": k_merge_indata,
                                        "outdata": f"k_matmul{mm_num}_out"
                                    }
                                }
                                prims_list.append(mmm_prim)
                                k_mmm_outdata = f"k_matmul{mm_num}_out"
                            k_merge_num += 1


                    if input_vars['k'] != 1 and input_vars['mn'] == 1:
                        for k_index in range(input_vars['k']-1):
                            if input_vars['mn'] == 1:
                                inout_indata = f"k_matmul{mm_num}_out"
                                inout_outdata = f"k_matmul{mm_num}_out"
                            else:
                                if num == 1 or num == 6:
                                    inout_indata = "" + f"k_matmul{mm_num}_out"
                                    inout_outdata = f"k_matmul{mm_num}_out"
                                else:
                                    inout_indata = f"matmul{mm_num}_out"
                                    inout_outdata = inout_indata

                            out_prim = {
                                "type": "parse_output",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": inout_indata,
                                    "outdata": inout_outdata
                                },
                                "cast": [{
                                    "dest": k_cast_id,
                                    "tag": k_cast_tag
                                }],
                            }
                            in_prim = {
                                "type": "parse_input",
                                "size": inout_size,
                                "sram_address": {
                                    "indata": inout_indata,
                                    "outdata": inout_outdata
                                },
                                "recv_cnt": 1,
                                "recv_tag": k_recv_tag,
                            }
                            prims_list += [out_prim, in_prim] if int(core_id / id_den) % 2 == 0 else [in_prim, out_prim]

                    if mm_time != 1:
                        sram_indata = sram_indata
                    elif input_vars['mn'] == 1 and input_vars['k'] == 1:
                        if "rope" in operates:
                            sram_indata = rope_outdata
                        else:
                            sram_indata = mm_outdata
                    elif input_vars['mn'] != 1 and input_vars['k'] == 1:
                        sram_indata = mn_mmm_outdata
                    elif  input_vars['k'] != 1:
                        sram_indata = k_mmm_outdata

                    if mm_time != 1:
                        if input_vars['mn'] == 1 and input_vars['k'] == 1:
                            swiglu_indata += mm_outdata + " "
                        elif input_vars['mn'] != 1 and input_vars['k'] == 1:
                            swiglu_indata += mn_mmm_outdata + " "
                        elif input_vars['k'] != 1:
                            swiglu_indata += k_mmm_outdata + " "

                    mm_num += 1
            elif "attention" in operates:
                ic_num = ic[:-1]
                if input_vars['mn'] != 1:
                    inout_size = f"{ic_num}BTC/{input_vars['mn'] * input_vars['k']}"

                    add_vars(input_vars, inout_size)
                    for mn_ndex in range(input_vars['mn']-1):
                        out_prim = {
                            "type": "parse_output",
                            "size": inout_size,
                            "sram_address": {
                                "indata": sram_indata,
                                "outdata": sram_indata
                            },
                            "cast": [{
                                "dest": mn_cast_id,
                                "tag": mn_cast_tag
                            }],
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
                        prims_list += [out_prim, in_prim] if core_id % 2 == 0 else [in_prim, out_prim]


                C = cal_size(f"C/{input_vars['mn'] * input_vars['k']}")
                add_vars(input_vars, C)
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
                        "outdata": f"attention{att_num}_out"
                    },
                    "dram_address": {
                        "data": f"attention{att_num}_data",
                        "out": "TODO"
                    }
                }
                prims_list.append(att_prim)
                sram_indata = f"attention{att_num}_out"

                if input_vars['mn'] != 1:
                    inout_size = f"BTC/{input_vars['mn'] * input_vars['k']}"

                    for mn_ndex in range(input_vars['mn']-1):
                        out_prim = {
                            "type": "parse_output",
                            "size": inout_size,
                            "sram_address": {
                                "indata": sram_indata,
                                "outdata": sram_indata
                            },
                            "cast": [{
                                "dest": mn_cast_id,
                                "tag": mn_cast_tag
                            }],
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
                        prims_list += [out_prim, in_prim] if core_id % 2 == 0 else [in_prim, out_prim]

                sram_indata = f"attention{att_num}_out"
                oc = "C"
                att_num += 1
            elif "residual" in operates:
                N = cal_size(B, "T", f"{ic}/{input_vars['mn']}")
                add_vars(input_vars, N)
                res_prim = {
                    "type": "Residual_f",
                    "N": N,
                    "sram_address": {
                        "indata": f"{res_start} {res_end}",
                        "outdata": f"residual{res_num}_out"
                    }
                }

                if num == 9 and layer_index == core_layer-1:
                    # No pp: always last layer, loop back to self                                                                                                                   
                    loop_id = core_id                                                                                                                                               
                    res_prim["cast"] = [{                                                                                                                                           
                                "dest": -1,                                                                                                                                             
                            "loopout": "true"                                                                                                                                       
                        },                                                                                                                                                          
                            {                                                                                                                                                           
                            "dest": loop_id,                                                                                                                                        
                                "loopout": "false"                                                                                                                                      
                            }]  

                prims_list.append(res_prim)
                sram_indata = f"_residual{res_num}_out"
                res_start = f"residual{res_num}_out"
                res_num += 1
            elif "gelu" in operates:
                N = cal_size(B, f"T/{input_vars['mn']}", f"{ic}/{input_vars['k']}")
                add_vars(input_vars, N)
                gelu_prim = {
                    "type": "Gelu_f",
                    "N": N,
                    "sram_address": {
                        "indata": sram_indata,
                        "outdata": f"gelu{gelu_num}_out"
                    },
                    "dram_address": {
                        "data": f"gelu{gelu_num}_data",
                        "out": "TODO"
                    }
                }
                prims_list.append(gelu_prim)
                sram_indata = f"gelu{gelu_num}_out"
                gelu_num += 1
            elif "swiglu" in operates:
                N = cal_size(B, f"T/{input_vars['mn']}", f"{ic}/{input_vars['k']}")
                add_vars(input_vars, N)
                swiglu_prim = {
                    "type": "swiglu_forward",
                    "N": N,
                    "sram_address": {
                        "indata": swiglu_indata[:-1],
                        "outdata": f"swiglu{swiglu_num}_out"
                    },
                    "dram_address": {
                        "input": 0,
                        "data": -1
                    }
                }
                prims_list.append(swiglu_prim)
                sram_indata = f"swiglu{swiglu_num}_out"
                swiglu_num += 1
            ic = oc

    return prims_list


def process_worklist_mnk(input_vars, core_id, core_layer, operation, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, last_layer):
    pirms = process_one_work_mnk(input_vars, operation, core_layer, core_id, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, last_layer)
    worklist = split_prims(pirms, core_id)
    return worklist


def generate_ring_allreduce_worklist(parallelism, core_index, size, is_first_core_type=True):
    """
    Generate ring all-reduce communication worklist.

    Args:
        parallelism: Number of parallel units (TP or EP)
        core_index: Index of the current core within its type (0 to parallelism-1)
        size: Size of data to transfer
        is_first_core_type: True for cores 0,2 pattern (send first), False for 1,3 pattern (recv first)

    Returns:
        List of worklist items for ring all-reduce
    """
    # Ring all-reduce tags: core i -> core (i+1)%parallelism
    # Tags pattern: 14, 21, 32, 43 for 4-way parallelism
    tags = []
    for i in range(parallelism):
        next_i = (i + 1) % parallelism
        tag = i * 10 + next_i + 10 + 4  # 14, 21, 32, 43
        tags.append(tag)

    next_core_offset = 1
    prev_core_offset = parallelism - 1

    send_tag = tags[core_index]
    recv_tag = tags[(core_index - 1 + parallelism) % parallelism]

    worklist = []

    # 2 * (parallelism - 1) steps for ring all-reduce
    for step in range(2 * (parallelism - 1)):
        if is_first_core_type:
            # Cores 0, 2: send first, then recv
            if step % 2 == 0:
                # Send step
                work = {
                    "recv_cnt": 0,
                    "cast": [{"dest": (core_index + next_core_offset) % parallelism, "tag": send_tag}],
                    "prims": [{
                        "type": "parse_output",
                        "size": size,
                        "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                    }]
                }
            else:
                # Recv step
                work = {
                    "recv_cnt": 1,
                    "recv_tag": recv_tag,
                    "cast": [],
                    "prims": [{
                        "type": "parse_input",
                        "size": size,
                        "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                    }]
                }
        else:
            # Cores 1, 3: recv first, then send
            if step % 2 == 0:
                # Recv + Send in same worklist item
                work = {
                    "recv_cnt": 1,
                    "recv_tag": recv_tag,
                    "cast": [{"dest": (core_index + next_core_offset) % parallelism, "tag": send_tag}],
                    "prims": [
                        {
                            "type": "parse_input",
                            "size": size,
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        },
                        {
                            "type": "parse_output",
                            "size": size,
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        }
                    ]
                }
                worklist.append(work)
                continue
            else:
                continue  # Already handled in previous step
        worklist.append(work)

    return worklist


def generate_ring_allreduce_worklist_ep(ep, core_index, size, base_core_id):
    """
    Generate ring all-reduce communication worklist for EP cores.
    Core IDs are offset by base_core_id (e.g., TP*2 for MoE cores).
    """
    tags = []
    for i in range(ep):
        next_i = (i + 1) % ep
        tag = i * 10 + next_i + 10 + 4
        tags.append(tag)

    send_tag = tags[core_index]
    recv_tag = tags[(core_index - 1 + ep) % ep]
    next_core = base_core_id + (core_index + 1) % ep

    worklist = []
    is_first_type = (core_index % 2 == 0)

    for step in range(2 * (ep - 1)):
        if is_first_type:
            if step % 2 == 0:
                work = {
                    "recv_cnt": 0,
                    "cast": [{"dest": next_core, "tag": send_tag}],
                    "prims": [{
                        "type": "parse_output",
                        "size": size,
                        "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                    }]
                }
            else:
                work = {
                    "recv_cnt": 1,
                    "recv_tag": recv_tag,
                    "cast": [],
                    "prims": [{
                        "type": "parse_input",
                        "size": size,
                        "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                    }]
                }
        else:
            if step % 2 == 0:
                work = {
                    "recv_cnt": 1,
                    "recv_tag": recv_tag,
                    "cast": [{"dest": next_core, "tag": send_tag}],
                    "prims": [
                        {
                            "type": "parse_input",
                            "size": size,
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        },
                        {
                            "type": "parse_output",
                            "size": size,
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        }
                    ]
                }
                worklist.append(work)
                continue
            else:
                continue
        worklist.append(work)

    return worklist


def process_attention_core_worklist_ep(input_vars, core_index, tp, moe_ep, moe_tp, core_layer, dp_index, base_id, moe_base_id, attn_cores, loop_count=1):
    """
    Generate worklist for Attention Core in EP mode without pp.

    Per layer, the structure is:
    - Main computation: parse_input(layer 0 only) + rmsnorm + Matmul(QKV) + rope + Attention + Matmul(O) + switch_data
    - Ring All-Reduce (TP dimension)
    - Residual + rmsnorm + gate_forward + cast to all MoE cores
    - recv from MoE + final Residual (with loop cast on last layer)
    """
    worklist = []

    # Add required variables
    add_vars(input_vars, "BTC")
    add_vars(input_vars, f"BTC/{tp}")
    add_vars(input_vars, f"T/{tp}")
    add_vars(input_vars, f"C/{tp}")
    add_vars(input_vars, f"3C-R/{tp}")
    add_vars(input_vars, f"NH/{tp}")
    add_vars(input_vars, "3BTC/4")

    total_layers = core_layer * loop_count

    for layer_idx in range(total_layers):
        is_first_layer = (layer_idx == 0)
        is_last_layer = (layer_idx == total_layers - 1)

        # --- Main computation worklist item ---
        if is_first_layer:
            main_prims = [
                {
                    "type": "parse_input",
                    "size": "BTC",
                    "sram_address": {"indata": "_residual2_out", "outdata": "rmsnorm1_in"}
                },
                {
                    "type": "rmsnorm_forward",
                    "B": "B",
                    "T": "T",
                    "C": "C",
                    "sram_address": {"indata": "_rmsnorm1_in", "outdata": "rmsnorm1_out"},
                    "dram_address": {"data": "rmsnorm1_data"}
                },
            ]
            main_recv_cnt = 1
        else:
            main_prims = [
                {
                    "type": "rmsnorm_forward",
                    "B": "B",
                    "T": "T",
                    "C": "C",
                    "sram_address": {"indata": "_residual2_out", "outdata": "rmsnorm1_out"},
                    "dram_address": {"data": "rmsnorm1_data"}
                },
            ]
            main_recv_cnt = 0

        main_prims += [
            {
                "type": "Matmul_f",
                "use_hw": False,
                "B": "B",
                "T": "T",
                "C": "C",
                "OC": f"3C-R/{tp}",
                "sram_address": {"indata": "rmsnorm1_out", "outdata": "matmul1_out"},
                "dram_address": {"data": "matmul1_data"}
            },
            {
                "type": "rope_forward",
                "B": "B",
                "T": "T",
                "C": f"3C-R/{tp}",
                "NH": f"NH/{tp}",
                "sram_address": {"indata": "matmul1_out", "outdata": "rope1_out"},
                "dram_address": {"data": "rope1_data"}
            },
            {
                "type": "Attention_f",
                "B": "B",
                "T": "T",
                "C": f"3C-R/{tp}",
                "R": "R",
                "NH": f"NH/{tp}",
                "sram_address": {"indata": "rope1_out", "outdata": "attention1_out"},
                "dram_address": {"data": "attention1_data", "out": "TODO"}
            },
            {
                "type": "Matmul_f",
                "use_hw": False,
                "B": "B",
                "T": "T",
                "C": f"C/{tp}",
                "OC": "C",
                "sram_address": {"indata": "attention1_out", "outdata": "matmul2_out"},
                "dram_address": {"data": "matmul2_data"}
            },
            {
                "type": "switch_data",
                "IN": "BTC",
                "OUT": f"BTC/{tp}",
                "sram_address": {"indata": "_matmul2_out", "outdata": "switch_out"}
            }
        ]

        main_work = {
            "recv_cnt": main_recv_cnt,
            "cast": [],
            "prims": main_prims
        }
        if is_first_layer:
            main_work["recv_tag"] = base_id + core_index
        worklist.append(main_work)

        # --- Ring All-Reduce for TP ---
        if tp > 1:
            for phase in range(tp):
                if phase == core_index:
                    cast_list = [{"dest": base_id + i, "tag": 50} for i in range(tp) if i != core_index]
                    worklist.append({
                        "recv_cnt": 0,
                        "cast": cast_list,
                        "prims": [{
                            "type": "parse_output",
                            "size": f"BTC/{tp}",
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        }]
                    })
                else:
                    worklist.append({
                        "recv_cnt": 1,
                        "recv_tag": 50,
                        "cast": [],
                        "prims": [{
                            "type": "parse_input",
                            "size": f"BTC/{tp}",
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        }]
                    })

        # --- Dispatch: Residual + rmsnorm + gate_forward + weighted cast to MoE cores ---
        moe_dests = [moe_base_id + i for i in range(moe_ep * moe_tp)]
        local_k_map = _moe_core_local_k_map(input_vars, layer_idx, moe_ep, moe_tp, moe_base_id)
        # Per-user model: each attention core sends (B*topk/attn_cores) expert units.
        # Convert expert units to traffic size with Bdp*T*C.
        per_attn_btck = (
            input_vars["Bdp"] * input_vars["T"] * input_vars["C"] * input_vars["B"] * input_vars["topk"]
        ) // attn_cores
        dispatch_size_map = _split_size_across_peers(
            per_attn_btck,
            moe_dests,
            {dest: local_k_map.get(dest, 0) for dest in moe_dests}
        )

        if is_first_layer:
            res1_indata = "rmsnorm1_in matmul2_out"
        else:
            res1_indata = "residual2_out matmul2_out"

        dispatch_prims = [
            {
                "type": "Residual_f",
                "N": "BTC",
                "sram_address": {"indata": res1_indata, "outdata": "residual1_out"},
                "dram_address": {"data": -1, "out": "TODO"}
            },
            {
                "type": "rmsnorm_forward",
                "B": "B",
                "T": f"T/{tp}",
                "C": "C",
                "sram_address": {"indata": "_residual1_out", "outdata": "rmsnorm2_out"},
                "dram_address": {"data": "rmsnorm2_data"}
            },
            {
                "type": "gate_forward",
                "B": "B",
                "T": f"T/{tp}",
                "C": "C",
                "K": "Kglobal",
                "E_N": "E_N",
                "sram_address": {"indata": "_rmsnorm2_out", "outdata": "gate_out"},
                "dram_address": {"data": -1}
            },
        ]

        worklist.append({
            "recv_cnt": 0,
            "cast": [],
            "prims": dispatch_prims
        })

        for dest in moe_dests:
            worklist.append({
                "recv_cnt": 0,
                "cast": [{"dest": dest, "weight": tp, "tag": 80}],
                "prims": [{
                    "type": "parse_output",
                    "size": dispatch_size_map.get(dest, 0),
                    "sram_address": {"indata": "rmsnorm2_out", "outdata": "rmsnorm2_out"}
                }]
            })

        # --- Padding to align with MoE return ---
        pad_steps = moe_ep * moe_tp if (moe_ep * moe_tp) > 1 else 0
        for _ in range(pad_steps):
            worklist.append({
                "recv_cnt": 0,
                "cast": [],
                "prims": []
            })

        # --- Final: Recv from MoE + Residual ---
        final_prims = [
            {
                "type": "Residual_f",
                "N": "BTC",
                "sram_address": {"indata": "_input_label residual1_out", "outdata": "residual2_out"},
                "dram_address": {"data": -1, "out": "residual2_out"}
            }
        ]

        if is_last_layer:
            final_cast = [{"dest": -1}]
        else:
            final_cast = []

        worklist.append({
            "recv_cnt": moe_ep * moe_tp,
            "recv_tag": 81,
            "cast": final_cast,
            "prims": final_prims
        })

    return worklist


def process_moe_core_worklist_ep(input_vars, moe_core_index, tp, moe_ep, moe_tp, core_layer, attn_cores, moe_base_id, loop_count=1):
    """
    Generate worklist for MoE Core in EP mode without pp.

    Per layer, the structure is:
    - recv_cnt=TP + MoE computation (matmul_moe x3 + swiglu + switch_data)
    - Ring All-Reduce (EP dimension)
    - cast result back to corresponding Attention core
    Repeated for each layer, followed by an empty worklist.
    """
    worklist = []

    dp = input_vars['dp']
    btc_dp = f"BTC/{dp}" if dp > 1 else "BTC"
    btc_dp_mtp = f"BTC/{dp * moe_tp}" if (dp > 1 and moe_tp > 1) else (btc_dp if dp > 1 else (f"BTC/{moe_tp}" if moe_tp > 1 else "BTC"))

    add_vars(input_vars, f"moeIS/{moe_tp}")
    add_vars(input_vars, btc_dp)
    if moe_tp > 1:
        add_vars(input_vars, btc_dp_mtp)

    total_layers = core_layer * loop_count
    ep_idx = moe_core_index // moe_tp
    tp_idx = moe_core_index % moe_tp
    en_ep_str = f"E_N/{moe_ep}" if moe_ep > 1 else "E_N"
    experts_per_ep = input_vars['experts'] // moe_ep if moe_ep > 1 else input_vars['experts']
    local_begin = ep_idx * experts_per_ep

    for layer_idx in range(total_layers):
        global_k = min(input_vars.get('Kglobal', input_vars['topk']), input_vars['experts'])
        route_seed = (MOE_ROUTE_BASE_SEED + layer_idx * 1000003) & 0x7fffffff
        chosen_global = _sample_unique_lcg(input_vars['experts'], global_k, route_seed)
        local_end = local_begin + experts_per_ep
        local_k = sum(1 for e in chosen_global if local_begin <= e < local_end)
        moe_core_abs_id = moe_base_id + moe_core_index
        moe_dests = [moe_base_id + i for i in range(moe_ep * moe_tp)]
        local_k_map = _moe_core_local_k_map(input_vars, layer_idx, moe_ep, moe_tp, moe_base_id)
        per_attn_btck = (
            input_vars["Bdp"] * input_vars["T"] * input_vars["C"] * input_vars["B"] * input_vars["topk"]
        ) // attn_cores
        recv_size_from_attn = _split_size_across_peers(
            per_attn_btck,
            moe_dests,
            {dest: local_k_map.get(dest, 0) for dest in moe_dests}
        ).get(moe_core_abs_id, 0)

        # --- Padding to align with Attention dispatch ---
        pad_steps = 1 + (tp if tp > 1 else 0)
        for _ in range(pad_steps):
            worklist.append({
                "recv_cnt": 0,
                "cast": [],
                "prims": []
            })

        # --- MoE computation (B=Bdp: each MoE core processes one dp group's tokens) ---
        moe_prims = [
            {
                "type": "parse_input",
                "size": recv_size_from_attn,
                "sram_address": {"indata": "input_label", "outdata": "input_label"}
            },
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": "C",
                "OC": f"moeIS/{moe_tp}",
                "K": local_k,
                "E_N": en_ep_str,
                "need_choose": True,
                "is_merge": False,
                "route_mode": 1,
                "route_seed": route_seed,
                "route_global_en": input_vars['experts'],
                "route_global_k": global_k,
                "route_local_begin": local_begin,
                "sram_address": {"indata": "_input_label", "outdata": "matmul_moe1_out"},
                "dram_address": {"data": "matmul_moe1_data", "out": "TODO"}
            },
            {
                "type": "matmul_forward_moe",
                "use_hw": False,
                "B": 1,
                "T": "T",
                "need_choose": False,
                "C": "C",
                "OC": f"moeIS/{moe_tp}",
                "K": local_k,
                "is_merge": False,
                "E_N": en_ep_str,
                "sram_address": {"indata": "input_label", "outdata": "matmul_moe2_out"},
                "dram_address": {"data": "matmul_moe2_data", "out": "TODO"}
            },
            {
                "type": "swiglu_forward",
                "N": f"moeIS/{moe_tp}",
                "sram_address": {"indata": "matmul_moe1_out matmul_moe2_out", "outdata": "swiglu1_out"},
                "dram_address": {"input": 0, "data": -1}
            },
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": f"moeIS/{moe_tp}",
                "need_choose": False,
                "OC": "C",
                "K": local_k,
                "E_N": en_ep_str,
                "is_merge": True,
                "sram_address": {"indata": "swiglu1_out", "outdata": "matmul_moe3_out"},
                "dram_address": {"data": "matmul_moe3_data", "out": "TODO"}
            },
            {
                "type": "switch_data",
                "IN": btc_dp,
                "OUT": btc_dp_mtp,
                "sram_address": {"indata": "_matmul_moe3_out", "outdata": "switch_out"}
            }
        ]

        worklist.append({
            "cast": [],
            "recv_cnt": attn_cores,
            "recv_tag": 80,
            "prims": moe_prims
        })

        # --- Ring All-Reduce for EP ---
        if moe_tp > 1:
            ep_group_base = moe_base_id + ep_idx * moe_tp
            for phase in range(moe_tp):
                if phase == tp_idx:
                    cast_list = [{"dest": ep_group_base + i, "tag": 60} for i in range(moe_tp) if i != tp_idx]
                    worklist.append({
                        "recv_cnt": 0,
                        "cast": cast_list,
                        "prims": [{
                            "type": "parse_output",
                            "size": btc_dp_mtp,
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        }]
                    })
                else:
                    worklist.append({
                        "recv_cnt": 1,
                        "recv_tag": 60,
                        "cast": [],
                        "prims": [{
                            "type": "parse_input",
                            "size": btc_dp_mtp,
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        }]
                    })

        # --- Combine: cast result back to ALL Attention cores ---
        # Keep combine symmetric with dispatch: per-destination message size
        # equals the per-attention dispatch contribution to this MoE core.
        cast_list = [{"dest": i, "tag": 81} for i in range(attn_cores)]
        worklist.append({
            "recv_cnt": 0,
            "cast": cast_list,
            "prims": [{
                "type": "parse_output",
                "size": recv_size_from_attn,
                "sram_address": {"indata": "matmul_moe3_out", "outdata": "matmul_moe3_out"}
            }]
        })

    return worklist


def process_cores_ep_mode(input_vars):
    """
    Process cores in EP (Expert Parallelism) mode without pp.

    Core ID scheme (no pp):
    - Per-dp-group: TP attention cores + EP MoE cores = (TP + EP)
    - Total: dp * (TP + EP)

    For dp_index d:
    - base = d * (TP + EP)
    - Attention core i: base + i (i = 0..TP-1)
    - MoE core j: base + TP + j (j = 0..EP-1)
    """
    tp = input_vars['mn'] * input_vars['k']
    moe_ep = input_vars.get('moe_ep', 1)
    moe_tp = input_vars.get('moe_tp', 1)
    moe_total = moe_ep * moe_tp
    dp = input_vars['dp']
    attn_cores = dp * tp

    # All layers in a single stage (no pp)
    core_layer = input_vars['L']
    loop_count = input_vars.get('loop', 1)

    cores = []
    moe_base = attn_cores
    for d in range(dp):
        attn_base = d * tp

        # Generate Attention Cores
        for i in range(tp):
            core = {
                "id": attn_base + i,
                "worklist": process_attention_core_worklist_ep(
                    input_vars, i, tp, moe_ep, moe_tp, core_layer,
                    d, attn_base, moe_base, attn_cores, loop_count)
            }
            cores.append(core)

    # Generate shared MoE Cores (across all dp groups)
    for j in range(moe_total):
        core = {
            "id": moe_base + j,
            "worklist": process_moe_core_worklist_ep(
                input_vars, j, tp, moe_ep, moe_tp, core_layer,
                attn_cores, moe_base, loop_count)
        }
        cores.append(core)

    return cores


def process_cores(input_vars):
    # Check for EP mode
    moe_ep = input_vars.get('moe_ep', 1)
    moe_tp = input_vars.get('moe_tp', 1)
    if moe_ep * moe_tp > 1:
        return process_cores_ep_mode(input_vars)

    global core_id, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id

    # All layers in a single stage (no pp)
    core_layer = input_vars['L']

    decoder = gpt
    if input_vars['model'] == "qwen":
        decoder = moe_qwen
    elif input_vars['model'] == "gpt":
        decoder = moe_gpt

    cores = []
    for dp_index in range(input_vars['dp']):
        for k_index in range(input_vars["k"]):
            for mn_index in range(input_vars["mn"]):
                dp_base = dp_index * input_vars['k'] * input_vars['mn']
                if input_vars['k'] != 1 and input_vars["mn"] != 1:
                    core_id = dp_base + k_index * input_vars["mn"] + mn_index
                    if mn_index < input_vars["mn"] - 1:
                        mn_cast_id = dp_base + k_index * input_vars["mn"] + mn_index + 1
                    else:
                        mn_cast_id = dp_base + (k_index-1) * input_vars["mn"] + mn_index + 1

                    if mn_index > 0:
                        mn_recv_id = dp_base + k_index * input_vars["mn"] + mn_index - 1
                    else:
                        mn_recv_id = dp_base + (k_index+1) * input_vars["mn"] + mn_index - 1

                    if k_index < input_vars["k"] - 1 :
                        k_cast_id = dp_base + (k_index + 1) * input_vars["mn"] + mn_index
                    else:
                        k_cast_id = dp_base + mn_index

                    if k_index > 0:
                        k_recv_id = dp_base + (k_index - 1) * input_vars["mn"] + mn_index
                    else:
                        k_recv_id = dp_base + (input_vars["k"] - 1) * input_vars["mn"] + mn_index
                elif input_vars['k'] == 1 and input_vars["mn"] != 1:
                    core_id = dp_base + mn_index
                    mn_cast_id = dp_base + mn_index + 1 if mn_index < input_vars["mn"] - 1 else dp_base
                    mn_recv_id = dp_base + mn_index - 1 if mn_index > 0 else dp_base + input_vars["mn"] - 1
                    k_cast_id = None
                    k_recv_id = None
                elif input_vars['k'] != 1 and input_vars["mn"] == 1:
                    core_id = dp_base + k_index
                    mn_cast_id = None
                    mn_recv_id = None
                    k_cast_id = dp_base + k_index + 1 if k_index < input_vars["k"] - 1 else dp_base
                    k_recv_id = dp_base + k_index - 1 if k_index > 0 else dp_base + input_vars["k"] - 1
                else:
                    core_id = dp_index
                    mn_recv_id = None
                    mn_cast_id = None
                    k_recv_id = None
                    k_cast_id = None

                core = {"id": core_id,
                        "loop": "loop",
                        "worklist": process_worklist_mnk(input_vars, core_id, core_layer, decoder, mn_cast_id, mn_recv_id, k_cast_id, k_recv_id, True)}
                cores.append(core)
    return cores


def apply_prim_copy_optimization(cores):
    """
    Post-process cores to use prim_copy for cores with identical prims.
    When two cores share the same prims in all worklist items, subsequent cores
    use prim_copy referencing the first core's id, and their worklist only
    contains recv_cnt and cast fields (no prims).
    """
    def get_prims_signature(core):
        worklist = core.get("worklist", [])
        prims_list = []
        for work_item in worklist:
            prims = work_item.get("prims", [])
            prims_list.append(prims)
        return json.dumps(prims_list, sort_keys=True)

    signature_to_first_id = {}
    optimized_cores = []

    for core in cores:
        sig = get_prims_signature(core)

        if sig not in signature_to_first_id:
            signature_to_first_id[sig] = core["id"]
            optimized_cores.append(core)
        else:
            new_core = {
                "id": core["id"],
                "prim_copy": signature_to_first_id[sig],
                "loop": core.get("loop"),
            }
            stripped_worklist = []
            for work_item in core.get("worklist", []):
                stripped = {"recv_cnt": work_item.get("recv_cnt", 0)}
                if "recv_tag" in work_item:
                    stripped["recv_tag"] = work_item["recv_tag"]
                if "cast" in work_item:
                    stripped["cast"] = work_item["cast"]
                stripped_worklist.append(stripped)
            new_core["worklist"] = stripped_worklist
            optimized_cores.append(new_core)

    return optimized_cores


def process_chips(input_vars):
    cores = process_cores(input_vars)
    chips = {"chip_id": 0, "cores": cores}
    return [chips]


# Preset MoE model configurations
# Format: { DH, NH, KVH, HS, L, IS, experts, topk, model }
MOE_PRESETS = {
    # DeepSeek-V3 / R1: 671B total, 37B active params
    "deepseek-v3": {
        "DH": 128, "NH": 128, "KVH": 128, "HS": 7168, "L": 61,
        "IS": 2048, "experts": 256, "topk": 8, "model": "qwen",
        "desc": "DeepSeek-V3 / R1 (671B-a37B, 256 experts, top-8)"
    },
    # Qwen3-235B-A22B (MoE)
    "qwen3-235b": {
        "DH": 128, "NH": 64, "KVH": 4, "HS": 4096, "L": 94,
        "IS": 2560, "experts": 128, "topk": 8, "model": "qwen",
        "desc": "Qwen3-235B-A22B (128 experts, top-8)"
    },
    # Mixtral 8x7B: ~46.7B total, ~12.9B active
    "mixtral-8x7b": {
        "DH": 128, "NH": 32, "KVH": 8, "HS": 4096, "L": 32,
        "IS": 14336, "experts": 8, "topk": 2, "model": "qwen",
        "desc": "Mixtral 8x7B (~46.7B total, ~12.9B active, 8 experts, top-2)"
    },
    # Mixtral 8x22B: ~176B total, ~39B active
    "mixtral-8x22b": {
        "DH": 128, "NH": 48, "KVH": 8, "HS": 6144, "L": 56,
        "IS": 16384, "experts": 8, "topk": 2, "model": "qwen",
        "desc": "Mixtral 8x22B (~176B total, ~39B active, 8 experts, top-2)"
    },
    # DBRX: 132B total, ~36B active
    "dbrx": {
        "DH": 128, "NH": 48, "KVH": 8, "HS": 6144, "L": 40,
        "IS": 10752, "experts": 16, "topk": 4, "model": "gpt",
        "desc": "DBRX (132B total, ~36B active, 16 experts, top-4)"
    },
    # Qwen2-57B-A14B (MoE)
    "qwen2-57b": {
        "DH": 128, "NH": 28, "KVH": 4, "HS": 3584, "L": 28,
        "IS": 2560, "experts": 64, "topk": 8, "model": "qwen",
        "desc": "Qwen2-57B-A14B (64 experts, top-8)"
    },
    # Qwen3-30B-A3B (MoE)
    "qwen3-30b": {
        "DH": 128, "NH": 32, "KVH": 4, "HS": 2048, "L": 48,
        "IS": 1024, "experts": 128, "topk": 8, "model": "qwen",
        "desc": "Qwen3-30B-A3B (128 experts, top-8)"
    },
}


def list_presets():
    print("Available MoE model presets:")
    print("-" * 70)
    for name, cfg in MOE_PRESETS.items():
        print(f"  {name:20s} {cfg['desc']}")
        print(f"  {' ':20s} DH={cfg['DH']}, NH={cfg['NH']}, KVH={cfg['KVH']}, "
              f"HS={cfg['HS']}, L={cfg['L']}, IS={cfg['IS']}")
        print(f"  {' ':20s} experts={cfg['experts']}, topk={cfg['topk']}")
        print()


def main():
    global MOE_ROUTE_BASE_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, help="name of output traces", default="./moe.json", required=False)
    parser.add_argument("--preset", type=str, help="preset MoE model name (use --list_presets to see all)", default=None, required=False)
    parser.add_argument("--list_presets", action="store_true", help="list all available preset models and exit")
    parser.add_argument("--B", type=int, help="Batch_size", default=1, required=False)
    parser.add_argument("--T", type=int, help="seq length", default=54, required=False)
    parser.add_argument("--DH", type=int, help="dimension of head", default=128, required=False)
    parser.add_argument("--NH", type=int, help="number of heads", default=32, required=False)
    parser.add_argument("--KVH", type=int, help="KV heads", default=8, required=False)
    parser.add_argument("--HS", type=int, help="hidden size", default=2560, required=False)
    parser.add_argument("--L", type=int, help="transformer layers", default=1, required=False)
    parser.add_argument("--dp", type=int, help="dataset parallel", default=1, required=False)
    parser.add_argument("--tp", type=str, help="tensor parallel, mn_k", default="1_1", required=False)
    parser.add_argument("--IS", type=int, help="intermediate size (expert hidden size for moe)", default=14336, required=False)
    parser.add_argument("--avg_output", type=int, help="average output tokens", default=10, required=False)
    parser.add_argument("--model", type=str, help="gpt, qwen", default="qwen", required=False)
    parser.add_argument("--experts", type=int, help="number of experts", default=8, required=False)
    parser.add_argument("--topk", type=int, help="top k experts", default=2, required=False)
    parser.add_argument("--moe_ep", type=int, help="moe expert parallel degree", default=1, required=False)
    parser.add_argument("--moe_tp", type=int, help="moe tensor parallel degree", default=1, required=False)
    parser.add_argument("--route_seed", type=int, help="base random seed for MoE routing/all2all expert selection", default=MOE_ROUTE_BASE_SEED, required=False)

    args = parser.parse_args()
    MOE_ROUTE_BASE_SEED = args.route_seed & 0x7fffffff

    if args.list_presets:
        list_presets()
        return

    input_vars = vars(args)
    input_vars['moe_total'] = input_vars.get('moe_ep', 1) * input_vars.get('moe_tp', 1)
    if input_vars['experts'] % input_vars['moe_ep'] != 0:
        raise ValueError(f"experts ({input_vars['experts']}) must be divisible by moe_ep ({input_vars['moe_ep']})")
    if input_vars['IS'] % input_vars['moe_tp'] != 0:
        raise ValueError(f"IS ({input_vars['IS']}) must be divisible by moe_tp ({input_vars['moe_tp']})")

    # Apply preset if specified (preset values override defaults, but explicit CLI args override preset)
    if args.preset:
        preset_name = args.preset.lower()
        if preset_name not in MOE_PRESETS:
            print(f"Error: unknown preset '{args.preset}'. Use --list_presets to see available presets.")
            return
        preset = MOE_PRESETS[preset_name]
        # Get the parser defaults to detect which args were explicitly provided
        defaults = {k: v for k, v in vars(parser.parse_args([])).items()}
        for key in ["DH", "NH", "KVH", "HS", "L", "IS", "experts", "topk", "model"]:
            # Only apply preset value if the user didn't explicitly override it
            if input_vars[key] == defaults[key]:
                input_vars[key] = preset[key]
        print(f"Using preset: {preset['desc']}")
    input_vars.pop("preset", None)
    input_vars.pop("list_presets", None)

    varitations = {
        "layernorm1_data": 0,
        "split_matmul1_out": 0,
        "matmul1_data": 0,
        "attention1_data": 0,
        "matmul2_data": 0,
        "matmul2_out": 0,
        "merge_matmul1_in": 0,
        "residual1_out": 0,
        "matmul3_data": 0,
        "matmul4_data": 0,
        "matmul4_out": 0,
        "layernorm2_data": 0,
        "split_matmul2_out": 0,
        "residual2_out": 0
    }
    input_vars = input_vars | varitations

    # No pp: force pp=1 for init_vars compatibility
    input_vars['pp'] = 1
    init_vars(input_vars)

    # Build file name (include EP, dp if > 1)

    input_vars['tp'] = input_vars['tp'].split("_")
    input_vars['mn'], input_vars['k'] = [int(i) for i in input_vars['tp']]
    configs = {"random": False,
               "vars": input_vars,
               "pipeline": 1,
               "source": process_source(input_vars),
               "chips": process_chips(input_vars)
               }

    input_vars.pop('tp')
    input_vars.pop('model')
    input_vars.pop('file_name', None)  # Remove string field that breaks C++ parser
    input_vars.pop('route_seed', None)
    if 'moe_ep' in input_vars and input_vars['moe_ep'] <= 1:
        input_vars.pop('moe_ep')
    if 'moe_tp' in input_vars and input_vars['moe_tp'] <= 1:
        input_vars.pop('moe_tp')
    if 'moe_total' in input_vars and input_vars['moe_total'] <= 1:
        input_vars.pop('moe_total')
    input_vars.pop('pp', None)  # Remove pp from output vars
    input_vars.pop('loop', None)  # Remove loop from output vars (unrolled into worklist)  
    # Only remove loop when in EP mode (loop is unrolled into worklist)
    # In non-EP mode, cores reference "loop" variable for iteration count

    with open(args.file_name, "w", encoding="utf-8") as f:
        f.write(_compact_json(configs))


if __name__ == '__main__':
    main()
