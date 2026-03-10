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
    dp = input_vars['dp']
    if dp > 1:
        input_vars["Bdp"] = input_vars["B"] // dp
    else:
        input_vars["Bdp"] = input_vars["B"]
    input_vars["C"] = int(input_vars['DH'] * input_vars['NH'])
    input_vars["R"] = int(input_vars['NH'] / input_vars['KVH'])
    input_vars["P"] = int(input_vars['HS'])
    input_vars["J"] = int(input_vars['IS'])
    input_vars["chunk"] = 1
    input_vars["loop"] = int(input_vars["avg_output"] + 1)

    input_vars['G'] = int(input_vars["C"] + 2 * input_vars["C"] / input_vars["R"])
    add_vars(input_vars, "BTC")

    # Parse TP from tp string
    tp_parts = input_vars['tp'].split("_")
    tp = int(tp_parts[0]) * int(tp_parts[1])

    # MoEntwine: effective EP = dp (FTD size), NOT dp*tp
    ep_eff = dp
    total_cores = dp * tp
    input_vars['ep_eff'] = ep_eff
    input_vars['total_cores'] = total_cores

    # MoE-related variables
    input_vars['moeIS'] = input_vars['IS']
    input_vars['K'] = input_vars['topk']
    input_vars['Kglobal'] = min(input_vars["B"] * input_vars['topk'], input_vars['experts'])
    input_vars['E_N'] = input_vars['experts']

    # Expert distribution across all cores: E_N/(dp*tp)
    total_cores = dp * tp
    if total_cores > 1:
        input_vars[f'E_N/{total_cores}'] = input_vars['experts'] // total_cores

    # QKV dimension (3C-R means Q + K + V for GQA)
    qkv_dim = int(input_vars["C"] + 2 * input_vars["C"] / input_vars["R"])
    input_vars['3C-R'] = qkv_dim
    if tp > 1:
        input_vars[f'3C-R/{tp}'] = qkv_dim // tp

    # Additional derived variables
    input_vars['3C'] = 3 * input_vars['C']
    if tp > 1:
        input_vars[f'T/{tp}'] = input_vars['T'] // tp
        input_vars[f'C/{tp}'] = input_vars['C'] // tp
        input_vars[f'NH/{tp}'] = input_vars['NH'] // tp
        input_vars[f'BTC/{tp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // tp
    input_vars[f'C/2'] = input_vars['C'] // 2
    input_vars[f'NH/2'] = input_vars['NH'] // 2
    input_vars['3BTC/4'] = 3 * input_vars['B'] * input_vars['T'] * input_vars['C'] // 4

    # FTD-local All-to-All data sizes
    if dp > 1:
        input_vars[f'BTC/{dp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // dp
        input_vars[f'BTCK/{dp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] * input_vars['topk'] // dp
        input_vars[f'BTCK/{dp*dp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] * input_vars['topk'] // (dp * dp)
    if tp > 1 and dp > 1:
        input_vars[f'BTC/{dp*tp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // (dp * tp)


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
                    value *= input_vars[key]

            value = int(value / int(keys_num))
            if value == 0:
                value = 1
            input_vars[keys] = value


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


def cal_size(word1, word2=None, word3=None, mul_num=None, div_num=None):
    [word1_nun, word1_den], word1_type = find_const(word1)
    if word2 is not None:
        [word2_num, word2_den], word2_type = find_const(word2)
    else:
        [word2_num, word2_den], word2_type = [1, 1], ""
    if word3 is not None:
        [word3_num, word3_den], word3_type = find_const(word3)
    else:
        [word3_num, word3_den], word3_type = [1, 1], ""

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
        return_word = f"{int(num / den)}{word1_type}{word2_type}{word3_type}"
    elif den % num == 0:
        return_word = f"{word1_type}{word2_type}{word3_type}/{int(den / num)}"
    else:
        return_word = f"{num}{word1_type}{word2_type}{word3_type}/{den}"

    return return_word


def process_source(input_vars):
    """
    MoEntwine mode: all cores receive source data.
    Core layout: dp * tp cores total.
    """
    tp = input_vars['mn'] * input_vars['k']
    dp = input_vars['dp']
    source = []
    for d in range(dp):
        base = d * tp
        for i in range(tp):
            source.append({"dest": base + i, "size": f"BTC/{dp}" if dp > 1 else "BTC"})
    add_vars(input_vars, "BTC")
    if dp > 1:
        add_vars(input_vars, f"BTC/{dp}")
    return source


def get_ftd_peers(core_abs_id, tp, dp):
    """
    Compute FTD peer core IDs for a given core.

    Core abs_id c = d * tp + j, where:
      - d = c // tp  (DP group index, i.e. which TP group)
      - j = c % tp   (intra-TP position = FTD index)

    FTD j = {d2 * tp + j for d2 in range(dp)}
    Returns: sorted list of FTD peer core IDs (excluding self)
    """
    ftd_idx = core_abs_id % tp
    peers = []
    for d in range(dp):
        peer = d * tp + ftd_idx
        if peer != core_abs_id:
            peers.append(peer)
    return sorted(peers)


def _split_size_across_peers(total_size, peers, weights=None):
    """
    Split a sender's total traffic size across peers (integer exact sum).
    If weights are provided, split proportionally to weights[peer].
    Keep integer sizes and preserve total sum exactly.
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


def process_core_worklist_moentwine(input_vars, core_index, tp, core_layer, dp_index, base_id, total_cores, loop_count=1, phase="both"):
    """
    Generate worklist for a core in MoEntwine FTD-based mode.
    Each core handles both attention and MoE computation.

    Key difference from expert_wise: All-to-All dispatch/combine only happens
    within each FTD (dp cores), not across all cores (dp*tp).

    FTD (Full Token Domain): After TP All-Reduce, each core in a TP group holds
    identical data. An FTD groups one core from each TP group. Each FTD
    independently processes all experts via local All-to-All.

    Per layer structure:
    1. Attention: rmsnorm + QKV matmul + rope + attention + O matmul + switch_data
    2. TP All-Reduce (sequential broadcast within TP group, tag 50)
    3a. Pre-MoE: Residual + rmsnorm + gate_forward + switch_data (prepare dispatch)
    3b. Dispatch All-to-All within FTD (tag 70, dp phases)
    3c. MoE computation: load_expert(E_N/dp) + moe_up x2 + swiglu + moe_down + switch_data
    3d. Combine All-to-All within FTD (tag 80, dp phases)
    4. Final Residual
    """
    worklist = []

    dp = input_vars['dp']
    ep_eff = dp  # effective EP = FTD size
    core_abs_id = base_id + core_index

    # Pre-compute required variables
    add_vars(input_vars, "BTC")
    if tp > 1:
        add_vars(input_vars, f"BTC/{tp}")
        add_vars(input_vars, f"T/{tp}")
        add_vars(input_vars, f"C/{tp}")
        add_vars(input_vars, f"3C-R/{tp}")
        add_vars(input_vars, f"NH/{tp}")
    if dp > 1:
        add_vars(input_vars, f"BTC/{dp}")
        add_vars(input_vars, f"BTCK/{dp*dp}")
    add_vars(input_vars, "3BTC/4")

    # TP size strings
    c_r_tp = f"3C-R/{tp}" if tp > 1 else "3C-R"
    nh_tp = f"NH/{tp}" if tp > 1 else "NH"
    c_tp = f"C/{tp}" if tp > 1 else "C"
    btc_dp = f"BTC/{dp}" if dp > 1 else "BTC"
    btc_tp_dp = f"BTC/{dp*tp}" if (dp > 1 and tp > 1) else (btc_dp if dp > 1 else (f"BTC/{tp}" if tp > 1 else "BTC"))
    btc_tp = f"BTC/{tp}" if tp > 1 else "BTC"

    # FTD-local EP size strings (key difference from expert_wise)
    moe_is = "moeIS"  # full IS (complete experts)
    btck_dp = f"BTCK/{dp*dp}" if dp > 1 else "BTC"  # fallback fixed size
    # Local expert count per core: E_N/(dp*tp)
    en_local = input_vars['experts'] // total_cores if total_cores > 1 else input_vars['experts']
    if total_cores > 1:
        input_vars[f'E_N/{total_cores}'] = en_local
    en_dp_str = f"E_N/{total_cores}" if total_cores > 1 else "E_N"

    # FTD peers for this core
    ftd_peers = get_ftd_peers(core_abs_id, tp, dp) if dp > 1 else []
    ftd_members = sorted(ftd_peers + [core_abs_id]) if dp > 1 else [core_abs_id]

    total_layers = core_layer * loop_count

    for layer_idx in range(total_layers):
        is_first_layer = (layer_idx == 0)
        is_last_layer = (layer_idx == total_layers - 1)

        # Determine job_type: 0 = prefill, 1 = decode
        loop_iter = layer_idx // core_layer
        if phase == "prefill":
            job_type = 0
        elif phase == "decode":
            job_type = 0 if layer_idx == 0 else 1
        else:  # both
            job_type = 0 if loop_iter == 0 else 1

        # Route selection is global across all cores.
        global_k = min(input_vars.get('Kglobal', input_vars['topk']), input_vars['experts'])
        if total_cores > 1:
            experts_per_core = input_vars['experts'] // total_cores
            local_begin = core_abs_id * experts_per_core
            local_end = local_begin + experts_per_core
            route_seed = (MOE_ROUTE_BASE_SEED + layer_idx * 1000003) & 0x7fffffff
            chosen_global = _sample_unique_lcg(input_vars['experts'], global_k, route_seed)
            local_k = sum(1 for e in chosen_global if local_begin <= e < local_end)
        else:
            route_seed = (MOE_ROUTE_BASE_SEED + layer_idx * 1000003) & 0x7fffffff
            local_begin = 0
            local_k = min(global_k, input_vars['experts'])
        local_btck = input_vars["Bdp"] * input_vars["T"] * input_vars["C"] * local_k

        # Build per-core traffic size for this FTD region in this layer.
        # FTD communication model:
        # 1) Compute FTD total traffic from sum(local_k) within this FTD.
        # 2) Split that total evenly as sender budgets across FTD members.
        # 3) For each sender, split sender budget to peers by destination local_k.
        #    This is NOT full broadcast. Sender total is divided across peers.
        #    If peer local_k are equal, peer shares become (near) equal.
        ftd_sender_btck = {}
        ftd_core_k = {}
        if dp > 1:
            experts_per_core = input_vars['experts'] // total_cores
            for sender_core in ftd_members:
                s_begin = sender_core * experts_per_core
                s_end = s_begin + experts_per_core
                s_k = sum(1 for e in chosen_global if s_begin <= e < s_end)
                ftd_core_k[sender_core] = s_k
            # User-requested model:
            # 1) FTD total traffic is based on sum(local_k) within this FTD.
            # 2) Sender totals are evenly split across all cores in this FTD.
            unit_btck = input_vars["Bdp"] * input_vars["T"] * input_vars["C"]
            ftd_total_btck = unit_btck * sum(ftd_core_k.values())
            ftd_sender_btck = _split_size_across_peers(ftd_total_btck, ftd_members)
        ftd_sender_to_peer_btck = {}
        if dp > 1:
            for sender_core in ftd_members:
                sender_peers = [p for p in ftd_members if p != sender_core]
                # Split each sender's traffic by destination local_k so receive
                # amount tracks destination expert activity.
                peer_weights = {p: ftd_core_k.get(p, 0) for p in sender_peers}
                split_map = _split_size_across_peers(
                    ftd_sender_btck.get(sender_core, 0),
                    sender_peers,
                    peer_weights
                )
                for peer_core, peer_size in split_map.items():
                    ftd_sender_to_peer_btck[(sender_core, peer_core)] = peer_size

        # =============================================
        # 1. Attention computation worklist item
        # =============================================
        if is_first_layer:
            main_prims = [
                {
                    "type": "parse_input",
                    "size": btc_dp,
                    "sram_address": {"indata": "_residual2_out", "outdata": "rmsnorm1_in"}
                },
                {
                    "type": "rmsnorm_forward",
                    "B": "Bdp",
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
                    "B": "Bdp",
                    "T": "T",
                    "C": "C",
                    "sram_address": {"indata": "_residual2_out", "outdata": "rmsnorm1_out"},
                    "dram_address": {"data": "rmsnorm1_data"}
                },
            ]
            main_recv_cnt = 0

        main_prims += [
            {
                "type": "matmul_forward_pd",
                "use_hw": False,
                "B": "Bdp",
                "T": "T",
                "C": "C",
                "OC": c_r_tp,
                "R": "R",
                "chunk": "chunk",
                "job_type": job_type,
                "sram_address": {"indata": "rmsnorm1_out", "outdata": "matmul1_out"},
                "dram_address": {"data": "matmul1_data"}
            },
            {
                "type": "rope_forward_pd",
                "B": "Bdp",
                "T": "T",
                "C": c_r_tp,
                "NH": nh_tp,
                "R": "R",
                "job_type": job_type,
                "sram_address": {"indata": "matmul1_out", "outdata": "rope1_out"},
                "dram_address": {"data": "rope1_data"}
            },
            {
                "type": "Attention_f_pd",
                "B": "Bdp",
                "T": "T",
                "C": c_tp,
                "NH": nh_tp,
                "DH": "DH",
                "R": "R",
                "job_type": job_type,
                "sram_address": {"indata": "rope1_out", "outdata": "attention1_out"},
                "dram_address": {"data": "attention1_data", "out": "TODO"}
            },
            {
                "type": "Matmul_f",
                "use_hw": False,
                "B": "Bdp",
                "T": "T",
                "C": c_tp,
                "OC": "C",
                "sram_address": {"indata": "attention1_out", "outdata": "matmul2_out"},
                "dram_address": {"data": "matmul2_data"}
            },
        ]

        # switch_data for TP allreduce (only if tp > 1)
        if tp > 1:
            main_prims.append({
                "type": "switch_data",
                "IN": btc_dp,
                "OUT": btc_tp_dp,
                "sram_address": {"indata": "_matmul2_out", "outdata": "switch_out"}
            })

        main_work = {
            "recv_cnt": main_recv_cnt,
            "cast": [],
            "prims": main_prims
        }
        if is_first_layer:
            main_work["recv_tag"] = base_id + core_index
        worklist.append(main_work)

        # =============================================
        # 2. TP All-Reduce (sequential broadcast within TP group)
        # =============================================
        if tp > 1:
            for tp_phase in range(tp):
                if tp_phase == core_index:
                    cast_list = [{"dest": base_id + i, "tag": 50} for i in range(tp) if i != core_index]
                    worklist.append({
                        "recv_cnt": 0,
                        "cast": cast_list,
                        "prims": [{
                            "type": "parse_output",
                            "size": btc_tp_dp,
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
                            "size": btc_tp_dp,
                            "sram_address": {"indata": "switch_out", "outdata": "switch_out"}
                        }]
                    })

        # =============================================
        # 3a. Pre-MoE: Residual + RMSNorm + Gate + switch_data (prepare dispatch)
        # =============================================
        if is_first_layer:
            res1_indata = "rmsnorm1_in matmul2_out"
        else:
            res1_indata = "residual2_out matmul2_out"

        pre_moe_prims = [
            {
                "type": "Residual_f",
                "N": btc_dp,
                "sram_address": {"indata": res1_indata, "outdata": "residual1_out"},
                "dram_address": {"data": -1, "out": "TODO"}
            },
            {
                "type": "rmsnorm_forward",
                "B": "Bdp",
                "T": "T",
                "C": "C",
                "sram_address": {"indata": "_residual1_out", "outdata": "rmsnorm2_out"},
                "dram_address": {"data": "rmsnorm2_data"}
            },
            {
                "type": "gate_forward",
                "B": "Bdp",
                "T": "T",
                "C": "C",
                "K": local_k,
                "E_N": "E_N",
                "sram_address": {"indata": "_rmsnorm2_out", "outdata": "gate_out"},
                "dram_address": {"data": -1}
            },
        ]

        # switch_data to prepare dispatch data (only if dp > 1, FTD-local)
        if dp > 1:
            pre_moe_prims.append({
                "type": "switch_data",
                "IN": btc_dp,
                "OUT": local_btck,
                "sram_address": {"indata": "_rmsnorm2_out", "outdata": "dispatch_switch_out"}
            })

        worklist.append({
            "recv_cnt": 0,
            "cast": [],
            "prims": pre_moe_prims
        })

        # =============================================
        # 3b. Dispatch All-to-All WITHIN FTD (tag 70, dp phases)
        # =============================================
        if dp > 1:
            for ftd_phase_core in ftd_members:
                if ftd_phase_core == core_abs_id:
                    # This core sends dispatch data to each FTD peer (split by peer).
                    for peer in ftd_peers:
                        peer_size = ftd_sender_to_peer_btck.get((core_abs_id, peer), 0)
                        worklist.append({
                            "recv_cnt": 0,
                            "cast": [{"dest": peer, "tag": 70}],
                            "prims": [{
                                "type": "parse_output",
                                "size": peer_size,
                                "sram_address": {"indata": "dispatch_switch_out", "outdata": "dispatch_switch_out"}
                            }]
                        })
                else:
                    # This core receives dispatch data from FTD peer
                    peer_size = ftd_sender_to_peer_btck.get((ftd_phase_core, core_abs_id), 0)
                    worklist.append({
                        "recv_cnt": 1,
                        "recv_tag": 70,
                        "cast": [],
                        "prims": [{
                            "type": "parse_input",
                            "size": peer_size,
                            "sram_address": {"indata": "dispatch_switch_out", "outdata": "dispatch_switch_out"}
                        }]
                    })

        # =============================================
        # 3c. MoE computation (E_N/dp local experts with full IS)
        # =============================================
        load_expert_indata = "gate_out dispatch_switch_out matmul_moe1_data matmul_moe2_data matmul_moe3_data" if dp > 1 else "gate_out matmul_moe1_data matmul_moe2_data matmul_moe3_data"
        moe_compute_prims = [
            {
                "type": "load_expert",
                "E_N": en_dp_str,
                "K": local_k,
                "C": "C",
                "OC": moe_is,
                "need_choose": False,
                "strategy": 2,
                "sram_address": {
                    "indata": load_expert_indata,
                    "outdata": "load_expert_out"
                }
            },
            # MoE up matmul 1 (with expert selection)
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": "C",
                "OC": moe_is,
                "K": local_k,
                "E_N": en_dp_str,
                "need_choose": True,
                "is_merge": False,
                "route_mode": 1,
                "route_seed": route_seed,
                "route_global_en": input_vars['experts'],
                "route_global_k": global_k,
                "route_local_begin": local_begin,
                "sram_address": {"indata": "_rmsnorm2_out", "outdata": "matmul_moe1_out"},
                "dram_address": {"data": "matmul_moe1_data", "out": "TODO"}
            },
            # MoE up matmul 2 (for swiglu, reuse expert selection)
            {
                "type": "matmul_forward_moe",
                "use_hw": False,
                "B": 1,
                "T": "T",
                "C": "C",
                "OC": moe_is,
                "K": local_k,
                "E_N": en_dp_str,
                "need_choose": False,
                "is_merge": False,
                "sram_address": {"indata": "rmsnorm2_out", "outdata": "matmul_moe2_out"},
                "dram_address": {"data": "matmul_moe2_data", "out": "TODO"}
            },
            # swiglu
            {
                "type": "swiglu_forward",
                "N": moe_is,
                "sram_address": {"indata": "matmul_moe1_out matmul_moe2_out", "outdata": "swiglu1_out"},
                "dram_address": {"input": 0, "data": -1}
            },
            # MoE down matmul (merge expert results)
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": moe_is,
                "OC": "C",
                "K": local_k,
                "E_N": en_dp_str,
                "need_choose": False,
                "is_merge": True,
                "sram_address": {"indata": "swiglu1_out", "outdata": "matmul_moe3_out"},
                "dram_address": {"data": "matmul_moe3_data", "out": "TODO"}
            },
        ]

        # switch_data to prepare combine data (only if dp > 1)
        if dp > 1:
            moe_compute_prims.append({
                "type": "switch_data",
                "IN": btc_dp,
                "OUT": local_btck,
                "sram_address": {"indata": "_matmul_moe3_out", "outdata": "combine_switch_out"}
            })

        worklist.append({
            "recv_cnt": 0,
            "cast": [],
            "prims": moe_compute_prims
        })

        # =============================================
        # 3d. Combine All-to-All WITHIN FTD (tag 80, dp phases)
        # =============================================
        if dp > 1:
            for ftd_phase_core in ftd_members:
                if ftd_phase_core == core_abs_id:
                    # This core sends combine data to each FTD peer (split by peer).
                    for peer in ftd_peers:
                        peer_size = ftd_sender_to_peer_btck.get((core_abs_id, peer), 0)
                        worklist.append({
                            "recv_cnt": 0,
                            "cast": [{"dest": peer, "tag": 80}],
                            "prims": [{
                                "type": "parse_output",
                                "size": peer_size,
                                "sram_address": {"indata": "combine_switch_out", "outdata": "combine_switch_out"}
                            }]
                        })
                else:
                    # This core receives combine data from FTD peer
                    peer_size = ftd_sender_to_peer_btck.get((ftd_phase_core, core_abs_id), 0)
                    worklist.append({
                        "recv_cnt": 1,
                        "recv_tag": 80,
                        "cast": [],
                        "prims": [{
                            "type": "parse_input",
                            "size": peer_size,
                            "sram_address": {"indata": "combine_switch_out", "outdata": "combine_switch_out"}
                        }]
                    })

        # =============================================
        # 4. Final Residual
        # =============================================
        moe_result = "combine_switch_out" if dp > 1 else "matmul_moe3_out"
        residual_prim = {
            "type": "Residual_f",
            "N": btc_dp,
            "sram_address": {"indata": f"{moe_result} residual1_out", "outdata": "residual2_out"},
            "dram_address": {"data": -1, "out": "residual2_out"}
        }

        final_cast = [{"dest": -1}] if is_last_layer else []

        worklist.append({
            "recv_cnt": 0,
            "cast": final_cast,
            "prims": [residual_prim]
        })

    return worklist


def process_cores(input_vars):
    """
    MoEntwine mode: all cores handle both attention and MoE.
    Core layout: dp groups, each with tp cores. Total = dp * tp.
    FTDs: tp FTDs, each with dp cores (one from each TP group).
    """
    tp = input_vars['mn'] * input_vars['k']
    dp = input_vars['dp']
    core_layer = input_vars['L']
    total_cores = dp * tp
    loop_count = input_vars.get('loop', 1)
    phase = input_vars.get('phase', 'both')

    cores = []
    for d in range(dp):
        base = d * tp
        for i in range(tp):
            core = {
                "id": base + i,
                "worklist": process_core_worklist_moentwine(
                    input_vars, i, tp, core_layer, d, base, total_cores, loop_count, phase)
            }
            cores.append(core)

    return cores


def process_chips(input_vars):
    cores = process_cores(input_vars)
    chips = {"chip_id": 0, "cores": cores}
    return [chips]


# Preset MoE model configurations
MOE_PRESETS = {
    "deepseek-v3": {
        "DH": 128, "NH": 128, "KVH": 128, "HS": 7168, "L": 61,
        "IS": 2048, "experts": 256, "topk": 8, "model": "qwen",
        "desc": "DeepSeek-V3 / R1 (671B-a37B, 256 experts, top-8)"
    },
    "qwen3-235b": {
        "DH": 128, "NH": 64, "KVH": 4, "HS": 4096, "L": 94,
        "IS": 2560, "experts": 128, "topk": 8, "model": "qwen",
        "desc": "Qwen3-235B-A22B (128 experts, top-8)"
    },
    "mixtral-8x7b": {
        "DH": 128, "NH": 32, "KVH": 8, "HS": 4096, "L": 32,
        "IS": 14336, "experts": 8, "topk": 2, "model": "qwen",
        "desc": "Mixtral 8x7B (~46.7B total, ~12.9B active, 8 experts, top-2)"
    },
    "mixtral-8x22b": {
        "DH": 128, "NH": 48, "KVH": 8, "HS": 6144, "L": 56,
        "IS": 16384, "experts": 8, "topk": 2, "model": "qwen",
        "desc": "Mixtral 8x22B (~176B total, ~39B active, 8 experts, top-2)"
    },
    "dbrx": {
        "DH": 128, "NH": 48, "KVH": 8, "HS": 6144, "L": 40,
        "IS": 10752, "experts": 16, "topk": 4, "model": "gpt",
        "desc": "DBRX (132B total, ~36B active, 16 experts, top-4)"
    },
    "qwen2-57b": {
        "DH": 128, "NH": 28, "KVH": 4, "HS": 3584, "L": 28,
        "IS": 2560, "experts": 64, "topk": 8, "model": "qwen",
        "desc": "Qwen2-57B-A14B (64 experts, top-8)"
    },
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
    parser = argparse.ArgumentParser(
        description="MoE workload generator - MoEntwine FTD-based mode (FTD-local All-to-All)")
    parser.add_argument("--file_name", type=str, help="name of output traces", default="./moe_moentwine.json", required=False)
    parser.add_argument("--preset", type=str, help="preset MoE model name (use --list_presets to see all)", default=None, required=False)
    parser.add_argument("--list_presets", action="store_true", help="list all available preset models and exit")
    parser.add_argument("--B", type=int, help="Batch_size", default=1, required=False)
    parser.add_argument("--T", type=int, help="seq length", default=54, required=False)
    parser.add_argument("--DH", type=int, help="dimension of head", default=128, required=False)
    parser.add_argument("--NH", type=int, help="number of heads", default=32, required=False)
    parser.add_argument("--KVH", type=int, help="KV heads", default=8, required=False)
    parser.add_argument("--HS", type=int, help="hidden size", default=2560, required=False)
    parser.add_argument("--L", type=int, help="transformer layers", default=1, required=False)
    parser.add_argument("--dp", type=int, help="dataset parallel (= FTD size = effective EP)", default=1, required=False)
    parser.add_argument("--tp", type=str, help="tensor parallel, mn_k", default="1_1", required=False)
    parser.add_argument("--IS", type=int, help="intermediate size (expert hidden size for moe)", default=14336, required=False)
    parser.add_argument("--avg_output", type=int, help="average output tokens", default=10, required=False)
    parser.add_argument("--model", type=str, help="gpt, qwen or moe", default="moe", required=False)
    parser.add_argument("--experts", type=int, help="number of experts", default=8, required=False)
    parser.add_argument("--topk", type=int, help="top k experts", default=2, required=False)
    parser.add_argument("--phase", type=str, choices=["prefill", "decode", "both"],
                        help="workload phase: prefill (job_type=0), decode (job_type=1), or both", default="both", required=False)

    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    input_vars = vars(args)

    # Apply preset if specified
    if args.preset:
        preset_name = args.preset.lower()
        if preset_name not in MOE_PRESETS:
            print(f"Error: unknown preset '{args.preset}'. Use --list_presets to see available presets.")
            return
        preset = MOE_PRESETS[preset_name]
        defaults = {k: v for k, v in vars(parser.parse_args([])).items()}
        for key in ["DH", "NH", "KVH", "HS", "L", "IS", "experts", "topk", "model"]:
            if input_vars[key] == defaults[key]:
                input_vars[key] = preset[key]
        print(f"Using preset: {preset['desc']}")
    input_vars.pop("preset", None)
    input_vars.pop("list_presets", None)

    if args.dp > 1 and args.B % args.dp != 0:
        print(f"Error: batch size B ({args.B}) must be divisible by dp ({args.dp})")
        return

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
    init_vars(input_vars)

    # Validate MoEntwine constraints
    dp = input_vars['dp']
    tp_parts = input_vars['tp'].split("_")
    tp = int(tp_parts[0]) * int(tp_parts[1])
    total_cores = dp * tp
    if total_cores > 1 and input_vars['experts'] % total_cores != 0:
        print(f"Error: experts ({input_vars['experts']}) must be divisible by total cores ({total_cores}) in MoEntwine mode")
        return
    if dp <= 1:
        print("Warning: dp=1 means no FTD parallelism. MoE computation is local only.")
    if tp <= 1:
        print("Warning: tp=1 means each FTD contains all cores. Equivalent to expert_wise mode.")

    # Adjust loop count based on phase
    if input_vars['phase'] == "prefill":
        input_vars['loop'] = 1
    elif input_vars['phase'] == "decode":
        input_vars['loop'] = input_vars['avg_output']
    # else "both": keep loop = avg_output + 1 (set by init_vars)

    input_vars['tp'] = input_vars['tp'].split("_")
    input_vars['mn'], input_vars['k'] = [int(i) for i in input_vars['tp']]

    configs = {
        "random": False,
        "vars": input_vars,
        "pipeline": 1,
        "source": process_source(input_vars),
        "chips": process_chips(input_vars)
    }

    input_vars.pop('tp')
    input_vars.pop('model')
    input_vars.pop('file_name', None)
    input_vars.pop('loop', None)
    input_vars.pop('phase', None)
    input_vars.pop('ep_eff', None)
    input_vars.pop('total_cores', None)

    with open(args.file_name, "w", encoding="utf-8") as f:
        f.write(_compact_json(configs))


if __name__ == '__main__':
    main()
