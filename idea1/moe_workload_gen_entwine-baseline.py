import os
import json
import argparse
import re
global input_vars
SUMMARY_SENTINEL_TYPE = "Residual_f"
SUMMARY_SENTINEL_OUT = "residual2_out"
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


def _resolve_value(expr, vars_map):
    """Return numeric value for a size expression if possible."""
    if expr is None:
        return None
    if isinstance(expr, (int, float)):
        return expr
    if expr in vars_map:
        return vars_map[expr]
    try:
        return int(expr)
    except Exception:
        pass
    try:
        [num, den], word_type = find_const(expr)
        if word_type and word_type in vars_map:
            return int(num * vars_map[word_type] / den)
    except Exception:
        pass
    return None


def _fmt_size(expr, vars_map):
    val = _resolve_value(expr, vars_map)
    return f"{expr} (= {val})" if val is not None else str(expr)


def _pick_size_expr(prims):
    for prim in prims:
        for key in ("size", "OUT", "IN", "OC", "C", "N"):
            if key in prim:
                return prim[key]
    return None


def _is_layer_boundary(prim):
    return prim.get("type") == SUMMARY_SENTINEL_TYPE and prim.get("sram_address", {}).get("outdata") == SUMMARY_SENTINEL_OUT


def _print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _infer_dims(prim, vars_map):
    """Return (in_shape, out_shape, in_elems, out_elems)."""
    def gv(key, default=1):
        return _resolve_value(prim.get(key), vars_map) or default

    B = gv("B")
    T = gv("T")
    C = gv("C")
    OC = gv("OC")
    N = gv("N")

    ptype = prim.get("type", "")
    if "matmul" in ptype.lower():
        in_e = B * T * C
        out_e = B * T * OC
        return (f"[{B*T}, {_fmt_size(prim.get('C','C'), vars_map)}]",
                f"[{B*T}, {_fmt_size(prim.get('OC','OC'), vars_map)}]",
                in_e, out_e)
    if "attention" in ptype.lower():
        in_e = B * T * C
        out_e = B * T * C
        return (f"[{B*T}, {_fmt_size(prim.get('C','C'), vars_map)}]",
                f"[{B*T}, {_fmt_size(prim.get('C','C'), vars_map)}]",
                in_e, out_e)
    if "rmsnorm" in ptype.lower():
        in_e = B * T * C
        return (f"[{B*T}, {_fmt_size(prim.get('C','C'), vars_map)}]",
                f"[{B*T}, {_fmt_size(prim.get('C','C'), vars_map)}]",
                in_e, in_e)
    if "residual" in ptype.lower():
        in_e = gv("N")
        return (f"[{in_e}]", f"[{in_e}]", in_e, in_e)
    if "gate_forward" in ptype.lower():
        in_e = B * T * C
        K = gv("K")
        out_e = B * T * K
        return (f"[{B*T}, {_fmt_size(prim.get('C','C'), vars_map)}]",
                f"[{B*T}, {_fmt_size(prim.get('K','K'), vars_map)}]",
                in_e, out_e)
    if "swiglu" in ptype.lower():
        in_e = gv("N")
        return (f"[{in_e}]", f"[{in_e}]", in_e, in_e)
    # fallback 1D
    size_expr = prim.get("size") or prim.get("OUT") or prim.get("IN") or prim.get("OC") or prim.get("C") or prim.get("N")
    elems = _resolve_value(size_expr, vars_map) or 0
    shape = f"[{size_expr}]"
    return (shape, shape, elems, elems)


def print_layer_summary(configs, vars_map, core_id=0, layer_idx=0):
    """
    Best-effort textual summary of one layer on one core:
    - Per-prim input/output names
    - Size expressions and resolved element counts if possible
    - Send/recv across cores (shape & size expression)
    """
    chips = configs.get("chips", [])
    if not chips:
        print("[summary] no chips found")
        return
    cores = chips[0].get("cores", [])
    core = next((c for c in cores if c.get("id") == core_id), None)
    if core is None:
        print(f"[summary] core {core_id} not found")
        return

    worklist = core.get("worklist", [])
    cur_layer = 0
    layer_items = []
    for item in worklist:
        layer_items.append(item)
        if any(_is_layer_boundary(p) for p in item.get("prims", [])):
            if cur_layer == layer_idx:
                break
            cur_layer += 1
            layer_items = []
    if cur_layer != layer_idx:
        print(f"[summary] layer {layer_idx} not found on core {core_id}")
        return

    print(f"=== Layer {layer_idx} summary (core {core_id}) ===")
    rows = []
    dtype_bytes = vars_map.get("dtype_bytes", 2)  # assume fp16 unless provided
    for idx, item in enumerate(layer_items):
        prims = item.get("prims", [])
        cast = item.get("cast", [])
        recv_cnt = item.get("recv_cnt", 0)
        recv_tag = item.get("recv_tag", "-")
        recv_size = _fmt_size(_pick_size_expr(prims), vars_map)

        for p_i, prim in enumerate(prims):
            in_shape, out_shape, in_e, out_e = _infer_dims(prim, vars_map)
            traffic_mb = (in_e + out_e) * dtype_bytes / (1024 * 1024)
            routing = ""
            if cast:
                send_expr = _pick_size_expr(prims)
                routing = f"send {_fmt_size(send_expr, vars_map)} -> {[c['dest'] for c in cast]}"
            if recv_cnt > 0:
                routing = f"recv tag {recv_tag} x{recv_cnt} {_fmt_size(recv_size, vars_map)}"
            rows.append([
                prim.get("type"),
                in_shape,
                out_shape,
                f"{traffic_mb:.3f} MB" if traffic_mb > 0 else "-",
                routing or "-"
            ])
    _print_table(["Kernel", "Input dim (per core)", "Output dim (per core)", "Est. traffic", "Routing"], rows)


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

    # Parse TP from tp string; in entwine mode ep == dp * tp (all cores)
    tp_parts = input_vars['tp'].split("_")
    tp = int(tp_parts[0]) * int(tp_parts[1])
    ep = dp * tp  # entwine: all experts distributed across all cores
    input_vars['ep'] = ep

    # MoE-related variables
    input_vars['moeIS'] = input_vars['IS']
    if ep > 1:
        input_vars[f'moeIS/{ep}'] = input_vars['IS'] // ep
    input_vars['K'] = input_vars['topk']
    # Aggregate expert choices per DP group: Bdp * topk.
    # Cap by total experts to match current simulator behavior (unique selection only).
    input_vars['Kdp'] = min(input_vars["Bdp"] * input_vars['topk'], input_vars['experts'])
    input_vars['E_N'] = input_vars['experts']

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
    if dp > 1:
        input_vars[f'BTC/{dp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // dp
    if tp > 1 and dp > 1:
        input_vars[f'BTC/{dp*tp}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // (dp * tp)
    if ep > 1:
        input_vars[f'BTC/{ep}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] // ep
        # All-to-All data size for expert_wise mode: B*T*C*K/ep (full batch)
        input_vars[f'BTCK/{ep}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] * input_vars['topk'] // ep
        if dp > 1:
            # Per-core All-to-All data: each core has Bdp tokens, per-message = Bdp*T*C*K/ep
            input_vars[f'BTCK/{dp*ep}'] = input_vars['B'] * input_vars['T'] * input_vars['C'] * input_vars['topk'] // (dp * ep)


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
    Entwine mode: all cores receive source data.
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


def process_core_worklist_entwine(input_vars, core_index, tp, core_layer, dp_index, base_id, total_cores, loop_count=1, phase="both"):
    """
    Generate worklist for a core in entwine mode (no EP separation).
    Each core handles both attention and MoE computation.
    ep == dp * tp: all experts are evenly distributed across ALL cores.

    Per layer structure:
    1. Attention: rmsnorm + QKV matmul + rope + attention + O matmul + switch_data
    2. TP All-Reduce (sequential broadcast within dp group)
    3. MoE computation: Residual + rmsnorm + gate_forward + moe_up x2 + swiglu + moe_down + switch_data
    4. EP All-Reduce (sequential broadcast across ALL cores, ep == dp * tp)
    5. Final Residual (with loop cast on last layer)
    """
    worklist = []

    dp = input_vars['dp']
    ep = total_cores  # entwine: ep == dp * tp
    core_abs_id = base_id + core_index  # absolute core id across all dp groups

    # Pre-compute required variables
    add_vars(input_vars, "BTC")
    if tp > 1:
        add_vars(input_vars, f"BTC/{tp}")
        add_vars(input_vars, f"T/{tp}")
        add_vars(input_vars, f"C/{tp}")
        add_vars(input_vars, f"3C-R/{tp}")
        add_vars(input_vars, f"NH/{tp}")
    if ep > 1:
        add_vars(input_vars, f"moeIS/{ep}")
        add_vars(input_vars, f"BTC/{ep}")
    add_vars(input_vars, "3BTC/4")

    # TP size strings
    tp_str = tp if tp > 1 else 1
    c_r_tp = f"3C-R/{tp}" if tp > 1 else "3C-R"
    nh_tp = f"NH/{tp}" if tp > 1 else "NH"
    c_tp = f"C/{tp}" if tp > 1 else "C"
    btc_dp = f"BTC/{dp}" if dp > 1 else "BTC"
    btc_tp_dp = f"BTC/{dp*tp}" if (dp > 1 and tp > 1) else (btc_dp if dp > 1 else (f"BTC/{tp}" if tp > 1 else "BTC"))
    btc_tp = f"BTC/{tp}" if tp > 1 else "BTC"
    # EP size strings
    moe_is_ep = f"moeIS/{ep}" if ep > 1 else "moeIS"
    btc_ep = f"BTC/{ep}" if ep > 1 else "BTC"

    total_layers = core_layer * loop_count

    for layer_idx in range(total_layers):
        is_first_layer = (layer_idx == 0)
        is_last_layer = (layer_idx == total_layers - 1)

        # Determine job_type: 0 = prefill, 1 = decode
        # For decode phase, only the first layer (layer_idx==0) uses job_type=0
        # to initialize the shared KV cache with T tokens. All other layers use
        # job_type=1. Compute cost is identical since exu_ops/vec_ops don't
        # depend on job_type; only KV cache SRAM write size differs.
        loop_iter = layer_idx // core_layer
        if phase == "prefill":
            job_type = 0
        elif phase == "decode":
            job_type = 0 if layer_idx == 0 else 1
        else:  # both
            job_type = 0 if loop_iter == 0 else 1

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
        # 2. TP All-Reduce (sequential broadcast)
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
        # 3. MoE dispatch + computation
        # =============================================
        if is_first_layer:
            res1_indata = "rmsnorm1_in matmul2_out"
        else:
            res1_indata = "residual2_out matmul2_out"

        moe_prims = [
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
                "K": "Kdp",
                "E_N": "E_N",
                "sram_address": {"indata": "_rmsnorm2_out", "outdata": "gate_out"},
                "dram_address": {"data": -1}
            },
            {
                "type": "load_expert",
                "E_N": "E_N",
                "K": "Kdp",
                "C": "C",
                "OC": moe_is_ep,
                "need_choose": False,
                "strategy": 2,
                "sram_address": {
                    "indata": "gate_out matmul_moe1_data matmul_moe2_data matmul_moe3_data",
                    "outdata": "load_expert_out"
                }
            },
            # MoE up matmul 1 (with expert selection)
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": "C",
                "OC": moe_is_ep,
                "K": "Kdp",
                "E_N": "E_N",
                "need_choose": True,
                "is_merge": False,
                # Reuse rmsnorm2_out for the next MoE matmul (avoid deletion after this prim)
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
                "OC": moe_is_ep,
                "K": "Kdp",
                "E_N": "E_N",
                "need_choose": False,
                "is_merge": False,
                "sram_address": {"indata": "rmsnorm2_out", "outdata": "matmul_moe2_out"},
                "dram_address": {"data": "matmul_moe2_data", "out": "TODO"}
            },
            # swiglu
            {
                "type": "swiglu_forward",
                "N": moe_is_ep,
                "sram_address": {"indata": "matmul_moe1_out matmul_moe2_out", "outdata": "swiglu1_out"},
                "dram_address": {"input": 0, "data": -1}
            },
            # MoE down matmul (merge expert results)
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": moe_is_ep,
                "OC": "C",
                "K": "Kdp",
                "E_N": "E_N",
                "need_choose": False,
                "is_merge": True,
                "sram_address": {"indata": "swiglu1_out", "outdata": "matmul_moe3_out"},
                "dram_address": {"data": "matmul_moe3_data", "out": "TODO"}
            },
        ]

        # switch_data for EP allreduce (only if ep > 1)
        if ep > 1:
            moe_prims.append({
                "type": "switch_data",
                "IN": btc_dp,
                "OUT": btc_ep,
                "sram_address": {"indata": "_matmul_moe3_out", "outdata": "ep_switch_out"}
            })

        worklist.append({
            "recv_cnt": 0,
            "cast": [],
            "prims": moe_prims
        })

        # =============================================
        # 4. EP All-Reduce (sequential broadcast across ALL cores, ep == dp * tp)
        # =============================================
        if ep > 1:
            for ep_phase in range(ep):
                if ep_phase == core_abs_id:
                    # This core broadcasts to all other cores
                    cast_list = [{"dest": i, "tag": 60} for i in range(ep) if i != core_abs_id]
                    worklist.append({
                        "recv_cnt": 0,
                        "cast": cast_list,
                        "prims": [{
                            "type": "parse_output",
                            "size": btc_ep,
                            "sram_address": {"indata": "ep_switch_out", "outdata": "ep_switch_out"}
                        }]
                    })
                else:
                    # This core receives from broadcasting core
                    worklist.append({
                        "recv_cnt": 1,
                        "recv_tag": 60,
                        "cast": [],
                        "prims": [{
                            "type": "parse_input",
                            "size": btc_ep,
                            "sram_address": {"indata": "ep_switch_out", "outdata": "ep_switch_out"}
                        }]
                    })

        # =============================================
        # 5. Final Residual
        # =============================================
        residual_prim = {
            "type": "Residual_f",
            "N": btc_dp,
            "sram_address": {"indata": "matmul_moe3_out residual1_out", "outdata": "residual2_out"},
            "dram_address": {"data": -1, "out": "residual2_out"}
        }

        final_cast = [{"dest": -1}] if is_last_layer else []

        worklist.append({
            "recv_cnt": 0,
            "cast": final_cast,
            "prims": [residual_prim]
        })

    return worklist


def process_core_worklist_expert_wise(input_vars, core_index, tp, core_layer, dp_index, base_id, total_cores, loop_count=1, phase="both"):
    """
    Generate worklist for a core in expert-wise mode.
    Each core handles both attention and MoE computation.
    Each core holds E_N/ep COMPLETE experts (full IS), tokens are routed via All-to-All.

    Per layer structure:
    1. Attention: rmsnorm + QKV matmul + rope + attention + O matmul + switch_data
    2. TP All-Reduce (sequential broadcast within dp group)
    3a. Pre-MoE: Residual + rmsnorm + gate_forward + switch_data (prepare dispatch)
    3b. Dispatch All-to-All (sequential broadcast across ALL cores, tag 70)
    3c. MoE computation: load_expert(E_N/ep, full IS) + moe_up x2 + swiglu + moe_down + switch_data (prepare combine)
    3d. Combine All-to-All (sequential broadcast across ALL cores, tag 80)
    4. Final Residual (with loop cast on last layer)
    """
    worklist = []

    dp = input_vars['dp']
    ep = total_cores  # entwine: ep == dp * tp
    core_abs_id = base_id + core_index  # absolute core id across all dp groups

    # Pre-compute required variables
    add_vars(input_vars, "BTC")
    if tp > 1:
        add_vars(input_vars, f"BTC/{tp}")
        add_vars(input_vars, f"T/{tp}")
        add_vars(input_vars, f"C/{tp}")
        add_vars(input_vars, f"3C-R/{tp}")
        add_vars(input_vars, f"NH/{tp}")
    if ep > 1:
        add_vars(input_vars, f"moeIS/{ep}")
        add_vars(input_vars, f"BTC/{ep}")
        add_vars(input_vars, f"BTCK/{ep}")
        if dp > 1:
            add_vars(input_vars, f"BTCK/{dp*ep}")
    add_vars(input_vars, "3BTC/4")

    # TP size strings
    c_r_tp = f"3C-R/{tp}" if tp > 1 else "3C-R"
    nh_tp = f"NH/{tp}" if tp > 1 else "NH"
    c_tp = f"C/{tp}" if tp > 1 else "C"
    btc_dp = f"BTC/{dp}" if dp > 1 else "BTC"
    btc_tp_dp = f"BTC/{dp*tp}" if (dp > 1 and tp > 1) else (btc_dp if dp > 1 else (f"BTC/{tp}" if tp > 1 else "BTC"))
    btc_tp = f"BTC/{tp}" if tp > 1 else "BTC"
    # EP size strings - expert_wise: each core holds complete experts with full IS
    moe_is_ep = "moeIS"
    # All-to-All data size per message: each core has Bdp tokens, per-message = Bdp*T*C*K/ep
    btck_ep = f"BTCK/{dp*ep}" if (ep > 1 and dp > 1) else (f"BTCK/{ep}" if ep > 1 else "BTC")
    # Local expert count for load_expert
    en_local = input_vars['experts'] // ep if ep > 1 else input_vars['experts']
    input_vars[f'E_N/{ep}'] = en_local
    en_ep_str = f"E_N/{ep}" if ep > 1 else "E_N"

    total_layers = core_layer * loop_count

    for layer_idx in range(total_layers):
        is_first_layer = (layer_idx == 0)
        is_last_layer = (layer_idx == total_layers - 1)

        # Determine job_type: 0 = prefill, 1 = decode
        # For decode phase, only the first layer (layer_idx==0) uses job_type=0
        # to initialize the shared KV cache with T tokens. All other layers use
        # job_type=1. Compute cost is identical since exu_ops/vec_ops don't
        # depend on job_type; only KV cache SRAM write size differs.
        loop_iter = layer_idx // core_layer
        if phase == "prefill":
            job_type = 0
        elif phase == "decode":
            job_type = 0 if layer_idx == 0 else 1
        else:  # both
            job_type = 0 if loop_iter == 0 else 1

        # Expert-wise routing: sample global experts per (dp group, layer),
        # then map to local experts owned by this core.
        global_k = min(input_vars.get('Kdp', input_vars['topk']), input_vars['experts'])
        if ep > 1:
            experts_per_core = input_vars['experts'] // ep
            local_begin = core_abs_id * experts_per_core
            local_end = local_begin + experts_per_core
            route_seed = (MOE_ROUTE_BASE_SEED + layer_idx * 1000003 + dp_index * 9176) & 0x7fffffff
            chosen_global = _sample_unique_lcg(input_vars['experts'], global_k, route_seed)
            local_ids = [e - local_begin for e in chosen_global if local_begin <= e < local_end]
            local_k = len(local_ids)
        else:
            route_seed = (MOE_ROUTE_BASE_SEED + layer_idx * 1000003 + dp_index * 9176) & 0x7fffffff
            local_begin = 0
            local_k = min(global_k, input_vars['experts'])

        # =============================================
        # 1. Attention computation worklist item (same as is_split)
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
        # 2. TP All-Reduce (sequential broadcast, same as is_split)
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

        # switch_data to prepare dispatch data (only if ep > 1)
        if ep > 1:
            pre_moe_prims.append({
                "type": "switch_data",
                "IN": btc_dp,
                "OUT": btck_ep,
                "sram_address": {"indata": "_rmsnorm2_out", "outdata": "dispatch_switch_out"}
            })

        worklist.append({
            "recv_cnt": 0,
            "cast": [],
            "prims": pre_moe_prims
        })

        # =============================================
        # 3b. Dispatch All-to-All (sequential broadcast, per-msg = BTCK/ep²)
        # Phase ordering: Core ep-1 sends first (reversed), so the slowest core
        # (last to exit Combine) acts as an implicit barrier before Dispatch begins.
        # This prevents timing desync where fast cores (low IDs) start Dispatch
        # while slow cores (high IDs) are still in TP All-Reduce.
        # =============================================
        if ep > 1:
            for ep_phase in range(ep):
                if ep_phase == (ep - 1 - core_abs_id):
                    cast_list = [{"dest": i, "tag": 70} for i in range(ep) if i != core_abs_id]
                    worklist.append({
                        "recv_cnt": 0,
                        "cast": cast_list,
                        "prims": [{
                            "type": "parse_output",
                            "size": btck_ep,
                            "sram_address": {"indata": "dispatch_switch_out", "outdata": "dispatch_switch_out"}
                        }]
                    })
                else:
                    worklist.append({
                        "recv_cnt": 1,
                        "recv_tag": 70,
                        "cast": [],
                        "prims": [{
                            "type": "parse_input",
                            "size": btck_ep,
                            "sram_address": {"indata": "dispatch_switch_out", "outdata": "dispatch_switch_out"}
                        }]
                    })

        # =============================================
        # 3c. MoE computation (local experts with full IS)
        # =============================================
        moe_compute_prims = [
            {
                "type": "load_expert",
                "E_N": en_ep_str,
                "K": local_k,
                "C": "C",
                "OC": "moeIS",
                "need_choose": False,
                "strategy": 2,
                "sram_address": {
                    "indata": "gate_out dispatch_switch_out matmul_moe1_data matmul_moe2_data matmul_moe3_data",
                    "outdata": "load_expert_out"
                }
            },
            # MoE up matmul 1 (with expert selection)
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": "C",
                "OC": moe_is_ep,
                "K": local_k,
                "E_N": en_ep_str,
                "need_choose": True,
                "is_merge": False,
                # Route metadata lets simulator reconstruct the exact same local expert IDs.
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
                "OC": moe_is_ep,
                "K": local_k,
                "E_N": en_ep_str,
                "need_choose": False,
                "is_merge": False,
                "sram_address": {"indata": "rmsnorm2_out", "outdata": "matmul_moe2_out"},
                "dram_address": {"data": "matmul_moe2_data", "out": "TODO"}
            },
            # swiglu
            {
                "type": "swiglu_forward",
                "N": moe_is_ep,
                "sram_address": {"indata": "matmul_moe1_out matmul_moe2_out", "outdata": "swiglu1_out"},
                "dram_address": {"input": 0, "data": -1}
            },
            # MoE down matmul (merge expert results)
            {
                "type": "matmul_forward_moe",
                "B": 1,
                "T": "T",
                "C": moe_is_ep,
                "OC": "C",
                "K": local_k,
                "E_N": en_ep_str,
                "need_choose": False,
                "is_merge": True,
                "sram_address": {"indata": "swiglu1_out", "outdata": "matmul_moe3_out"},
                "dram_address": {"data": "matmul_moe3_data", "out": "TODO"}
            },
        ]

        # switch_data to prepare combine data (only if ep > 1)
        if ep > 1:
            moe_compute_prims.append({
                "type": "switch_data",
                "IN": btc_dp,
                "OUT": btck_ep,
                "sram_address": {"indata": "_matmul_moe3_out", "outdata": "combine_switch_out"}
            })

        worklist.append({
            "recv_cnt": 0,
            "cast": [],
            "prims": moe_compute_prims
        })

        # =============================================
        # 3d. Combine All-to-All (sequential broadcast, per-msg = BTCK/ep²)
        # Same reversed phase ordering as Dispatch for consistency.
        # =============================================
        if ep > 1:
            for ep_phase in range(ep):
                if ep_phase == (ep - 1 - core_abs_id):
                    cast_list = [{"dest": i, "tag": 80} for i in range(ep) if i != core_abs_id]
                    worklist.append({
                        "recv_cnt": 0,
                        "cast": cast_list,
                        "prims": [{
                            "type": "parse_output",
                            "size": btck_ep,
                            "sram_address": {"indata": "combine_switch_out", "outdata": "combine_switch_out"}
                        }]
                    })
                else:
                    worklist.append({
                        "recv_cnt": 1,
                        "recv_tag": 80,
                        "cast": [],
                        "prims": [{
                            "type": "parse_input",
                            "size": btck_ep,
                            "sram_address": {"indata": "combine_switch_out", "outdata": "combine_switch_out"}
                        }]
                    })

        # =============================================
        # 4. Final Residual
        # =============================================
        moe_result = "combine_switch_out" if ep > 1 else "matmul_moe3_out"
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
    Entwine mode: all cores are identical, handling both attention and MoE.
    Core layout: dp groups, each with tp cores. ep == dp * tp.
    Total cores = dp * tp.
    """
    tp = input_vars['mn'] * input_vars['k']
    dp = input_vars['dp']
    core_layer = input_vars['L']
    total_cores = dp * tp
    loop_count = input_vars.get('loop', 1)
    phase = input_vars.get('phase', 'both')
    ep_mode = input_vars.get('ep_mode', 'is_split')

    worklist_fn = process_core_worklist_expert_wise if ep_mode == "expert_wise" else process_core_worklist_entwine

    cores = []
    for d in range(dp):
        base = d * tp
        for i in range(tp):
            core = {
                "id": base + i,
                "worklist": worklist_fn(
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
    parser = argparse.ArgumentParser(description="MoE workload generator - entwine mode (no EP separation)")
    parser.add_argument("--file_name", type=str, help="name of output traces", default="./moe_entwine.json", required=False)
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
    parser.add_argument("--model", type=str, help="gpt, qwen or moe", default="moe", required=False)
    parser.add_argument("--experts", type=int, help="number of experts", default=8, required=False)
    parser.add_argument("--topk", type=int, help="top k experts", default=2, required=False)
    parser.add_argument("--phase", type=str, choices=["prefill", "decode", "both"],
                        help="workload phase: prefill (job_type=0), decode (job_type=1), or both", default="both", required=False)
    #   --phase prefill: 只生成 prefill 阶段，job_type=0，loop=1
    #   --phase decode: 只生成 decode 阶段，job_type=1，loop=avg_output
    #   --phase both (默认): prefill + decode，第一个 loop 迭代 job_type=0，之后 job_type=1，loop=avg_output+1
    parser.add_argument("--ep_mode", type=str, choices=["is_split", "expert_wise"],
                        help="EP parallelism mode: is_split (split IS across all cores) or expert_wise (each core holds E_N/ep complete experts)",
                        default="expert_wise", required=False)

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

    # Validate expert_wise mode constraints
    if input_vars.get('ep_mode') == 'expert_wise':
        ep = input_vars.get('ep', 1)
        if ep > 1 and input_vars['experts'] % ep != 0:
            print(f"Error: experts ({input_vars['experts']}) must be divisible by ep ({ep}) in expert_wise mode")
            return

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
    if 'ep' in input_vars and input_vars['ep'] <= 1:
        input_vars.pop('ep')
    input_vars.pop('loop', None)  # Remove loop from output vars (unrolled into worklist)
    input_vars.pop('phase', None)  # Remove phase from output vars
    input_vars.pop('ep_mode', None)  # Remove ep_mode from output vars

    with open(args.file_name, "w", encoding="utf-8") as f:
        f.write(_compact_json(configs))

    # --- summary for core0 layer0 -------------------------------------------------
    summary_vars = dict(input_vars)
    try:
        print_layer_summary(configs, summary_vars, core_id=0, layer_idx=0)
    except Exception as exc:  # best-effort; keep main output intact
        print(f"[summary skipped] {exc}")


if __name__ == '__main__':
    main()
