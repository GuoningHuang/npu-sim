#!/usr/bin/env python3
"""
Analyze communication events from Chrome trace event files.
Compares baseline vs optimized simulation traces.

Strategy:
- Parse the config JSON to get the worklist tag sequence per core.
- Parse the trace JSON to extract B/E (begin/end) pairs for communication events.
- Assign tags by matching the sequential worklist ordering with the temporal event order.
"""

import json
import sys
from collections import defaultdict, Counter
import time


def parse_config_comm_sequence(config_path):
    """Parse the config to get the ordered sequence of communication operations per core.
    
    Returns dict: core_id -> list of (type, tag, count) tuples in order
    """
    with open(config_path) as f:
        cfg = json.load(f)
    
    core_sequences = {}
    for core in cfg['chips'][0]['cores']:
        core_id = core['id']
        seq = []
        for w in core['worklist']:
            recv_cnt = w.get('recv_cnt', 0)
            recv_tag = w.get('recv_tag', None)
            if recv_cnt > 0 and recv_tag is not None:
                seq.append(('recv', recv_tag, recv_cnt))
            
            casts = w.get('cast', [])
            if casts:
                tag_counts = defaultdict(int)
                for c in casts:
                    tag = c.get('tag')
                    if tag is not None:
                        tag_counts[tag] += 1
                for tag, cnt in tag_counts.items():
                    seq.append(('send', tag, cnt))
        
        core_sequences[core_id] = seq
    
    return core_sequences


def parse_trace_comm_events(trace_path):
    """Parse the trace file and extract B/E pairs for communication events."""
    events_by_key = defaultdict(list)
    pid_to_core = {}
    
    print(f"  Parsing {trace_path}...")
    t0 = time.time()
    
    line_count = 0
    with open(trace_path) as f:
        for line in f:
            line_count += 1
            line = line.strip().rstrip(',')
            if not line or line in ('{', '}', '[', ']', ']}'):
                continue
            if '"traceEvents"' in line:
                idx = line.find('[')
                if idx >= 0:
                    line = line[idx+1:].strip().rstrip(',')
                    if not line:
                        continue
            
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            name = ev.get('name', '')
            ph = ev.get('ph', '')
            pid = ev.get('pid')
            tid = ev.get('tid')
            ts = ev.get('ts')
            
            if name == 'process_name' and ph == 'M':
                pid_to_core[pid] = ev.get('args', {}).get('name', f'pid_{pid}')
                continue
            
            if ph in ('B', 'E') and pid is not None and ts is not None:
                if tid == 0 or tid == 3:
                    events_by_key[(pid, tid, name)].append((ts, ph))
    
    t1 = time.time()
    print(f"  Parsed {line_count} lines in {t1-t0:.1f}s")
    
    comm_events = defaultdict(list)
    
    for (pid, tid, name), ts_list in events_by_key.items():
        if pid > 16:
            continue
        ts_list.sort(key=lambda x: x[0])
        stack = []
        for ts, ph in ts_list:
            if ph == 'B':
                stack.append(ts)
            elif ph == 'E' and stack:
                begin_ts = stack.pop(0)
                dur = ts - begin_ts
                comm_events[pid].append((begin_ts, ts, dur, name, tid))
    
    for pid in comm_events:
        comm_events[pid].sort(key=lambda x: x[0])
    
    return comm_events, pid_to_core


def assign_tags_to_events(comm_events, core_sequences, pid_to_core):
    """Assign tags to communication events based on worklist sequence."""
    tagged_events = defaultdict(list)
    
    for pid in sorted(comm_events.keys()):
        core_name = pid_to_core.get(pid, f'pid_{pid}')
        try:
            core_id = int(core_name.split()[-1])
        except:
            continue
        
        if core_id not in core_sequences:
            continue
        
        seq = core_sequences[core_id]
        
        send_events = [(b, e, d, n) for b, e, d, n, t in comm_events[pid] if t == 3]
        recv_events = [(b, e, d, n) for b, e, d, n, t in comm_events[pid] if t == 0]
        
        send_req_events = [(b, e, d, n) for b, e, d, n in send_events if 'SEND_REQ' in n]
        send_data_events = [(b, e, d, n) for b, e, d, n in send_events if 'SEND_DATA' in n]
        send_done_events = [(b, e, d, n) for b, e, d, n in send_events if 'SEND_DONE' in n]
        
        recv_ack_events = [(b, e, d, n) for b, e, d, n in recv_events if 'RECV_ACK' in n]
        recv_data_events = [(b, e, d, n) for b, e, d, n in recv_events if 'RECV_DATA' in n]
        recv_conf_events = [(b, e, d, n) for b, e, d, n in recv_events if 'RECV_CONF' in n]
        recv_start_events = [(b, e, d, n) for b, e, d, n in recv_events if 'RECV_START' in n]
        recv_weight_events = [(b, e, d, n) for b, e, d, n in recv_events if 'RECV_WEIGHT' in n]
        
        send_req_idx = 0
        send_data_idx = 0
        send_done_idx = 0
        recv_ack_idx = 0
        recv_data_idx = 0
        recv_conf_idx = 0
        recv_start_idx = 0
        recv_weight_idx = 0
        
        for op_type, tag, count in seq:
            # Initial startup recvs (tag 0-15) - consume corresponding events
            if op_type == 'recv' and tag >= 0 and tag <= 15:
                for _ in range(count):
                    if recv_ack_idx < len(recv_ack_events):
                        recv_ack_idx += 1
                    if recv_data_idx < len(recv_data_events):
                        recv_data_idx += 1
                    if recv_conf_idx < len(recv_conf_events):
                        recv_conf_idx += 1
                    if recv_start_idx < len(recv_start_events):
                        recv_start_idx += 1
                    if recv_weight_idx < len(recv_weight_events):
                        recv_weight_idx += 1
                continue
            
            if op_type == 'send':
                for _ in range(count):
                    if send_req_idx < len(send_req_events):
                        tagged_events[('send_req', tag)].append(
                            (pid, core_id, *send_req_events[send_req_idx]))
                        send_req_idx += 1
                    if send_data_idx < len(send_data_events):
                        tagged_events[('send_data', tag)].append(
                            (pid, core_id, *send_data_events[send_data_idx]))
                        send_data_idx += 1
                    if send_done_idx < len(send_done_events):
                        tagged_events[('send_done', tag)].append(
                            (pid, core_id, *send_done_events[send_done_idx]))
                        send_done_idx += 1
            
            elif op_type == 'recv':
                for _ in range(count):
                    if recv_ack_idx < len(recv_ack_events):
                        tagged_events[('recv_ack', tag)].append(
                            (pid, core_id, *recv_ack_events[recv_ack_idx]))
                        recv_ack_idx += 1
                    if recv_data_idx < len(recv_data_events):
                        tagged_events[('recv_data', tag)].append(
                            (pid, core_id, *recv_data_events[recv_data_idx]))
                        recv_data_idx += 1
                    if recv_conf_idx < len(recv_conf_events):
                        tagged_events[('recv_conf', tag)].append(
                            (pid, core_id, *recv_conf_events[recv_conf_idx]))
                        recv_conf_idx += 1
                    if recv_start_idx < len(recv_start_events):
                        tagged_events[('recv_start', tag)].append(
                            (pid, core_id, *recv_start_events[recv_start_idx]))
                        recv_start_idx += 1
                    if recv_weight_idx < len(recv_weight_events):
                        tagged_events[('recv_weight', tag)].append(
                            (pid, core_id, *recv_weight_events[recv_weight_idx]))
                        recv_weight_idx += 1
        
        if core_id == 0:
            print(f"  Core {core_id}: send_req={len(send_req_events)} (consumed {send_req_idx}), "
                  f"send_data={len(send_data_events)} (consumed {send_data_idx}), "
                  f"recv_ack={len(recv_ack_events)} (consumed {recv_ack_idx}), "
                  f"recv_data={len(recv_data_events)} (consumed {recv_data_idx})")
    
    return tagged_events


def analyze_tagged_events(tagged_events):
    """Compute statistics for tagged events."""
    tag_stats = {}
    
    for (event_type, tag), events in tagged_events.items():
        if tag not in tag_stats:
            tag_stats[tag] = {}
        
        durations = [ev[4] for ev in events]
        
        core_durations = defaultdict(float)
        core_counts = defaultdict(int)
        for ev in events:
            core_id = ev[1]
            core_durations[core_id] += ev[4]
            core_counts[core_id] += 1
        
        if durations:
            tag_stats[tag][event_type] = {
                'count': len(durations),
                'total_dur': sum(durations),
                'avg_dur': sum(durations) / len(durations),
                'min_dur': min(durations),
                'max_dur': max(durations),
                'per_core_dur': dict(core_durations),
                'per_core_cnt': dict(core_counts),
            }
    
    return tag_stats


def print_comparison(stats_baseline, stats_opt):
    """Print a comparison table."""
    tag_names = {50: "TP All-Reduce", 70: "Dispatch All-to-All", 80: "Combine All-to-All"}
    
    all_tags = sorted(set(list(stats_baseline.keys()) + list(stats_opt.keys())))
    
    print("\n" + "=" * 120)
    print(f"{'COMMUNICATION EVENT COMPARISON':^120}")
    print("=" * 120)
    
    for tag in all_tags:
        if tag not in tag_names:
            continue
        tag_name = tag_names.get(tag, f"Tag {tag}")
        print(f"\n{'─' * 120}")
        print(f"  TAG {tag}: {tag_name}")
        print(f"{'─' * 120}")
        
        types_b = stats_baseline.get(tag, {})
        types_o = stats_opt.get(tag, {})
        all_types = sorted(set(list(types_b.keys()) + list(types_o.keys())))
        
        for etype in all_types:
            sb = types_b.get(etype, {})
            so = types_o.get(etype, {})
            
            print(f"\n  Event: {etype}")
            print(f"  {'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Diff':>15} {'Ratio':>10}")
            print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*15} {'─'*10}")
            
            for metric in ['count', 'total_dur', 'avg_dur', 'min_dur', 'max_dur']:
                vb = sb.get(metric, 0)
                vo = so.get(metric, 0)
                diff = vo - vb
                ratio = vo / vb if vb != 0 else float('inf')
                
                if metric == 'count':
                    print(f"  {metric:<25} {vb:>15d} {vo:>15d} {diff:>+15d} {ratio:>10.3f}x")
                else:
                    print(f"  {metric + ' (cycles)':<25} {vb:>15.2f} {vo:>15.2f} {diff:>+15.2f} {ratio:>10.3f}x")
            
            cores_b = sb.get('per_core_dur', {})
            cores_o = so.get('per_core_dur', {})
            if cores_b or cores_o:
                all_cores = sorted(set(list(cores_b.keys()) + list(cores_o.keys())))
                print(f"\n  Per-core total duration:")
                print(f"  {'Core':<10} {'Baseline':>15} {'Optimized':>15} {'Diff':>15} {'Ratio':>10}")
                print(f"  {'─'*10} {'─'*15} {'─'*15} {'─'*15} {'─'*10}")
                for core in all_cores:
                    cb = cores_b.get(core, 0)
                    co = cores_o.get(core, 0)
                    d = co - cb
                    r = co / cb if cb != 0 else float('inf')
                    print(f"  Core {core:<4} {cb:>15.2f} {co:>15.2f} {d:>+15.2f} {r:>10.3f}x")
    
    # Aggregate summary
    print(f"\n{'=' * 120}")
    print(f"{'AGGREGATE SUMMARY':^120}")
    print(f"{'=' * 120}")
    
    print(f"\n  {'Tag':<30} {'Event Type':<15} {'Baseline Total':>18} {'Opt Total':>18} {'Speedup':>10}")
    print(f"  {'─'*30} {'─'*15} {'─'*18} {'─'*18} {'─'*10}")
    
    for tag in all_tags:
        if tag not in tag_names:
            continue
        tag_name = tag_names.get(tag, f"Tag {tag}")
        types_b = stats_baseline.get(tag, {})
        types_o = stats_opt.get(tag, {})
        
        for etype in ['send_data', 'recv_data', 'recv_ack', 'send_req', 'send_done', 'recv_conf', 'recv_start', 'recv_weight']:
            sb = types_b.get(etype, {})
            so = types_o.get(etype, {})
            vb = sb.get('total_dur', 0)
            vo = so.get('total_dur', 0)
            if vb == 0 and vo == 0:
                continue
            speedup = vb / vo if vo != 0 else float('inf')
            print(f"  {tag_name + f' (tag {tag})':<30} {etype:<15} {vb:>18.2f} {vo:>18.2f} {speedup:>10.3f}x")


def main():
    config_path = '/home/code/npu-sim/entewine/30B-A3B-ftd.json'
    baseline_path = '/home/code/npu-sim/entewine/events-baseline-ftd.json'
    opt_path = '/home/code/npu-sim/entewine/events-opt-ftd.json'
    
    print("=" * 80)
    print("Communication Event Analysis: Baseline vs Optimized")
    print("=" * 80)
    
    print("\n[1] Parsing config for worklist tag sequences...")
    core_sequences = parse_config_comm_sequence(config_path)
    print(f"  Found {len(core_sequences)} cores")
    
    send_tags = Counter()
    recv_tags = Counter()
    for core_id, seq in core_sequences.items():
        for op_type, tag, count in seq:
            if op_type == 'send':
                send_tags[tag] += count
            else:
                recv_tags[tag] += count
    print(f"  Send operations by tag: {dict(send_tags)}")
    print(f"  Recv operations by tag: {dict(recv_tags)}")
    
    print("\n[2] Parsing baseline trace...")
    comm_baseline, pid_to_core_b = parse_trace_comm_events(baseline_path)
    
    print("\n[3] Parsing optimized trace...")
    comm_opt, pid_to_core_o = parse_trace_comm_events(opt_path)
    
    print("\n[4] Assigning tags to baseline events...")
    tagged_baseline = assign_tags_to_events(comm_baseline, core_sequences, pid_to_core_b)
    
    print("\n[5] Assigning tags to optimized events...")
    tagged_opt = assign_tags_to_events(comm_opt, core_sequences, pid_to_core_o)
    
    print("\n[6] Computing statistics...")
    stats_baseline = analyze_tagged_events(tagged_baseline)
    stats_opt = analyze_tagged_events(tagged_opt)
    
    print_comparison(stats_baseline, stats_opt)
    
    # Overall latency
    print(f"\n{'=' * 120}")
    print(f"{'OVERALL SIMULATION LATENCY':^120}")
    print(f"{'=' * 120}")
    
    max_ts_b = 0
    for pid, events in comm_baseline.items():
        for ev in events:
            max_ts_b = max(max_ts_b, ev[1])
    
    max_ts_o = 0
    for pid, events in comm_opt.items():
        for ev in events:
            max_ts_o = max(max_ts_o, ev[1])
    
    print(f"\n  Baseline max comm timestamp: {max_ts_b:.2f} cycles")
    print(f"  Optimized max comm timestamp: {max_ts_o:.2f} cycles")
    if max_ts_o > 0:
        print(f"  Ratio: {max_ts_b / max_ts_o:.3f}x")


if __name__ == '__main__':
    main()
