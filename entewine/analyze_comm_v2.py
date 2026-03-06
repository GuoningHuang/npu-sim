#!/usr/bin/env python3
"""
Communication Event Analysis: Baseline vs Optimized (v2)
Analyzes Chrome trace event files and groups communications by tag (50/70/80).
Also computes per-core total communication time and computation time breakdown.
"""

import json
import sys
from collections import defaultdict, Counter
import time


def parse_config_comm_sequence(config_path):
    """Parse config to get ordered sequence of comm ops per core."""
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


def parse_trace_all_events(trace_path):
    """Parse the trace file. Collect ALL B/E events for all threads."""
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
            
            if ph in ('B', 'E') and pid is not None and ts is not None and pid <= 16:
                events_by_key[(pid, tid, name)].append((ts, ph))
    
    t1 = time.time()
    print(f"  Parsed {line_count} lines in {t1-t0:.1f}s")
    
    # Match B/E pairs
    all_events = defaultdict(list)  # pid -> list of (begin, end, dur, name, tid)
    for (pid, tid, name), ts_list in events_by_key.items():
        ts_list.sort(key=lambda x: x[0])
        stack = []
        for ts, ph in ts_list:
            if ph == 'B':
                stack.append(ts)
            elif ph == 'E' and stack:
                begin_ts = stack.pop(0)
                dur = ts - begin_ts
                all_events[pid].append((begin_ts, ts, dur, name, tid))
    
    for pid in all_events:
        all_events[pid].sort(key=lambda x: x[0])
    
    return all_events, pid_to_core


def assign_tags(all_events, core_sequences, pid_to_core):
    """Assign tags to send/recv events. Also return comp events."""
    tagged = defaultdict(list)  # (event_category, tag) -> [(pid, core_id, begin, end, dur, name)]
    comp_events = defaultdict(list)  # pid -> [(begin, end, dur, name)]
    
    for pid in sorted(all_events.keys()):
        core_name = pid_to_core.get(pid, f'pid_{pid}')
        try:
            core_id = int(core_name.split()[-1])
        except:
            continue
        
        if core_id not in core_sequences:
            continue
        
        seq = core_sequences[core_id]
        
        # Separate by tid
        tid_events = defaultdict(list)
        for b, e, d, n, t in all_events[pid]:
            tid_events[t].append((b, e, d, n))
        
        # Computation events (tid=1)
        for b, e, d, n in tid_events.get(1, []):
            comp_events[pid].append((b, e, d, n))
        
        # Send sub-events (tid=3)
        send_req = [x for x in tid_events.get(3, []) if 'SEND_REQ' in x[3]]
        send_data = [x for x in tid_events.get(3, []) if 'SEND_DATA' in x[3]]
        send_done = [x for x in tid_events.get(3, []) if 'SEND_DONE' in x[3]]
        
        # Recv sub-events (tid=0)
        recv_ack = [x for x in tid_events.get(0, []) if 'RECV_ACK' in x[3]]
        recv_data = [x for x in tid_events.get(0, []) if 'RECV_DATA' in x[3]]
        recv_conf = [x for x in tid_events.get(0, []) if 'RECV_CONF' in x[3]]
        recv_start = [x for x in tid_events.get(0, []) if 'RECV_START' in x[3]]
        recv_weight = [x for x in tid_events.get(0, []) if 'RECV_WEIGHT' in x[3]]
        
        # Index counters
        idx = {
            'send_req': 0, 'send_data': 0, 'send_done': 0,
            'recv_ack': 0, 'recv_data': 0, 'recv_conf': 0,
            'recv_start': 0, 'recv_weight': 0
        }
        evlists = {
            'send_req': send_req, 'send_data': send_data, 'send_done': send_done,
            'recv_ack': recv_ack, 'recv_data': recv_data, 'recv_conf': recv_conf,
            'recv_start': recv_start, 'recv_weight': recv_weight
        }
        
        for op_type, tag, count in seq:
            if op_type == 'recv' and 0 <= tag <= 15:
                # Startup recvs - consume but don't tag
                for _ in range(count):
                    for k in ['recv_ack', 'recv_data', 'recv_conf', 'recv_start', 'recv_weight']:
                        if idx[k] < len(evlists[k]):
                            idx[k] += 1
                continue
            
            if op_type == 'send':
                for _ in range(count):
                    for k in ['send_req', 'send_data', 'send_done']:
                        if idx[k] < len(evlists[k]):
                            ev = evlists[k][idx[k]]
                            tagged[(k, tag)].append((pid, core_id, ev[0], ev[1], ev[2], ev[3]))
                            idx[k] += 1
            elif op_type == 'recv':
                for _ in range(count):
                    for k in ['recv_ack', 'recv_data', 'recv_conf', 'recv_start', 'recv_weight']:
                        if idx[k] < len(evlists[k]):
                            ev = evlists[k][idx[k]]
                            tagged[(k, tag)].append((pid, core_id, ev[0], ev[1], ev[2], ev[3]))
                            idx[k] += 1
    
    return tagged, comp_events


def compute_stats(tagged):
    """Compute per-tag, per-event-type statistics."""
    stats = {}
    for (etype, tag), events in tagged.items():
        if tag not in stats:
            stats[tag] = {}
        durs = [e[4] for e in events]
        per_core = defaultdict(float)
        per_core_cnt = defaultdict(int)
        for e in events:
            per_core[e[1]] += e[4]
            per_core_cnt[e[1]] += 1
        
        stats[tag][etype] = {
            'count': len(durs),
            'total': sum(durs),
            'avg': sum(durs) / len(durs) if durs else 0,
            'min': min(durs) if durs else 0,
            'max': max(durs) if durs else 0,
            'per_core': dict(per_core),
            'per_core_cnt': dict(per_core_cnt),
        }
    return stats


def print_results(stats_b, stats_o, comp_b, comp_o, pid_to_core_b, pid_to_core_o):
    tag_names = {50: "TP All-Reduce (tag 50)", 70: "Dispatch All-to-All (tag 70)", 80: "Combine All-to-All (tag 80)"}
    focus_etypes = ['send_data', 'recv_data', 'recv_ack']
    
    print("\n" + "=" * 130)
    print("COMMUNICATION ANALYSIS: BASELINE vs OPTIMIZED")
    print("=" * 130)
    
    # ── Per-tag summary ──
    for tag in [50, 70, 80]:
        name = tag_names[tag]
        print(f"\n{'─'*130}")
        print(f"  {name}")
        print(f"{'─'*130}")
        
        tb = stats_b.get(tag, {})
        to = stats_o.get(tag, {})
        
        print(f"\n  {'Event Type':<18} {'Metric':<12} {'Baseline':>14} {'Optimized':>14} {'Diff':>14} {'B/O Ratio':>10}")
        print(f"  {'─'*18} {'─'*12} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
        
        for etype in focus_etypes:
            sb = tb.get(etype, {})
            so = to.get(etype, {})
            for m in ['count', 'total', 'avg', 'min', 'max']:
                vb = sb.get(m, 0)
                vo = so.get(m, 0)
                d = vo - vb
                r = vb / vo if vo != 0 else float('inf')
                label = etype if m == 'count' else ''
                if m == 'count':
                    print(f"  {label:<18} {m:<12} {vb:>14d} {vo:>14d} {d:>+14d} {r:>10.3f}x")
                else:
                    unit = 'cycles'
                    print(f"  {label:<18} {m + f' ({unit})':<12} {vb:>14.2f} {vo:>14.2f} {d:>+14.2f} {r:>10.3f}x")
            print()
        
        # Per-core recv_ack (this shows waiting time, which is the interesting part)
        print(f"  Per-core recv_ack total (waiting/synchronization time):")
        sb = tb.get('recv_ack', {})
        so = to.get('recv_ack', {})
        cb = sb.get('per_core', {})
        co = so.get('per_core', {})
        all_cores = sorted(set(list(cb.keys()) + list(co.keys())))
        print(f"  {'Core':<8} {'Baseline':>14} {'Optimized':>14} {'Diff':>14} {'B/O':>10}")
        print(f"  {'─'*8} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
        for c in all_cores:
            vb = cb.get(c, 0)
            vo = co.get(c, 0)
            d = vo - vb
            r = vb / vo if vo != 0 else float('inf')
            print(f"  Core {c:<3} {vb:>14.2f} {vo:>14.2f} {d:>+14.2f} {r:>10.3f}x")
    
    # ── Grand summary ──
    print(f"\n{'='*130}")
    print("GRAND SUMMARY")
    print(f"{'='*130}")
    
    print(f"\n  {'Communication Type':<35} {'Baseline (cycles)':>18} {'Optimized (cycles)':>18} {'B/O Ratio':>12}")
    print(f"  {'─'*35} {'─'*18} {'─'*18} {'─'*12}")
    
    grand_b = 0
    grand_o = 0
    for tag in [50, 70, 80]:
        name = tag_names[tag]
        tb = stats_b.get(tag, {})
        to = stats_o.get(tag, {})
        
        # Sum all sub-event types for total comm time per tag
        total_b = sum(s.get('total', 0) for s in tb.values())
        total_o = sum(s.get('total', 0) for s in to.values())
        grand_b += total_b
        grand_o += total_o
        r = total_b / total_o if total_o != 0 else float('inf')
        print(f"  {name:<35} {total_b:>18.2f} {total_o:>18.2f} {r:>12.3f}x")
        
        # Also show key sub-totals
        for etype in ['send_data', 'recv_data', 'recv_ack']:
            vb = tb.get(etype, {}).get('total', 0)
            vo = to.get(etype, {}).get('total', 0)
            r2 = vb / vo if vo != 0 else float('inf')
            print(f"    {etype:<33} {vb:>18.2f} {vo:>18.2f} {r2:>12.3f}x")
    
    r_grand = grand_b / grand_o if grand_o != 0 else float('inf')
    print(f"  {'─'*35} {'─'*18} {'─'*18} {'─'*12}")
    print(f"  {'TOTAL ALL COMM':<35} {grand_b:>18.2f} {grand_o:>18.2f} {r_grand:>12.3f}x")
    
    # ── Computation time comparison ──
    print(f"\n{'='*130}")
    print("PER-CORE COMPUTATION TIME (Comp_prim thread, tid=1)")
    print(f"{'='*130}")
    
    # Sum comp time per core
    comp_core_b = {}
    comp_core_o = {}
    for pid, evts in comp_b.items():
        core_name = pid_to_core_b.get(pid, '')
        try:
            cid = int(core_name.split()[-1])
        except:
            continue
        comp_core_b[cid] = sum(e[2] for e in evts)
    
    for pid, evts in comp_o.items():
        core_name = pid_to_core_o.get(pid, '')
        try:
            cid = int(core_name.split()[-1])
        except:
            continue
        comp_core_o[cid] = sum(e[2] for e in evts)
    
    all_cores = sorted(set(list(comp_core_b.keys()) + list(comp_core_o.keys())))
    print(f"\n  {'Core':<8} {'Baseline':>14} {'Optimized':>14} {'Diff':>14} {'B/O':>10}")
    print(f"  {'─'*8} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
    total_comp_b = 0
    total_comp_o = 0
    for c in all_cores:
        vb = comp_core_b.get(c, 0)
        vo = comp_core_o.get(c, 0)
        total_comp_b += vb
        total_comp_o += vo
        d = vo - vb
        r = vb / vo if vo != 0 else float('inf')
        print(f"  Core {c:<3} {vb:>14.2f} {vo:>14.2f} {d:>+14.2f} {r:>10.3f}x")
    
    r_comp = total_comp_b / total_comp_o if total_comp_o != 0 else float('inf')
    print(f"  {'TOTAL':<8} {total_comp_b:>14.2f} {total_comp_o:>14.2f} {total_comp_o - total_comp_b:>+14.2f} {r_comp:>10.3f}x")
    
    # ── End-to-end latency from logs ──
    print(f"\n{'='*130}")
    print("END-TO-END SIMULATION LATENCY (from log files)")
    print(f"{'='*130}")
    # ftd versions
    lat_b_ftd = 1656104507424  # ps
    lat_o_ftd = 1657254887680  # ps
    lat_b_noftd = 1957853062312  # ps
    lat_o_noftd = 1958547581384  # ps
    
    print(f"\n  {'Config':<25} {'Baseline (us)':>18} {'Optimized (us)':>18} {'B/O Ratio':>12}")
    print(f"  {'─'*25} {'─'*18} {'─'*18} {'─'*12}")
    r1 = lat_b_ftd / lat_o_ftd
    r2 = lat_b_noftd / lat_o_noftd
    print(f"  {'FTD traces':<25} {lat_b_ftd/1e6:>18.2f} {lat_o_ftd/1e6:>18.2f} {r1:>12.4f}x")
    print(f"  {'Non-FTD traces':<25} {lat_b_noftd/1e6:>18.2f} {lat_o_noftd/1e6:>18.2f} {r2:>12.4f}x")
    
    print(f"\n  NOTE: B/O Ratio > 1 means baseline is faster, < 1 means optimized is faster.")
    print(f"  Both configurations show nearly identical end-to-end latency (~0.07% difference).")
    print(f"  The mapping optimization (MOE-baseline.spec vs MOE-opt.spec) does not change")
    print(f"  the total data transfer volume, only the core-to-core routing topology.")


def main():
    config_path = '/home/code/npu-sim/entewine/30B-A3B-ftd.json'
    baseline_path = '/home/code/npu-sim/entewine/events-baseline-ftd.json'
    opt_path = '/home/code/npu-sim/entewine/events-opt-ftd.json'
    
    print("=" * 80)
    print("Communication Event Analysis v2: Baseline vs Optimized")
    print("=" * 80)
    
    print("\n[1] Parsing config...")
    core_sequences = parse_config_comm_sequence(config_path)
    print(f"  {len(core_sequences)} cores")
    
    print("\n[2] Parsing baseline trace...")
    evts_b, ptc_b = parse_trace_all_events(baseline_path)
    
    print("\n[3] Parsing optimized trace...")
    evts_o, ptc_o = parse_trace_all_events(opt_path)
    
    print("\n[4] Assigning tags to baseline...")
    tagged_b, comp_b = assign_tags(evts_b, core_sequences, ptc_b)
    
    print("\n[5] Assigning tags to optimized...")
    tagged_o, comp_o = assign_tags(evts_o, core_sequences, ptc_o)
    
    print("\n[6] Computing statistics...")
    stats_b = compute_stats(tagged_b)
    stats_o = compute_stats(tagged_o)
    
    print_results(stats_b, stats_o, comp_b, comp_o, ptc_b, ptc_o)


if __name__ == '__main__':
    main()
