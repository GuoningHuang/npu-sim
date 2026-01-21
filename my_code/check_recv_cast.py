import json
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print("usage: check_recv_cast.py <workload.json>")
    sys.exit(1)

path = sys.argv[1]
with open(path) as f:
    j = json.load(f)

# Collect all casts as (dest, tag) -> list of (src_core, work_idx, location)
cast_map = defaultdict(list)
source_dests = {s.get("dest") for s in j.get("source", []) if "dest" in s}
empty_cores = []
empty_work_items = []
first_sender = {}
first_recv = {}
last_sender = {}
last_recv = {}
cast_count = defaultdict(int)
recv_count = defaultdict(int)

def add_cast(src_core, work_idx, cast, location):
    if not isinstance(cast, dict):
        return
    dest = cast.get("dest")
    tag = cast.get("tag")
    if dest is None or tag is None:
        return
    cast_map[(dest, tag)].append((src_core, work_idx, location))
    cast_count[(dest, tag)] += 1
    key = (dest, tag)
    if key not in first_sender or work_idx < first_sender[key][1]:
        first_sender[key] = (src_core, work_idx, location)
    if key not in last_sender or work_idx > last_sender[key][1]:
        last_sender[key] = (src_core, work_idx, location)

for chip in j.get("chips", []):
    for core in chip.get("cores", []):
        cid = core.get("id")
        for wi, w in enumerate(core.get("worklist", [])):
            for cast in w.get("cast", []) if isinstance(w.get("cast", []), list) else []:
                add_cast(cid, wi, cast, "worklist")
            for pi, p in enumerate(w.get("prims", [])):
                for cast in p.get("cast", []) if isinstance(p.get("cast", []), list) else []:
                    add_cast(cid, wi, cast, f"prim#{pi}")
            recv_cnt = w.get("recv_cnt", 0)
            recv_tag = w.get("recv_tag")
            if recv_cnt and recv_tag is not None:
                key = (cid, recv_tag)
                recv_count[key] += recv_cnt
                if key not in first_recv or wi < first_recv[key][0]:
                    first_recv[key] = (wi, recv_cnt)
                if key not in last_recv or wi > last_recv[key][0]:
                    last_recv[key] = (wi, recv_cnt)

missing = []
missing_tag = []
host_recv_issues = []
for chip in j.get("chips", []):
    for core in chip.get("cores", []):
        cid = core.get("id")
        worklist = core.get("worklist", [])
        if not worklist:
            empty_cores.append(cid)
            continue
        for wi, w in enumerate(core.get("worklist", [])):
            if not w.get("prims"):
                empty_work_items.append((cid, wi))
            recv_cnt = w.get("recv_cnt", 0)
            recv_tag = w.get("recv_tag")
            if recv_cnt and recv_tag is None:
                missing_tag.append((cid, wi, recv_cnt))
                continue
            if recv_cnt:
                if recv_tag == cid:
                    if cid not in source_dests:
                        host_recv_issues.append((cid, wi, recv_tag, "non-source core uses host tag"))
                    elif wi != 0:
                        host_recv_issues.append((cid, wi, recv_tag, "host tag used outside first work"))
                key = (cid, recv_tag)
                if key not in cast_map:
                    if cid in source_dests and recv_tag == cid and wi == 0:
                        continue
                    missing.append((cid, wi, recv_cnt, recv_tag))

print("recv_cnt without recv_tag:", len(missing_tag))
for cid, wi, recv_cnt in missing_tag[:50]:
    print(f"  core {cid} work {wi} recv_cnt {recv_cnt} recv_tag None")
if len(missing_tag) > 50:
    print("  ...")

print("recv_cnt with no matching cast:", len(missing))
for cid, wi, recv_cnt, recv_tag in missing[:50]:
    print(f"  core {cid} work {wi} recv_cnt {recv_cnt} recv_tag {recv_tag}")
if len(missing) > 50:
    print("  ...")

print("empty cores:", len(empty_cores))
for cid in empty_cores[:50]:
    print(f"  core {cid}")
if len(empty_cores) > 50:
    print("  ...")

print("empty worklist items:", len(empty_work_items))
for cid, wi in empty_work_items[:50]:
    print(f"  core {cid} work {wi}")
if len(empty_work_items) > 50:
    print("  ...")

print("host recv tag issues:", len(host_recv_issues))
for cid, wi, recv_tag, reason in host_recv_issues[:50]:
    print(f"  core {cid} work {wi} recv_tag {recv_tag} ({reason})")
if len(host_recv_issues) > 50:
    print("  ...")

# Potential startup stalls: recv at work 0 but earliest send is in a later work item.
startup_stalls = []
for (cid, recv_tag), (wi, recv_cnt) in first_recv.items():
    if wi != 0:
        continue
    if cid in source_dests and recv_tag == cid:
        continue
    sender = first_sender.get((cid, recv_tag))
    if sender is None:
        continue
    src_core, src_wi, src_loc = sender
    if src_wi > wi:
        startup_stalls.append((cid, recv_tag, recv_cnt, src_core, src_wi, src_loc))

print("startup recv waits on later send:", len(startup_stalls))
for cid, recv_tag, recv_cnt, src_core, src_wi, src_loc in startup_stalls[:50]:
    print(
        f"  core {cid} work 0 recv_tag {recv_tag} recv_cnt {recv_cnt} "
        f"first send by core {src_core} work {src_wi} ({src_loc})"
    )
if len(startup_stalls) > 50:
    print("  ...")

# Trace dependency chains from work 0 recv to detect cycles not rooted at source.
first_work_recv = {}
for chip in j.get("chips", []):
    for core in chip.get("cores", []):
        cid = core.get("id")
        worklist = core.get("worklist", [])
        if not worklist:
            continue
        w0 = worklist[0]
        recv_cnt = w0.get("recv_cnt", 0)
        recv_tag = w0.get("recv_tag")
        if recv_cnt and recv_tag is not None:
            first_work_recv[cid] = recv_tag

deadlock_cycles = []
unresolved_chains = []
for cid, recv_tag in sorted(first_work_recv.items()):
    if cid in source_dests and recv_tag == cid:
        continue
    path = []
    seen = set()
    cur_core = cid
    cur_tag = recv_tag
    while True:
        state = (cur_core, cur_tag)
        if state in seen:
            path.append(state)
            deadlock_cycles.append(path)
            break
        seen.add(state)
        path.append(state)
        if cur_core in source_dests and cur_tag == cur_core:
            break
        sender = first_sender.get((cur_core, cur_tag))
        if sender is None:
            unresolved_chains.append(path)
            break
        next_core = sender[0]
        next_tag = first_work_recv.get(next_core)
        if next_tag is None:
            unresolved_chains.append(path)
            break
        cur_core, cur_tag = next_core, next_tag

print("startup dependency cycles:", len(deadlock_cycles))
for path in deadlock_cycles[:10]:
    preview = " -> ".join(f"{c}:{t}" for c, t in path[:10])
    print(f"  {preview}")
if len(deadlock_cycles) > 10:
    print("  ...")

print("startup dependency unresolved:", len(unresolved_chains))
for path in unresolved_chains[:10]:
    preview = " -> ".join(f"{c}:{t}" for c, t in path[:10])
    print(f"  {preview}")
if len(unresolved_chains) > 10:
    print("  ...")

# Tail recv sanity: check if the last recv for a tag happens after the last send.
tail_missing = []
for key, (r_wi, r_cnt) in last_recv.items():
    s = last_sender.get(key)
    if s is None:
        continue
    if r_wi > s[1]:
        tail_missing.append((key, r_wi, r_cnt, s[0], s[1], s[2]))

print("tail recv after last send:", len(tail_missing))
for (dest, tag), r_wi, r_cnt, s_cid, s_wi, s_loc in tail_missing[:50]:
    print(
        f"  core {dest} recv_tag {tag} last recv work {r_wi} "
        f"(recv_cnt {r_cnt}) last send by core {s_cid} work {s_wi} ({s_loc})"
    )
if len(tail_missing) > 50:
    print("  ...")

# Count mismatch sanity: total sends vs recvs per tag.
count_mismatch = []
for key, c in cast_count.items():
    r = recv_count.get(key, 0)
    if c != r:
        count_mismatch.append((key, c, r))
for key, r in recv_count.items():
    if key not in cast_count:
        count_mismatch.append((key, 0, r))

print("send/recv count mismatch:", len(count_mismatch))
for (dest, tag), c, r in count_mismatch[:50]:
    print(f"  core {dest} tag {tag} sends {c} recvs {r}")
if len(count_mismatch) > 50:
    print("  ...")

# Pipeline tag sanity: check recv_tag 3000+cid has a matching cast.
pipeline_missing = []
for chip in j.get("chips", []):
    for core in chip.get("cores", []):
        cid = core.get("id")
        for wi, w in enumerate(core.get("worklist", [])):
            recv_cnt = w.get("recv_cnt", 0)
            recv_tag = w.get("recv_tag")
            if recv_cnt and recv_tag == 3000 + cid:
                if (cid, recv_tag) not in cast_map:
                    pipeline_missing.append((cid, wi, recv_tag))

print("pipeline recv missing cast:", len(pipeline_missing))
for cid, wi, recv_tag in pipeline_missing[:50]:
    print(f"  core {cid} work {wi} recv_tag {recv_tag}")
if len(pipeline_missing) > 50:
    print("  ...")

# Optional: report casts that have no receiver
# Build receiver set
recv_set = set()
for chip in j.get("chips", []):
    for core in chip.get("cores", []):
        cid = core.get("id")
        for w in core.get("worklist", []):
            recv_cnt = w.get("recv_cnt", 0)
            recv_tag = w.get("recv_tag")
            if recv_cnt and recv_tag is not None:
                recv_set.add((cid, recv_tag))

orphan = []
for key, srcs in cast_map.items():
    if key not in recv_set:
        orphan.append((key, srcs))

print("casts with no matching recv:", len(orphan))
for (dest, tag), srcs in orphan[:50]:
    src_preview = ", ".join(f"{cid}:{wi}" for cid, wi, _ in srcs[:3])
    print(f"  dest {dest} tag {tag} from {src_preview}")
if len(orphan) > 50:
    print("  ...")

# ============================================================
# New checks for deadlock diagnosis
# ============================================================

print("\n" + "=" * 60)
print("DEADLOCK DIAGNOSIS CHECKS")
print("=" * 60)

# 1. Count loopout (dest: -1) distribution
loopout_cores = defaultdict(list)  # core_id -> list of work_idx
for chip in j.get("chips", []):
    for core in chip.get("cores", []):
        cid = core.get("id")
        for wi, w in enumerate(core.get("worklist", [])):
            for cast in w.get("cast", []) if isinstance(w.get("cast", []), list) else []:
                if cast.get("dest") == -1:
                    loopout_cores[cid].append(wi)
                    break  # Only count once per work item

end_cores_count = sum(len(v) for v in loopout_cores.values())
print(f"\n[Loopout (dest=-1) Distribution]")
print(f"  Total work items with loopout: {end_cores_count}")
print(f"  Cores with loopout: {len(loopout_cores)}")
if loopout_cores:
    for cid in sorted(loopout_cores.keys())[:20]:
        print(f"    Core {cid}: work items {loopout_cores[cid]}")
    if len(loopout_cores) > 20:
        print(f"    ... and {len(loopout_cores) - 20} more cores")

# 2. Calculate expected DONE count
pipeline_val = j.get("pipeline", 1)
end_count_sources = 0
for src in j.get("source", []):
    if src.get("is_end", False):
        end_count_sources += src.get("loop", 1)

expected_done = end_cores_count * pipeline_val * max(1, end_count_sources)
print(f"\n[Expected DONE Count]")
print(f"  end_cores (work items with dest=-1): {end_cores_count}")
print(f"  pipeline: {pipeline_val}")
print(f"  end_count_sources: {end_count_sources}")
print(f"  Expected DONE = {end_cores_count} * {pipeline_val} * max(1, {end_count_sources}) = {expected_done}")

# 3. Check vars for zero values (can cause FPE)
vars_dict = j.get("vars", {})
zero_vars = [(k, v) for k, v in vars_dict.items() if v == 0]
print(f"\n[Zero Value Variables (can cause FPE)]")
if zero_vars:
    print(f"  WARNING: {len(zero_vars)} variables have value 0:")
    for k, v in zero_vars[:30]:
        print(f"    {k} = {v}")
    if len(zero_vars) > 30:
        print(f"    ... and {len(zero_vars) - 30} more")
else:
    print("  No zero-value variables found")

# 4. Check pipeline stage distribution
config_vars = j.get("vars", {})
pp = config_vars.get("pp", 1)
mn = config_vars.get("mn", 1)
k = config_vars.get("k", 1)
cores_per_stage = mn * k
total_cores = len(j.get("chips", [{}])[0].get("cores", []))

print(f"\n[Parallelism Config]")
print(f"  pp (pipeline parallelism): {pp}")
print(f"  mn: {mn}, k: {k}")
print(f"  cores per stage: {cores_per_stage}")
print(f"  total cores: {total_cores}")

# Check which stages have loopout
stages_with_loopout = set()
for cid in loopout_cores.keys():
    stage = cid // cores_per_stage if cores_per_stage > 0 else 0
    stages_with_loopout.add(stage)

print(f"\n[Pipeline Stages with Loopout]")
print(f"  Total stages: {pp}")
print(f"  Stages with loopout: {sorted(stages_with_loopout)}")
if len(stages_with_loopout) == 1 and pp - 1 in stages_with_loopout:
    print(f"  WARNING: Only the last stage (stage {pp-1}) has loopout!")
    print(f"           This may cause deadlock if simulation expects DONE from other stages.")

# 5. Check if T/mn could be zero in decode phase
t_vars = {k: v for k, v in vars_dict.items() if k.startswith("T") and "/" in k}
problematic_t = [(k, v) for k, v in t_vars.items() if v == 0]
if problematic_t:
    print(f"\n[Sequence Length Division Issues]")
    print(f"  WARNING: The following T/mn variables are 0 (will cause FPE in simulation):")
    for k, v in problematic_t:
        print(f"    {k} = {v}")
    print(f"  This typically happens when decode sequence length (T=1) is divided by mn>1")

# 6. Summary
print(f"\n[Summary]")
issues = []
if zero_vars:
    issues.append(f"Zero-value variables: {len(zero_vars)}")
if len(stages_with_loopout) < pp:
    issues.append(f"Missing loopout in {pp - len(stages_with_loopout)} stages")
if problematic_t:
    issues.append(f"T/mn=0 issues: {len(problematic_t)}")

if issues:
    print("  POTENTIAL ISSUES FOUND:")
    for issue in issues:
        print(f"    - {issue}")
else:
    print("  No obvious issues detected")

# 7. Check loopback flow (tag 4000+cid)
print(f"\n[Loopback Flow Analysis (tag 4000+cid)]")
loopback_sends = defaultdict(list)  # (dest, tag) -> [(src_core, work_idx)]
loopback_recvs = defaultdict(list)  # (core_id, tag) -> [work_idx]

for chip in j.get("chips", []):
    for core in chip.get("cores", []):
        cid = core.get("id")
        for wi, w in enumerate(core.get("worklist", [])):
            # Check for loopback recv (tag 4000+cid)
            recv_tag = w.get("recv_tag")
            recv_cnt = w.get("recv_cnt", 0)
            if recv_cnt and recv_tag is not None and recv_tag >= 4000:
                loopback_recvs[(cid, recv_tag)].append(wi)

            # Check for loopback send
            for cast in w.get("cast", []) if isinstance(w.get("cast", []), list) else []:
                dest = cast.get("dest")
                tag = cast.get("tag")
                if dest is not None and tag is not None and tag >= 4000 and dest != -1:
                    loopback_sends[(dest, tag)].append((cid, wi))

print(f"  Loopback sends: {len(loopback_sends)} unique (dest, tag) pairs")
print(f"  Loopback recvs: {len(loopback_recvs)} unique (core, tag) pairs")

# Check matching
loopback_issues = []
for (cid, tag), recv_works in loopback_recvs.items():
    send_list = loopback_sends.get((cid, tag), [])
    if len(send_list) != len(recv_works):
        loopback_issues.append((cid, tag, len(recv_works), len(send_list)))

if loopback_issues:
    print(f"  WARNING: Loopback send/recv count mismatch:")
    for cid, tag, recv_cnt, send_cnt in loopback_issues[:20]:
        print(f"    Core {cid} tag {tag}: {recv_cnt} recvs, {send_cnt} sends")
else:
    print(f"  Loopback send/recv counts match")

# Show loopback details for first stage cores
first_stage_cores = list(range(cores_per_stage)) if cores_per_stage > 0 else [0]
print(f"\n  Loopback details for first stage (cores {first_stage_cores}):")
for cid in first_stage_cores:
    tag = 4000 + cid
    recv_works = loopback_recvs.get((cid, tag), [])
    send_list = loopback_sends.get((cid, tag), [])
    if recv_works or send_list:
        send_info = ", ".join(f"core{s[0]}:work{s[1]}" for s in send_list[:5])
        recv_info = ", ".join(f"work{w}" for w in recv_works[:5])
        print(f"    Core {cid} tag {tag}:")
        print(f"      Sends: {send_info if send_info else 'none'}")
        print(f"      Recvs: {recv_info if recv_info else 'none'}")
