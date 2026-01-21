import json
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print("usage: check_workload_fatal.py <workload.json>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    j = json.load(f)

chips = j.get("chips", [])
sources = j.get("source", [])
vars_dict = j.get("vars", {})

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
source_dests = {s.get("dest") for s in sources if "dest" in s}

def iter_cores():
    for chip in chips:
        for core in chip.get("cores", []):
            yield core

def iter_works():
    for core in iter_cores():
        cid = core.get("id")
        for wi, w in enumerate(core.get("worklist", [])):
            yield cid, wi, w

# ------------------------------------------------------------
# Collect casts and recvs
# ------------------------------------------------------------
first_sender = {}   # (dest, tag) -> (src_core, work_idx)
cast_map = defaultdict(list)
cast_count = defaultdict(int)
recv_count = defaultdict(int)

first_recv = {}    # (core, tag) -> work_idx

def iter_casts(w):
    for cast in w.get("cast", []) if isinstance(w.get("cast", []), list) else []:
        if isinstance(cast, dict):
            yield cast
    for p in w.get("prims", []) if isinstance(w.get("prims", []), list) else []:
        for cast in p.get("cast", []) if isinstance(p.get("cast", []), list) else []:
            if isinstance(cast, dict):
                yield cast

for cid, wi, w in iter_works():
    for cast in iter_casts(w):
        dest = cast.get("dest")
        tag = cast.get("tag")
        if dest is None or tag is None:
            continue
        cast_map[(dest, tag)].append((cid, wi))
        cast_count[(dest, tag)] += 1
        if (dest, tag) not in first_sender or wi < first_sender[(dest, tag)][1]:
            first_sender[(dest, tag)] = (cid, wi)

    recv_cnt = w.get("recv_cnt", 0)
    recv_tag = w.get("recv_tag")
    if recv_cnt and recv_tag is not None:
        key = (cid, recv_tag)
        recv_count[key] += recv_cnt
        if key not in first_recv or wi < first_recv[key]:
            first_recv[key] = wi

# ------------------------------------------------------------
# FATAL 1: recv_cnt without recv_tag
# ------------------------------------------------------------
fatal_recv_no_tag = []
for cid, wi, w in iter_works():
    if w.get("recv_cnt", 0) and w.get("recv_tag") is None:
        fatal_recv_no_tag.append((cid, wi, w.get("recv_cnt")))

# ------------------------------------------------------------
# FATAL 2: startup recv waits on later send
# ------------------------------------------------------------
startup_stall = []
for (cid, tag), wi in first_recv.items():
    if wi != 0:
        continue
    # source 注入路径例外
    if cid in source_dests and tag == cid:
        continue
    sender = first_sender.get((cid, tag))
    if sender is None:
        continue
    src_core, src_wi = sender
    if src_wi > 0:
        startup_stall.append((cid, tag, src_core, src_wi))

# ------------------------------------------------------------
# FATAL 2b: recv waits on later send (any work item)
# ------------------------------------------------------------
late_send_stall = []
for cid, wi, w in iter_works():
    recv_cnt = w.get("recv_cnt", 0)
    recv_tag = w.get("recv_tag")
    if not recv_cnt or recv_tag is None:
        continue
    if cid in source_dests and recv_tag == cid and wi == 0:
        continue
    sender = first_sender.get((cid, recv_tag))
    if sender is None:
        continue
    src_core, src_wi = sender
    if src_wi > wi:
        late_send_stall.append((cid, wi, recv_tag, src_core, src_wi))

# ------------------------------------------------------------
# FATAL 3: startup dependency cycle (non-source)
# ------------------------------------------------------------
# build first-work recv map
first_work_recv = {}
for core in iter_cores():
    cid = core.get("id")
    worklist = core.get("worklist", [])
    if not worklist:
        continue
    w0 = worklist[0]
    if w0.get("recv_cnt", 0) and w0.get("recv_tag") is not None:
        first_work_recv[cid] = w0.get("recv_tag")

deadlock_cycles = []

for start_core, start_tag in first_work_recv.items():
    if start_core in source_dests and start_tag == start_core:
        continue

    seen = set()
    path = []

    cur_core, cur_tag = start_core, start_tag
    while True:
        state = (cur_core, cur_tag)
        if state in seen:
            path.append(state)
            deadlock_cycles.append(path)
            break
        seen.add(state)
        path.append(state)

        sender = first_sender.get((cur_core, cur_tag))
        if sender is None:
            break

        next_core = sender[0]
        next_tag = first_work_recv.get(next_core)
        if next_tag is None:
            break

        # source 作为根，合法
        if next_core in source_dests and next_tag == next_core:
            break

        cur_core, cur_tag = next_core, next_tag

# ------------------------------------------------------------
# FATAL 4: T/mn == 0  (FPE)
# ------------------------------------------------------------
t_div_zero = [(k, v) for k, v in vars_dict.items()
              if k.startswith("T") and "/" in k and v == 0]

# ------------------------------------------------------------
# FATAL 5: pipeline stages missing loopout
# ------------------------------------------------------------
pipeline = j.get("pipeline", 1)
mn = vars_dict.get("mn", 1)
k = vars_dict.get("k", 1)
cores_per_stage = mn * k

loopout_stages = set()
for cid, wi, w in iter_works():
    for cast in w.get("cast", []) if isinstance(w.get("cast", []), list) else []:
        if cast.get("dest") == -1:
            stage = cid // cores_per_stage if cores_per_stage else 0
            loopout_stages.add(stage)

missing_stages = []
if pipeline > 1:
    missing_stages = [s for s in range(pipeline) if s not in loopout_stages]

# ------------------------------------------------------------
# FATAL 6: recv with no matching cast / send-recv mismatch
# ------------------------------------------------------------
missing_cast = []
for cid, wi, w in iter_works():
    recv_cnt = w.get("recv_cnt", 0)
    recv_tag = w.get("recv_tag")
    if not recv_cnt or recv_tag is None:
        continue
    if cid in source_dests and recv_tag == cid and wi == 0:
        continue
    if (cid, recv_tag) not in cast_map:
        missing_cast.append((cid, wi, recv_cnt, recv_tag))

count_mismatch = []
for key, c in cast_count.items():
    r = recv_count.get(key, 0)
    if c != r:
        count_mismatch.append((key, c, r))
for key, r in recv_count.items():
    if key not in cast_count:
        count_mismatch.append((key, 0, r))

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FATAL WORKLOAD LOGIC ERRORS")
print("=" * 60)

def report(title, items, fmt):
    print(f"\n[{title}] ({len(items)})")
    for x in items[:20]:
        print("  " + fmt(x))
    if len(items) > 20:
        print("  ...")

if fatal_recv_no_tag:
    report(
        "recv_cnt without recv_tag",
        fatal_recv_no_tag,
        lambda x: f"core {x[0]} work {x[1]} recv_cnt {x[2]}"
    )

if startup_stall:
    report(
        "startup recv waits on later send",
        startup_stall,
        lambda x: f"core {x[0]} recv_tag {x[1]} first send by core {x[2]} work {x[3]}"
    )

if late_send_stall:
    report(
        "recv waits on later send (any work)",
        late_send_stall,
        lambda x: f"core {x[0]} work {x[1]} recv_tag {x[2]} first send by core {x[3]} work {x[4]}"
    )

if deadlock_cycles:
    report(
        "startup dependency cycles (non-source)",
        deadlock_cycles,
        lambda p: " -> ".join(f"{c}:{t}" for c, t in p)
    )

if t_div_zero:
    report(
        "T/mn == 0 (FPE)",
        t_div_zero,
        lambda x: f"{x[0]} = {x[1]}"
    )

if missing_stages:
    print(f"\n[Pipeline stages missing loopout] ({len(missing_stages)})")
    print(f"  Missing stages: {missing_stages}")

if missing_cast:
    report(
        "recv without matching cast",
        missing_cast,
        lambda x: f"core {x[0]} work {x[1]} recv_cnt {x[2]} recv_tag {x[3]}"
    )

if count_mismatch:
    report(
        "send/recv count mismatch",
        count_mismatch,
        lambda x: f"core {x[0][0]} tag {x[0][1]} sends {x[1]} recvs {x[2]}"
    )

if not any([fatal_recv_no_tag, startup_stall, late_send_stall, deadlock_cycles, t_div_zero, missing_stages, missing_cast, count_mismatch]):
    print("\nNo fatal workload logic errors detected.")
