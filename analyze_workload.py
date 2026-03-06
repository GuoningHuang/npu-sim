#!/usr/bin/env python3
import json
import sys
from collections import defaultdict, deque

def analyze_workload_communication(workload_file):
    """分析workload中的通信模式，检查死锁风险"""
    with open(workload_file, 'r') as f:
        data = json.load(f)

    print(f"分析 workload: {workload_file}")
    print(f"参数: loop={data['vars']['loop']}, L={data['vars']['L']}, T={data['vars']['T']}")

    # 统计每个chip的core数量
    for chip in data['chips']:
        chip_id = chip['chip_id']
        core_count = len(chip['cores'])
        print(f"Chip {chip_id}: {core_count} cores")

        # 分析每个core的通信模式
        for core in chip['cores']:
            core_id = core['id']
            print(f"\nCore {core_id}:")

            # 统计通信操作
            send_count = 0
            recv_count = 0
            cast_count = 0

            for work_item in core['worklist']:
                # 检查cast操作
                if 'cast' in work_item and work_item['cast']:
                    cast_count += len(work_item['cast'])
                    print(f"  Cast operations: {len(work_item['cast'])}")

                # 检查recv_cnt
                if 'recv_cnt' in work_item:
                    recv_count += work_item['recv_cnt']

                # 检查prims中的通信操作
                for prim in work_item.get('prims', []):
                    if prim['type'] in ['send_data', 'recv_data', 'SEND_DATA', 'RECV_DATA']:
                        print(f"  Communication prim: {prim['type']}")

            print(f"  Total recv_cnt: {recv_count}")
            print(f"  Total cast operations: {cast_count}")

    # 检查是否有循环依赖
    print("\n检查通信依赖...")

    # 构建通信图
    comm_graph = defaultdict(list)

    for chip in data['chips']:
        for core in chip['cores']:
            core_id = core['id']
            for work_item in core['worklist']:
                if 'cast' in work_item:
                    for cast_op in work_item['cast']:
                        dest = cast_op['dest']
                        comm_graph[core_id].append(dest)

    # 检查是否有循环
    def has_cycle(graph):
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    if has_cycle(comm_graph):
        print("⚠️  发现通信循环依赖，可能存在死锁风险！")
    else:
        print("✅ 没有发现通信循环依赖")

    print(f"\n通信图: {dict(comm_graph)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_workload.py <workload.json>")
        sys.exit(1)

    analyze_workload_communication(sys.argv[1])