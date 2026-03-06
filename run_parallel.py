import os
import subprocess
import shutil
import argparse
import signal
import sys
import atexit
import time
from concurrent.futures import ProcessPoolExecutor

# --- 配置区 ---
SH_FILE = "run.sh"
TOTAL_CORES = 16
TEMP_BASE_PREFIX = "temp"

# 全局变量，记录创建的文件夹
created_folders = []

def get_combinations(target):
    """生成满足 (EP + mn * k) * DP * PP = 16 的组合"""
    configs = []
    factors = [i for i in range(1, target + 1) if target % i == 0]
    for dp in factors:
        for pp in [p for p in factors if p > 0]:
            if target % (dp * pp) != 0: continue
            sum_total = target // (dp * pp)
            for ep in range(1, sum_total):
                tp_val = sum_total - ep
                for mn in range(1, tp_val + 1):
                    if tp_val % mn == 0:
                        k = tp_val // mn
                        configs.append({
                            "ep": ep, "tp_str": f"{mn}_{k}",
                            "dp": dp, "pp": pp,
                            "folder": f"{TEMP_BASE_PREFIX}DP{dp}_PP{pp}_EP{ep}_TP{mn}_{k}"
                        })
    return configs

def cleanup():
    """退出时的自动清理逻辑"""
    if not created_folders: return
    print("\n" + "-"*30)
    print("🧹 正在执行自动清理...")
    curr_dir = os.getcwd()
    for folder in created_folders:
        path = os.path.join(curr_dir, folder)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except: pass
    print("✨ 临时文件夹清理完毕。")

def signal_handler(sig, frame):
    """捕获 Ctrl+C"""
    print("\n\n⚠️  检测到中断！正在强制停止所有任务并清理环境...")
    sys.exit(0)

def execute_job(config):
    """执行任务并统计耗时"""
    curr_dir = os.getcwd()
    run_dir = os.path.join(curr_dir, config['folder'])
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 1. 准备目录和脚本
        if os.path.exists(run_dir): shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        shutil.copy2(os.path.join(curr_dir, SH_FILE), os.path.join(run_dir, SH_FILE))
        
        # 2. 运行脚本
        print(f"🚀 [开始] {config['folder']}")
        result = subprocess.run(
            ["bash", SH_FILE, str(config['dp']), str(config['pp']), str(config['ep']), config['tp_str']],
            cwd=run_dir,
            capture_output=True,
            text=True
        )
        
        # 3. 计算耗时
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            status = "✅ 成功"
        else:
            status = f"❌ 失败 (Code: {result.returncode})"
            
        # 打印单项任务完成结果和时间
        print(f"{status}: {config['folder']} | 耗时: {elapsed_time:.2f}s")
        return (True, config['folder'], elapsed_time)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"💥 异常: {config['folder']} | 耗时: {elapsed_time:.2f}s | 错误: {str(e)}")
        return (False, config['folder'], elapsed_time)

def main():
    # 注册清理和信号
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="NPU 并行仿真自动化工具")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="并行进程数")
    parser.add_argument("--keep", action="store_true", help="任务结束后保留 temp 文件夹")
    args = parser.parse_args()

    if not os.path.exists(SH_FILE):
        print(f"错误: 找不到 {SH_FILE}")
        return

    configs = get_combinations(TOTAL_CORES)
    global created_folders
    created_folders = [c['folder'] for c in configs]

    print(f"🔍 扫描到 {len(configs)} 种配置情况")
    print(f"⚙️  并行进程数: {args.jobs}")
    print("-" * 40)

    total_start_time = time.time()

    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(execute_job, configs))
    except KeyboardInterrupt:
        pass

    total_elapsed = time.time() - total_start_time

    # 统计汇总
    print("\n" + "="*40)
    print("📋 所有任务执行报告")
    print("-" * 40)
    for success, folder, duration in results:
        mark = "✔" if success else "✘"
        print(f"[{mark}] {folder:40} | 耗时: {duration:8.2f}s")
    print("-" * 40)
    print(f"总计耗时: {total_elapsed:.2f}s")
    print("=" * 40)

    if args.keep:
        atexit.unregister(cleanup)
        print("💾 临时文件夹已保留。")

if __name__ == "__main__":
    main()