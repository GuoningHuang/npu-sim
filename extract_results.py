import os
import re

# --- 配置区 ---
LOG_BASE_DIR = "./run_logs2" 

def clean_ansi(text):
    """清理 ANSI 颜色转义代码"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_log_file(file_path):
    """解析日志，提取 Latency 或识别错误"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines: return 2, "Empty"
        
        tail_text = clean_ansi("\n".join(lines[-10:]))
        if "Error:" in tail_text or "sc_start" in tail_text:
            return 1, "Sim Error"

        for line in reversed(lines):
            clean_line = clean_ansi(line)
            match = re.search(r'\|\s*(\d+)\s*ps', clean_line)
            if match:
                ps_val = int(match.group(1))
                if ps_val > 0: return 0, ps_val / 1_000_000_000.0
        
        if "| 0 s" in tail_text: return 1, "Failed(0s)"
        return 2, "Incomplete"
    except: return 2, "Read Err"

def extract_params(folder_name):
    """从文件夹名中提取 DP, PP, EP, TP 数值"""
    dp = re.search(r'_DP(\d+)', folder_name)
    pp = re.search(r'_PP(\d+)', folder_name)
    ep = re.search(r'_EP(\d+)', folder_name)
    tp = re.search(r'_TP([\d_]+)', folder_name)

    return (
        dp.group(1) if dp else "-",
        pp.group(1) if pp else "-",
        ep.group(1) if ep else "-",
        tp.group(1) if tp else "-"
    )

def main():
    if not os.path.exists(LOG_BASE_DIR):
        print(f"❌ 找不到目录: {LOG_BASE_DIR}")
        return

    all_data = []
    folders = sorted([f for f in os.listdir(LOG_BASE_DIR) if os.path.isdir(os.path.join(LOG_BASE_DIR, f))])

    for folder in folders:
        folder_path = os.path.join(LOG_BASE_DIR, folder)
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if not txt_files: continue

        dp, pp, ep, tp = extract_params(folder)

        results = []
        for f in txt_files:
            code, val = parse_log_file(os.path.join(folder_path, f))
            results.append((code, val))
        
        results.sort(key=lambda x: x[0])
        final_val = results[0][1]

        all_data.append({
            'dp': dp, 'pp': pp, 'ep': ep, 'tp': tp,
            'latency': final_val,
            'folder': folder
        })

    # 按 Latency 排序 (数值优先，错误信息排在后面)
    all_data.sort(key=lambda x: (not isinstance(x['latency'], float), x['latency']))

    # --- 打印表格 ---
    header = f"{'ID':>3} | {'DP':>4} | {'PP':>4} | {'EP':>4} | {'TP':>8} | {'Latency (ms)':>15} | {'Folder Name'}"
    print("\n" + header)
    print("-" * (len(header) + 15))

    error_count = 0
    success_count = 0

    for idx, d in enumerate(all_data, 1):
        l_val = d['latency']
        if isinstance(l_val, float):
            latency_str = f"{l_val:>12.4f} ms"
            success_count += 1
        else:
            latency_str = f"{l_val:>15}"
            error_count += 1
            
        print(f"{idx:>3} | {d['dp']:>4} | {d['pp']:>4} | {d['ep']:>4} | {d['tp']:>8} | {latency_str} | {d['folder']}")

    # --- 总结报告 ---
    print("-" * (len(header) + 15))
    print(f"📊 统计总结:")
    print(f"  - 总测试项 (Total):   {len(all_data)}")
    print(f"  - 成功完成 (Success): {success_count}")
    print(f"  - 异常错误 (Errors):  {error_count}")
    
    if len(all_data) > 0:
        error_rate = (error_count / len(all_data)) * 100
        print(f"  - 错误率 (Error Rate): {error_rate:.2f}%")
    print("-" * (len(header) + 15) + "\n")

if __name__ == "__main__":
    main()