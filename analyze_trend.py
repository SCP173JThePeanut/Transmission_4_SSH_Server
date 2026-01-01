import torch
import os
import numpy as np

# === 配置 ===
layer_idx = 31       # 重点观察深层，因为语义最丰富
base_step = 20       # 基准 Step (SparseD 生成 Mask 的时间点)
max_step = 128
interval = 4         # 采样间隔 (需要与 modeling 中 PROBE_STEPS 一致)
top_k_ratio = 0.3    # 稀疏保留比例
# ============

def get_iou(set_a, set_b):
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def analyze_trend():
    print(f"=== Attention Mask Drift Analysis (Layer {layer_idx}) ===")
    print(f"Base Step: {base_step} | Top-K: {top_k_ratio}")
    print("-" * 40)
    print(f"{'Step':<10} | {'IoU vs Base':<15} | {'Status'}")
    print("-" * 40)

    # 1. 加载基准 Mask
    path_base = f"debug_data/attn_layer{layer_idx}_step{base_step}.pt"
    if not os.path.exists(path_base):
        print(f"Error: Base file {path_base} not found. Please run generation first.")
        return

    attn_base = torch.load(path_base).float()
    L = attn_base.shape[0]
    k = int(L * top_k_ratio)
    
    # 获取基准的 Top-K 索引集合
    _, indices_base = torch.topk(attn_base, k, dim=-1)
    base_sets = [set(indices_base[i].tolist()) for i in range(L)]

    # 2. 遍历所有 Step
    steps = list(range(0, max_step + 1, interval))
    
    for step in steps:
        if step == base_step:
            print(f"{step:<10} | {1.0000:<15.4f} | Base")
            continue

        path_curr = f"debug_data/attn_layer{layer_idx}_step{step}.pt"
        if not os.path.exists(path_curr):
            # 如果文件不存在，静默跳过或打印提示
            # print(f"{step:<10} | {'N/A':<15} | File Missing")
            continue

        attn_curr = torch.load(path_curr).float()
        
        # 确保形状一致（防止不同 prompt 长度混合）
        if attn_curr.shape[0] != L:
            print(f"{step:<10} | {'Shape Mismatch'} | Skip")
            continue

        _, indices_curr = torch.topk(attn_curr, k, dim=-1)
        
        # 计算平均 IoU
        row_ious = []
        for i in range(L):
            curr_set = set(indices_curr[i].tolist())
            row_ious.append(get_iou(base_sets[i], curr_set))
        
        avg_iou = np.mean(row_ious)
        
        # 状态标记
        status = ""
        if avg_iou < 0.4: status = "⚠️ Drifted"
        elif avg_iou < 0.6: status = "📉 Low"
        
        print(f"{step:<10} | {avg_iou:<15.4f} | {status}")

if __name__ == "__main__":
    analyze_trend()
