import torch
import os

# === 配置 ===
target_layers = [0, 15, 31]
check_step = 100  # 检查生成后期的注意力分布
# ============

def analyze_layers():
    print(f"\n=== Layer-wise Sink Analysis (at Step {check_step}) ===")
    print(f"{'Layer':<6} | {'Sink Mass (First 4)':<20} | {'Local Mass (+-5)':<20} | {'Dominant Token (Idx/Score)'}")
    print("-" * 85)

    for layer in target_layers:
        path = f"debug_data/attn_layer{layer}_step{check_step}.pt"
        if not os.path.exists(path):
            print(f"{layer:<6} | File not found")
            continue

        # 加载 [L, L] 矩阵
        attn = torch.load(path).float()
        L = attn.shape[0]

        # 1. Sink Mass: 所有 Token 对前 4 个 Token 的关注度总和占比
        # attn[:, :4] 是取前4列，sum() 是总能量，除以 L 是平均每个 Token 给 Sink 多少分
        # 因为 attn 每一行和为 1，所以所有元素和为 L。
        # 这里计算：平均每个 Query 分配给 Sink 的权重和
        sink_mass = attn[:, :4].sum(dim=-1).mean().item()

        # 2. Local Mass: 对角线附近关注度 (Sliding Window 假设验证)
        local_sum = 0.0
        window = 5
        for i in range(L):
            start = max(0, i - window)
            end = min(L, i + window + 1)
            local_sum += attn[i, start:end].sum().item()
        local_mass = local_sum / L

        # 3. Global Dominant Token: 寻找全图最受关注的 Token
        # 对列求和，看哪一列（Key）得分最高
        col_sum = attn.sum(dim=0)
        top_idx = col_sum.argmax().item()
        # 归一化得分：平均每个 Query 给了它多少权重
        top_score = col_sum[top_idx].item() / L

        print(f"{layer:<6} | {sink_mass:<20.2%} | {local_mass:<20.2%} | Idx {top_idx} ({top_score:.2%})")

    print("-" * 85)
    print("指标说明:")
    print("1. Sink Mass: 若接近 0%，说明 DLM 不需要 Attention Sink。")
    print("2. Local Mass: 若很高 (>50%)，说明主要是局部注意力，适合 Sliding Window。")
    print("3. Dominant Token: 若 Sink Mass 低但此处分高，说明存在非首位的‘隐形 Sink’。")

if __name__ == "__main__":
    analyze_layers()
