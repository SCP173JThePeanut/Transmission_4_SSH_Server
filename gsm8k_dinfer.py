#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dInfer(vLLM backend) 离线 GSM8K：LLaDA-MoE (FusedMoE) 正确性 vs block_length
- 对齐你现有 Fast-dLLM / SGLang 脚本：offline_datasets/gsm8k + 4-shot prompt + CUDA Event 计时
- 单卡单请求（不启用 TP），用于扫 block_length
"""

import os
import re
import json
import time
import argparse
import warnings

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm import distributed

from dinfer.model import LLaDAMoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder, HierarchyDecoder
from dinfer import BlockWiseDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, IterSmoothDiffusionLLM

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= 提示词模板（对齐 Fast-dLLM 的 4-shot）=================
GSM8K_FEW_SHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: {question}
A:"""

def extract_answer_gsm8k(text: str):
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return None
    return nums[-1]

def extract_gold_gsm8k(gold_text: str):
    if "####" in gold_text:
        return extract_answer_gsm8k(gold_text.split("####")[1].strip())
    return extract_answer_gsm8k(gold_text)

def is_correct_gsm8k(pred: str, gold: str) -> bool:
    p = extract_answer_gsm8k(pred)
    g = extract_gold_gsm8k(gold)
    if p is None or g is None:
        return False
    try:
        return float(p) == float(g)
    except Exception:
        return str(p) == str(g)

def resolve_data_dir(base_path: str):
    candidates = [
        base_path,
        os.path.abspath(base_path),
        os.path.join(os.getcwd(), base_path),
        "../../offline_datasets",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return base_path

def load_gsm8k_dataset(data_dir: str):
    real_path = resolve_data_dir(data_dir)
    print(f"Data Path: {real_path}")
    return load_from_disk(os.path.join(real_path, "gsm8k"))["test"]

def build_gsm8k_prompt(question: str) -> str:
    return GSM8K_FEW_SHOT.format(question=question)

def build_chat_input_if_possible(tokenizer, raw_prompt: str) -> str:
    # 尽量对齐 Fast-dLLM：能用 chat_template 就用，否则回退 raw
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return raw_prompt

def create_dllm(model, args, mask_id: int, eos_id: int):
    # decoder
    if args.parallel_decoding == "threshold":
        decoder = ThresholdParallelDecoder(
            temperature=args.temperature,
            threshold=args.threshold,
            mask_id=mask_id,
            eos_id=eos_id,
        )
    else:
        decoder = HierarchyDecoder(
            temperature=args.temperature,
            threshold=args.threshold,
            low_threshold=args.low_threshold,
            mask_id=mask_id,
            eos_id=eos_id,
        )

    # cache
    cache_factory = None
    if args.cache in ("prefix", "dual"):
        cache_factory = KVCacheFactory(args.cache, is_bd_model=False)

    # diffusion runner
    use_sw = args.cont_weight > 0 and (args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0)
    if args.cont_weight > 0:
        if use_sw:
            return IterSmoothWithVicinityCacheDiffusionLLM(
                model,
                decoder,
                BlockIteratorFactory(start_block_align=True),
                cache_factory=cache_factory,
                early_stop=True,
                cont_weight=args.cont_weight,
                prefix_look=args.prefix_look,
                after_look=args.after_look,
                warmup_steps=args.warmup_times,
            )
        else:
            return IterSmoothDiffusionLLM(
                model,
                decoder,
                BlockIteratorFactory(start_block_align=True),
                cache_factory=cache_factory,
                early_stop=True,
                cont_weight=args.cont_weight,
            )
    else:
        # cont_weight=0 时，按 dInfer benchmark 默认：BlockWise + 可选 vicinity（这里给个开关）
        if args.use_vicinity_cache:
            return VicinityCacheDiffusionLLM(
                model,
                decoder,
                BlockIteratorFactory(start_block_align=True),
                cache_factory=cache_factory,
                early_stop=True,
                prefix_look=args.prefix_look,
                after_look=args.after_look,
                warmup_steps=args.warmup_times,
            )
        return BlockWiseDiffusionLLM(
            model,
            decoder,
            BlockIteratorFactory(start_block_align=True),
            cache_factory=cache_factory,
            early_stop=True,
            use_shift=args.use_shift,
        )

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("dInfer offline GSM8K for LLaDA-MoE (FusedMoE)")

    parser.add_argument("--model_name", type=str, default="/home/*")
    parser.add_argument("--data_dir", type=str, default="/home/*")

    parser.add_argument("--gen_len", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--low_threshold", type=float, default=0.3)
    parser.add_argument("--parallel_decoding", type=str, default="threshold", choices=["threshold", "hierarchy_faster"])

    parser.add_argument("--cache", type=str, default="dual", choices=["", "prefix", "dual"])
    parser.add_argument("--cont_weight", type=float, default=0.0)
    parser.add_argument("--prefix_look", type=int, default=16)
    parser.add_argument("--after_look", type=int, default=16)
    parser.add_argument("--warmup_times", type=int, default=4)

    parser.add_argument("--use_shift", action="store_true")
    parser.add_argument("--use_vicinity_cache", action="store_true")

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 才能做公平计时（CUDA Event）。")

    # ====== LLaDA-MoE 固定 token id（来自 dInfer benchmarks / 官方示例）======
    mask_id = 156895
    eos_id = 156892

    # 单卡单进程：world_size=1
    rank = 0
    world_size = 1
    gpu_id = 0  # 由 CUDA_VISIBLE_DEVICES 决定映射后的 0

    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    # init vllm distributed env (single rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(45601 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))
    distributed.init_distributed_environment(world_size, rank, "env://", rank, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # model (MoE fused)
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        vllm_config = get_current_vllm_config()
        print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)

        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
        model = model.to(device)

        # torch.compile 加速 Python/框架开销
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False, dynamic=True)

        dllm = create_dllm(model, args, mask_id=mask_id, eos_id=eos_id)

    # dataset
    dataset = load_gsm8k_dataset(args.data_dir)
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))
    print(f"Task: gsm8k | Samples: {len(dataset)}")

    # warmup（建议开，稳定 cudagraph/compile cache）
    if args.warmup:
        print("\nWarming up (3 dummy inferences)...")
        raw = "Warm up process start."
        prompt = build_chat_input_if_possible(tokenizer, raw)
        input_ids = torch.tensor(tokenizer(prompt)["input_ids"], device=device).unsqueeze(0)
        for _ in range(3):
            _ = dllm.generate(input_ids, gen_length=min(args.gen_len, 64), block_length=min(args.block_length, 32))
        torch.cuda.empty_cache()
        print("Warm-up done! Starting evaluation...\n")

    # output
    if args.output is None:
        args.output = f"results_gsm8k_dinfer_moe_gl{args.gen_len}_bl{args.block_length}_thr{str(args.threshold).replace('.','')}.jsonl"
    f_out = open(args.output, "w", encoding="utf-8")

    correct = 0
    total = 0
    latencies = []
    tps_list = []

    for i, item in enumerate(dataset):
        raw_prompt = build_gsm8k_prompt(item["question"])
        prompt = build_chat_input_if_possible(tokenizer, raw_prompt)
        gold = item["answer"]

        input_ids = torch.tensor(tokenizer(prompt)["input_ids"], device=device).unsqueeze(0)
        prompt_len = int(input_ids.shape[1])

        start_t = torch.cuda.Event(enable_timing=True)
        end_t = torch.cuda.Event(enable_timing=True)

        prev_forwards = dllm.num_forwards
        start_t.record()
        out = dllm.generate(input_ids, gen_length=args.gen_len, block_length=args.block_length)
        end_t.record()
        torch.cuda.synchronize()

        nfe = dllm.num_forwards - prev_forwards
        latency_s = start_t.elapsed_time(end_t) / 1000.0
        latencies.append(latency_s)

        # decode：只拿生成部分（去掉 prompt）
        gen_ids = out[0, prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        ok = is_correct_gsm8k(gen_text, gold)
        if ok:
            correct += 1
        total += 1
        acc = 100.0 * correct / total

        # 统计生成 token 数（剔除 mask/eos）
        gen_token_num = int(((gen_ids != mask_id) & (gen_ids != eos_id)).sum().item())
        tps = (gen_token_num / latency_s) if latency_s > 0 else 0.0
        tps_list.append(tps)

        if (i % args.log_interval == 0) or (i == len(dataset) - 1):
            avg_lat = sum(latencies) / len(latencies)
            avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
            print(
                f"[{i+1}/{len(dataset)}] "
                f"Latency: {latency_s:.2f}s (Avg: {avg_lat:.2f}s) | "
                f"TPS: {tps:.1f} (Avg: {avg_tps:.1f}) | "
                f"Acc: {acc:.2f}% | "
                f"prompt_len={prompt_len} gen_tokens={gen_token_num} nfe={nfe}"
            )

        f_out.write(
            json.dumps(
                {
                    "index": i,
                    "task": "gsm8k",
                    "latency": latency_s,
                    "tps": tps,
                    "nfe": int(nfe),
                    "prompt_len": prompt_len,
                    "gen_tokens": gen_token_num,
                    "gold": gold,
                    "gen": gen_text,
                    "correct": ok,
                    # 配置记录（方便画曲线）
                    "model_name": args.model_name,
                    "gen_len": args.gen_len,
                    "block_length": args.block_length,
                    "temperature": args.temperature,
                    "parallel_decoding": args.parallel_decoding,
                    "threshold": args.threshold,
                    "low_threshold": args.low_threshold,
                    "cache": args.cache,
                    "cont_weight": args.cont_weight,
                    "prefix_look": args.prefix_look,
                    "after_look": args.after_look,
                    "warmup_times": args.warmup_times,
                    "use_shift": bool(args.use_shift),
                    "use_vicinity_cache": bool(args.use_vicinity_cache),
                    "mask_id": mask_id,
                    "eos_id": eos_id,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f_out.flush()

    f_out.close()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    final_acc = 100.0 * correct / total if total else 0.0
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print("\n========== Final Summary ==========")
    print(f"Samples: {total}")
    print(f"Final Accuracy: {final_acc:.2f}%")
    print(f"Avg Latency: {avg_latency:.4f}s")
    print(f"Avg TPS: {avg_tps:.2f}")
    print(f"Peak Memory: {peak_memory_gb:.2f} GB")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()