import os
import re
import json
import time
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig

from model.modeling_llada import LLaDAModelLM
from generate import generate, generate_with_dual_cache

# ================= 提示词模板（沿用你 SparseD 脚本的 4-shot）=================
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

def load_local_data(task: str, data_dir: str):
    real_path = resolve_data_dir(data_dir)
    print(f"Data Path: {real_path}")

    if task == "gsm8k":
        return load_from_disk(os.path.join(real_path, "gsm8k"))["test"]
    if task in ("humaneval", "he"):
        return load_from_disk(os.path.join(real_path, "humaneval"))["test"]
    raise ValueError(f"Unsupported task: {task}")

def build_gsm8k_input(tokenizer, question: str):
    prompt = GSM8K_FEW_SHOT.format(question=question)
    # 尝试按 instruct 的 chat template；失败就回退到 raw
    try:
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        input_text = prompt
    return input_text

def warmup(model, tokenizer, device, gen_length, steps, block_length, mode, threshold, factor):
    print("\nWarming up (3 dummy inferences)...")
    dummy = tokenizer("Warm up process start.", return_tensors="pt").to(device).input_ids
    with torch.no_grad():
        for _ in range(3):
            if mode == "fast":
                _ = generate_with_dual_cache(
                    model,
                    dummy,
                    steps=min(steps, 16),
                    gen_length=min(gen_length, 32),
                    block_length=min(block_length, 32),
                    temperature=0.0,
                    threshold=threshold,
                    factor=factor,
                )
            else:
                _ = generate(
                    model,
                    dummy,
                    steps=min(steps, 16),
                    gen_length=min(gen_length, 32),
                    block_length=min(block_length, 32),
                    temperature=0.0,
                    threshold=threshold,
                    factor=factor,
                )
    torch.cuda.empty_cache()
    print("Warm-up done! Starting evaluation...\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/*")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "humaneval"])
    parser.add_argument("--data_dir", type=str, default="/home/*")

    # base/fast 开关
    parser.add_argument("--mode", type=str, default="fast", choices=["base", "fast"])

    # generation params（对齐 SparseD 命名：steps/gen_len）
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32, help="Generation block length (NOT SparseD sparse block).")

    # fast-dLLM 的“并行/稀疏”参数：二选一即可
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold mode (parallel sparsity knob).")
    parser.add_argument("--factor", type=float, default=None, help="Dynamic threshold factor mode (parallel sparsity knob).")

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) load model
    print(f"Loading model... mode={args.mode} model_path={args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 2) data
    dataset = load_local_data(args.task, args.data_dir)
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))
    print(f"Task: {args.task} | Samples: {len(dataset)}")

    # 3) warmup
    warmup(
        model,
        tokenizer,
        device,
        gen_length=args.gen_len,
        steps=args.steps,
        block_length=args.block_length,
        mode=args.mode,
        threshold=args.threshold,
        factor=args.factor,
    )

    # 4) output file name
    mode_tag = args.mode
    extra = []
    if args.threshold is not None:
        extra.append(f"th{str(args.threshold).replace('.', '')}")
    if args.factor is not None:
        extra.append(f"fac{str(args.factor).replace('.', '')}")
    extra_tag = ("_" + "_".join(extra)) if extra else ""
    output_file = f"results_{args.task}_{mode_tag}_gl{args.gen_len}_st{args.steps}_bl{args.block_length}{extra_tag}.jsonl"
    f_out = open(output_file, "w", encoding="utf-8")

    correct_count = 0
    total_processed = 0
    latencies = []

    for i, item in enumerate(dataset):
        if args.task == "gsm8k":
            input_text = build_gsm8k_input(tokenizer, item["question"])
            gold = item["answer"]
        else:
            # humaneval: 直接用 prompt
            input_text = item["prompt"]
            gold = item.get("canonical_solution", "")

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        prompt_len = int(inputs.input_ids.shape[1])

        # 计时（用 cuda event）
        start_t = torch.cuda.Event(enable_timing=True)
        end_t = torch.cuda.Event(enable_timing=True)

        start_t.record()
        with torch.no_grad():
            if args.mode == "fast":
                out_ids, nfe = generate_with_dual_cache(
                    model,
                    inputs.input_ids,
                    steps=args.steps,
                    gen_length=args.gen_len,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    threshold=args.threshold,
                    factor=args.factor,
                )
            else:
                out_ids, nfe = generate(
                    model,
                    inputs.input_ids,
                    steps=args.steps,
                    gen_length=args.gen_len,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    threshold=args.threshold,
                    factor=args.factor,
                )
        end_t.record()
        torch.cuda.synchronize()

        latency = start_t.elapsed_time(end_t) / 1000.0
        latencies.append(latency)

        gen_text = tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)

        is_correct = False
        if args.task == "gsm8k":
            is_correct = is_correct_gsm8k(gen_text, gold)
            if is_correct:
                correct_count += 1

        total_processed += 1
        acc = (correct_count / total_processed) * 100 if args.task == "gsm8k" else None

        if i % args.log_interval == 0 or i == len(dataset) - 1:
            avg_lat = sum(latencies) / len(latencies)
            if acc is None:
                print(f"[{i+1}/{len(dataset)}] Latency: {latency:.2f}s (Avg: {avg_lat:.2f}s) | NFE: {nfe}")
            else:
                print(f"[{i+1}/{len(dataset)}] Latency: {latency:.2f}s (Avg: {avg_lat:.2f}s) | Acc: {acc:.2f}% | NFE: {nfe}")

        f_out.write(
            json.dumps(
                {
                    "index": i,
                    "task": args.task,
                    "mode": args.mode,
                    "latency": latency,
                    "nfe": nfe,
                    "prompt_len": prompt_len,
                    "gold": gold,
                    "gen": gen_text,
                    "correct": is_correct if args.task == "gsm8k" else None,
                    "threshold": args.threshold,
                    "factor": args.factor,
                    "gen_len": args.gen_len,
                    "steps": args.steps,
                    "block_length": args.block_length,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f_out.flush()

    f_out.close()
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    if args.task == "gsm8k":
        final_acc = (correct_count / total_processed) * 100 if total_processed else 0.0
        print(f"\nFinal Accuracy: {final_acc:.2f}% | Avg Latency: {avg_latency:.4f}s")
    else:
        print(f"\nFinished {args.task}. Avg Latency: {avg_latency:.4f}s")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()