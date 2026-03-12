import os
import re
import json
import time
import argparse
import warnings

import torch
import sglang as sgl
from datasets import load_from_disk
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["NUMEXPR_MAX_THREADS"] = "16"

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
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return raw_prompt

def dump_yaml_config_to_tmp(d: dict) -> str:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("需要 PyYAML：pip install pyyaml") from e

    path = f"/tmp/sglang_dllm_config_{os.getpid()}_{int(time.time())}.yaml"
    with open(path, "w") as f:
        yaml.dump(d, f)
    return path

def get_meta_int(meta_info: dict, key: str):
    v = meta_info.get(key, None)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="sglang GSM8K correctness vs block_size (JointThreshold)")

    parser.add_argument("--model_path", type=str, default="/home/*")
    parser.add_argument("--data_dir", type=str, default="/home/*")

    parser.add_argument("--gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore_eos", action="store_true")

    parser.add_argument("--block_size", type=int, default=32, help="JointThreshold block_size (generation block size)")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--edit_threshold", type=float, default=0.0)
    parser.add_argument("--max_post_edit_steps", type=int, default=16)
    parser.add_argument("--penalty_lambda", type=float, default=0.0)

    parser.add_argument("--mem_fraction", type=float, default=0.8)
    parser.add_argument("--attention_backend", type=str, default="flashinfer", choices=["flashinfer", "triton"])
    parser.add_argument("--disable_cuda_graph", action="store_true", default=True)

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--warmup", action="store_true")

    parser.add_argument("--use_chat_template", action="store_true", default=False)

    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA。")

    # tokenizer（仅用于 apply_chat_template；不影响 sglang engine 的内部 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True) if args.use_chat_template else None

    dataset = load_gsm8k_dataset(args.data_dir)
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))
    print(f"Task: gsm8k | Samples: {len(dataset)}")

    algo_cfg = {
        "block_size": int(args.block_size),
        "threshold": float(args.threshold),
        "edit_threshold": float(args.edit_threshold),
        "max_post_edit_steps": int(args.max_post_edit_steps),
        "penalty_lambda": float(args.penalty_lambda),
    }
    dllm_cfg_path = dump_yaml_config_to_tmp(algo_cfg)
    print(f"Created dllm_algorithm_config: {dllm_cfg_path}")
    print(f"  -> {algo_cfg}")

    engine = sgl.Engine(
        model_path=args.model_path,
        dllm_algorithm="JointThreshold",
        dllm_algorithm_config=dllm_cfg_path,
        trust_remote_code=True,
        disable_cuda_graph=bool(args.disable_cuda_graph),
        mem_fraction_static=args.mem_fraction,
        attention_backend=args.attention_backend,
        max_running_requests=1,
        log_level="error",
    )

    if args.warmup:
        print("\nWarming up (3 dummy inferences)...")
        warmup_params = {"temperature": 0.0, "max_new_tokens": 16, "ignore_eos": True}
        for _ in range(3):
            _ = engine.generate("Warm up process start.", warmup_params)
        torch.cuda.empty_cache()
        print("Warm-up done! Starting evaluation...\n")

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.gen_len,
        "ignore_eos": args.ignore_eos,
    }

    if args.output is None:
        args.output = f"results_gsm8k_sglang_gl{args.gen_len}_bs{args.block_size}.jsonl"
    f_out = open(args.output, "w", encoding="utf-8")

    correct = 0
    latencies = []
    tps_list = []
    total = 0

    torch.cuda.reset_peak_memory_stats()

    for i, item in enumerate(dataset):
        raw_prompt = build_gsm8k_prompt(item["question"])
        prompt = build_chat_input_if_possible(tokenizer, raw_prompt) if tokenizer is not None else raw_prompt
        gold = item["answer"]

        start_t = torch.cuda.Event(enable_timing=True)
        end_t = torch.cuda.Event(enable_timing=True)

        start_t.record()
        output = engine.generate(prompt, sampling_params)
        end_t.record()
        torch.cuda.synchronize()

        latency_s = start_t.elapsed_time(end_t) / 1000.0
        latencies.append(latency_s)

        gen_text = output.get("text", "")
        meta = output.get("meta_info", {}) or {}
        prompt_tokens = get_meta_int(meta, "prompt_tokens")
        completion_tokens = get_meta_int(meta, "completion_tokens")
        nfe = meta.get("nfe", None)

        ok = is_correct_gsm8k(gen_text, gold)
        if ok:
            correct += 1
        total += 1
        acc = 100.0 * correct / total

        gen_tokens = completion_tokens if completion_tokens is not None else None
        tps = (gen_tokens / latency_s) if (gen_tokens is not None and latency_s > 0) else None
        if tps is not None:
            tps_list.append(float(tps))

        if (i % args.log_interval == 0) or (i == len(dataset) - 1):
            avg_lat = sum(latencies) / len(latencies)
            avg_tps = (sum(tps_list) / len(tps_list)) if tps_list else None
            tps_str = f"{tps:.1f} (Avg: {avg_tps:.1f})" if (tps is not None and avg_tps is not None) else "N/A"
            print(
                f"[{i+1}/{len(dataset)}] "
                f"Latency: {latency_s:.2f}s (Avg: {avg_lat:.2f}s) | "
                f"TPS: {tps_str} | "
                f"Acc: {acc:.2f}% | "
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} nfe={nfe}"
            )

        f_out.write(
            json.dumps(
                {
                    "index": i,
                    "task": "gsm8k",
                    "latency": latency_s,
                    "tps": tps,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "nfe": nfe,
                    "gold": gold,
                    "gen": gen_text,
                    "correct": ok,
                    "model_path": args.model_path,
                    "gen_len": args.gen_len,
                    "temperature": args.temperature,
                    "ignore_eos": args.ignore_eos,
                    "use_chat_template": bool(args.use_chat_template),
                    "dllm_algorithm": "JointThreshold",
                    "block_size": args.block_size,
                    "threshold": args.threshold,
                    "edit_threshold": args.edit_threshold,
                    "max_post_edit_steps": args.max_post_edit_steps,
                    "penalty_lambda": args.penalty_lambda,
                    "engine": {
                        "attention_backend": args.attention_backend,
                        "max_running_requests": 1,
                        "mem_fraction_static": args.mem_fraction,
                        "disable_cuda_graph": bool(args.disable_cuda_graph),
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f_out.flush()

    f_out.close()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tps = (sum(tps_list) / len(tps_list)) if tps_list else 0.0
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print("\n========== Final Summary ==========")
    print(f"Samples: {total}")
    print(f"Final Accuracy: {100.0*correct/total:.2f}%")
    print(f"Avg Latency: {avg_latency:.4f}s")
    print(f"Avg TPS: {avg_tps:.2f}")
    print(f"Peak Memory: {peak_memory_gb:.2f} GB")
    print(f"Saved to: {args.output}")

    try:
        if os.path.exists(dllm_cfg_path):
            os.remove(dllm_cfg_path)
    except Exception:
        pass

if __name__ == "__main__":
    main()