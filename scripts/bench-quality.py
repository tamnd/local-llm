#!/usr/bin/env python3
# bench-quality.py: run MMLU, GSM8K, and HumanEval against local models via
# the Ollama API and write a markdown results table to stdout.
#
# Datasets are loaded via the HuggingFace datasets library (installed
# automatically if missing). Results are also saved to /tmp/bench-quality.json
# so they survive terminal disconnect.
#
# Usage:
#   python3 scripts/bench-quality.py [options]
#
# Options:
#   --models   comma-separated Ollama model tags (default: qwen3.6:27b,qwen3.6:35b)
#   --ollama   Ollama base URL (default: http://127.0.0.1:11434)
#   --mmlu-n   questions per MMLU subject (default: 30, 10 subjects = 300 total)
#   --gsm8k-n  GSM8K test questions (default: 200)
#   --he-n     HumanEval problems, 0 = all 164 (default: 0)
#   --skip     comma-separated benchmarks to skip: mmlu, gsm8k, humaneval
#   --out      JSON results file (default: /tmp/bench-quality.json)
#
# Models are queried with thinking disabled (/no_think) so output is direct
# and timing is comparable. Re-run with --think to enable chain-of-thought.

import argparse
import gzip
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="qwen3.6:27b,qwen3.6:35b")
    p.add_argument("--ollama", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    p.add_argument("--mmlu-n", type=int, default=30, dest="mmlu_n")
    p.add_argument("--gsm8k-n", type=int, default=200, dest="gsm8k_n")
    p.add_argument("--he-n", type=int, default=0, dest="he_n")
    p.add_argument("--skip", default="")
    p.add_argument("--think", action="store_true", help="enable chain-of-thought")
    p.add_argument("--out", default="/tmp/bench-quality.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def ollama_gen(base_url, model, prompt, system=None, max_tokens=512, temperature=0.0,
               think=False):
    # Use the chat API so we can pass think:false, which the generate API ignores
    # for Qwen3.x models (they output <think> preamble that silently eats tokens).
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "seed": 42,
        },
    }
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        d = json.loads(resp.read())
    msg = d.get("message", {})
    text = msg.get("content", "")
    toks = d.get("eval_count", 0)
    ns = d.get("eval_duration", 1)
    return text, toks, ns


def measure_tps(base_url, model, n_tokens=256):
    prompt = (
        "Explain in detail how a write-ahead log guarantees durability in a "
        "database engine. Cover log records, checkpoints, redo/undo phases, "
        "group commit, and fsync ordering."
    )
    _, toks, ns = ollama_gen(base_url, model, prompt, max_tokens=n_tokens, think=False)
    return round(toks / (ns / 1e9), 1) if ns > 0 and toks > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataset loading (HuggingFace datasets library)
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "high_school_mathematics",
    "college_mathematics",
    "abstract_algebra",
    "high_school_computer_science",
    "machine_learning",
    "formal_logic",
    "high_school_world_history",
    "philosophy",
    "high_school_physics",
    "professional_law",
]


def ensure_datasets():
    try:
        import datasets as _  # noqa
    except ImportError:
        print("installing huggingface datasets...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])


def load_mmlu(n_per_subject=30, seed=42):
    from datasets import load_dataset

    rng = random.Random(seed)
    rows = []
    dev_bank = {}

    for subj in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=False)
        val = load_dataset("cais/mmlu", subj, split="validation", trust_remote_code=False)

        choices_key = "choices"
        answer_key = "answer"
        question_key = "question"

        # Build dev examples from validation split (5 examples)
        dev = []
        for item in list(val)[:5]:
            dev.append((item[question_key], item[choices_key], item[answer_key]))
        dev_bank[subj] = dev

        # Sample test questions
        test_list = list(ds)
        rng.shuffle(test_list)
        for item in test_list[:n_per_subject]:
            rows.append({
                "subject": subj,
                "question": item[question_key],
                "choices": item[choices_key],
                "answer": item[answer_key],
                "dev": dev,
            })
    return rows


def load_gsm8k(n=200, seed=42):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=False)
    items = list(ds)
    random.Random(seed).shuffle(items)
    return items[:n]


def load_humaneval(n=0, seed=42):
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=False)
    items = list(ds)
    if n and n < len(items):
        random.Random(seed).shuffle(items)
        items = items[:n]
    return items


# ---------------------------------------------------------------------------
# MMLU evaluation
# ---------------------------------------------------------------------------

CHOICE_LABELS = ["A", "B", "C", "D"]


def fmt_mmlu(question, choices, dev, think=False):
    lines = []
    for dq, dc, da in dev:
        lines.append(f"Question: {dq}")
        for i, c in enumerate(dc):
            lines.append(f"{CHOICE_LABELS[i]}. {c}")
        lines.append(f"Answer: {CHOICE_LABELS[da]}\n")
    lines.append(f"Question: {question}")
    for i, c in enumerate(choices):
        lines.append(f"{CHOICE_LABELS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_choice(text):
    m = re.search(r"\b([A-D])\b", text.strip())
    return m.group(1) if m else None


def run_mmlu(base_url, model, items, think=False):
    sys_prompt = (
        "Answer each multiple-choice question with just one letter: A, B, C, or D."
    )
    _ = think  # thinking controlled via ollama_gen think= param, not system prompt

    correct = 0
    errors = 0
    t0 = time.time()

    for i, item in enumerate(items):
        prompt = fmt_mmlu(item["question"], item["choices"], item["dev"], think)
        try:
            # With thinking on, the <think> block can be hundreds of tokens
            # before the answer letter appears. Use a generous budget.
            mtok = 1024 if think else 16
            resp, _, _ = ollama_gen(base_url, model, prompt, system=sys_prompt,
                                     max_tokens=mtok, temperature=0.0, think=think)
            pred = extract_choice(resp)
            gold = CHOICE_LABELS[item["answer"]]
            if pred == gold:
                correct += 1
        except Exception as e:
            errors += 1
            print(f"  mmlu error {i}: {e}", file=sys.stderr, flush=True)

        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            elapsed = time.time() - t0
            print(f"  mmlu {i+1}/{len(items)}  acc={correct/(i+1-errors):.1%}  {elapsed:.0f}s",
                  flush=True)

    total = len(items) - errors
    acc = correct / total if total else 0.0
    return acc, total


# ---------------------------------------------------------------------------
# GSM8K evaluation
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text):
    # Model should end with #### <number>; fall back to last number
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"(?<![a-zA-Z])([\d,]+\.?\d*)(?![a-zA-Z])", text)
    return nums[-1].replace(",", "").strip() if nums else None


def run_gsm8k(base_url, model, items, think=False):
    sys_prompt = (
        "Solve the math problem step by step. "
        "Put your final answer after '####' like this: #### 42"
    )
    _ = think  # thinking controlled via ollama_gen think= param, not system prompt

    correct = 0
    errors = 0
    t0 = time.time()

    for i, item in enumerate(items):
        question = item["question"]
        gold_raw = item["answer"]
        gold = extract_gsm8k_answer(gold_raw)

        try:
            mtok = 2048 if think else 512
            resp, _, _ = ollama_gen(base_url, model, question, system=sys_prompt,
                                     max_tokens=mtok, temperature=0.0, think=think)
            pred = extract_gsm8k_answer(resp)
            if pred and gold and pred == gold:
                correct += 1
        except Exception as e:
            errors += 1
            print(f"  gsm8k error {i}: {e}", file=sys.stderr, flush=True)

        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            elapsed = time.time() - t0
            print(f"  gsm8k {i+1}/{len(items)}  acc={correct/(i+1-errors):.1%}  {elapsed:.0f}s",
                  flush=True)

    total = len(items) - errors
    acc = correct / total if total else 0.0
    return acc, total


# ---------------------------------------------------------------------------
# HumanEval evaluation
# ---------------------------------------------------------------------------

def extract_code(text, prompt):
    # Strip markdown fences
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text


def run_humaneval(base_url, model, items, think=False):
    sys_prompt = (
        "Complete the Python function. Return only the implementation, "
        "no explanation, no markdown fences."
    )
    _ = think  # thinking controlled via ollama_gen think= param, not system prompt

    passed = 0
    errors = 0
    t0 = time.time()

    for i, item in enumerate(items):
        task_id = item["task_id"]
        prompt = item["prompt"]
        entry = item["entry_point"]
        test_code = item["test"]
        canonical = item.get("canonical_solution", "")

        try:
            mtok = 2048 if think else 512
            resp, _, _ = ollama_gen(base_url, model, prompt, system=sys_prompt,
                                     max_tokens=mtok, temperature=0.0, think=think)
            impl = extract_code(resp, prompt)

            # Build executable: prompt (has imports + signature) + impl + tests
            full = prompt + "\n" + impl + "\n\n" + test_code + "\n\ncheck(" + entry + ")\n"
            result = subprocess.run(
                [sys.executable, "-c", full],
                capture_output=True,
                timeout=10,
                text=True,
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            errors += 1
        except Exception as e:
            errors += 1
            print(f"  humaneval error {task_id}: {e}", file=sys.stderr, flush=True)

        if (i + 1) % 20 == 0 or (i + 1) == len(items):
            elapsed = time.time() - t0
            total_so_far = i + 1 - errors
            rate = passed / total_so_far if total_so_far else 0
            print(f"  humaneval {i+1}/{len(items)}  pass@1={rate:.1%}  {elapsed:.0f}s",
                  flush=True)

    total = len(items) - errors
    acc = passed / total if total else 0.0
    return acc, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ensure_datasets()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    think = args.think

    print(f"models: {models}", flush=True)
    print(f"think: {think}", flush=True)
    print(f"skip: {skip or 'none'}", flush=True)
    print()

    # Load datasets once
    mmlu_items = gsm8k_items = he_items = None
    if "mmlu" not in skip:
        print("loading MMLU...", flush=True)
        mmlu_items = load_mmlu(n_per_subject=args.mmlu_n)
        print(f"  {len(mmlu_items)} questions across {len(MMLU_SUBJECTS)} subjects", flush=True)
    if "gsm8k" not in skip:
        print("loading GSM8K...", flush=True)
        gsm8k_items = load_gsm8k(n=args.gsm8k_n)
        print(f"  {len(gsm8k_items)} problems", flush=True)
    if "humaneval" not in skip:
        print("loading HumanEval...", flush=True)
        he_items = load_humaneval(n=args.he_n)
        print(f"  {len(he_items)} problems", flush=True)
    print()

    results = {}
    out_path = Path(args.out)

    for model in models:
        print(f"=== {model} ===", flush=True)
        r = {"model": model, "think": think}

        print("measuring tok/s (256 decode tokens)...", flush=True)
        tps = measure_tps(args.ollama, model)
        r["tps"] = tps
        print(f"  {tps} tok/s", flush=True)

        if mmlu_items is not None:
            print(f"running MMLU ({len(mmlu_items)} questions)...", flush=True)
            acc, n = run_mmlu(args.ollama, model, mmlu_items, think=think)
            r["mmlu"] = {"acc": round(acc, 4), "n": n}
            print(f"  MMLU: {acc:.1%}  (n={n})", flush=True)

        if gsm8k_items is not None:
            print(f"running GSM8K ({len(gsm8k_items)} problems)...", flush=True)
            acc, n = run_gsm8k(args.ollama, model, gsm8k_items, think=think)
            r["gsm8k"] = {"acc": round(acc, 4), "n": n}
            print(f"  GSM8K: {acc:.1%}  (n={n})", flush=True)

        if he_items is not None:
            print(f"running HumanEval ({len(he_items)} problems)...", flush=True)
            acc, n = run_humaneval(args.ollama, model, he_items, think=think)
            r["humaneval"] = {"acc": round(acc, 4), "n": n}
            print(f"  HumanEval pass@1: {acc:.1%}  (n={n})", flush=True)

        results[model] = r
        # Save after each model so results survive a disconnect
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved to {out_path}", flush=True)
        print()

    # Print summary table
    print_table(results, skip)


def print_table(results, skip):
    print("\n" + "=" * 72, flush=True)
    print("RESULTS", flush=True)
    print("=" * 72, flush=True)

    header = f"{'model':<30} {'tok/s':>7}"
    if "mmlu" not in skip:
        header += f" {'MMLU':>8}"
    if "gsm8k" not in skip:
        header += f" {'GSM8K':>8}"
    if "humaneval" not in skip:
        header += f" {'HumanEval':>10}"
    print(header, flush=True)
    print("-" * 72, flush=True)

    for model, r in results.items():
        row = f"{model:<30} {r.get('tps', 0):>7.1f}"
        if "mmlu" not in skip:
            v = r.get("mmlu", {})
            row += f" {v.get('acc', 0):.1%}".rjust(9) if v else f" {'N/A':>8}"
        if "gsm8k" not in skip:
            v = r.get("gsm8k", {})
            row += f" {v.get('acc', 0):.1%}".rjust(9) if v else f" {'N/A':>8}"
        if "humaneval" not in skip:
            v = r.get("humaneval", {})
            row += f" {v.get('acc', 0):.1%}".rjust(11) if v else f" {'N/A':>10}"
        print(row, flush=True)

    print()
    print("MMLU: 10 subjects × 30 q, 5-shot", flush=True)
    print("GSM8K: grade-school math, 200 random test problems", flush=True)
    print("HumanEval: code generation pass@1, all 164 problems", flush=True)
    print("think=False for all runs (no_think prompt)", flush=True)


if __name__ == "__main__":
    main()
