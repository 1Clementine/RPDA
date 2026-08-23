import argparse
import json
import gc
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_model_specs(specs):
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"Bad model spec: {s}. Expected name=path")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def load_jsonl(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_tok(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, device, dtype):
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


def normalize_answer_index(ans, labels=None):
    if isinstance(ans, int):
        return ans
    if isinstance(ans, str):
        s = ans.strip()
        if s.isdigit():
            v = int(s)
            if v in [1, 2]:
                return v - 1
            return v
        if labels and s in labels:
            return labels.index(s)
        letters = ["A", "B", "C", "D", "E"]
        if s.upper() in letters:
            return letters.index(s.upper())
    raise ValueError(f"Cannot parse answer: {ans}")


def extract_example(task, x):
    task = task.lower()

    if task == "arc_easy":
        q = x.get("question") or x.get("query") or ""
        choices_obj = x.get("choices")
        choices = []
        labels = []

        if isinstance(choices_obj, dict):
            texts = choices_obj.get("text", [])
            labs = choices_obj.get("label", [])
            choices = [str(t) for t in texts]
            labels = [str(l) for l in labs]
        elif isinstance(choices_obj, list):
            for c in choices_obj:
                if isinstance(c, dict):
                    choices.append(str(c.get("text", c.get("label", ""))))
                    labels.append(str(c.get("label", len(labels))))
                else:
                    choices.append(str(c))
                    labels.append(str(len(labels)))

        ans = x.get("answerKey", x.get("answer", x.get("label")))
        gold = normalize_answer_index(ans, labels)
        prompt = f"Question: {q}\nAnswer:"
        conts = [" " + c for c in choices]
        return prompt, conts, gold

    if task == "piqa":
        goal = x.get("goal", "")
        choices = [x.get("sol1", ""), x.get("sol2", "")]
        ans = x.get("label", x.get("answer"))
        gold = normalize_answer_index(ans)
        prompt = f"Goal: {goal}\nSolution:"
        conts = [" " + c for c in choices]
        return prompt, conts, gold

    if task == "hellaswag":
        ctx = x.get("ctx", x.get("context", ""))
        endings = x.get("endings", x.get("choices", []))
        choices = [str(e) for e in endings]
        ans = x.get("label", x.get("answer"))
        gold = normalize_answer_index(ans)
        prompt = ctx
        conts = [" " + c for c in choices]
        return prompt, conts, gold

    if task == "winogrande":
        sent = x.get("sentence", "")
        option1 = x.get("option1", "")
        option2 = x.get("option2", "")
        ans = x.get("answer", x.get("label"))
        gold = normalize_answer_index(ans)
        if "_" in sent:
            prefix, suffix = sent.split("_", 1)
            prompt = prefix
            conts = [option1 + suffix, option2 + suffix]
        else:
            prompt = sent + "\nAnswer:"
            conts = [" " + option1, " " + option2]
        return prompt, conts, gold

    # generic fallback
    prompt = x.get("prompt", x.get("question", ""))
    choices = x.get("choices", x.get("options", []))
    if isinstance(choices, dict):
        choices = choices.get("text", [])
    choices = [str(c) for c in choices]
    ans = x.get("answer", x.get("label", x.get("answerKey")))
    gold = normalize_answer_index(ans)
    conts = [" " + c for c in choices]
    return prompt, conts, gold


@torch.no_grad()
def score_continuation(model, tok, prompt, continuation, device, max_length):
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    cont_ids = tok(continuation, add_special_tokens=False).input_ids

    if len(cont_ids) == 0:
        return float("-inf"), float("-inf"), 0

    max_prompt_len = max_length - len(cont_ids)
    if max_prompt_len <= 1:
        cont_ids = cont_ids[-(max_length - 1):]
        prompt_ids = prompt_ids[-1:]
    elif len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    full_ids = prompt_ids + cont_ids
    labels = [-100] * len(prompt_ids) + cont_ids

    input_ids = torch.tensor([full_ids], device=device)
    labels_t = torch.tensor([labels], device=device)

    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels_t[:, 1:]

    mask = shift_labels.ne(-100)
    if mask.sum().item() == 0:
        return float("-inf"), float("-inf"), 0

    logp = F.log_softmax(shift_logits, dim=-1)
    gathered = logp.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    selected = gathered[mask]

    total = float(selected.sum().item())
    avg = float(selected.mean().item())
    n_tok = int(mask.sum().item())
    return total, avg, n_tok


@torch.no_grad()
def eval_task(model, tok, task_name, examples, device, max_length):
    correct_raw = 0
    correct_norm = 0
    used = 0

    for x in examples:
        try:
            prompt, conts, gold = extract_example(task_name, x)
            if not conts or gold < 0 or gold >= len(conts):
                continue

            raw_scores = []
            norm_scores = []
            for c in conts:
                total, avg, _ = score_continuation(model, tok, prompt, c, device, max_length)
                raw_scores.append(total)
                norm_scores.append(avg)

            pred_raw = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
            pred_norm = max(range(len(norm_scores)), key=lambda i: norm_scores[i])

            correct_raw += int(pred_raw == gold)
            correct_norm += int(pred_norm == gold)
            used += 1
        except Exception:
            continue

    return {
        "n": used,
        "acc": correct_raw / max(used, 1),
        "acc_norm": correct_norm / max(used, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--bench_dir", default="data/capability_benchmarks")
    ap.add_argument("--tasks", nargs="+", default=["arc_easy", "piqa", "hellaswag", "winogrande"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    task_files = {
        "arc_easy": "arc_easy_val.jsonl",
        "piqa": "piqa_val.jsonl",
        "hellaswag": "hellaswag_val.jsonl",
        "winogrande": "winogrande_val.jsonl",
    }

    datasets = {}
    for t in args.tasks:
        path = Path(args.bench_dir) / task_files[t]
        datasets[t] = load_jsonl(path, args.limit)
        print(f"[Loaded] {t}: {len(datasets[t])} from {path}")

    rows = []
    model_specs = parse_model_specs(args.models)

    for name, path in model_specs:
        print(f"\n===== Eval MC capability: {name} =====")
        tok = load_tok(path)
        model = load_model(path, args.device, args.dtype)

        for task in args.tasks:
            r = eval_task(model, tok, task, datasets[task], args.device, args.max_length)
            rows.append({
                "model": name,
                "path": path,
                "task": task,
                "n": r["n"],
                "acc": r["acc"],
                "acc_norm": r["acc_norm"],
            })
            print(f"{task}: n={r['n']} acc={r['acc']:.4f} acc_norm={r['acc_norm']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== MC capability ===")
    print(df.to_string(index=False))
    print("[Saved]", out)


if __name__ == "__main__":
    main()
