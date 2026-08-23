#!/usr/bin/env python3
"""ED-BA 5-seed evaluation matching frozen protocol: edba_tqa_binary_layers_6_8_10_logreg_5seed_v1.
Uses original TQA binary JSONL + per-seed train shuffle (matching e3_statistical_stability.py).
Layers: [6, 8, 10] (0-based). StandardScaler + LogisticRegression. 5 seeds with ddof=1 std.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEEDS = [0, 1, 2, 3, 4]  # Match original: range(5)
LAYERS = [6, 8, 10]

# TQA binary JSONL (TruthfulQA 二分类，正负平衡)
_DEFAULT_TQA = 'data/truthfulqa_binary.jsonl'


def load_tqa_binary(jsonl_path):
    items = []
    with open(jsonl_path) as f:
        for line in f:
            items.append(json.loads(line))
    items.sort(key=lambda x: x['id'])
    train = items[:2542]
    test = items[2542:2542+636]
    return (
        [x['text'] for x in train],
        [x['label'] for x in train],
        [x['text'] for x in test],
        [x['label'] for x in test],
    )


@torch.no_grad()
def extract_hidden_states(model, tokenizer, texts, layers, device, max_length=256):
    # Get transformer blocks
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        blocks = list(model.transformer.h)
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        blocks = list(model.gpt_neox.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        blocks = list(model.model.decoder.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        blocks = list(model.model.layers)
    else:
        raise ValueError(f"Cannot find blocks in {type(model).__name__}")

    model.eval()
    cache = {l: [] for l in layers}
    handles = []

    def make_hook(layer_id):
        def hook(module, inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            cache[layer_id].append(out[:, -1, :].detach().cpu())
        return hook

    for l in layers:
        handle = blocks[l].register_forward_hook(make_hook(l))
        handles.append(handle)

    tokenizer.padding_side = 'left'
    bs = 16
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        inp = tokenizer(batch, return_tensors='pt', padding=True,
                       truncation=True, max_length=max_length).to(device)
        _ = model(**inp)

    for h in handles:
        h.remove()

    return {l: torch.cat(cache[l], dim=0).float().numpy() for l in layers}


def train_and_eval(train_reps, train_labels, test_reps, test_labels, seed):
    """Match original: shuffle train with seed, then StandardScaler + LogisticRegression."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_reps))
    X_train = np.array(train_reps)[idx]
    y_train = np.array(train_labels)[idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(test_reps)
    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    return float(balanced_accuracy_score(test_labels, preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--tqa_jsonl', default=_DEFAULT_TQA)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device
    train_texts, train_labels, test_texts, test_labels = load_tqa_binary(args.tqa_jsonl)
    print(f"TQA binary: {len(train_texts)} train, {len(test_texts)} test")

    # Load source
    print(f"Source: {args.source}")
    src = AutoModelForCausalLM.from_pretrained(args.source).to(device)
    tok = AutoTokenizer.from_pretrained(args.source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Layers: {LAYERS}")

    # Extract source states (train + test)
    all_texts = train_texts + test_texts
    src_reps = extract_hidden_states(src, tok, all_texts, LAYERS, device)
    src_concat = np.concatenate([src_reps[l] for l in LAYERS], axis=1)
    src_train = src_concat[:len(train_texts)]
    src_test = src_concat[len(train_texts):]

    # Source self-eval (5 seeds)
    src_ba_seeds = []
    for s in SEEDS:
        ba = train_and_eval(src_train, train_labels, src_test, test_labels, s)
        src_ba_seeds.append(ba)
    src_mean = np.mean(src_ba_seeds)
    src_std = np.std(src_ba_seeds, ddof=1)
    print(f"  Source ED-BA: {src_mean:.4f} ± {src_std:.4f}")

    del src; torch.cuda.empty_cache()

    # Evaluate checkpoints
    rows = []
    for cp_path in args.checkpoints:
        cp_name = Path(cp_path).parent.name if 'best_checkpoint' in cp_path else Path(cp_path).name
        print(f"[{cp_name}] ", end='', flush=True)

        if not os.path.isdir(cp_path):
            print("SKIP (not found)")
            continue

        cand = AutoModelForCausalLM.from_pretrained(cp_path).to(device)
        cand_tok = AutoTokenizer.from_pretrained(cp_path)
        if cand_tok.pad_token is None:
            cand_tok.pad_token = cand_tok.eos_token

        cand_reps = extract_hidden_states(cand, cand_tok, test_texts, LAYERS, device)
        cand_concat = np.concatenate([cand_reps[l] for l in LAYERS], axis=1)

        ba_seeds = []
        for s in SEEDS:
            ba = train_and_eval(src_train, train_labels, cand_concat, test_labels, s)
            ba_seeds.append(ba)

        mean_ba = np.mean(ba_seeds)
        std_ba = np.std(ba_seeds, ddof=1)
        print(f"ED-BA = {mean_ba:.4f} ± {std_ba:.4f}")

        rows.append({
            'checkpoint': cp_name,
            'checkpoint_path': cp_path,
            'ed_ba_mean': round(mean_ba, 6),
            'ed_ba_std': round(std_ba, 6),
            'ed_ba_seeds': [round(b, 6) for b in ba_seeds],
            'layers': str(LAYERS),
            'protocol': 'edba_tqa_binary_layers_6_8_10_logreg_5seed_v1',
        })

        del cand; torch.cuda.empty_cache()

    # Save
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[Saved] {args.out_csv}")
    print(df[['checkpoint','ed_ba_mean','ed_ba_std']].to_string(index=False))


if __name__ == '__main__':
    main()
