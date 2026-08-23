#!/usr/bin/env python3
"""
Unified REEF Protocol Audit — extracts representations once, then scores under 6 protocols.

Protocol A: fixed_layer_L9_equivalent_depth — L9→L9 for 12L models, L18→L9 for 24L models
Protocol B: gpt2_L6_L11_mean — L6-L11 diagonal mean (GPT-2 only)
Protocol C: equivalent_depth_mean — explicit layer mapping with source→target mean
Protocol D: full_diagonal_mean — all diagonal layers mean
Protocol E: row_best_mean — per source layer max then average (sensitivity only)
Protocol F: DTW_alignment — DTW path average similarity (sensitivity only)

Single consistent extraction: max_length=512, left-padding, hook-based last-token,
center=True, scale=True (unit-variance) — matching REEF standard.

Usage:
    python scripts_final/reef_unified_protocol_audit.py \
        --source models/victim/base/gpt2 \
        --gpt2_checkpoints runs/twostage_raw/r04 runs/twostage_raw/r08 ... \
        --external_models models/external/gpt-neo-125m models/external/pythia-410m ... \
        --tqa_data data/truthfulqa.csv \
        --out_dir results/audit_final/reef_unified
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── CKA Kernels ──────────────────────────────────────────────────────────

def linear_cka_hsic(X, Y, eps=1e-8):
    """HSIC-style CKA: center features then scale to unit variance (REEF standard)."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    sx = X.std(axis=0, ddof=0); sy = Y.std(axis=0, ddof=0)
    X = X / (sx + eps); Y = Y / (sy + eps)
    XtY = X.T @ Y; XtX = X.T @ X; YtY = Y.T @ Y
    return float(np.sum(XtY**2) / (math.sqrt(np.sum(XtX**2) * np.sum(YtY**2)) + eps))


def linear_cka_center_only(X, Y, eps=1e-8):
    """Center-only CKA (no scaling)."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    X -= X.mean(axis=0, keepdims=True); Y -= Y.mean(axis=0, keepdims=True)
    XtY = X.T @ Y; XtX = X.T @ X; YtY = Y.T @ Y
    return float(np.sum(XtY**2) / (math.sqrt(np.sum(XtX**2) * np.sum(YtY**2)) + eps))


# ── Extraction ───────────────────────────────────────────────────────────

def get_blocks(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return list(model.gpt_neox.layers)
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return list(model.model.decoder.layers)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    raise ValueError(f"Cannot find blocks in {type(model).__name__}")


@torch.no_grad()
def extract_representations(model_path, texts, device='cuda', max_length=512):
    """Hook-based last-token extraction with left-padding (REEF standard)."""
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else '[PAD]'
    tok.padding_side = 'left'

    blocks = get_blocks(model)
    n_layers = len(blocks)
    cache = {l: [] for l in range(n_layers)}
    handles = []

    def hook_fn(lid):
        def fn(module, inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            cache[lid].append(out[:, -1, :].detach().cpu())
        return fn

    for l in range(n_layers):
        handles.append(blocks[l].register_forward_hook(hook_fn(l)))

    bs = 16
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        inp = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        _ = model(**inp)

    for h in handles: h.remove()
    del model; torch.cuda.empty_cache()

    return {l: torch.cat(cache[l], dim=0).float().numpy() for l in range(n_layers)}


# ── Protocol Scoring ─────────────────────────────────────────────────────

def compute_full_cka_matrix(src_reps, tgt_reps, cka_fn, n_src, n_tgt):
    """Compute full [n_src x n_tgt] CKA matrix."""
    S = np.zeros((n_src, n_tgt), dtype=np.float64)
    for i in range(n_src):
        for j in range(n_tgt):
            S[i, j] = cka_fn(src_reps[i], tgt_reps[j])
    return S


def protocol_A_fixed_L9(src_reps, tgt_reps, n_src, n_tgt, cka_fn):
    """Fixed layer L9→L9 for 12L, L18→L9 for 24L (equivalent depth)."""
    if n_src == 12 and n_tgt == 12:
        return cka_fn(src_reps[9], tgt_reps[9])
    elif n_src == 12 and n_tgt == 24:
        return cka_fn(src_reps[9], tgt_reps[18])
    elif n_src == 24 and n_tgt == 12:
        return cka_fn(src_reps[18], tgt_reps[9])
    elif n_src == 24 and n_tgt == 24:
        return cka_fn(src_reps[18], tgt_reps[18])
    else:
        # Default: proportional depth
        src_layer = int(9/12 * n_src)
        tgt_layer = int(9/12 * n_tgt)
        return cka_fn(src_reps[src_layer], tgt_reps[tgt_layer])


def protocol_B_L6_L11_mean(src_reps, tgt_reps, cka_fn):
    """GPT-2 L6-L11 diagonal mean (both must be 12 layers)."""
    vals = [cka_fn(src_reps[i], tgt_reps[i]) for i in range(6, 12)]
    return float(np.mean(vals))


def protocol_C_equivalent_depth_mean(src_reps, tgt_reps, n_src, n_tgt, cka_fn):
    """Equivalent depth mapping: for each source layer, map to target at same depth %."""
    mappings = []
    for i in range(n_src):
        src_depth = i / max(n_src - 1, 1)
        tgt_layer = int(round(src_depth * (n_tgt - 1)))
        mappings.append((i, tgt_layer))
    vals = [cka_fn(src_reps[s], tgt_reps[t]) for s, t in mappings]
    return float(np.mean(vals)), mappings


def protocol_D_full_diagonal_mean(src_reps, tgt_reps, n_min, cka_fn):
    """Diagonal mean over min(n_src, n_tgt) layers."""
    vals = [cka_fn(src_reps[i], tgt_reps[i]) for i in range(n_min)]
    return float(np.mean(vals))


def protocol_E_row_best_mean(src_reps, tgt_reps, n_src, n_tgt, cka_fn):
    """For each source layer, max over target layers, then average."""
    row_bests = []
    for i in range(n_src):
        best = max(cka_fn(src_reps[i], tgt_reps[j]) for j in range(n_tgt))
        row_bests.append(best)
    return float(np.mean(row_bests))


def protocol_F_dtw_alignment(src_reps, tgt_reps, n_src, n_tgt, cka_fn):
    """DTW alignment on CKA matrix — average path similarity."""
    S = compute_full_cka_matrix(src_reps, tgt_reps, cka_fn, n_src, n_tgt)
    # Convert similarity to distance for DTW
    D = 1.0 - S
    # Simple DTW
    cost = np.full((n_src, n_tgt), np.inf)
    cost[0, 0] = D[0, 0]
    for i in range(1, n_src):
        cost[i, 0] = cost[i-1, 0] + D[i, 0]
    for j in range(1, n_tgt):
        cost[0, j] = cost[0, j-1] + D[0, j]
    for i in range(1, n_src):
        for j in range(1, n_tgt):
            cost[i, j] = D[i, j] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    # Traceback
    path = []
    i, j = n_src - 1, n_tgt - 1
    while i > 0 or j > 0:
        path.append((i, j, float(S[i, j])))
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            d = min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            if d == cost[i-1, j-1]: i -= 1; j -= 1
            elif d == cost[i-1, j]: i -= 1
            else: j -= 1
    path.append((0, 0, float(S[0, 0])))
    path_sims = [p[2] for p in path]
    return float(np.mean(path_sims)), path


# ── Main ─────────────────────────────────────────────────────────────────

def load_tqa(path, n=200):
    df = pd.read_csv(path)
    if 'statement' in df.columns:
        return df['statement'].tolist()[:n]
    if 'question' in df.columns:
        return df['question'].tolist()[:n]
    return df.iloc[:,0].tolist()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='Source GPT-2 model path')
    ap.add_argument('--gpt2_checkpoints', nargs='*', default=[], help='GPT-2 RPDA checkpoint paths')
    ap.add_argument('--external_models', nargs='*', default=[], help='External model paths')
    ap.add_argument('--tqa_data', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_length', type=int, default=512)
    ap.add_argument('--cka_mode', default='hsic', choices=['hsic', 'center_only'],
                    help='hsic=center+scale (REEF), center_only=center only')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cka_fn = linear_cka_hsic if args.cka_mode == 'hsic' else linear_cka_center_only

    texts = load_tqa(args.tqa_data, 200)
    print(f"TQA200: {len(texts)} questions | CKA: {args.cka_mode} | max_length: {args.max_length}")

    # Extract source
    print(f"\n[Extract source] {args.source}")
    src_reps = extract_representations(args.source, texts, dev, args.max_length)
    n_src = len(src_reps)
    print(f"  Layers: {n_src}")

    # Collect all targets
    targets = []
    for cp in args.gpt2_checkpoints:
        targets.append(('gpt2_rpda', os.path.basename(cp), cp))
    for em in args.external_models:
        targets.append(('external', os.path.basename(em), em))

    # Self-check
    print("\n[Self-check] Source vs Source")
    self_A = protocol_A_fixed_L9(src_reps, src_reps, n_src, n_src, cka_fn)
    self_B = protocol_B_L6_L11_mean(src_reps, src_reps, cka_fn) if n_src == 12 else None
    self_C, self_C_map = protocol_C_equivalent_depth_mean(src_reps, src_reps, n_src, n_src, cka_fn)
    self_D = protocol_D_full_diagonal_mean(src_reps, src_reps, n_src, cka_fn)
    self_E = protocol_E_row_best_mean(src_reps, src_reps, n_src, n_src, cka_fn)
    self_F, self_F_path = protocol_F_dtw_alignment(src_reps, src_reps, n_src, n_src, cka_fn)
    print(f"  A(fixed L9 eq depth) = {self_A:.10f}")
    print(f"  B(L6-L11 mean)       = {self_B}")
    print(f"  C(eq depth mean)     = {self_C:.10f}")
    print(f"  D(full diag mean)    = {self_D:.10f}")
    print(f"  E(row_best mean)     = {self_E:.10f}")
    print(f"  F(DTW alignment)     = {self_F:.10f}")

    # Consistency check: self must be 1.0
    tol = 1e-6
    checks = [(self_A, 'A'), (self_C, 'C'), (self_D, 'D'), (self_F, 'F')]
    for val, name in checks:
        if abs(val - 1.0) > tol:
            print(f"  *** FAIL: Protocol {name} self-CKA = {val} != 1.0 (diff={abs(val-1.0):.2e})")
        else:
            print(f"  PASS: Protocol {name} self-CKA = {val}")

    # Score all targets
    rows = []
    for cat, name, path in targets:
        print(f"\n[{cat}] {name}")
        try:
            tgt_reps = extract_representations(path, texts, dev, args.max_length)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        n_tgt = len(tgt_reps)
        print(f"  Layers: {n_tgt}")

        score_A = protocol_A_fixed_L9(src_reps, tgt_reps, n_src, n_tgt, cka_fn)
        score_B = protocol_B_L6_L11_mean(src_reps, tgt_reps, cka_fn) if (n_src == 12 and n_tgt == 12) else None
        score_C, score_C_map = protocol_C_equivalent_depth_mean(src_reps, tgt_reps, n_src, n_tgt, cka_fn)
        n_min = min(n_src, n_tgt)
        score_D = protocol_D_full_diagonal_mean(src_reps, tgt_reps, n_min, cka_fn)
        score_E = protocol_E_row_best_mean(src_reps, tgt_reps, n_src, n_tgt, cka_fn)
        score_F, score_F_path = protocol_F_dtw_alignment(src_reps, tgt_reps, n_src, n_tgt, cka_fn)

        print(f"  A={score_A:.6f} B={score_B} C={score_C:.6f} D={score_D:.6f} E={score_E:.6f} F={score_F:.6f}")

        rows.append({
            'category': cat, 'model': name, 'checkpoint_path': path,
            'n_layers_src': n_src, 'n_layers_tgt': n_tgt,
            'protocol_A_fixed_L9_eq_depth': score_A,
            'protocol_B_L6_L11_mean': score_B,
            'protocol_C_eq_depth_mean': score_C,
            'protocol_C_layer_mapping': str(score_C_map),
            'protocol_D_full_diag_mean': score_D,
            'protocol_E_row_best_mean': score_E,
            'protocol_F_dtw_alignment': score_F,
            'protocol_F_path': str(score_F_path[:5]) + '...' + str(score_F_path[-5:]),
            'cka_mode': args.cka_mode, 'max_length': args.max_length,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / 'reef_unified_all_protocols.csv', index=False)
    print(f"\n[Saved] {out / 'reef_unified_all_protocols.csv'}")
    print(df[['model','protocol_A_fixed_L9_eq_depth','protocol_B_L6_L11_mean',
              'protocol_C_eq_depth_mean','protocol_D_full_diag_mean',
              'protocol_E_row_best_mean','protocol_F_dtw_alignment']].to_string(index=False))

    # Save metadata
    meta = {
        'source_model': args.source, 'cka_mode': args.cka_mode,
        'max_length': args.max_length, 'dataset': 'TruthfulQA-200',
        'extraction': 'hook-based last-token left-padding',
        'self_check_A': self_A, 'self_check_C': self_C,
        'self_check_D': self_D, 'self_check_E': self_E, 'self_check_F': self_F,
    }
    with open(out / 'reef_unified_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
