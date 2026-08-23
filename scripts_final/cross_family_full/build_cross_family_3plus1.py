#!/usr/bin/env python3
"""Generalized same-depth 3+1 structural perturbation for GPT-2, Pythia, and OPT.

Phase 1 (alpha-split): N layers → 2N layers with alpha attention split
Phase 2 (3+1 compression): 2N layers → N layers via shifted 3+1 window with gamma

Usage:
    python build_cross_family_3plus1.py --source models/external/pythia-410m \\
        --out_dir runs/cross_family_full_v1/checkpoints/pythia410m \\
        --alpha 0.10 --gamma 0.0 --rounds 30 \\
        --checkpoints r04 r08 r12 r16 r20 r24 r30
"""

import argparse, os, copy, math, json, time
from pathlib import Path
from collections import OrderedDict

import torch
from model_adapters import (
    detect_arch, get_num_layers, get_blocks, set_block,
    get_final_norm, set_final_norm, get_lm_head,
    get_token_embeddings, get_position_embeddings,
    copy_embeddings_and_norm, save_checkpoint,
    create_empty_model
)


def scale_conv1d_or_linear(tensor, scale):
    """Scale a linear/Conv1D weight tensor. Works for both GPT-2 Conv1D (out,in) and standard Linear (out,in)."""
    w = tensor.detach().clone()
    w = w * scale
    return w


def zero_conv1d_or_linear(tensor):
    """Zero out weight with matching shape."""
    return torch.zeros_like(tensor)


def build_alph_split(source_model, alpha=0.10):
    """Build 2N-layer intermediate model with alpha-split attention.

    For GPT-2:
        Each source block i becomes two blocks: A (scaled QKVO × sqrt(alpha)) + B (scaled QKVO × sqrt(1-alpha))
        FFN: A gets residual FFN, B gets zeroed FFN (acts as pass-through)

    For GPT-Neo (mixed local/global attention):
        Doubles attention_layers as well.
    """
    arch = detect_arch(source_model)
    n_layers = get_num_layers(source_model)
    cfg = copy.deepcopy(source_model.config)

    # Create doubled config
    cfg_dict = cfg.to_dict()
    doubled_layers = 2 * n_layers
    # Different config classes store num_layers differently
    cfg_attrs = ['num_hidden_layers', 'n_layer', 'num_layers']
    for attr in cfg_attrs:
        if attr in cfg_dict:
            cfg_dict[attr] = doubled_layers
    # GPT-Neo: double attention_layers (mixed local/global pattern)
    if 'attention_layers' in cfg_dict and cfg_dict['attention_layers'] is not None:
        if isinstance(cfg_dict['attention_layers'], list):
            cfg_dict['attention_layers'] = [v for v in cfg_dict['attention_layers'] for _ in (0,1)]
        # Remove attention_types to prevent recomputation from the original types
        cfg_dict.pop('attention_types', None)
    new_cfg = type(cfg)(**cfg_dict)

    arch_class = type(source_model)
    split_model = arch_class(new_cfg)

    # Copy embeddings and norm
    copy_embeddings_and_norm(source_model, split_model)

    # Split each source block into A and B
    src_blocks = get_blocks(source_model)
    dst_blocks = get_blocks(split_model)

    alpha = getattr(build_alph_split, '_alpha', 0.10)

    for i in range(n_layers):
        src = src_blocks[i]
        A = dst_blocks[2 * i]
        B = dst_blocks[2 * i + 1]

        # Deep copy source state
        A.load_state_dict(src.state_dict())
        B.load_state_dict(src.state_dict())

        # Scale attention weights
        scale_a = math.sqrt(alpha)
        scale_b = math.sqrt(1.0 - alpha)

        # Find attention QKVO params and scale them
        attn_params_A = _get_attn_weight_params(A, arch)
        attn_params_B = _get_attn_weight_params(B, arch)

        for name in attn_params_A:
            param_src = src.state_dict()[name]
            _set_param(A, name, param_src * scale_a)
        for name in attn_params_B:
            param_src = src.state_dict()[name]
            _set_param(B, name, param_src * scale_b)

        # B: zero out FFN (so it passes through)
        ffn_params_B = _get_ffn_weight_params(B, arch)
        for name in ffn_params_B:
            _set_param(B, name, torch.zeros_like(src.state_dict()[name]))

    # dst_blocks is a list reference to the same modules already in split_model, no reassignment needed
    return split_model


def _get_attn_weight_params(block, arch):
    """Get attention weight parameter names for a block."""
    params = []
    for name in block.state_dict():
        if any(k in name.lower() for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
                                              'c_attn', 'c_proj', 'qkv_proj',
                                              'query_key_value', 'attention']):
            if 'weight' in name:
                params.append(name)
    # For GPT-2 Conv1D-based attention
    if not params:
        for name in block.state_dict():
            if 'weight' in name and ('attn' in name.lower() or 'attention' in name.lower()):
                params.append(name)
    # Fallback: get all attention module params
    if not params:
        for name in block.state_dict():
            if 'weight' in name and 'attn' in name.lower():
                params.append(name)
    return params


def _get_ffn_weight_params(block, arch):
    """Get FFN weight parameter names for a block."""
    params = []
    for name in block.state_dict():
        if any(k in name.lower() for k in ['fc1', 'fc2', 'mlp', 'intermediate',
                                              'dense_4h_to_h', 'dense_h_to_4h',
                                              'dense']):
            if 'weight' in name:
                params.append(name)
    # For GPT-2
    if not params:
        for name in block.state_dict():
            if 'weight' in name and ('mlp' in name.lower() or 'ffn' in name.lower()):
                params.append(name)
    return params


def _set_param(block, name, tensor):
    """Set a parameter in a block's state dict."""
    sd = block.state_dict()
    sd[name].copy_(tensor)


def build_3plus1_compression(teacher_2N, student_N, gamma=0.0):
    """Shifted 3+1 compression: maps 2N teacher layers to N student layers.

    For each student position j (0..N-1):
        teacher indices: p0=2j, p1=2j+1, p2=2j+2, p3=2j+3
        student[j] = teacher[p1] (dominant copy)
        + gamma * (avg(p0, p2, p3) - teacher[p1])  # shifted window blend
    """
    arch = detect_arch(student_N)
    n = get_num_layers(student_N)
    t_blocks = get_blocks(teacher_2N)
    s_blocks = get_blocks(student_N)

    for j in range(n):
        p1_idx = 2 * j + 1  # dominant source

        # Get neighbors (shifted window)
        p0_idx = max(0, p1_idx - 1)
        p2_idx = min(len(t_blocks) - 1, p1_idx + 1)
        p3_idx = min(len(t_blocks) - 1, p1_idx + 2)

        p0 = t_blocks[p0_idx]
        p1 = t_blocks[p1_idx]
        p2 = t_blocks[p2_idx]
        p3 = t_blocks[p3_idx]

        # Compute blended state dict
        p1_sd = p1.state_dict()
        p0_sd = p0.state_dict()
        p2_sd = p2.state_dict()
        p3_sd = p3.state_dict()

        new_sd = OrderedDict()
        for k in p1_sd:
            if 'weight' in k or 'bias' in k:
                avg_other = (p0_sd[k].float() + p2_sd[k].float() + p3_sd[k].float()) / 3.0
                new_sd[k] = p1_sd[k].float() + gamma * (avg_other - p1_sd[k].float())
            else:
                new_sd[k] = p1_sd[k].clone()

        s_blocks[j].load_state_dict(new_sd)

    return student_N


def run_3plus1_attack(source_path, out_dir, alpha=0.10, gamma=0.0,
                       rounds=30, checkpoints=None):
    """Run iterative 3+1 attack: for each round, alph-split then 3+1 compress."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load source
    source = AutoModelForCausalLM.from_pretrained(source_path).to(device)
    source.eval()
    tokenizer = AutoTokenizer.from_pretrained(source_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    arch = detect_arch(source)
    n_layers = get_num_layers(source)
    print(f"Source: {source_path}")
    print(f"  Architecture: {arch}, layers: {n_layers}")
    print(f"  Output: {out_dir}")
    print(f"  Alpha: {alpha}, Gamma: {gamma}, Rounds: {rounds}")

    if checkpoints is None:
        checkpoints = []

    os.makedirs(out_dir, exist_ok=True)

    # Save source as r00
    r00_dir = os.path.join(out_dir, 'r00')
    if not os.path.exists(r00_dir):
        save_checkpoint(source, tokenizer, r00_dir)
        print(f"  r00: saved")

    # Build alpha split models
    model_24 = None  # 2N-layer intermediate
    build_alph_split._alpha = alpha
    model_12 = copy.deepcopy(source)  # N-layer current

    total_start = time.time()

    for r in range(1, rounds + 1):
        round_start = time.time()

        # Step 1: Alpha-split N → 2N
        model_24 = build_alph_split(model_12)
        model_24.to(device)

        # Step 2: 3+1 compress 2N → N
        new_model = create_empty_model(source.config)
        new_model.to(device)
        copy_embeddings_and_norm(model_12, new_model)

        new_model = build_3plus1_compression(model_24, new_model, gamma=gamma)

        # Replace
        del model_24
        del model_12
        torch.cuda.empty_cache()

        model_12 = new_model
        model_12.eval()

        elapsed = time.time() - round_start
        rname = f'r{r:02d}'

        # Save checkpoint if requested
        if rname in checkpoints or str(r) in checkpoints:
            save_dir = os.path.join(out_dir, rname)
            save_checkpoint(model_12, tokenizer, save_dir)
            print(f"  {rname}: saved ({elapsed:.1f}s)")
        elif r % 4 == 0:
            # Save every 4th round for visibility
            save_dir = os.path.join(out_dir, rname)
            save_checkpoint(model_12, tokenizer, save_dir)
            print(f"  {rname}: saved ({elapsed:.1f}s)")
        else:
            print(f"  {rname}: {elapsed:.1f}s (not saved)")

    total_elapsed = time.time() - total_start
    print(f"  Total: {total_elapsed/60:.1f} min")

    # Clean up GPU
    del model_12
    torch.cuda.empty_cache()

    print("Done.")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='Source model path')
    ap.add_argument('--out_dir', required=True, help='Output directory for checkpoints')
    ap.add_argument('--alpha', type=float, default=0.10, help='Alpha attention split (default: 0.10)')
    ap.add_argument('--gamma', type=float, default=0.0, help='Gamma blend (default: 0.0)')
    ap.add_argument('--rounds', type=int, default=30, help='Number of rounds (default: 30)')
    ap.add_argument('--checkpoints', nargs='*', default=None,
                    help='Checkpoint round names to save (e.g. r12 r16 r20 r24 r30). If none, saves every 4th.')
    ap.add_argument('--start_round', type=int, default=1, help='Starting round number')
    ap.add_argument('--resume_from', default=None, help='Resume from a previous checkpoint (path to rXX)')
    ap.add_argument('--from_iter', type=int, default=0, help='Continue from this iteration (if resuming from saved checkpoint)')
    args = ap.parse_args()

    # Determine which checkpoints to save
    if args.checkpoints:
        checkpoints = args.checkpoints
    else:
        checkpoints = [f'r{i:02d}' for i in range(4, args.rounds + 1, 4)]

    start_round = args.start_round
    source_path = args.source
    current_model = None

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from: {args.resume_from} at round {args.from_iter + 1}")
        from transformers import AutoModelForCausalLM
        current_model = AutoModelForCausalLM.from_pretrained(args.resume_from)
        start_round = args.from_iter + 1
        source_path = args.source  # source still used for tokenizer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_model = current_model.to(device)
        current_model.eval()

    run_3plus1_attack(
        source_path=source_path,
        out_dir=args.out_dir,
        alpha=args.alpha,
        gamma=args.gamma,
        rounds=args.rounds,
        checkpoints=checkpoints
    )


if __name__ == '__main__':
    main()
