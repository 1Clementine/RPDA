#!/usr/bin/env python3
"""Rebuild GPT-2 original gamma-correct 3+1 trajectory.
Uses the original alpha-split + gamma-correct 3+1 compression scripts.
Parameters from stage35 final_attack_stage_summary.md:
  alpha = 0.10, gamma = [0.01, 0.01, 0.005, 0.005, 0.0, 0.0]
"""
import argparse, os, shutil, time
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from common import alpha_split, gamma_compress

TOKENIZER_FILES = [
    "tokenizer.json", "vocab.json", "merges.txt",
    "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
]

def copy_tokenizer_files(src_dir, dst_dir):
    for name in TOKENIZER_FILES:
        src = Path(src_dir) / name
        if src.exists():
            shutil.copy2(str(src), str(Path(dst_dir) / name))

def save_model(model, tokenizer_src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    copy_tokenizer_files(tokenizer_src, out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--gamma_schedule', default='0.01,0.01,0.005,0.005,0.0,0.0')
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--checkpoints', nargs='*',
                    default=['r04','r08','r12','r16','r20','r24','r30'])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    gamma = [float(x.strip()) for x in args.gamma_schedule.split(',')]
    assert len(gamma) == 6, f"Expected 6 gamma values, got {len(gamma)}"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Source: {args.source} | Alpha: {args.alpha} | Gamma: {gamma}")
    print(f"Rounds: {args.rounds} | Checkpoints: {args.checkpoints}")

    os.makedirs(args.out_dir, exist_ok=True)

    # R00: save source
    r00 = os.path.join(args.out_dir, 'r00')
    if not os.path.exists(r00):
        src_m = GPT2LMHeadModel.from_pretrained(args.source).to(device).eval()
        save_model(src_m, args.source, r00)
        del src_m; torch.cuda.empty_cache()
        print(f"  r00: saved")
    else:
        print(f"  r00: exists")

    current_path = r00
    t0 = time.time()

    for r in range(1, args.rounds + 1):
        rt = time.time()
        rname = f'r{r:02d}'

        # Load current
        model_12 = GPT2LMHeadModel.from_pretrained(current_path).to(device).eval()

        # Alpha-split: 12L -> 24L
        model_24 = alpha_split(model_12, args.alpha)
        del model_12; torch.cuda.empty_cache()

        # Gamma compress: 24L -> 12L
        new_12 = gamma_compress(model_24, gamma)
        del model_24; torch.cuda.empty_cache()

        # Save checkpoint if requested
        round_dir = os.path.join(args.out_dir, rname)
        if rname in args.checkpoints or str(r) in args.checkpoints:
            save_model(new_12, args.source, round_dir)
            print(f"  {rname}: saved ({time.time()-rt:.1f}s)")

        # Update current for next round
        if os.path.exists(round_dir):
            current_path = round_dir
        else:
            # Need a path for next round — save to temp and clean up after
            save_model(new_12, args.source, round_dir)
            current_path = round_dir

        del new_12; torch.cuda.empty_cache()

    # Clean up non-checkpoint rounds
    for r in range(1, args.rounds + 1):
        rname = f'r{r:02d}'
        if rname not in args.checkpoints and str(r) not in args.checkpoints:
            d = os.path.join(args.out_dir, rname)
            if os.path.exists(d):
                shutil.rmtree(d)

    print(f"Total: {(time.time()-t0)/60:.1f} min")

    # Print final models
    import glob
    for d in sorted(glob.glob(os.path.join(args.out_dir, 'r*'))):
        print(f"  {os.path.basename(d)}: {d}")


if __name__ == '__main__':
    main()
