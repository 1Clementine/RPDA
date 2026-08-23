#!/usr/bin/env python3
"""Extend RPDA trajectory from a checkpoint by N additional rounds using same alpha/gamma config."""
import argparse, os, shutil, time
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel
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
    ap.add_argument('--start_checkpoint', required=True, help='r30 checkpoint path')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--gamma_schedule', default='0.01,0.01,0.005,0.005,0.0,0.0')
    ap.add_argument('--n_rounds', type=int, default=5)
    ap.add_argument('--start_round', type=int, default=30, help='Round number of start_checkpoint')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    gamma = [float(x.strip()) for x in args.gamma_schedule.split(',')]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Start: {args.start_checkpoint} (r{args.start_round}) | Rounds: {args.n_rounds} | Alpha: {args.alpha} | Gamma: {gamma}")

    current_path = args.start_checkpoint
    t0 = time.time()

    for step in range(1, args.n_rounds + 1):
        rt = time.time()
        rname = f'r{args.start_round + step:02d}'
        print(f"\n[{rname}] Loading {current_path} ...")
        model_12 = GPT2LMHeadModel.from_pretrained(current_path).to(device).eval()

        model_24 = alpha_split(model_12, args.alpha)
        del model_12; torch.cuda.empty_cache()

        new_12 = gamma_compress(model_24, gamma)
        del model_24; torch.cuda.empty_cache()

        round_dir = os.path.join(args.out_dir, rname)
        save_model(new_12, args.start_checkpoint, round_dir)
        print(f"  {rname}: saved ({time.time()-rt:.1f}s)")

        current_path = round_dir
        del new_12; torch.cuda.empty_cache()

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    for d in sorted(Path(args.out_dir).glob('r*')):
        print(f"  {d.name}: {d}")


if __name__ == '__main__':
    main()
