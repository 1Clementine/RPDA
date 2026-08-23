#!/usr/bin/env python3
"""Build recovery-every-4 trajectory: r16-rec, r17-r19, r20-rec, r21-r23, ..., r60-rec.
Attack uses same 3+1 alpha-split + gamma-correct compression.
Recovery uses same formal_output_recovery_v2 config (last8, lr=5e-5, 1000 steps, KL).
"""
import argparse, os, shutil, subprocess, sys, time
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

def attack_one_round(model_path, out_dir, alpha, gamma, tokenizer_src, device_str):
    device = torch.device(device_str)
    model_12 = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
    model_24 = alpha_split(model_12, alpha)
    del model_12; torch.cuda.empty_cache()
    new_12 = gamma_compress(model_24, gamma)
    del model_24; torch.cuda.empty_cache()
    save_model(new_12, tokenizer_src, out_dir)
    del new_12; torch.cuda.empty_cache()

def run_recovery(student_path, out_dir, teacher, device_str):
    """Run formal_output_recovery_v2.py via subprocess."""
    cmd = [
        sys.executable, "scripts_final/formal_output_recovery_v2.py",
        "--teacher_model", teacher,
        "--student_model", student_path,
        "--out_dir", out_dir,
        "--train_file", "data/formal_recovery_wikitext2/train.txt",
        "--val_file", "data/formal_recovery_wikitext2/test.txt",
        "--train_scope", "last8",
        "--max_steps", "1000",
        "--lr", "5e-5",
        "--block_size", "256",
        "--batch_size", "1",
        "--grad_accum", "16",
        "--kl_weight", "1.0",
        "--ce_weight", "0.0",
        "--eval_every", "1000",
        "--save_every", "1000",
        "--device", device_str,
    ]
    print(f"  [RECOVERY] {Path(student_path).name} -> {Path(out_dir).name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [RECOVERY ERROR] {result.stderr[-300:]}")
        raise RuntimeError(f"Recovery failed for {student_path}")
    # best checkpoint is in out_dir/best_checkpoint
    return os.path.join(out_dir, "best_checkpoint")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start_checkpoint', required=True, help='r16 checkpoint from pure trajectory')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--teacher', required=True, help='Source/teacher model for recovery')
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--gamma_schedule', default='0.01,0.01,0.005,0.005,0.0,0.0')
    ap.add_argument('--max_round', type=int, default=60)
    ap.add_argument('--recovery_interval', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    gamma = [float(x.strip()) for x in args.gamma_schedule.split(',')]
    tokenizer_src = args.teacher
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    current_path = args.start_checkpoint
    current_round = 16

    while current_round <= args.max_round:
        rt = time.time()

        # Apply recovery at current round
        # Save pre-recovery checkpoint first
        pre_dir = os.path.join(args.out_dir, f"r{current_round:02d}-pre")
        if not os.path.exists(os.path.join(pre_dir, "config.json")):
            print(f"\n[r{current_round:02d}-pre] Saving pre-recovery checkpoint from {Path(current_path).name}")
            import shutil
            shutil.copytree(current_path, pre_dir, symlinks=True)
        else:
            print(f"\n[r{current_round:02d}-pre] EXISTS")

        rec_dir = os.path.join(args.out_dir, f"r{current_round:02d}-rec")
        if os.path.exists(os.path.join(rec_dir, "best_checkpoint", "config.json")):
            print(f"[r{current_round:02d}-rec] EXISTS, skipping recovery")
            current_path = os.path.join(rec_dir, "best_checkpoint")
        else:
            print(f"[r{current_round:02d}-rec] Recovery from {Path(current_path).name}...")
            current_path = run_recovery(current_path, rec_dir, args.teacher, args.device)
            print(f"  Done ({time.time()-rt:.1f}s)")

        # Attack 4 rounds, reaching the next recovery round
        next_recovery_round = min(current_round + args.recovery_interval, args.max_round)
        for r in range(current_round + 1, next_recovery_round + 1):
            rt2 = time.time()
            rdir = os.path.join(args.out_dir, f"r{r:02d}")
            if os.path.exists(os.path.join(rdir, "config.json")):
                print(f"  r{r:02d}: EXISTS")
                current_path = rdir
            else:
                attack_one_round(current_path, rdir, args.alpha, gamma, tokenizer_src, args.device)
                current_path = rdir
                print(f"  r{r:02d}: saved ({time.time()-rt2:.1f}s)")

        current_round = next_recovery_round

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    for d in sorted(Path(args.out_dir).glob('r*')):
        print(f"  {d.name}")


if __name__ == '__main__':
    main()
