#!/usr/bin/env python3
"""Formal output recovery v2: KL + CE fine-tuning on WikiText-2 with train/val split.
Supports last4/last6/last8 scopes. Saves val_BPB, val_KL, val_CE per eval step.
"""
import argparse, json, math, os, random, gc, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_text_blocks(tokenizer, text_file, block_size, max_blocks):
    text = Path(text_file).read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer(text, add_special_tokens=False).input_ids
    blocks = []
    for start in range(0, max(0, len(ids) - block_size - 1), block_size):
        block = ids[start:start + block_size + 1]
        if len(block) == block_size + 1:
            blocks.append(torch.tensor(block, dtype=torch.long))
        if max_blocks > 0 and len(blocks) >= max_blocks:
            break
    if not blocks:
        raise RuntimeError(f"No blocks from: {text_file}")
    return blocks


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_scope(model, scope):
    """Unfreeze top N transformer layers + ln_f. Embeddings and lm_head stay frozen."""
    n = len(model.transformer.h)
    if scope == "last4": start = n - 4
    elif scope == "last6": start = n - 6
    elif scope == "last8": start = n - 8
    elif scope == "all": start = 0
    else: raise ValueError(f"Unknown scope: {scope}")
    start = max(0, start)
    for i in range(start, n):
        for p in model.transformer.h[i].parameters():
            p.requires_grad = True
    if hasattr(model.transformer, "ln_f"):
        for p in model.transformer.ln_f.parameters():
            p.requires_grad = True
    # Verify lm_head and embeddings frozen
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = False
    if hasattr(model.transformer, "wte"):
        for p in model.transformer.wte.parameters():
            p.requires_grad = False
    if hasattr(model.transformer, "wpe"):
        for p in model.transformer.wpe.parameters():
            p.requires_grad = False


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def validate(model, teacher, val_blocks, device, temperature, kl_weight, ce_weight, num_bytes=None):
    """Compute val_KL, val_CE, val_BPB."""
    model.eval(); teacher.eval()
    total_kl, total_ce, total_loss, n_batches = 0.0, 0.0, 0.0, 0

    gpt2_bpb = None  # base BPB for ratio
    with torch.no_grad():
        for block in val_blocks:
            x = block.unsqueeze(0).to(device)
            t_logits = teacher(input_ids=x).logits[:, :-1, :].float()
            s_logits = model(input_ids=x).logits[:, :-1, :].float()
            labels = x[:, 1:]

            # KL
            log_s = F.log_softmax(s_logits / temperature, dim=-1)
            p_t = F.softmax(t_logits / temperature, dim=-1)
            kl = F.kl_div(log_s, p_t, reduction="batchmean") * (temperature ** 2)

            # CE
            ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), labels.reshape(-1))

            total_kl += kl.item()
            total_ce += ce.item()
            total_loss += (kl_weight * kl.item() + ce_weight * ce.item())
            n_batches += 1

    model.train()
    avg_kl = total_kl / max(n_batches, 1)
    avg_ce = total_ce / max(n_batches, 1)
    avg_loss = total_loss / max(n_batches, 1)

    # BPB estimate: use CE for BPB
    val_bpb = avg_ce * math.log2(math.e) if num_bytes else None
    return avg_kl, avg_ce, avg_loss, val_bpb


def sample_batch(blocks, batch_size):
    batch = random.choices(blocks, k=batch_size)
    return torch.stack(batch, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--train_scope", choices=["last4", "last6", "last8", "all"], default="last6")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--max_blocks", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--kl_weight", type=float, default=1.0)
    ap.add_argument("--ce_weight", type=float, default=0.10)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Save config
    config = vars(args)
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load models
    print(f"[Teacher] {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, ).to(device)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    print(f"[Student] {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(args.student_model, ).to(device)
    student.train()

    freeze_all(student)
    unfreeze_scope(student, args.train_scope)
    n_trainable, n_total = count_params(student)
    print(f"[Params] trainable={n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    # Data
    train_blocks = load_text_blocks(tok, args.train_file, args.block_size, args.max_blocks)
    val_blocks = load_text_blocks(tok, args.val_file, args.block_size, max(100, args.max_blocks//4))
    print(f"[Data] train={len(train_blocks)} val={len(val_blocks)}")

    # Trainable params
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    opt.zero_grad(set_to_none=True)

    T = args.temperature
    metrics_log = []
    best_val_loss = float("inf")
    best_step = 0
    patience = 0

    # GPT-2 reference BPB (computed once)
    gpt2_bpb = None

    for step in range(1, args.max_steps + 1):
        # Train
        x = sample_batch(train_blocks, args.batch_size).to(device)
        with torch.no_grad():
            t_logits = teacher(input_ids=x).logits[:, :-1, :].float()
        s_logits = student(input_ids=x).logits[:, :-1, :].float()
        labels = x[:, 1:]

        log_s = F.log_softmax(s_logits / T, dim=-1)
        p_t = F.softmax(t_logits / T, dim=-1)
        kl = F.kl_div(log_s, p_t, reduction="batchmean") * (T * T)
        ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), labels.reshape(-1))
        loss = (args.kl_weight * kl + args.ce_weight * ce) / args.grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] NaN/Inf at step {step}, skipping")
            opt.zero_grad(set_to_none=True); continue

        loss.backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

        # Eval
        if step % args.eval_every == 0 or step == args.max_steps:
            val_kl, val_ce, val_loss, val_bpb = validate(
                student, teacher, val_blocks, device, T, args.kl_weight, args.ce_weight)
            entry = {
                "step": step, "train_kl": float(kl.item()), "train_ce": float(ce.item()),
                "val_kl": val_kl, "val_ce": val_ce, "val_loss": val_loss, "val_bpb": val_bpb,
            }
            metrics_log.append(entry)
            student.train()
            print(f"[step {step:04d}] train_kl={kl.item():.4f} train_ce={ce.item():.4f} val_kl={val_kl:.4f} val_ce={val_ce:.4f} val_bpb={val_bpb:.4f}" if val_bpb else "")

            if val_loss < best_val_loss:
                best_val_loss = val_loss; best_step = step; patience = 0
                # Save best
                ckpt_dir = out_dir / "best_checkpoint"
                ckpt_dir.mkdir(exist_ok=True)
                student.save_pretrained(str(ckpt_dir))
                tok.save_pretrained(str(ckpt_dir))
                print(f"  -> NEW BEST (val_loss={best_val_loss:.4f})")
            else:
                patience += 1

        # Periodic save
        if step % args.save_every == 0:
            ckpt_dir = out_dir / f"checkpoint-{step:04d}"
            ckpt_dir.mkdir(exist_ok=True)
            student.save_pretrained(str(ckpt_dir))
            tok.save_pretrained(str(ckpt_dir))

        if patience >= args.early_stopping_patience:
            print(f"[Early stop] {patience} evals without improvement at step {step}")
            break

    # Save metrics
    with open(out_dir / "recovery_metrics.jsonl", "w") as f:
        for entry in metrics_log:
            f.write(json.dumps(entry) + "\n")

    best_info = {"best_step": best_step, "best_val_loss": best_val_loss, "best_checkpoint": str(out_dir / "best_checkpoint")}
    with open(out_dir / "best_checkpoint_info.json", "w") as f:
        json.dump(best_info, f, indent=2)

    # README
    with open(out_dir / "README.txt", "w") as f:
        f.write(f"teacher: {args.teacher_model}\nstudent: {args.student_model}\n")
        f.write(f"scope: {args.train_scope}\nsteps: {args.max_steps}\nlr: {args.lr}\n")
        f.write(f"kl_weight: {args.kl_weight}\nce_weight: {args.ce_weight}\nT: {args.temperature}\n")
        f.write(f"best_step: {best_step}\nbest_val_loss: {best_val_loss}\n")

    print(f"\n[Done] Best: step={best_step} val_loss={best_val_loss:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
