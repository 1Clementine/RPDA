#!/usr/bin/env python3
"""Compute BPB (bits-per-byte) and KL divergence for GPT-2 checkpoints on WikiText-2.
BPB = cross_entropy / log(2). KL computed against source model.
Matches twostage_wikitext_25600_continuous_v1 protocol.
"""
import argparse, math, os, sys
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def compute_bpb_kl(model_path, source_model, source_logits_cache, tok, device, blocks):
    """Compute BPB and KL for one checkpoint against pre-computed source logits."""
    # Load candidate model
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()

    total_nll = 0.0
    total_kl = 0.0
    total_tokens = 0

    for idx, blk in enumerate(blocks):
        blk = blk.to(device)
        inp = blk[:-1].unsqueeze(0)
        lbl = blk[1:].unsqueeze(0)

        lo = model(input_ids=inp).logits.float()

        # NLL for BPB
        nll = F.cross_entropy(lo.view(-1, lo.shape[-1]), lbl.view(-1), reduction='sum').item()
        total_nll += nll
        total_tokens += lbl.numel()

        # KL against source
        sl = source_logits_cache[idx].to(device)
        kl = F.kl_div(
            F.log_softmax(lo.view(-1, lo.shape[-1]), -1),
            F.softmax(sl.view(-1, sl.shape[-1]), -1),
            reduction='sum'
        ).item()
        total_kl += kl

    bpb = total_nll / max(total_tokens, 1) / math.log(2)
    kl = total_kl / max(total_tokens, 1)

    del model; torch.cuda.empty_cache()
    return bpb, kl, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_model', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_tokens', type=int, default=25600)
    ap.add_argument('--block_size', type=int, default=256)
    args = ap.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.source_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Pre-compute source logits
    print(f"[Source] {args.source_model}")
    source = AutoModelForCausalLM.from_pretrained(args.source_model).to(device).eval()

    # Use local WikiText-2 test file
    wt_path = "data/wikitext2_test_raw.txt"
    if not Path(wt_path).exists():
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([x["text"] for x in ds if x["text"].strip()])
        Path(wt_path).write_text(text, encoding="utf-8")
    else:
        text = Path(wt_path).read_text(encoding="utf-8", errors="ignore")

    ids = tok(text, add_special_tokens=False, truncation=True, max_length=1024*10).input_ids

    blocks = []
    for s in range(0, max(0, len(ids) - args.block_size - 1), args.block_size):
        b = ids[s:s+args.block_size+1]
        if len(b) == args.block_size+1:
            blocks.append(torch.tensor(b, dtype=torch.long))
        if len(blocks) >= args.max_tokens // args.block_size:
            break

    print(f"  Blocks: {len(blocks)}")

    source_logits_cache = []
    with torch.no_grad():
        for blk in blocks:
            inp = blk[:-1].unsqueeze(0).to(device)
            lo = source(input_ids=inp).logits.float().cpu()
            source_logits_cache.append(lo)

    # Source self-check
    source_bpb, source_kl, source_tokens = compute_bpb_kl(
        args.source_model, source, source_logits_cache, tok, device, blocks)
    print(f"  Source BPB: {source_bpb:.6f}, KL: {source_kl:.6f}, tokens: {source_tokens}")

    import csv
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['checkpoint', 'BPB', 'BPB_ratio', 'KL', 'tokens'])

        # Write source row
        w.writerow(['Source', source_bpb, 1.0, source_kl, source_tokens])
        print(f"  Source: BPB={source_bpb:.6f}, ratio=1.0, KL={source_kl:.6f}")

        for cp in args.checkpoints:
            print(f"[{Path(cp).name}] ", end='', flush=True)
            try:
                bpb, kl, nt = compute_bpb_kl(cp, source, source_logits_cache, tok, device, blocks)
                ratio = bpb / source_bpb if source_bpb > 0 else float('nan')
                w.writerow([cp, bpb, ratio, kl, nt])
                print(f"BPB={bpb:.6f}, ratio={ratio:.6f}, KL={kl:.6f}")
            except Exception as e:
                print(f"SKIP: {e}")
                w.writerow([cp, 'ERROR', 'ERROR', 'ERROR', 0])

    print(f"\n[Saved] {args.out_csv}")


if __name__ == '__main__':
    main()
