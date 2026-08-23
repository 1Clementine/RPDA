#!/usr/bin/env python3
"""Rebuild Pythia-410M every4 trajectory with L20 as REEF standard layer.
every4: layers [0,4,8,12,16,20], alpha=0.10, gamma=0.25, single-shot per round.
Recovery: 200 steps, lr=3e-5, last4 (L20-L23), KL distillation, no temperature.
Saves all checkpoints to runs/pythia_every4_L20/.
"""
import copy, math, os, sys, json, shutil
from pathlib import Path
from collections import OrderedDict
import torch, torch.nn.functional as F
import numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / 'cross_family_full'))
from model_adapters import detect_arch, get_num_layers, get_blocks, copy_embeddings_and_norm, create_empty_model
from build_cross_family_3plus1 import _get_attn_weight_params, _get_ffn_weight_params, _set_param
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM

DEVICE = 'cuda'
PYTHIA = 'models/external/pythia-410m'
OUT = 'runs/pythia_every4_L20'
RESULTS = 'results/pythia_every4_L20'
os.makedirs(OUT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

ALPHA = 0.10
GAMMA = 0.25
EVERY4 = [0, 4, 8, 12, 16, 20]

# ── Data ──
TOKENIZER_FILES = ["tokenizer.json", "vocab.json", "merges.txt",
                   "tokenizer_config.json", "special_tokens_map.json"]

def copy_tokenizer_files(src_dir, dst_dir):
    for name in TOKENIZER_FILES:
        src = Path(src_dir) / name
        if src.exists():
            shutil.copy2(str(src), str(Path(dst_dir) / name))

def save_model(model, tokenizer_src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    copy_tokenizer_files(tokenizer_src, out_dir)

def load_blocks(tokenizer, path, n=80):
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    ids = tokenizer(text, add_special_tokens=False).input_ids
    blocks = []
    for s in range(0, max(0, len(ids)-257), 256):
        b = ids[s:s+257]
        if len(b) == 257: blocks.append((torch.tensor(b[:-1]), torch.tensor(b[1:])))
        if len(blocks) >= n: break
    return blocks


class WikiDataset(Dataset):
    def __init__(self, tok, path, blk=256):
        t = Path(path).read_text(encoding='utf-8', errors='ignore')
        ids = tok(t, add_special_tokens=False).input_ids
        self.blocks = [ids[s:s+blk+1] for s in range(0, max(0, len(ids)-blk), blk)][:500]
    def __len__(self): return len(self.blocks)
    def __getitem__(self, i):
        b = self.blocks[i]
        return torch.tensor(b[:-1], dtype=torch.long), torch.tensor(b[1:], dtype=torch.long)


# ── RPDA Attack ──
def attack_one_round_scoped(model, alpha, gamma, perturb_layers, device='cuda'):
    """One round of layer-scope RPDA on Pythia. Only perturb specified layers."""
    c2 = copy.deepcopy(model).to(device)
    arch = detect_arch(c2); n = get_num_layers(c2)
    cfg = copy.deepcopy(c2.config); dbl = 2*n
    nc = type(cfg)(**{k: (v if k != 'num_hidden_layers' else dbl)
                      for k, v in cfg.to_dict().items()})
    split = type(c2)(nc)
    copy_embeddings_and_norm(c2, split)
    sb = get_blocks(c2); db = get_blocks(split)
    ps = set(perturb_layers)
    for i in range(n):
        s = sb[i]; A = db[2*i]; B = db[2*i+1]
        if i in ps:
            A.load_state_dict(s.state_dict()); B.load_state_dict(s.state_dict())
            for nm in _get_attn_weight_params(A, arch):
                _set_param(A, nm, s.state_dict()[nm] * math.sqrt(alpha))
            for nm in _get_attn_weight_params(B, arch):
                _set_param(B, nm, s.state_dict()[nm] * math.sqrt(1.0-alpha))
            for nm in _get_ffn_weight_params(A, arch):
                _set_param(A, nm, s.state_dict()[nm] * math.sqrt(gamma))
            for nm in _get_ffn_weight_params(B, arch):
                _set_param(B, nm, s.state_dict()[nm] * math.sqrt(1.0-gamma))
        else:
            A.load_state_dict(s.state_dict()); B.load_state_dict(s.state_dict())
    split.to(device)
    new12 = create_empty_model(c2.config).to(device)
    copy_embeddings_and_norm(c2, new12)
    tb = get_blocks(split); sb2 = get_blocks(new12); nt = len(tb); nn = len(sb2)
    for j in range(nn):
        p1_idx = 2*j + 1
        if p1_idx >= nt: break
        p1 = tb[p1_idx]; p1_sd = p1.state_dict()
        nbrs = []
        for off in [-1, 1, 2]:
            idx = max(0, min(nt-1, p1_idx+off))
            if idx != p1_idx: nbrs.append(tb[idx].state_dict())
        nsd = OrderedDict()
        for k in p1_sd:
            if ('weight' in k or 'bias' in k) and nbrs:
                avg_sd = sum(sd[k].float() for sd in nbrs) / len(nbrs)
                nsd[k] = p1_sd[k].float()
            else:
                nsd[k] = p1_sd[k].clone()
        sb2[j].load_state_dict(nsd)
    del split, c2; torch.cuda.empty_cache()
    return new12.eval()


# ── Recovery ──
def recovery_kl_distill(student, teacher, tokenizer, train_path, steps=200, lr=3e-5, last_n=4, device='cuda'):
    """Output-side KL distillation for Pythia, matching original e4 protocol."""
    student = copy.deepcopy(student)
    blocks_h = get_blocks(student); n = len(blocks_h)
    for p in student.parameters(): p.requires_grad = False
    trainables = []
    for i in range(max(0, n-last_n), n):
        for _, p in blocks_h[i].named_parameters():
            p.requires_grad = True; trainables.append(p)
    train_blocks = load_blocks(tokenizer, train_path, 80)
    opt = torch.optim.AdamW(trainables, lr=lr)
    student.train(); teacher.eval()
    for step in range(1, steps+1):
        tk = 0.0; nb = 0
        for inp, lbl in train_blocks:
            inp, lbl = inp.unsqueeze(0).to(device), lbl.unsqueeze(0).to(device)
            with torch.no_grad(): t_out = teacher(input_ids=inp).logits
            s_out = student(input_ids=inp).logits
            loss = F.kl_div(F.log_softmax(s_out, -1), F.softmax(t_out, -1), reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
            tk += loss.item(); nb += 1
    student.eval()
    return student


# ── REEF (fixed L20) ──
def linear_cka_hsic(X, Y, eps=1e-8):
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    sx = X.std(axis=0, ddof=0); sy = Y.std(axis=0, ddof=0)
    X = X / (sx + eps); Y = Y / (sy + eps)
    XtY = X.T @ Y; XtX = X.T @ X; YtY = Y.T @ Y
    return float(np.sum(XtY**2) / (math.sqrt(np.sum(XtX**2) * np.sum(YtY**2)) + eps))


@torch.no_grad()
def extract_hidden_states(model, tokenizer, texts, layers, device, max_length=512):
    """Hook-based last-token extraction with left-padding (matching REEF protocol)."""
    blocks = get_blocks(model)
    model.eval()
    tokenizer.padding_side = 'left'
    cache = {l: [] for l in layers}
    handles = []
    def make_hook(lid):
        def hook(module, inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            cache[lid].append(out[:, -1, :].detach().cpu())
        return hook
    for l in layers:
        handles.append(blocks[l].register_forward_hook(make_hook(l)))
    bs = 16
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        inp = tokenizer(batch, return_tensors='pt', padding=True,
                       truncation=True, max_length=max_length).to(device)
        _ = model(**inp)
    for h in handles: h.remove()
    return {l: torch.cat(cache[l], dim=0).float().numpy() for l in layers}


def compute_reef_fixed_L20(src_reps, cand_reps):
    """Fixed L20 CKA."""
    return linear_cka_hsic(src_reps[20], cand_reps[20])


# ── ED-BA ──
def load_tqa_binary(jsonl_path):
    items = []
    with open(jsonl_path) as f:
        for line in f:
            items.append(json.loads(line))
    items.sort(key=lambda x: x['id'])
    train = items[:2542]; test = items[2542:2542+636]
    return ([x['text'] for x in train], [x['label'] for x in train],
            [x['text'] for x in test], [x['label'] for x in test])

TQA_BIN = 'archive/project_legacy_20260608_104625/raw_models_and_runs/runs/stage4_baselines/easydetector_lite_v0/data/truthfulqa_binary.jsonl'

def compute_edba(src_m, cand_m, tokenizer, device):
    """ED-BA on midlate layers [14,18,22] (Pythia 24L protocol). Single seed=0."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    tr_t, tr_l, te_t, te_l = load_tqa_binary(TQA_BIN)
    midlate = [14, 18, 22]
    src_reps = extract_hidden_states(src_m, tokenizer, tr_t+te_t, midlate, device)
    X = np.concatenate([src_reps[l] for l in midlate], axis=1)
    scaler = StandardScaler().fit(X[:len(tr_t)])
    clf = LogisticRegression(max_iter=2000, random_state=0, class_weight='balanced')
    clf.fit(scaler.transform(X[:len(tr_t)]), tr_l)
    cand_reps = extract_hidden_states(cand_m, tokenizer, te_t, midlate, device)
    cc = np.concatenate([cand_reps[l] for l in midlate], axis=1)
    return float(balanced_accuracy_score(te_l, clf.predict(scaler.transform(cc))))


# ── BPB/KL ──
WIKI_TEST = 'data/formal_recovery_wikitext2/test.txt'

def compute_bpb_kl(model, ref_model, tokenizer, device):
    blocks = load_blocks(tokenizer, WIKI_TEST, 30)
    bpb_sum = 0.0; ref_bpb_sum = 0.0; kl_sum = 0.0; total_tokens = 0
    for inp, lbl in blocks[:20]:
        inp, lbl = inp.unsqueeze(0).to(device), lbl.unsqueeze(0).to(device)
        s_out = model(input_ids=inp).logits
        r_out = ref_model(input_ids=inp).logits
        bpb_sum += F.cross_entropy(s_out.view(-1, s_out.shape[-1]), lbl.view(-1), reduction='sum').item()
        ref_bpb_sum += F.cross_entropy(r_out.view(-1, r_out.shape[-1]), lbl.view(-1), reduction='sum').item()
        kl_sum += F.kl_div(F.log_softmax(s_out.view(-1, s_out.shape[-1]), -1),
                          F.softmax(r_out.view(-1, r_out.shape[-1]), -1), reduction='sum').item()
        total_tokens += lbl.numel()
    return bpb_sum / max(ref_bpb_sum, 1), kl_sum / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Pythia-410M every4: layers={EVERY4}, alpha={ALPHA}, gamma={GAMMA}")
    print(f"REEF: fixed L20 | ED-BA: [14,18,22] | BPB/KL: WikiText-2")
    print()

    # Load source
    print("[Source] Loading Pythia-410M...")
    src_m = AutoModelForCausalLM.from_pretrained(PYTHIA, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(PYTHIA, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    n_layers = get_num_layers(src_m)
    print(f"  Layers: {n_layers}, Arch: {detect_arch(src_m)}")

    # Extract source representations for REEF
    tqa_texts = pd.read_csv('data/truthfulqa.csv')['statement'].tolist()[:200]
    print(f"  TQA200 samples: {len(tqa_texts)}")

    # Save source as r00
    src_dir = os.path.join(OUT, 'r00')
    if not os.path.exists(os.path.join(src_dir, 'config.json')):
        save_model(src_m, PYTHIA, src_dir)
        print(f"  r00 saved")
    else:
        print(f"  r00 exists")

    # ── Build r01 ──
    r01_dir = os.path.join(OUT, 'r01')
    if not os.path.exists(os.path.join(r01_dir, 'config.json')):
        print("\n[Build] r01...")
        r01_m = attack_one_round_scoped(src_m, ALPHA, GAMMA, EVERY4, device)
        save_model(r01_m, PYTHIA, r01_dir)
        print(f"  r01 saved")
    else:
        print(f"\n  r01 exists")

    # ── Build r02 ──
    r02_dir = os.path.join(OUT, 'r02')
    if not os.path.exists(os.path.join(r02_dir, 'config.json')):
        print("\n[Build] r02...")
        r01_loaded = AutoModelForCausalLM.from_pretrained(r01_dir, trust_remote_code=True).to(device).eval()
        r02_m = attack_one_round_scoped(r01_loaded, ALPHA, GAMMA, EVERY4, device)
        del r01_loaded; torch.cuda.empty_cache()
        save_model(r02_m, PYTHIA, r02_dir)
        print(f"  r02 saved")
    else:
        print(f"\n  r02 exists")

    # ── Build r02_rec ──
    r02_rec_dir = os.path.join(OUT, 'r02_rec')
    if not os.path.exists(os.path.join(r02_rec_dir, 'config.json')):
        print("\n[Recovery] r02_rec (200 steps, lr=3e-5, last4)...")
        r02_loaded = AutoModelForCausalLM.from_pretrained(r02_dir, trust_remote_code=True).to(device).eval()
        teacher = AutoModelForCausalLM.from_pretrained(PYTHIA, trust_remote_code=True).to(device).eval()
        WIKI_TRAIN = 'data/formal_recovery_wikitext2/train.txt'
        r02_rec_m = recovery_kl_distill(r02_loaded, teacher, tokenizer, WIKI_TRAIN,
                                         steps=200, lr=3e-5, last_n=4, device=device)
        del r02_loaded, teacher; torch.cuda.empty_cache()
        save_model(r02_rec_m, PYTHIA, r02_rec_dir)
        print(f"  r02_rec saved")
    else:
        print(f"\n  r02_rec exists")

    # ── Evaluate ──
    print("\n" + "="*70)
    print("EVALUATION (REEF fixed L20)")
    print("="*70)

    src_reps = extract_hidden_states(src_m, tokenizer, tqa_texts, list(range(24)), device)

    models = {
        'Pythia_Source': src_m,
        'Pythia_every4_r01': AutoModelForCausalLM.from_pretrained(r01_dir, trust_remote_code=True).to(device).eval(),
        'Pythia_every4_r02': AutoModelForCausalLM.from_pretrained(r02_dir, trust_remote_code=True).to(device).eval(),
        'Pythia_every4_r02_rec': AutoModelForCausalLM.from_pretrained(r02_rec_dir, trust_remote_code=True).to(device).eval(),
    }

    rows = []
    for name, m in models.items():
        print(f"\n[{name}]")
        # REEF L20
        cand_reps = extract_hidden_states(m, tokenizer, tqa_texts, [20], device)
        reef = linear_cka_hsic(src_reps[20], cand_reps[20])
        print(f"  REEF L20 = {reef:.6f}")

        # BPB/KL
        bpb, kl = compute_bpb_kl(m, src_m, tokenizer, device)
        print(f"  BPB ratio = {bpb:.6f}, KL = {kl:.6f}")

        # ED-BA
        edba = compute_edba(src_m, m, tokenizer, device)
        print(f"  ED-BA = {edba:.4f}")

        rows.append({
            'checkpoint': name,
            'checkpoint_path': '',
            'round': 0 if 'Source' in name else (1 if 'r01' in name else 2),
            'REEF_L20': round(reef, 6),
            'BPB_ratio': round(bpb, 6),
            'KL': round(kl, 6),
            'ED_BA': round(edba, 4),
            'reef_protocol': 'reef_fixed_L20_max512_HSIC_v1',
            'output_protocol': 'pythia_output_wikitext2_v1',
            'edba_layers': '[14,18,22]',
            'status': 'frozen_20260731',
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS, 'pythia_every4_L20.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")
    print(df[['checkpoint','REEF_L20','BPB_ratio','KL','ED_BA']].to_string(index=False))

    print(f"\nCheckpoints saved to: {OUT}/")
    for d in sorted(Path(OUT).glob('r*')):
        print(f"  {d.name}")


if __name__ == '__main__':
    main()
