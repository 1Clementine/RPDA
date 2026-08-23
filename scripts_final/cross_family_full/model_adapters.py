#!/usr/bin/env python3
"""Unified model adapter for GPT-2, Pythia/GPTNeoX, and OPT architectures.

Supports:
- GPT-2:      transformer.h, transforemr.ln_f, lm_head, transformer.wte, transformer.wpe
- GPTNeoX:     gpt_neox.layers, gpt_neox.final_layer_norm, embed_out, gpt_neox.embed_in
- OPT:         model.decoder.layers, model.decoder.final_layer_norm, lm_head
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ============================================================
# Architecture detection
# ============================================================

def detect_arch(model):
    """Return 'gpt2', 'gpt_neox', or 'opt' based on model structure."""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return 'gpt2'
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return 'gpt_neox'
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return 'opt'
    raise ValueError(f"Cannot detect architecture for {type(model).__name__}")


# ============================================================
# Block access
# ============================================================

def get_num_layers(model):
    arch = detect_arch(model)
    if arch == 'gpt2':
        return len(model.transformer.h)
    elif arch == 'gpt_neox':
        return len(model.gpt_neox.layers)
    elif arch == 'opt':
        return len(model.model.decoder.layers)


def get_blocks(model):
    arch = detect_arch(model)
    if arch == 'gpt2':
        return list(model.transformer.h)
    elif arch == 'gpt_neox':
        return list(model.gpt_neox.layers)
    elif arch == 'opt':
        return list(model.model.decoder.layers)


def set_block(model, idx, new_block):
    arch = detect_arch(model)
    if arch == 'gpt2':
        model.transformer.h[idx] = new_block
    elif arch == 'gpt_neox':
        model.gpt_neox.layers[idx] = new_block
    elif arch == 'opt':
        model.model.decoder.layers[idx] = new_block


# ============================================================
# Norm and head access
# ============================================================

def get_final_norm(model):
    arch = detect_arch(model)
    if arch == 'gpt2':
        return model.transformer.ln_f
    elif arch == 'gpt_neox':
        return model.gpt_neox.final_layer_norm
    elif arch == 'opt':
        norm = model.model.decoder.final_layer_norm
        if norm is not None:
            return norm
        # Some OPT versions have per-layer norms but no global final norm.
        # Return the last layer's final_layer_norm as an approximation.
        if hasattr(model.model.decoder, 'layers'):
            last = model.model.decoder.layers[-1]
            if hasattr(last, 'final_layer_norm'):
                return last.final_layer_norm
        return None


def set_final_norm(model, new_norm):
    arch = detect_arch(model)
    if arch == 'gpt2':
        model.transformer.ln_f = new_norm
    elif arch == 'gpt_neox':
        model.gpt_neox.final_layer_norm = new_norm
    elif arch == 'opt':
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = new_norm
        elif hasattr(model.model.decoder, 'layers'):
            model.model.decoder.layers[-1].final_layer_norm = new_norm


def get_lm_head(model):
    arch = detect_arch(model)
    if arch == 'gpt2':
        return model.lm_head
    elif arch == 'gpt_neox':
        return model.embed_out
    elif arch == 'opt':
        return model.lm_head


def get_token_embeddings(model):
    arch = detect_arch(model)
    if arch == 'gpt2':
        return model.transformer.wte
    elif arch == 'gpt_neox':
        return model.gpt_neox.embed_in
    elif arch == 'opt':
        return model.model.decoder.embed_tokens


def get_position_embeddings(model):
    arch = detect_arch(model)
    if arch == 'gpt2':
        return model.transformer.wpe
    elif arch == 'gpt_neox':
        return None  # RoPE, no learned PE
    elif arch == 'opt':
        return model.model.decoder.embed_positions


# ============================================================
# Model creation
# ============================================================

def create_config_double_layers(source_config_or_path):
    """Create a config with 2*n_layers for alpha-split intermediate model."""
    if isinstance(source_config_or_path, str):
        cfg = AutoConfig.from_pretrained(source_config_or_path)
    else:
        cfg = source_config_or_path

    new_cfg = type(cfg)(**cfg.to_dict())
    n_layers = cfg.num_hidden_layers
    new_cfg.num_hidden_layers = 2 * n_layers
    new_cfg.n_layer = 2 * n_layers  # some configs use n_layer
    return new_cfg


def create_empty_model(source_config, double_layers=False):
    """Create a model with same config. Optionally double the layers for alpha-split."""
    if isinstance(source_config, str):
        src_cfg = AutoConfig.from_pretrained(source_config)
    else:
        src_cfg = source_config

    if double_layers:
        cfg_dict = src_cfg.to_dict()
        cfg_dict['num_hidden_layers'] = 2 * cfg_dict.get('num_hidden_layers', cfg_dict.get('n_layer', 12))
        if 'n_layer' in cfg_dict:
            cfg_dict['n_layer'] = cfg_dict['num_hidden_layers']
        new_cfg = type(src_cfg)(**cfg_dict)
    else:
        new_cfg = type(src_cfg)(**src_cfg.to_dict())

    arch_class = type(AutoModelForCausalLM.from_config(new_cfg))
    model = arch_class(new_cfg)
    return model


# ============================================================
# Weight transfer
# ============================================================

def copy_embeddings_and_norm(src, dst):
    """Copy token embeddings, position embeddings (if any), and final norm from src to dst."""
    src_arch = detect_arch(src)
    dst_arch = detect_arch(dst)

    # Token embeddings
    src_wte = get_token_embeddings(src)
    dst_wte = get_token_embeddings(dst)
    if src_wte is not None and dst_wte is not None:
        dst_wte.load_state_dict(src_wte.state_dict())

    # Position embeddings (GPT-2 and OPT have learned PE; Pythia uses RoPE)
    src_wpe = get_position_embeddings(src)
    dst_wpe = get_position_embeddings(dst)
    if src_wpe is not None and dst_wpe is not None:
        dst_wpe.load_state_dict(src_wpe.state_dict())

    # Final norm
    src_norm = get_final_norm(src)
    dst_norm = get_final_norm(dst)
    if src_norm is not None and dst_norm is not None:
        dst_norm.load_state_dict(src_norm.state_dict())

    # LM head
    if hasattr(src, 'lm_head') and hasattr(dst, 'lm_head'):
        dst.lm_head.load_state_dict(src.lm_head.state_dict())
    elif src_arch == 'gpt_neox' and dst_arch == 'gpt_neox':
        src_head = get_lm_head(src)
        dst_head = get_lm_head(dst)
        if hasattr(src_head, 'state_dict') and hasattr(dst_head, 'state_dict'):
            dst_head.load_state_dict(src_head.state_dict())

    # Tie weights if applicable
    if hasattr(dst, 'config') and getattr(dst.config, 'tie_word_embeddings', False):
        if hasattr(dst, 'get_output_embeddings'):
            dst._tied_weights_keys = []


# ============================================================
# Save / Load
# ============================================================

def save_checkpoint(model, tokenizer, path):
    """Save model + tokenizer as a HuggingFace checkpoint."""
    import os
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)


def load_checkpoint(path, device='cpu'):
    """Load a model from a checkpoint path."""
    model = AutoModelForCausalLM.from_pretrained(path).to(device).eval()
    return model


# ============================================================
# Hidden state extraction for REEF
# ============================================================

def get_hidden_states(model, tokenizer, prompts, layer_indices=None, mode='last_token'):
    """Extract hidden states using model's native forward with output_hidden_states=True.
    This approach is architecture-agnostic and handles attention masks correctly per model type.
    """
    import numpy as np

    device = next(model.parameters()).device

    if layer_indices is None:
        layer_indices = list(range(get_num_layers(model)))

    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        all_hidden = outputs.hidden_states  # tuple of (n_layers+1) tensors: embedding + each layer output

    hidden_states = {}
    for i in layer_indices:
        h = all_hidden[i + 1]  # skip embedding layer (index 0)
        if mode == 'last_token':
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1) - 1
                rep = h[torch.arange(h.shape[0], device=device), seq_lens].cpu().numpy()
            else:
                rep = h[:, -1, :].cpu().numpy()
        elif mode == 'mean':
            if inputs.get('attention_mask') is not None:
                mask = inputs['attention_mask'].unsqueeze(-1).float().to(h.dtype)
                rep = (h * mask).sum(dim=1) / mask.sum(dim=1)
                rep = rep.cpu().numpy()
            else:
                rep = h.mean(dim=1).cpu().numpy()
        else:
            rep = h.cpu().numpy()
        hidden_states[i] = rep

    return hidden_states


# ============================================================
# Adapter test
# ============================================================

def test_adapter(model_path, arch_name):
    """Quick test: load model, check architecture, run forward pass."""
    print(f"\n=== Testing {arch_name}: {model_path} ===")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    arch = detect_arch(model)
    n = get_num_layers(model)
    blocks = get_blocks(model)
    norm = get_final_norm(model)
    head = get_lm_head(model)
    wte = get_token_embeddings(model)
    wpe = get_position_embeddings(model)

    print(f"  Detected arch: {arch}")
    print(f"  Layers: {n}")
    print(f"  Blocks type: {type(blocks[0]).__name__}")
    print(f"  Final norm: {type(norm).__name__ if norm else 'None'}")
    print(f"  LM head: {type(head).__name__ if head else 'None'}")
    print(f"  Token emb: {type(wte).__name__ if wte else 'None'}")
    print(f"  Position emb: {type(wpe).__name__ if wpe else 'None (RoPE?)'}")

    # Forward pass test
    prompts = ["The capital of France is", "Machine learning is great"]
    hidden = get_hidden_states(model, tokenizer, prompts, layer_indices=[0, n//2, n-1], mode='last_token')
    print(f"  Hidden states: layers={list(hidden.keys())}, shape[0]={hidden[0].shape}")
    print(f"  Forward pass: OK")

    # Checkpoint save/load test
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(model, tokenizer, tmp)
        m2 = load_checkpoint(tmp)
        a2 = detect_arch(m2)
        print(f"  Save/load: OK (arch={a2}, layers={get_num_layers(m2)})")

    return arch, n


if __name__ == '__main__':
    import sys
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python model_adapters.py <model_path> [<model_path2> ...]")
        sys.exit(1)
    for p in paths:
        test_adapter(p, p)
