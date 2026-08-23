#!/usr/bin/env python3
"""RPDA 核心共享函数：3+1 路径重组的权重操作（alpha_split + gamma_compress）。

被 rebuild_gpt2_gamma_correct.py / extend_rpda_trajectory.py /
build_recovery_trajectory.py 共同 import，避免方法代码三处重复。
"""
import copy
import torch
from transformers import GPT2LMHeadModel


def scale_conv1d(conv, scale):
    with torch.no_grad():
        conv.weight.mul_(scale)
        conv.bias.mul_(scale)


def zero_conv1d(conv):
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()


def weighted_conv1d(dst, src_a, src_b, gamma):
    with torch.no_grad():
        dst.weight.copy_((1.0 - gamma) * src_a.weight + gamma * src_b.weight)
        dst.bias.copy_((1.0 - gamma) * src_a.bias + gamma * src_b.bias)


def alpha_split(model_12L, alpha):
    """12L -> 24L alpha attention split. Returns 24L model on same device."""
    device = next(model_12L.parameters()).device
    old_n = model_12L.config.n_layer
    new_n = old_n * 2
    beta = 1.0 - alpha

    cfg = copy.deepcopy(model_12L.config)
    cfg.n_layer = new_n
    split = GPT2LMHeadModel(cfg).to(device).eval()

    split.transformer.wte.load_state_dict(model_12L.transformer.wte.state_dict())
    split.transformer.wpe.load_state_dict(model_12L.transformer.wpe.state_dict())
    split.transformer.ln_f.load_state_dict(model_12L.transformer.ln_f.state_dict())

    for i in range(old_n):
        src = model_12L.transformer.h[i]
        A = split.transformer.h[2 * i]
        B = split.transformer.h[2 * i + 1]
        A.load_state_dict(src.state_dict())
        B.load_state_dict(src.state_dict())
        scale_conv1d(A.attn.c_proj, alpha)
        zero_conv1d(A.mlp.c_fc)
        zero_conv1d(A.mlp.c_proj)
        scale_conv1d(B.attn.c_proj, beta)

    split.tie_weights()
    return split


def gamma_compress(teacher_24L, gamma_schedule):
    """24L -> 12L gamma-correct 3+1 compression. Returns 12L model."""
    device = next(teacher_24L.parameters()).device
    tee = teacher_24L
    old_n = tee.config.n_layer
    new_n = old_n // 2

    cfg = copy.deepcopy(tee.config)
    cfg.n_layer = new_n
    student = GPT2LMHeadModel(cfg).to(device).eval()

    student.transformer.wte.load_state_dict(tee.transformer.wte.state_dict())
    student.transformer.wpe.load_state_dict(tee.transformer.wpe.state_dict())
    student.transformer.ln_f.load_state_dict(tee.transformer.ln_f.state_dict())

    for g in range(old_n // 4):
        gamma = gamma_schedule[g]
        p0_idx = 4 * g + 0; p1_idx = 4 * g + 1
        p2_idx = 4 * g + 2; p3_idx = 4 * g + 3
        q0_idx = 2 * g + 0; q1_idx = 2 * g + 1

        p0 = tee.transformer.h[p0_idx]; p1 = tee.transformer.h[p1_idx]
        p2 = tee.transformer.h[p2_idx]; p3 = tee.transformer.h[p3_idx]
        q0 = student.transformer.h[q0_idx]; q1 = student.transformer.h[q1_idx]

        q0.load_state_dict(p1.state_dict())
        q1.load_state_dict(p3.state_dict())

        # Gamma-weighted attention blending on q0
        weighted_conv1d(q0.attn.c_attn, p1.attn.c_attn, p2.attn.c_attn, gamma)
        with torch.no_grad():
            q0.attn.c_proj.weight.copy_(
                p0.attn.c_proj.weight + p1.attn.c_proj.weight + gamma * p2.attn.c_proj.weight)
            q0.attn.c_proj.bias.copy_(
                p0.attn.c_proj.bias + p1.attn.c_proj.bias + gamma * p2.attn.c_proj.bias)

    student.tie_weights()
    return student
