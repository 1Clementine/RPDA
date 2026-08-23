# RPDA Frozen Final Results

**Date**: 2026-08-02
**Protocols**: REEF fixed L9 (max512 HSIC CKA), BPB/KL (WikiText-2, 256-token blocks), ED-BA (TQA binary, layers [6,8,10], 5-seed)

## File Inventory

| # | File | Content |
|---|------|---------|
| 1 | `1_pure_rpda_trajectory.csv` | Pure RPDA r01-r60 — REEF L9, BPB ratio, KL |
| 2 | `2_recovery_every4_trajectory.csv` | Stage-wise recovery r16-rec→r60-rec — REEF L9, BPB ratio, KL |
| 3 | `3_unrelated_models.csv` | 12-layer non-source reference models |
| 4 | `4_gpt2_operation_spectrum.csv` | GPT-2 operation spectrum |
| 5 | `5_edba_trajectory.csv` | ED-BA 40-round for both trajectories |
| 6 | `6_protocol_spec.json` | Full protocol specifications |
| 7 | `7_pythia_every4_L20.csv` | Pythia-410M every4 (REEF fixed L20) |
| — | `layerwise_reef_to_source.csv` | Layer-wise REEF: C-r16 and S-r16(rec) vs Source (L0-L11) |
| — | `single_node_r16_metrics.csv` | r16 single-node recovery: 7 states |
| — | `single_node_r16_paired_deltas.csv` | Paired deltas: immediate, continuation-4, continuation-8 |
| — | `fixed_l9_recovery_mean.csv` | Fixed-L9 recovery at r16 — 3-seed mean |
| — | `js_divergence_summary.csv` | Per-sample JS divergence summary |

## Key Numbers

### Decoupling at r16
| Model | REEF L9 | BPB ratio | KL |
|---|:---:|:---:|:---:|
| RPDA-C-r16 | 0.5080 | 1.255 | 0.861 |
| Fixed-L9 recovery | 0.5080 | 1.098 | 0.434 |
| Full SOR (S-r16(rec)) | 0.4394 | 1.060 | 0.342 |

### r16 Single-Node Recovery
| Comparison | Δ REEF L9 | Δ BPB ratio | Δ KL |
|---|:---:|:---:|:---:|
| Immediate (r16) | -0.069 | -0.194 | -0.519 |
| Continuation-4 (r20) | -0.066 | -0.218 | -0.621 |
| Continuation-8 (r24) | -0.067 | -0.244 | -0.730 |

### Reference Models (12L, L9→L9)
| Model | REEF-L9 | ED-BA |
|---|:---:|:---:|
| GPT-Neo-125M | 0.4651 | 0.5000 |
| OPT-125M | 0.2574 | 0.5000 |
| Pythia-160M | 0.1866 | 0.4925 |

### JS Divergence (per-sample mean)
| Model | Mean JS |
|---|:---:|
| RPDA-C-r16 | 0.1352 |
| Fixed-L9 recovery | 0.0681 |
| Full SOR | 0.0633 |
