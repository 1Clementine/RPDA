# RPDA Project Context

## Project Overview

RPDA (Representation Path Disentanglement Attack) — a structural recombination attack on GPT-2 that reduces forward fingerprints (REEF, ED-BA) while maintaining output quality through staged output-side recovery.

## Key Directories

| Directory | Content |
|------|------|
| `results/frozen/` | Final frozen data (8 CSV files) |
| `results/audit_final/` | Audit reports |
| `results/archive/` | All historical data |
| `paper_tables_final/` | Generated LaTeX tables |
| `scripts_final/` | Experiment scripts |
| `runs/twostage_staged/` | Staged RPDA checkpoints (SR-r00 to SR-r30 + recovery) |
| `runs/twostage_raw/` | RAW RPDA checkpoints (r00 to r30) |
| `/home/syh/Work/Other/Research/REEF/` | Official REEF repository |
| `/home/syh/Work/Other/Research/GhostSpec/` | GhostSpec repository |

## Protocols

| Metric | Protocol ID |
|--------|-------------|
| REEF | `official_reef_max512_center_scale_HSIC_fixed_layer_v1` |
| ED-BA | `edba_tqa_binary_layers_6_8_10_logreg_5seed_v1` |
| BPB/KL | `twostage_wikitext_25600_continuous_v1` |
| GhostSpec | `GhostSpec_v1` (correlation + transformed MSE) |

## GPT-2 RPDA Configuration

- **Attack**: 3+1 alpha-split + gamma-correct compression
- **Alpha**: 0.10
- **Gamma schedule**: [0.01, 0.01, 0.005, 0.005, 0.0, 0.0]
- **Window**: All 12 layers every round (global)
- **Rounds**: 30 (r01-r30)
- **Cumulative rule**: rN = gamma_compress(alpha_split(rN-1))

## Recovery Configuration

- **Script**: `run_staged_recovery_trajectory.py`
- **LR**: 5e-5, **Steps**: 1000, **Scope**: last8
- **Optimizer**: AdamW, **Objective**: KL divergence (log_target=True)
- **Teacher**: Source (SR-r00)
- **Recovery points**: r16, r24, r30

## Key Results (GPT-2 Frozen)

### RAW r30
- REEF L6-11: 0.499, BPB: 1.413, KL: 1.470
- ED-BA: 0.521 ± 0.038

### STAGED r30_rec
- REEF L6-11: 0.433, BPB: 1.072, KL: 0.459
- ED-BA: 0.563 ± 0.035

### Recovery Decoupling
- r16: BPB 1.258→1.026 (-18%), KL 0.872→0.300 (-66%)
- REEF and ED-BA do NOT recover

## E1 Controls (Official REEF)

| Configuration | REEF L6-11 | ED-BA | BPB |
|------|:---:|:---:|:---:|
| Source | 1.000 | 0.698 | 1.000 |
| 2+2 control | 1.000 | 0.698 | 1.000 |
| Boundary-only | 1.000 | 0.698 | 1.000 |
| 3+1 RPDA raw | 0.796 | 0.660 | 1.007 |

## Pythia-410M every4 (Official REEF, 24L, L12-23 mean)

| Checkpoint | REEF L12-23 | ED-BA | BPB |
|------|:---:|:---:|:---:|
| Source | 1.000 | 0.683 | 1.000 |
| r01 | 0.888 | 0.580 | 1.050 |
| r02 | 0.500 | 0.532 | 1.359 |
| r02_rec | 0.510 | 0.505 | 1.289 |

Every4 config: layers [0,4,8,12,16,20], alpha=0.10, gamma=0.25, single-shot per round.

## Cross-Model CKA (GPT-2 L9 vs equivalent depth)

| Model | Layer | CKA |
|------|:---:|:---:|
| GPT-2 Source | L9 | 1.000 |
| GPT-2 RAW r30 | L9 | 0.395 |
| GPT-Neo-125M | L9 | 0.391 |
| Pythia-410M | L18 | 0.332 |
| Qwen2.5-0.5B | L18 | 0.283 |
| OPT-350M | L18 | 0.278 |

## GhostSpec (Correlation vs Source)

| Checkpoint | Correlation |
|------|:---:|
| Source | 1.000000 |
| r04 RAW | 0.999990 |
| r30 RAW | 0.999344 |
| r16 REC | 0.985303 |
| r30 REC | 0.926785 |

## Frozen Data Files

| # | File | Content |
|:---:|------|------|
| 1 | `1_gpt2_raw_trajectory.csv` | RAW REEF L0-L11 + BPB/KL (9 rows) |
| 2 | `2_gpt2_staged_trajectory.csv` | STAGED REEF + BPB/KL (8 rows) |
| 3 | `3_gpt2_recovery_pairs.csv` | Recovery before/after (3 pairs) |
| 4 | `4_gpt2_edba_final.csv` | ED-BA 5-seed ddof=1 (17 rows) |
| 5 | `5_e1_controls_final.csv` | E1 controls (4 rows) |
| 6 | `6_pythia_auxiliary.csv` | Pythia official REEF (4 rows) |
| 7 | `7_unrelated_reference.csv` | Cross-model CKA (5 rows) |
| 8 | `8_ghost_boundary.csv` | GhostSpec boundary (2 rows) |

## Known Gaps

- MC Avg: lm_eval not available (per-task scores missing)
- TinyLlama: not in project
- Cross-model REEF for Qwen/OPT/GPT-Neo: computed in-session, need permanent CSV save
