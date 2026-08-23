# RPDA 核心流程（冻结副本）

本目录是 RPDA（Representation Path Disentanglement Attack）项目的核心流程冻结副本，
从 `/home/syh/Work/Slim-transformer` 复制而来，供论文复现与归档使用。

**冻结日期**: 2026-08-15
**方法名称**: 论文统一使用 `Ghost`（源代码/结果字段中的 `GhostSpec` 仅为内部命名）

---

## 仓库内容说明（投稿精简版）

本仓库作为论文投稿附带的源码，**不含模型权重**。为保证可复现，保留以下内容：

- **核心代码**：`scripts_final/`（RPDA 重组、SOR 恢复、三项评价协议）+ `experiments/`（辅助实验）
- **冻结结果**：`results/frozen_final/`（全部论文数值的唯一事实来源）
- **评估数据**：`data/`（WikiText-2 test、TruthfulQA、SOR 恢复训练集）
- **论文图表**：`Final/figures/`、`Final/tables/`
- **模型定义**：`models/victim/base/gpt2/` 仅保留 `config.json` + tokenizer 文件

**已移除（权重，可再生成）**：

| 原内容 | 获取方式 |
|---|---|
| Source GPT-2 权重 `model.safetensors`（124M） | 从 HuggingFace 下载 `gpt2`，或从完整项目 `Slim-transformer/models/victim/base/gpt2/` 复制 |
| RPDA 检查点 `checkpoints/C-r16` 等 | 用 `scripts_final/rebuild_gpt2_gamma_correct.py` 从 Source 重建 |
| SOR 恢复检查点 `checkpoints/S-r16-rec` 等 | 用 `scripts_final/build_recovery_trajectory.py` 重建 |

> 注：脚本内路径硬编码为完整项目的相对路径（`runs/twostage_*`、`models/victim/base/gpt2`）。在完整项目环境（Slim-transformer）中运行即可；本仓库用于论文审阅的代码与结果归档。

---

## 核心流程

RPDA 由三部分组成：

### 1. 路径重组攻击（3+1 α-split + γ-correct compression）

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 构建轨迹 | `scripts_final/rebuild_gpt2_gamma_correct.py` | 从 Source 出发，每轮 `rN = gamma_compress(alpha_split(rN−1))` |
| 扩展轨迹 | `scripts_final/extend_rpda_trajectory.py` | 从任意 checkpoint 继续若干轮 |
| 构建恢复轨迹 | `scripts_final/build_recovery_trajectory.py` | 每 4 轮执行一次 SOR，后续从恢复后 checkpoint 继续 |

> 三个脚本共享 `scripts_final/common.py` 中的 `alpha_split` / `gamma_compress`（方法唯一实现，避免代码重复）。

关键配置：
- α = 0.10
- γ schedule = [0.01, 0.01, 0.005, 0.005, 0.0, 0.0]
- 每轮作用于全部 12 层

### 2. 输出侧恢复（SOR, Stage-wise Output Recovery）

| 脚本 | `scripts_final/formal_output_recovery_v2.py` |
|---|---|
| 目标 | KL 蒸馏（teacher = Source） |
| 训练范围 | last8（L4–L11 + final LayerNorm） |
| 配置 | lr=5e-5, steps=1000, temperature=2.0, AdamW |
| 数据 | WikiText-2 train（`data/formal_recovery_wikitext2/train.txt`） |

### 3. 三项评价协议

| 指标 | 脚本 | 协议 |
|---|---|---|
| REEF-L9 | `scripts_final/reef_unified_protocol_audit.py` | fixed L9, max512, HSIC CKA, TQA200, last-token, left-padding |
| BPT ratio / KL | `scripts_final/compute_bpb_kl_batch.py` | WikiText-2 test, 256-token blocks, 25,600 tokens |
| ED-BA | `scripts_final/run_edba_5seed.py` | TQA binary, layers [6,8,10], 5-seed, StandardScaler + LogisticRegression |

共享：
- `scripts_final/common.py` — 3+1 权重操作（alpha_split / gamma_compress）
- `scripts_final/cross_family_full/model_adapters.py` — 多架构模型适配器

辅助（`experiments/`，非核心流程）：
- `experiments/eval_mc_capability.py` — downstream utility sanity check（ARC-Easy / PIQA 零样本）
- `experiments/rebuild_pythia_every4_L20.py` — Pythia-410M 跨架构辅助实验

---

## 关键 checkpoint 路径映射

原项目路径 → 本副本路径：

| 原路径 | 副本路径 | 说明 |
|---|---|---|
| `models/victim/base/gpt2` | `models/victim/base/gpt2` | Source GPT-2 (124M, 12L) |
| `runs/twostage_raw/r16` | `checkpoints/C-r16` | 连续重组第 16 轮 |
| `runs/twostage_raw_extended/r40` | `checkpoints/C-r40` | 连续重组第 40 轮 |
| `runs/twostage_recovery_every4/r16-rec/best_checkpoint` | `checkpoints/S-r16-rec` | 恢复后 r16 |
| `runs/twostage_recovery_every4/r40-rec/best_checkpoint` | `checkpoints/S-r40-rec` | 恢复后 r40（完整 RPDA 终点） |

> 注意：脚本中的路径硬编码为原项目相对路径。若要在本副本中直接运行，
> 需将脚本内 `runs/twostage_*` 路径替换为 `checkpoints/*`。

---

## 冻结结果

全部冻结数据在 `results/frozen_final/`：

| 文件 | 内容 |
|---|---|
| `1_pure_rpda_trajectory.csv` | 纯 RPDA r01–r60（REEF/BPT/KL） |
| `2_recovery_every4_trajectory.csv` | 阶段性恢复 r16-rec→r60-rec |
| `3_unrelated_models.csv` | 12 层非同源参照 |
| `4_gpt2_operation_spectrum.csv` | GPT-2 操作光谱（微调/剪枝/量化/插值） |
| `5_edba_trajectory.csv` | ED-BA 40 轮 |
| `6_protocol_spec.json` | 完整协议规格 |
| `7_pythia_every4_L20.csv` | Pythia-410M every4 |
| `README.md` | 冻结结果汇总 |

---

## 论文图表

`figures/` 与 `Final/figures/` 保存核心论文图：

- `gpt2_reef_spectrum.pdf` — REEF-L9 光谱/轨迹图
- `trajectory_C_vs_S.pdf` — 连续 vs 阶段恢复轨迹
- `representation_output_decoupling_r16_v2.pdf` — r16 解耦（左：逐层 REEF；右：输出行为）

`Final/tables/` 保存论文 LaTeX 表格。

---

## 冻结规则

任何对论文、表格、图注、实验分析、摘要、引言或结论的修改，必须先完整阅读
`EXPERIMENT_FREEZE.md` 与 `results/frozen_final/` 下的冻结清单。

唯一事实来源为冻结清单，数值冲突时不得自行消解，必须报告并等待用户决定。
