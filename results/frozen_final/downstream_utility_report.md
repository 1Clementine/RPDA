# Downstream Utility Sanity Check

## 目的

轻量 sanity check：RPDA 连续路径重组与 SOR 对 GPT-2 基本 downstream utility 的影响。不扩展新主结论，仅判断 utility 是否退化、SOR 是否恢复、r40(rec) 是否仍可用。

## 任务与协议

- **Harness**: 项目现有 `scripts_final/eval_mc_capability.py`（零样本 likelihood evaluation，无 fine-tuning）
- **任务**: ARC-Easy（500 例，val）、PIQA（500 例，val）— 已下载本地数据
- **评估方式**: 每个选项作为 prompt continuation，取负对数似然最高者为预测（`acc` 为原始选择，`acc_norm` 为按 token 数归一化）
- **统一设置**: 同一数据、同一 prompt、fp16、max_length=512、同随机种子（无采样）
- **模型**: Source, C-r16, S-r16(rec), C-r40, S-r40(rec)

## 结果

| 模型 | ARC-Easy acc | PIQA acc | BPT ratio | KL |
|---|---:|---:|---:|---:|
| Source | 0.447 | 0.626 | 1.000 | 0.000 |
| C-r16 | 0.394 | 0.602 | 1.255 | 0.861 |
| S-r16(rec) | 0.427 | 0.594 | 1.060 | 0.342 |
| C-r40 | 0.354 | 0.574 | 1.555 | 2.050 |
| S-r40(rec) | 0.429 | 0.606 | 1.099 | 0.538 |

（ARC-Easy 使用原始 `acc`，PIQA 使用原始 `acc`）

## 观察

1. **continuous reorganization 导致 utility 退化**：C-r40 在两个任务上均低于 Source（ARC-Easy 0.354 vs 0.447，PIQA 0.574 vs 0.626），且随轮次退化加重（r16 < r40 退化幅度更大）。与 BPT ratio 从 1.25 升至 1.56、KL 从 0.86 升至 2.05 的方向一致。

2. **SOR 后 utility 出现恢复**：S-r40(rec) 相比 C-r40 在 ARC-Easy 上从 0.354 恢复到 0.429（接近 Source 0.447），PIQA 从 0.574 恢复到 0.606（接近 Source 0.626）。BPT 从 1.56 降至 1.10，KL 从 2.05 降至 0.54。

3. **r40(rec) 保持基本可用性**：S-r40(rec) 的 downstream 准确率（ARC-Easy 0.429，PIQA 0.606）均接近 Source（0.447，0.626），说明 SOR 基本恢复了 utility。

## 结论（限定表述）

- 连续路径重组会单调降低下游任务准确率，但绝对退化幅度有限（约 2–10 个百分点）。
- SOR 能有效恢复 downstream utility，恢复后的准确率接近 Source 水平。
- 上述结果**不构成** "full capability preservation" 的证明——本 sanity check 仅覆盖 2 个零样本 likelihood 任务，未覆盖生成质量、长程推理等更广泛的能力维度。
- downstream utility 的恢复方向与 BPT ratio / KL 的输出侧改善方向一致，支持"输出侧恢复"的定性判断。

## 输出文件

- `downstream_utility.csv`
- `downstream_utility_raw.csv`（逐任务明细）
- 本报告
