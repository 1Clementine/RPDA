# RPDA 计算开销报告

**硬件**: NVIDIA GeForce RTX 3090 (24 GiB)
**软件**: PyTorch + HuggingFace Transformers（与主实验一致）
**测量方法**: 每个 timing 前执行 `torch.cuda.synchronize()`；单次操作重复 3 次取均值（SOR step 重复 20 次，预热后取稳态）。

---

## 1. 单次操作

| 操作 | wall-clock | peak GPU memory |
|---|---:|---:|
| 单次 local residual-path reorganization | 5.52 s | 1819 MiB |
| 单次 SOR step（1 block） | 0.044 s | 2195 MiB |

reorganization 包含 `alpha_split`（12L→24L）与 `gamma_compress`（24L→12L）两个子步骤，是全模型纯权重重排，无反向传播。

## 2. SOR 可训练参数

| 项 | 值 |
|---|---:|
| 可训练参数 | 56,704,512 |
| 总参数 | 124,439,808 |
| 比例 | 45.57% |

SOR 更新最后 8 个 Transformer block（L4–L11）+ final LayerNorm，冻结 embedding、L0–L3 与 lm_head。

## 3. 完整 GPT-2 RPDA-r40

| 项 | 时间 |
|---|---:|
| reorganization 总时间（40 轮 × 5.52s） | 221 s ≈ 3.7 min |
| SOR 总时间（7 个恢复节点 × 44s） | 308 s ≈ 5.1 min |
| **总 wall-clock** | **529 s ≈ 8.8 min** |
| SOR 占总成本比例 | 58.2% |

注：recovery-every-4 轨迹在 r16–r40 共 7 个恢复节点（r16、r20、r24、r28、r32、r36、r40）。

## 4. Fine-tuning baseline（参考）

| 项 | 值 |
|---|---:|
| 可训练参数比例 | 22.78%（last4） |
| 3 epoch（batch8, WikiText-2 train） | ~31 s ≈ 0.5 min |

---

## 结论

1. **path reorganization 本身是低成本操作**：单次仅 5.5s、1.8 GiB 显存，且是纯权重拷贝/缩放/混合，不涉及反向传播。40 轮总共 3.7 min。

2. **RPDA 的主要计算开销来自 SOR**：SOR 占完整 r40 总成本的 58%（5.1 min / 8.8 min）。原因是 SOR 需要反向传播 + 1000 步优化。

3. **SOR 相比完整 fine-tuning 更新了 45.6% 参数**：SOR 更新 last8（45.6%），而普通 last4 fine-tuning 仅更新 22.8%。SOR 的可训练范围更大，是 RPDA 输出侧恢复的主力成本来源。

4. **完整 RPDA 的实际额外成本约 8.8 min**（单张 RTX 3090），其中重组 3.7 min、SOR 5.1 min。相对于 fine-tuning baseline（0.5 min）仍更高，但绝对量级小，处于可接受的实验成本范围。
