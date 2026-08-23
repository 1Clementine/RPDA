# EasyDetector 原始协议核对报告

## 依据

- 论文: `rele/EasyDetector_Using_Linear_Probe_to_Detect_the_Provenance_of_Large_Language_Models.pdf` (TrustCom 2024)
- **论文未提供源码**。当前项目实现为独立实现，本文档仅以论文 Section III（Methodology）文本为唯一权威依据逐项核对。

---

## 逐项核对

### 1. Representation hook

| 项 | 原始论文 | 当前实现 |
|---|---|---|
| 位置 | **MLP 输出**（"embedding of the last token from the multi-layer perceptron"） | **完整 Transformer block 输出**（含 attention + MLP + 残差） |
| 理由 | 论文明确选 MLP 而非 self-attention block，因 MLP 提供非线性变换，增强表达能力 | block 输出含残差，与 REEF 使用同一 hook 位置 |

**差异：MLP output → block output 是主要 adaptation。**

### 2. Token position

| 项 | 原始论文 | 当前实现 |
|---|---|---|
| 位置 | 最后一个 token（$r^{(l)} = h^{(l)}_n$，n 为序列长度） | 最后一个有效 token（左 padding + attention mask） |
| padding | 论文未明确 | 左 padding |

**差异：基本一致，均取最后 token。当前实现额外处理 padding（更严谨），不构成实质性偏离。**

### 3. Layer usage

| 项 | 原始论文 | 当前实现 |
|---|---|---|
| 方式 | **每层独立训练 probe**（"train the separate linear probe for each layer"） | **L6+L8+L10 拼接后训练一个 probe** |
| 层选择 | middle layers（32 层模型取 layer 12–24） | L6、L8、L10（GPT-2 12 层的中后段） |

**差异：multi-layer concatenation 是第二处主要 adaptation。** 原始为每层独立 probe，当前为多层拼接 probe。

### 4. Probe protocol

| 项 | 原始论文 | 当前实现 |
|---|---|---|
| 类型 | 线性 probe：$f_l(r)=\sigma(w_l^\top r + b_l)$（sigmoid + binary cross-entropy，即 logistic regression） | LogisticRegression + StandardScaler |
| 标准化 | 论文未明确 | 在 source 训练集上 fit StandardScaler |
| train/test | 4:1 随机划分 | 2542:636 ≈ 4:1 |
| 迁移 | probe 在 source 上训练，应用到 candidate | 一致 |

**差异：StandardScaler 是当前实现添加的，论文未明确是否标准化。属必要适配，不改变核心协议。**

### 5. Metric

| 项 | 原始论文 | 当前实现 |
|---|---|---|
| 指标 | **classification accuracy**（"accuracy of around 50%"） | **balanced accuracy** |

**差异：metric 从 accuracy 改为 balanced accuracy。** 因数据正负平衡（train 1272/1270），两者数值几乎相同，非实质性差异。

---

## 明确结论

1. **当前实现与原始 EasyDetector 的三处具体差异：**
   - (a) MLP output → block output（主要）
   - (b) 每层独立 probe → 多层拼接 probe（主要）
   - (c) accuracy → balanced accuracy（次要，平衡数据下几乎等价）

2. **MLP output → block output 是否主要 adaptation？** 是，但**不是唯一**。multi-layer concatenation 同样是主要 adaptation——原始论文每层独立训练 probe，当前拼接 L6+L8+L10 训练单一 probe。

3. **是否还存在其他 adaptation？** 有：metric（accuracy→balanced accuracy）、StandardScaler 标准化（论文未明确）。

4. **原始协议下 RPDA 主要趋势是否仍成立？** **成立。** 两种协议下趋势一致：
   - Source 最高（block 0.671 / mlp 0.637）
   - C-r40 相对 Source 下降（block 0.511 / mlp 0.538）
   - S-r40(rec) 向 Source 回升但低于 Source（block 0.584 / mlp 0.591）
   - 非源模型均接近 0.5（随机）

5. **论文主文应使用哪套结果？**
   - 当前实现有两处主要偏离（block output + multi-layer concat），**不能声称"完全复现 EasyDetector"**。
   - 建议：主文使用**当前 adapted block-output 结果**（因为：(a) 与 REEF 共用同一 hook 位置，内部一致；(b) 已是冻结主结果），并全文明确命名为 **"EasyDetector-style multi-layer linear probe"** 或 **"adapted EasyDetector"**。
   - 若需强调与原始协议的关系，可在附录给出本报告的 mlp-output 结果作为对照，说明趋势一致。

## 输出文件

- `easydetector_protocol_audit.md`（本报告）
- `easydetector_original_protocol.csv`（MLP-output 协议，5-seed 均值）
- `easydetector_protocol_comparison.csv`（block vs mlp 对比）
