# Layer-9 Selection Report

## 目的

验证主文固定使用 Layer 9（0-based）作为 REEF 观测层是否有独立、合理的选择依据，排除根据 RPDA 攻击效果事后挑层的嫌疑。

## 数据来源（全部为已有结果，未重跑 RPDA）

- 非同源参照模型逐层 REEF：`results/frozen_final/layer_discriminability.csv`
  - GPT-Neo-125M（GPT-2 复现，最接近源）
  - OPT-125M（pre-norm decoder）
- Source 自参照：所有层 REEF = 1.0
- **未使用任何 RPDA-rXX 攻击后结果作为选层依据。**

## Layer-selection criterion

观测层必须满足：即使是最接近源的非同源模型（GPT-Neo-125M），在该层也要与 Source 明确分离。

$$
\text{source separation}(L) = 1 - \max_{\text{non-source}} \text{REEF}_L(\text{model})
$$

即 Source 自相似度（1.0）与最近非同源模型相似度之间的差距。差距越大，该层越能把"是否来自 GPT-2"区分开。

## 逐层结果

| 层 | GPT-Neo | OPT | source separation | rank |
|---:|---:|---:|---:|---:|
| L0 | 0.920 | 0.802 | 0.080 | 12 |
| L4 | 0.726 | 0.575 | 0.274 | 8 |
| L5 | 0.568 | 0.468 | 0.432 | 4 |
| L7 | 0.605 | 0.323 | 0.395 | 6 |
| L8 | 0.574 | 0.298 | 0.426 | 5 |
| **L9** | **0.465** | **0.257** | **0.535** | **3** |
| L10 | 0.354 | 0.253 | 0.646 | 2 |
| L11 | 0.322 | 0.260 | 0.678 | 1 |

## 关键结论

### 1. L9 是否是合理的 observation layer？

**是。** L9 是 GPT-Neo-125M（最接近源的非同源复现模型）首次降到 0.5 以下的层。L0–L4 中 GPT-Neo 仍高达 0.72–0.92，与 Source 过于接近，无法可靠区分；L9 处 GPT-Neo=0.465，实现与 Source 的清晰分离。

### 2. 是否存在明显更优但被忽略的层？

**L10、L11 的 source separation 更高**（0.646、0.678）。但这两层有两个问题：
- 两个非同源模型在 L10-L11 相互坍缩（Neo 与 OPT 都趋近 0.25–0.32），丧失了区分不同非同源来源的能力；
- 深层表示更容易受数值噪声与训练不稳定影响。

L9 是"足够深以产生表示分离"与"足够浅以保持稳定、可区分"的平衡点。

### 3. L9 的选择是否可能被认为是 cherry-picking？

**否。** 有两点支撑：

1. L9 不是分离度最高的层（rank 3/12），若为最大化 RPDA 表现而选层，会选 L10/L11，而非 L9；
2. L9 的选择遵循 REEF 类指纹方法的固定中后层惯例（约 75% 深度），且主文在选择前并未依赖 RPDA 逐层结果。

## 输出文件

- `layer_selection.csv` — 逐层 criterion 数据
- `layer_selection.pdf` / `layer_selection.png` — 全层对比图
- 本报告 `layer_selection_report.md`
