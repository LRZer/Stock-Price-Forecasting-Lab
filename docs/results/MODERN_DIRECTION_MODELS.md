# 八种现代结构：次日涨跌教学实验

这八种网络使用与原有方向分类实验相同的六个 OHLCV 相对特征、20 日输入窗口、时间切分、训练期归一化、0.5 判断阈值和训练参数。它们借鉴时间序列预测、分类或通用序列建模论文的核心结构，这里统一改为输出**次日上涨概率**。这些是轻量教学改编，不是论文的精确复现，也不继承论文中的成绩结论。

| 项目名称 | 借鉴的结构 | 在本项目中做什么 | 原论文 |
| --- | --- | --- | --- |
| `dlinear-direction` | DLinear | 把每个特征的历史拆成平滑趋势与剩余变化，分别用线性层汇总，再做二分类 | [Zeng 等，AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317) |
| `tsmixer-direction` | TSMixer | 用残差 MLP 轮流混合时间位置和六个特征，再做二分类 | [Chen 等，TMLR 2023](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/) |
| `patchtst-direction` | PatchTST | 按每个特征分别把 20 日切成重叠短片，用共享注意力编码，再汇总六个特征 | [Nie 等，ICLR 2023](https://arxiv.org/abs/2211.14730) |
| `itransformer-direction` | iTransformer | 把每个特征的 20 日历史当作一个 token，让注意力在特征之间交换信息 | [Liu 等，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html) |
| `nhits-direction` | N-HiTS | 用多个时间尺度汇总历史，逐层重建并扣除已解释部分，合成涨跌 logit | [Challu 等，AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25854) |
| `tide-direction` | TiDE | 将逐日特征投影后交给残差 MLP 编码器和解码器，加入线性捷径 | [Das 等，TMLR 2023](https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/) |
| `timesnet-direction` | TimesNet | 按几个适合 20 日窗口的周期折成二维网格，以卷积提取变化，再按每个样本的频谱强度加权 | [Wu 等，ICLR 2023](https://arxiv.org/abs/2210.02186) |
| `mamba-style-direction` | Mamba 风格选择性状态空间 | 输入决定每天保留或更新多少隐藏状态；使用纯 PyTorch 顺序计算 | [Gu 与 Dao，COLM 2024](https://arxiv.org/abs/2312.00752) |

这些模型单列于 `stocklab/modern_direction_models.py`，训练入口是 `stocklab/modern_direction.py`。N-HiTS 的原始多步预测、TiDE 的未来协变量、TimesNet 的完整自适应周期模块，以及 Mamba 的专用并行扫描和完整结构都未照搬；尤其 `mamba-style-direction` **不是官方 Mamba 实现**。已冻结的正式六候选、未来评测代码和哈希清单均未修改。下列 2025 年日期早已被项目查看，新增网络的成绩只能用于**回顾性学习比较**，不能作为真正未见数据上的优势证据。

## 固定测试：2025-06-02 至 2025-12-31

正确天数 / 148 日；每个模型的最佳训练轮次由验证期 Brier 误差决定。每行是一个预先命名的结构，不能看下面的测试分数再挑“赢家”。

| 模型 | GOOG | AAPL | TSLA |
| --- | ---: | ---: | ---: |
| 训练期多数方向 | 85/148 = 57.4% | 82/148 = 55.4% | 79/148 = 53.4% |
| DLinear 改编 | 78/148 = 52.7% | 58/148 = 39.2% | 72/148 = 48.6% |
| TSMixer 改编 | 84/148 = 56.8% | 72/148 = 48.6% | 75/148 = 50.7% |
| PatchTST 改编 | 83/148 = 56.1% | 73/148 = 49.3% | 71/148 = 48.0% |
| iTransformer 改编 | 85/148 = 57.4% | 76/148 = 51.4% | 76/148 = 51.4% |
| N-HiTS 改编 | 85/148 = 57.4% | 82/148 = 55.4% | 73/148 = 49.3% |
| TiDE 改编 | 83/148 = 56.1% | 80/148 = 54.1% | 73/148 = 49.3% |
| TimesNet 改编 | 90/148 = 60.8% | 79/148 = 53.4% | 67/148 = 45.3% |
| Mamba 风格 | 86/148 = 58.1% | 82/148 = 55.4% | 77/148 = 52.0% |

## 三轮按时间向前测试

三轮的测试期各 100 日，同一股票的三轮不重叠。下面把**同一结构**的正确天数相加，显示它在不同时间段的波动；这不是根据测试分数选模的正式结论。

| 模型 | GOOG（300 日） | AAPL（300 日） | TSLA（300 日） |
| --- | ---: | ---: | ---: |
| 训练期多数方向 | 162/300 = 54.0% | 161/300 = 53.7% | 151/300 = 50.3% |
| DLinear 改编 | 158/300 = 52.7% | 122/300 = 40.7% | 147/300 = 49.0% |
| TSMixer 改编 | 148/300 = 49.3% | 135/300 = 45.0% | 144/300 = 48.0% |
| PatchTST 改编 | 157/300 = 52.3% | 164/300 = 54.7% | 136/300 = 45.3% |
| iTransformer 改编 | 170/300 = 56.7% | 150/300 = 50.0% | 140/300 = 46.7% |
| N-HiTS 改编 | 160/300 = 53.3% | 151/300 = 50.3% | 159/300 = 53.0% |
| TiDE 改编 | 159/300 = 53.0% | 150/300 = 50.0% | 146/300 = 48.7% |
| TimesNet 改编 | 165/300 = 55.0% | 161/300 = 53.7% | 143/300 = 47.7% |
| Mamba 风格 | 161/300 = 53.7% | 163/300 = 54.3% | 148/300 = 49.3% |

GOOG 固定期的 TimesNet 为 90/148，超过基线的 85/148；但 AAPL、TSLA 的同一结构分别为 79/148、67/148，低于基线。滚动合计中，GOOG 的 iTransformer、AAPL 的 PatchTST、TSLA 的 N-HiTS 各有高于基线的分数，不过三个股票的较好结构不同。**不能按测试成绩在这些结构中挑赢家。**

只根据**验证期**在八种网络与多数方向之间逐轮选模，固定测试的 GOOG、AAPL、TSLA 分别为 90、79、67 天正确；对应基线为 85、82、79 天。三轮滚动合计分别为 167、165、157 天正确；对应基线为 162、161、151 天。这些差距较小，且新结构是在看过 2025 年结果后加入的；固定测试与后两轮滚动测试日期还重叠，不能视作独立复验。正式未来检查继续遵守原来的六候选冻结规则。

## 复现

```powershell
python -m stocklab.modern_direction --mode fixed-2025
python -m stocklab.modern_direction --mode rolling-2025
python -m scripts.verify_modern_direction
```

`runs/modern-direction/` 保存每一轮的 `metrics.csv`、`predictions.csv`、`confusion.csv`、`calibration.csv`、训练历史和模型图。固定测试还保存权重。`summary.json` 记录分割日期、训练设置及原始数据哈希；图像可用于查看模型是否几乎总猜同一方向，以及预测概率与真实上涨比例的关系。
