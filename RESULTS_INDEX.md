# 全部测试结果与指标索引

本页从已保存的 CSV 生成，展示 2025 年主实验、现代结构实验和两只外部股票的每个模型。
分数是历史测试表现，不是未来收益或预测可靠性的证明。固定测试与滚动测试有重叠日期；现代模型是在看过 2025 年结果后加入的。

**指标读法：**正确天数 / 准确率越高越好；平衡准确率分别计算上涨日和未上涨日的命中率后平均；Brier 是上涨概率与 0/1 真实标签的平均平方误差，越低越好；预测上涨比例显示模型是否几乎只猜一个方向。
滚动表的 Brier 是三个 100 日测试折的均值；平衡准确率、预测上涨比例、混淆矩阵和校准分箱请看每折 CSV。

## 20 类直接分类网络 + 3 个基线

### 固定测试

#### GOOG

[metrics.csv](runs/neural-classifiers/fixed-2025/goog/metrics.csv) · [predictions.csv](runs/neural-classifiers/fixed-2025/goog/predictions.csv) · [confusion.csv](runs/neural-classifiers/fixed-2025/goog/confusion.csv) · [calibration.csv](runs/neural-classifiers/fixed-2025/goog/calibration.csv) · [summary.json](runs/neural-classifiers/fixed-2025/goog/summary.json) · [diagnostics.png](runs/neural-classifiers/fixed-2025/goog/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gru-attention` | 92/148 = 62.2% | 58.2% | 0.2365 | 77.7% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/gru-attention.png) |
| `gru-2path` | 84/148 = 56.8% | 49.4% | 0.2489 | 99.3% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/gru-2path.png) |
| `gru` | 84/148 = 56.8% | 49.8% | 0.2443 | 96.6% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/gru.png) |
| `bidirectional-vanilla` | 84/148 = 56.8% | 49.4% | 0.2488 | 99.3% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/bidirectional-vanilla.png) |
| `gbdt-window` | 76/148 = 51.4% | 49.8% | 0.2514 | 60.1% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/gbdt-window.png) |
| `bidirectional-gru` | 87/148 = 58.8% | 51.6% | 0.2482 | 98.6% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/bidirectional-gru.png) |
| `lstm-2path` | 85/148 = 57.4% | 50.0% | 0.2488 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/lstm-2path.png) |
| `bidirectional-lstm` | 85/148 = 57.4% | 50.0% | 0.2486 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/bidirectional-lstm.png) |
| `lstm` | 85/148 = 57.4% | 50.0% | 0.2469 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/lstm.png) |
| `cnn-seq2seq` | 85/148 = 57.4% | 50.0% | 0.2499 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/cnn-seq2seq.png) |
| `dilated-cnn-seq2seq` | 85/148 = 57.4% | 50.0% | 0.2499 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/dilated-cnn-seq2seq.png) |
| `train-majority` | 85/148 = 57.4% | 50.0% | 0.2460 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/train-majority.png) |
| `vanilla-2path` | 86/148 = 58.1% | 51.0% | 0.2488 | 98.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/vanilla-2path.png) |
| `tcn-residual` | 80/148 = 54.1% | 47.5% | 0.2468 | 93.9% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/tcn-residual.png) |
| `logit-window` | 84/148 = 56.8% | 53.1% | 0.2405 | 75.0% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/logit-window.png) |
| `lstm-seq2seq-vae` | 84/148 = 56.8% | 50.6% | 0.2498 | 91.2% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/lstm-seq2seq-vae.png) |
| `vanilla` | 88/148 = 59.5% | 54.4% | 0.2490 | 84.5% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/vanilla.png) |
| `attention-is-all-you-need` | 83/148 = 56.1% | 50.9% | 0.2461 | 85.1% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/attention-is-all-you-need.png) |
| `bidirectional-gru-seq2seq` | 79/148 = 53.4% | 47.3% | 0.2495 | 90.5% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/bidirectional-gru-seq2seq.png) |
| `gru-seq2seq-vae` | 83/148 = 56.1% | 51.1% | 0.2497 | 83.8% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/gru-seq2seq-vae.png) |
| `lstm-seq2seq` | 87/148 = 58.8% | 54.1% | 0.2497 | 82.4% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/lstm-seq2seq.png) |
| `gru-seq2seq` | 80/148 = 54.1% | 48.5% | 0.2495 | 87.2% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/gru-seq2seq.png) |
| `bidirectional-lstm-seq2seq` | 89/148 = 60.1% | 55.0% | 0.2498 | 85.1% | [查看](runs/neural-classifiers/fixed-2025/goog/diagnostics/models/bidirectional-lstm-seq2seq.png) |

#### AAPL

[metrics.csv](runs/neural-classifiers/fixed-2025/aapl/metrics.csv) · [predictions.csv](runs/neural-classifiers/fixed-2025/aapl/predictions.csv) · [confusion.csv](runs/neural-classifiers/fixed-2025/aapl/confusion.csv) · [calibration.csv](runs/neural-classifiers/fixed-2025/aapl/calibration.csv) · [summary.json](runs/neural-classifiers/fixed-2025/aapl/summary.json) · [diagnostics.png](runs/neural-classifiers/fixed-2025/aapl/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `cnn-seq2seq` | 79/148 = 53.4% | 49.8% | 0.2510 | 83.1% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/cnn-seq2seq.png) |
| `dilated-cnn-seq2seq` | 81/148 = 54.7% | 50.7% | 0.2529 | 87.2% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/dilated-cnn-seq2seq.png) |
| `gbdt-window` | 67/148 = 45.3% | 42.8% | 0.2691 | 72.3% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/gbdt-window.png) |
| `bidirectional-gru` | 82/148 = 55.4% | 50.0% | 0.2484 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/bidirectional-gru.png) |
| `attention-is-all-you-need` | 82/148 = 55.4% | 50.7% | 0.2496 | 93.2% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/attention-is-all-you-need.png) |
| `bidirectional-vanilla` | 81/148 = 54.7% | 49.4% | 0.2492 | 99.3% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/bidirectional-vanilla.png) |
| `lstm-seq2seq` | 82/148 = 55.4% | 50.0% | 0.2490 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/lstm-seq2seq.png) |
| `bidirectional-lstm-seq2seq` | 82/148 = 55.4% | 50.0% | 0.2496 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/bidirectional-lstm-seq2seq.png) |
| `logit-window` | 79/148 = 53.4% | 50.1% | 0.2532 | 80.4% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/logit-window.png) |
| `lstm` | 81/148 = 54.7% | 49.7% | 0.2493 | 96.6% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/lstm.png) |
| `gru-seq2seq-vae` | 81/148 = 54.7% | 49.4% | 0.2495 | 99.3% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/gru-seq2seq-vae.png) |
| `lstm-seq2seq-vae` | 80/148 = 54.1% | 48.8% | 0.2491 | 98.6% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/lstm-seq2seq-vae.png) |
| `tcn-residual` | 80/148 = 54.1% | 48.8% | 0.2496 | 98.6% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/tcn-residual.png) |
| `gru-attention` | 77/148 = 52.0% | 48.0% | 0.2547 | 87.2% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/gru-attention.png) |
| `lstm-2path` | 82/148 = 55.4% | 50.0% | 0.2482 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/lstm-2path.png) |
| `train-majority` | 82/148 = 55.4% | 50.0% | 0.2478 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/train-majority.png) |
| `bidirectional-lstm` | 82/148 = 55.4% | 50.0% | 0.2490 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/bidirectional-lstm.png) |
| `bidirectional-gru-seq2seq` | 82/148 = 55.4% | 50.0% | 0.2495 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/bidirectional-gru-seq2seq.png) |
| `gru-seq2seq` | 81/148 = 54.7% | 49.5% | 0.2497 | 98.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/gru-seq2seq.png) |
| `gru-2path` | 82/148 = 55.4% | 50.0% | 0.2487 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/gru-2path.png) |
| `gru` | 82/148 = 55.4% | 50.0% | 0.2491 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/gru.png) |
| `vanilla` | 81/148 = 54.7% | 51.6% | 0.2496 | 79.1% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/vanilla.png) |
| `vanilla-2path` | 82/148 = 55.4% | 52.1% | 0.2496 | 81.1% | [查看](runs/neural-classifiers/fixed-2025/aapl/diagnostics/models/vanilla-2path.png) |

#### TSLA

[metrics.csv](runs/neural-classifiers/fixed-2025/tsla/metrics.csv) · [predictions.csv](runs/neural-classifiers/fixed-2025/tsla/predictions.csv) · [confusion.csv](runs/neural-classifiers/fixed-2025/tsla/confusion.csv) · [calibration.csv](runs/neural-classifiers/fixed-2025/tsla/calibration.csv) · [summary.json](runs/neural-classifiers/fixed-2025/tsla/summary.json) · [diagnostics.png](runs/neural-classifiers/fixed-2025/tsla/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `bidirectional-gru-seq2seq` | 72/148 = 48.6% | 49.9% | 0.2499 | 31.8% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/bidirectional-gru-seq2seq.png) |
| `cnn-seq2seq` | 66/148 = 44.6% | 47.7% | 0.2500 | 3.4% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/cnn-seq2seq.png) |
| `logit-window` | 76/148 = 51.4% | 53.4% | 0.2609 | 19.6% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/logit-window.png) |
| `gru-seq2seq` | 70/148 = 47.3% | 48.9% | 0.2500 | 26.4% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/gru-seq2seq.png) |
| `lstm` | 68/148 = 45.9% | 46.9% | 0.2500 | 35.8% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/lstm.png) |
| `lstm-2path` | 75/148 = 50.7% | 51.5% | 0.2500 | 37.8% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/lstm-2path.png) |
| `bidirectional-vanilla` | 66/148 = 44.6% | 43.7% | 0.2501 | 62.8% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/bidirectional-vanilla.png) |
| `bidirectional-lstm` | 72/148 = 48.6% | 51.4% | 0.2501 | 8.8% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/bidirectional-lstm.png) |
| `attention-is-all-you-need` | 72/148 = 48.6% | 49.2% | 0.2508 | 41.2% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/attention-is-all-you-need.png) |
| `lstm-seq2seq-vae` | 69/148 = 46.6% | 48.5% | 0.2500 | 21.6% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/lstm-seq2seq-vae.png) |
| `gru-seq2seq-vae` | 79/148 = 53.4% | 54.7% | 0.2499 | 31.1% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/gru-seq2seq-vae.png) |
| `gbdt-window` | 69/148 = 46.6% | 48.1% | 0.2883 | 28.4% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/gbdt-window.png) |
| `dilated-cnn-seq2seq` | 69/148 = 46.6% | 50.0% | 0.2500 | 0.0% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/dilated-cnn-seq2seq.png) |
| `gru` | 70/148 = 47.3% | 50.3% | 0.2502 | 6.1% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/gru.png) |
| `gru-2path` | 73/148 = 49.3% | 50.3% | 0.2500 | 35.1% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/gru-2path.png) |
| `gru-attention` | 68/148 = 45.9% | 46.7% | 0.2502 | 38.5% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/gru-attention.png) |
| `tcn-residual` | 69/148 = 46.6% | 46.4% | 0.2502 | 52.7% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/tcn-residual.png) |
| `vanilla-2path` | 65/148 = 43.9% | 43.9% | 0.2501 | 50.0% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/vanilla-2path.png) |
| `bidirectional-lstm-seq2seq` | 70/148 = 47.3% | 49.6% | 0.2500 | 15.5% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/bidirectional-lstm-seq2seq.png) |
| `lstm-seq2seq` | 71/148 = 48.0% | 50.2% | 0.2500 | 17.6% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/lstm-seq2seq.png) |
| `bidirectional-gru` | 65/148 = 43.9% | 46.0% | 0.2504 | 18.9% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/bidirectional-gru.png) |
| `train-majority` | 79/148 = 53.4% | 50.0% | 0.2492 | 100.0% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/train-majority.png) |
| `vanilla` | 64/148 = 43.2% | 44.0% | 0.2506 | 38.5% | [查看](runs/neural-classifiers/fixed-2025/tsla/diagnostics/models/vanilla.png) |

### 三轮滚动测试

#### GOOG

[三轮合计 CSV](runs/neural-classifiers/rolling-2025/goog/rolling_metrics.csv)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `attention-is-all-you-need` | 166/300 = 55.3% | 逐轮 CSV | 0.2480 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/attention-is-all-you-need.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/attention-is-all-you-need.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/attention-is-all-you-need.png) |
| `dilated-cnn-seq2seq` | 164/300 = 54.7% | 逐轮 CSV | 0.2506 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/dilated-cnn-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/dilated-cnn-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/dilated-cnn-seq2seq.png) |
| `lstm-seq2seq` | 170/300 = 56.7% | 逐轮 CSV | 0.2512 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/lstm-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/lstm-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/lstm-seq2seq.png) |
| `bidirectional-gru` | 164/300 = 54.7% | 逐轮 CSV | 0.2514 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/bidirectional-gru.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/bidirectional-gru.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/bidirectional-gru.png) |
| `gru` | 167/300 = 55.7% | 逐轮 CSV | 0.2496 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/gru.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/gru.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/gru.png) |
| `bidirectional-lstm-seq2seq` | 170/300 = 56.7% | 逐轮 CSV | 0.2498 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/bidirectional-lstm-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/bidirectional-lstm-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/bidirectional-lstm-seq2seq.png) |
| `lstm-2path` | 165/300 = 55.0% | 逐轮 CSV | 0.2506 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/lstm-2path.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/lstm-2path.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/lstm-2path.png) |
| `lstm` | 163/300 = 54.3% | 逐轮 CSV | 0.2517 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/lstm.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/lstm.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/lstm.png) |
| `vanilla` | 165/300 = 55.0% | 逐轮 CSV | 0.2480 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/vanilla.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/vanilla.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/vanilla.png) |
| `cnn-seq2seq` | 163/300 = 54.3% | 逐轮 CSV | 0.2496 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/cnn-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/cnn-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/cnn-seq2seq.png) |
| `vanilla-2path` | 165/300 = 55.0% | 逐轮 CSV | 0.2464 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/vanilla-2path.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/vanilla-2path.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/vanilla-2path.png) |
| `train-majority` | 162/300 = 54.0% | 逐轮 CSV | 0.2484 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/train-majority.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/train-majority.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/train-majority.png) |
| `gru-2path` | 163/300 = 54.3% | 逐轮 CSV | 0.2499 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/gru-2path.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/gru-2path.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/gru-2path.png) |
| `bidirectional-gru-seq2seq` | 162/300 = 54.0% | 逐轮 CSV | 0.2516 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/bidirectional-gru-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/bidirectional-gru-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/bidirectional-gru-seq2seq.png) |
| `lstm-seq2seq-vae` | 160/300 = 53.3% | 逐轮 CSV | 0.2507 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/lstm-seq2seq-vae.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/lstm-seq2seq-vae.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/lstm-seq2seq-vae.png) |
| `bidirectional-vanilla` | 164/300 = 54.7% | 逐轮 CSV | 0.2455 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/bidirectional-vanilla.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/bidirectional-vanilla.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/bidirectional-vanilla.png) |
| `bidirectional-lstm` | 163/300 = 54.3% | 逐轮 CSV | 0.2495 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/bidirectional-lstm.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/bidirectional-lstm.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/bidirectional-lstm.png) |
| `gru-seq2seq` | 163/300 = 54.3% | 逐轮 CSV | 0.2517 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/gru-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/gru-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/gru-seq2seq.png) |
| `gru-attention` | 169/300 = 56.3% | 逐轮 CSV | 0.2466 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/gru-attention.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/gru-attention.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/gru-attention.png) |
| `gru-seq2seq-vae` | 164/300 = 54.7% | 逐轮 CSV | 0.2503 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/gru-seq2seq-vae.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/gru-seq2seq-vae.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/gru-seq2seq-vae.png) |
| `tcn-residual` | 162/300 = 54.0% | 逐轮 CSV | 0.2479 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/tcn-residual.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/tcn-residual.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/tcn-residual.png) |
| `logit-window` | 165/300 = 55.0% | 逐轮 CSV | 0.2538 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/logit-window.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/logit-window.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/logit-window.png) |
| `gbdt-window` | 151/300 = 50.3% | 逐轮 CSV | 0.2549 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics/models/gbdt-window.png) · [2](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics/models/gbdt-window.png) · [3](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics/models/gbdt-window.png) |

- 第 1 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/goog/fold-1/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/goog/fold-1/diagnostics.png)
- 第 2 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/goog/fold-2/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/goog/fold-2/diagnostics.png)
- 第 3 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/goog/fold-3/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/goog/fold-3/diagnostics.png)

#### AAPL

[三轮合计 CSV](runs/neural-classifiers/rolling-2025/aapl/rolling_metrics.csv)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `vanilla` | 157/300 = 52.3% | 逐轮 CSV | 0.2495 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/vanilla.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/vanilla.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/vanilla.png) |
| `gru-seq2seq` | 151/300 = 50.3% | 逐轮 CSV | 0.2523 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/gru-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/gru-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/gru-seq2seq.png) |
| `train-majority` | 161/300 = 53.7% | 逐轮 CSV | 0.2487 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/train-majority.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/train-majority.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/train-majority.png) |
| `bidirectional-lstm` | 161/300 = 53.7% | 逐轮 CSV | 0.2489 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/bidirectional-lstm.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/bidirectional-lstm.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/bidirectional-lstm.png) |
| `lstm-2path` | 160/300 = 53.3% | 逐轮 CSV | 0.2487 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/lstm-2path.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/lstm-2path.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/lstm-2path.png) |
| `lstm` | 160/300 = 53.3% | 逐轮 CSV | 0.2495 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/lstm.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/lstm.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/lstm.png) |
| `bidirectional-gru` | 161/300 = 53.7% | 逐轮 CSV | 0.2486 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/bidirectional-gru.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/bidirectional-gru.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/bidirectional-gru.png) |
| `gru` | 157/300 = 52.3% | 逐轮 CSV | 0.2497 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/gru.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/gru.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/gru.png) |
| `gru-attention` | 156/300 = 52.0% | 逐轮 CSV | 0.2507 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/gru-attention.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/gru-attention.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/gru-attention.png) |
| `bidirectional-gru-seq2seq` | 154/300 = 51.3% | 逐轮 CSV | 0.2538 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/bidirectional-gru-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/bidirectional-gru-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/bidirectional-gru-seq2seq.png) |
| `cnn-seq2seq` | 159/300 = 53.0% | 逐轮 CSV | 0.2522 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/cnn-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/cnn-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/cnn-seq2seq.png) |
| `bidirectional-lstm-seq2seq` | 151/300 = 50.3% | 逐轮 CSV | 0.2542 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/bidirectional-lstm-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/bidirectional-lstm-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/bidirectional-lstm-seq2seq.png) |
| `lstm-seq2seq-vae` | 162/300 = 54.0% | 逐轮 CSV | 0.2524 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/lstm-seq2seq-vae.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/lstm-seq2seq-vae.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/lstm-seq2seq-vae.png) |
| `gru-seq2seq-vae` | 146/300 = 48.7% | 逐轮 CSV | 0.2567 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/gru-seq2seq-vae.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/gru-seq2seq-vae.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/gru-seq2seq-vae.png) |
| `attention-is-all-you-need` | 161/300 = 53.7% | 逐轮 CSV | 0.2516 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/attention-is-all-you-need.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/attention-is-all-you-need.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/attention-is-all-you-need.png) |
| `dilated-cnn-seq2seq` | 163/300 = 54.3% | 逐轮 CSV | 0.2532 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/dilated-cnn-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/dilated-cnn-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/dilated-cnn-seq2seq.png) |
| `lstm-seq2seq` | 152/300 = 50.7% | 逐轮 CSV | 0.2562 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/lstm-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/lstm-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/lstm-seq2seq.png) |
| `tcn-residual` | 154/300 = 51.3% | 逐轮 CSV | 0.2505 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/tcn-residual.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/tcn-residual.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/tcn-residual.png) |
| `gru-2path` | 161/300 = 53.7% | 逐轮 CSV | 0.2488 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/gru-2path.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/gru-2path.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/gru-2path.png) |
| `gbdt-window` | 153/300 = 51.0% | 逐轮 CSV | 0.2628 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/gbdt-window.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/gbdt-window.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/gbdt-window.png) |
| `bidirectional-vanilla` | 157/300 = 52.3% | 逐轮 CSV | 0.2498 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/bidirectional-vanilla.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/bidirectional-vanilla.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/bidirectional-vanilla.png) |
| `vanilla-2path` | 167/300 = 55.7% | 逐轮 CSV | 0.2503 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/vanilla-2path.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/vanilla-2path.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/vanilla-2path.png) |
| `logit-window` | 158/300 = 52.7% | 逐轮 CSV | 0.2573 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics/models/logit-window.png) · [2](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics/models/logit-window.png) · [3](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics/models/logit-window.png) |

- 第 1 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/aapl/fold-1/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/aapl/fold-1/diagnostics.png)
- 第 2 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/aapl/fold-2/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/aapl/fold-2/diagnostics.png)
- 第 3 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/aapl/fold-3/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/aapl/fold-3/diagnostics.png)

#### TSLA

[三轮合计 CSV](runs/neural-classifiers/rolling-2025/tsla/rolling_metrics.csv)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `vanilla-2path` | 156/300 = 52.0% | 逐轮 CSV | 0.2506 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/vanilla-2path.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/vanilla-2path.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/vanilla-2path.png) |
| `bidirectional-gru` | 147/300 = 49.0% | 逐轮 CSV | 0.2515 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/bidirectional-gru.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/bidirectional-gru.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/bidirectional-gru.png) |
| `lstm-2path` | 162/300 = 54.0% | 逐轮 CSV | 0.2554 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/lstm-2path.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/lstm-2path.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/lstm-2path.png) |
| `train-majority` | 151/300 = 50.3% | 逐轮 CSV | 0.2499 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/train-majority.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/train-majority.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/train-majority.png) |
| `gru-seq2seq-vae` | 157/300 = 52.3% | 逐轮 CSV | 0.2498 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/gru-seq2seq-vae.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/gru-seq2seq-vae.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/gru-seq2seq-vae.png) |
| `lstm` | 160/300 = 53.3% | 逐轮 CSV | 0.2551 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/lstm.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/lstm.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/lstm.png) |
| `gbdt-window` | 155/300 = 51.7% | 逐轮 CSV | 0.2682 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/gbdt-window.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/gbdt-window.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/gbdt-window.png) |
| `bidirectional-gru-seq2seq` | 158/300 = 52.7% | 逐轮 CSV | 0.2498 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/bidirectional-gru-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/bidirectional-gru-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/bidirectional-gru-seq2seq.png) |
| `gru-seq2seq` | 156/300 = 52.0% | 逐轮 CSV | 0.2498 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/gru-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/gru-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/gru-seq2seq.png) |
| `bidirectional-lstm-seq2seq` | 164/300 = 54.7% | 逐轮 CSV | 0.2505 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/bidirectional-lstm-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/bidirectional-lstm-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/bidirectional-lstm-seq2seq.png) |
| `lstm-seq2seq` | 152/300 = 50.7% | 逐轮 CSV | 0.2498 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/lstm-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/lstm-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/lstm-seq2seq.png) |
| `gru-2path` | 154/300 = 51.3% | 逐轮 CSV | 0.2578 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/gru-2path.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/gru-2path.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/gru-2path.png) |
| `gru` | 144/300 = 48.0% | 逐轮 CSV | 0.2566 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/gru.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/gru.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/gru.png) |
| `tcn-residual` | 147/300 = 49.0% | 逐轮 CSV | 0.2552 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/tcn-residual.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/tcn-residual.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/tcn-residual.png) |
| `lstm-seq2seq-vae` | 162/300 = 54.0% | 逐轮 CSV | 0.2499 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/lstm-seq2seq-vae.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/lstm-seq2seq-vae.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/lstm-seq2seq-vae.png) |
| `logit-window` | 155/300 = 51.7% | 逐轮 CSV | 0.2627 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/logit-window.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/logit-window.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/logit-window.png) |
| `bidirectional-lstm` | 152/300 = 50.7% | 逐轮 CSV | 0.2522 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/bidirectional-lstm.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/bidirectional-lstm.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/bidirectional-lstm.png) |
| `bidirectional-vanilla` | 149/300 = 49.7% | 逐轮 CSV | 0.2515 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/bidirectional-vanilla.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/bidirectional-vanilla.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/bidirectional-vanilla.png) |
| `cnn-seq2seq` | 154/300 = 51.3% | 逐轮 CSV | 0.2500 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/cnn-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/cnn-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/cnn-seq2seq.png) |
| `vanilla` | 133/300 = 44.3% | 逐轮 CSV | 0.2517 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/vanilla.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/vanilla.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/vanilla.png) |
| `gru-attention` | 151/300 = 50.3% | 逐轮 CSV | 0.2519 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/gru-attention.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/gru-attention.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/gru-attention.png) |
| `dilated-cnn-seq2seq` | 154/300 = 51.3% | 逐轮 CSV | 0.2500 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/dilated-cnn-seq2seq.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/dilated-cnn-seq2seq.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/dilated-cnn-seq2seq.png) |
| `attention-is-all-you-need` | 144/300 = 48.0% | 逐轮 CSV | 0.2526 | 逐轮 CSV | [1](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics/models/attention-is-all-you-need.png) · [2](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics/models/attention-is-all-you-need.png) · [3](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics/models/attention-is-all-you-need.png) |

- 第 1 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/tsla/fold-1/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/tsla/fold-1/diagnostics.png)
- 第 2 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/tsla/fold-2/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/tsla/fold-2/diagnostics.png)
- 第 3 轮：[metrics.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/predictions.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/confusion.csv) · [calibration.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/calibration.csv) · [summary.json](runs/neural-classifiers/rolling-2025/tsla/fold-3/summary.json) · [diagnostics.png](runs/neural-classifiers/rolling-2025/tsla/fold-3/diagnostics.png)

## 8 种现代结构 + 多数方向基线

### 固定测试

#### GOOG

[metrics.csv](runs/modern-direction/fixed-2025/goog/metrics.csv) · [predictions.csv](runs/modern-direction/fixed-2025/goog/predictions.csv) · [confusion.csv](runs/modern-direction/fixed-2025/goog/confusion.csv) · [calibration.csv](runs/modern-direction/fixed-2025/goog/calibration.csv) · [summary.json](runs/modern-direction/fixed-2025/goog/summary.json) · [diagnostics.png](runs/modern-direction/fixed-2025/goog/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `timesnet-direction` | 90/148 = 60.8% | 60.1% | 0.2396 | 56.1% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/timesnet-direction.png) |
| `patchtst-direction` | 83/148 = 56.1% | 52.9% | 0.2454 | 71.6% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/patchtst-direction.png) |
| `mamba-style-direction` | 86/148 = 58.1% | 50.8% | 0.2477 | 99.3% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/mamba-style-direction.png) |
| `dlinear-direction` | 78/148 = 52.7% | 51.8% | 0.2499 | 56.1% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/dlinear-direction.png) |
| `train-majority` | 85/148 = 57.4% | 50.0% | 0.2460 | 100.0% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/train-majority.png) |
| `nhits-direction` | 85/148 = 57.4% | 50.4% | 0.2497 | 97.3% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/nhits-direction.png) |
| `itransformer-direction` | 85/148 = 57.4% | 55.1% | 0.2434 | 66.2% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/itransformer-direction.png) |
| `tsmixer-direction` | 84/148 = 56.8% | 55.8% | 0.2480 | 57.4% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/tsmixer-direction.png) |
| `tide-direction` | 83/148 = 56.1% | 51.9% | 0.2466 | 78.4% | [查看](runs/modern-direction/fixed-2025/goog/diagnostics/models/tide-direction.png) |

#### AAPL

[metrics.csv](runs/modern-direction/fixed-2025/aapl/metrics.csv) · [predictions.csv](runs/modern-direction/fixed-2025/aapl/predictions.csv) · [confusion.csv](runs/modern-direction/fixed-2025/aapl/confusion.csv) · [calibration.csv](runs/modern-direction/fixed-2025/aapl/calibration.csv) · [summary.json](runs/modern-direction/fixed-2025/aapl/summary.json) · [diagnostics.png](runs/modern-direction/fixed-2025/aapl/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `timesnet-direction` | 79/148 = 53.4% | 49.4% | 0.2556 | 87.2% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/timesnet-direction.png) |
| `mamba-style-direction` | 82/148 = 55.4% | 50.0% | 0.2478 | 100.0% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/mamba-style-direction.png) |
| `nhits-direction` | 82/148 = 55.4% | 50.0% | 0.2498 | 100.0% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/nhits-direction.png) |
| `train-majority` | 82/148 = 55.4% | 50.0% | 0.2478 | 100.0% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/train-majority.png) |
| `patchtst-direction` | 73/148 = 49.3% | 45.7% | 0.2564 | 83.1% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/patchtst-direction.png) |
| `tsmixer-direction` | 72/148 = 48.6% | 45.7% | 0.2530 | 77.0% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/tsmixer-direction.png) |
| `dlinear-direction` | 58/148 = 39.2% | 36.8% | 0.2537 | 70.3% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/dlinear-direction.png) |
| `itransformer-direction` | 76/148 = 51.4% | 48.6% | 0.2515 | 75.7% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/itransformer-direction.png) |
| `tide-direction` | 80/148 = 54.1% | 50.4% | 0.2501 | 83.8% | [查看](runs/modern-direction/fixed-2025/aapl/diagnostics/models/tide-direction.png) |

#### TSLA

[metrics.csv](runs/modern-direction/fixed-2025/tsla/metrics.csv) · [predictions.csv](runs/modern-direction/fixed-2025/tsla/predictions.csv) · [confusion.csv](runs/modern-direction/fixed-2025/tsla/confusion.csv) · [calibration.csv](runs/modern-direction/fixed-2025/tsla/calibration.csv) · [summary.json](runs/modern-direction/fixed-2025/tsla/summary.json) · [diagnostics.png](runs/modern-direction/fixed-2025/tsla/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `timesnet-direction` | 67/148 = 45.3% | 48.1% | 0.2524 | 8.1% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/timesnet-direction.png) |
| `tsmixer-direction` | 75/148 = 50.7% | 50.5% | 0.2498 | 52.7% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/tsmixer-direction.png) |
| `itransformer-direction` | 76/148 = 51.4% | 51.1% | 0.2509 | 53.4% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/itransformer-direction.png) |
| `nhits-direction` | 73/148 = 49.3% | 49.1% | 0.2500 | 52.7% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/nhits-direction.png) |
| `tide-direction` | 73/148 = 49.3% | 49.0% | 0.2492 | 55.4% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/tide-direction.png) |
| `patchtst-direction` | 71/148 = 48.0% | 47.9% | 0.2499 | 51.4% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/patchtst-direction.png) |
| `mamba-style-direction` | 77/148 = 52.0% | 53.4% | 0.2500 | 29.7% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/mamba-style-direction.png) |
| `train-majority` | 79/148 = 53.4% | 50.0% | 0.2492 | 100.0% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/train-majority.png) |
| `dlinear-direction` | 72/148 = 48.6% | 50.1% | 0.2502 | 29.1% | [查看](runs/modern-direction/fixed-2025/tsla/diagnostics/models/dlinear-direction.png) |

### 三轮滚动测试

#### GOOG

[三轮合计 CSV](runs/modern-direction/rolling-2025/goog/rolling_metrics.csv)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `patchtst-direction` | 157/300 = 52.3% | 逐轮 CSV | 0.2479 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/patchtst-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/patchtst-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/patchtst-direction.png) |
| `mamba-style-direction` | 161/300 = 53.7% | 逐轮 CSV | 0.2476 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/mamba-style-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/mamba-style-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/mamba-style-direction.png) |
| `timesnet-direction` | 165/300 = 55.0% | 逐轮 CSV | 0.2461 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/timesnet-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/timesnet-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/timesnet-direction.png) |
| `train-majority` | 162/300 = 54.0% | 逐轮 CSV | 0.2484 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/train-majority.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/train-majority.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/train-majority.png) |
| `tide-direction` | 159/300 = 53.0% | 逐轮 CSV | 0.2519 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/tide-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/tide-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/tide-direction.png) |
| `tsmixer-direction` | 148/300 = 49.3% | 逐轮 CSV | 0.2514 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/tsmixer-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/tsmixer-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/tsmixer-direction.png) |
| `nhits-direction` | 160/300 = 53.3% | 逐轮 CSV | 0.2522 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/nhits-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/nhits-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/nhits-direction.png) |
| `itransformer-direction` | 170/300 = 56.7% | 逐轮 CSV | 0.2481 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/itransformer-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/itransformer-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/itransformer-direction.png) |
| `dlinear-direction` | 158/300 = 52.7% | 逐轮 CSV | 0.2470 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics/models/dlinear-direction.png) · [2](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics/models/dlinear-direction.png) · [3](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics/models/dlinear-direction.png) |

- 第 1 轮：[metrics.csv](runs/modern-direction/rolling-2025/goog/fold-1/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/goog/fold-1/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/goog/fold-1/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/goog/fold-1/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/goog/fold-1/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/goog/fold-1/diagnostics.png)
- 第 2 轮：[metrics.csv](runs/modern-direction/rolling-2025/goog/fold-2/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/goog/fold-2/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/goog/fold-2/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/goog/fold-2/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/goog/fold-2/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/goog/fold-2/diagnostics.png)
- 第 3 轮：[metrics.csv](runs/modern-direction/rolling-2025/goog/fold-3/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/goog/fold-3/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/goog/fold-3/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/goog/fold-3/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/goog/fold-3/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/goog/fold-3/diagnostics.png)

#### AAPL

[三轮合计 CSV](runs/modern-direction/rolling-2025/aapl/rolling_metrics.csv)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `patchtst-direction` | 164/300 = 54.7% | 逐轮 CSV | 0.2501 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/patchtst-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/patchtst-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/patchtst-direction.png) |
| `train-majority` | 161/300 = 53.7% | 逐轮 CSV | 0.2487 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/train-majority.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/train-majority.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/train-majority.png) |
| `mamba-style-direction` | 163/300 = 54.3% | 逐轮 CSV | 0.2486 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/mamba-style-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/mamba-style-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/mamba-style-direction.png) |
| `timesnet-direction` | 161/300 = 53.7% | 逐轮 CSV | 0.2532 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/timesnet-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/timesnet-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/timesnet-direction.png) |
| `tide-direction` | 150/300 = 50.0% | 逐轮 CSV | 0.2535 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/tide-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/tide-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/tide-direction.png) |
| `nhits-direction` | 151/300 = 50.3% | 逐轮 CSV | 0.2516 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/nhits-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/nhits-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/nhits-direction.png) |
| `dlinear-direction` | 122/300 = 40.7% | 逐轮 CSV | 0.2524 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/dlinear-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/dlinear-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/dlinear-direction.png) |
| `itransformer-direction` | 150/300 = 50.0% | 逐轮 CSV | 0.2522 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/itransformer-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/itransformer-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/itransformer-direction.png) |
| `tsmixer-direction` | 135/300 = 45.0% | 逐轮 CSV | 0.2599 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics/models/tsmixer-direction.png) · [2](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics/models/tsmixer-direction.png) · [3](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics/models/tsmixer-direction.png) |

- 第 1 轮：[metrics.csv](runs/modern-direction/rolling-2025/aapl/fold-1/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/aapl/fold-1/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/aapl/fold-1/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/aapl/fold-1/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/aapl/fold-1/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/aapl/fold-1/diagnostics.png)
- 第 2 轮：[metrics.csv](runs/modern-direction/rolling-2025/aapl/fold-2/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/aapl/fold-2/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/aapl/fold-2/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/aapl/fold-2/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/aapl/fold-2/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/aapl/fold-2/diagnostics.png)
- 第 3 轮：[metrics.csv](runs/modern-direction/rolling-2025/aapl/fold-3/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/aapl/fold-3/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/aapl/fold-3/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/aapl/fold-3/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/aapl/fold-3/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/aapl/fold-3/diagnostics.png)

#### TSLA

[三轮合计 CSV](runs/modern-direction/rolling-2025/tsla/rolling_metrics.csv)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `itransformer-direction` | 140/300 = 46.7% | 逐轮 CSV | 0.2541 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/itransformer-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/itransformer-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/itransformer-direction.png) |
| `dlinear-direction` | 147/300 = 49.0% | 逐轮 CSV | 0.2503 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/dlinear-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/dlinear-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/dlinear-direction.png) |
| `timesnet-direction` | 143/300 = 47.7% | 逐轮 CSV | 0.2503 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/timesnet-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/timesnet-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/timesnet-direction.png) |
| `train-majority` | 151/300 = 50.3% | 逐轮 CSV | 0.2499 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/train-majority.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/train-majority.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/train-majority.png) |
| `nhits-direction` | 159/300 = 53.0% | 逐轮 CSV | 0.2510 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/nhits-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/nhits-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/nhits-direction.png) |
| `tide-direction` | 146/300 = 48.7% | 逐轮 CSV | 0.2521 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/tide-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/tide-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/tide-direction.png) |
| `patchtst-direction` | 136/300 = 45.3% | 逐轮 CSV | 0.2532 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/patchtst-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/patchtst-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/patchtst-direction.png) |
| `tsmixer-direction` | 144/300 = 48.0% | 逐轮 CSV | 0.2510 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/tsmixer-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/tsmixer-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/tsmixer-direction.png) |
| `mamba-style-direction` | 148/300 = 49.3% | 逐轮 CSV | 0.2506 | 逐轮 CSV | [1](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics/models/mamba-style-direction.png) · [2](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics/models/mamba-style-direction.png) · [3](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics/models/mamba-style-direction.png) |

- 第 1 轮：[metrics.csv](runs/modern-direction/rolling-2025/tsla/fold-1/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/tsla/fold-1/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/tsla/fold-1/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/tsla/fold-1/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/tsla/fold-1/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/tsla/fold-1/diagnostics.png)
- 第 2 轮：[metrics.csv](runs/modern-direction/rolling-2025/tsla/fold-2/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/tsla/fold-2/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/tsla/fold-2/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/tsla/fold-2/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/tsla/fold-2/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/tsla/fold-2/diagnostics.png)
- 第 3 轮：[metrics.csv](runs/modern-direction/rolling-2025/tsla/fold-3/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/tsla/fold-3/predictions.csv) · [confusion.csv](runs/modern-direction/rolling-2025/tsla/fold-3/confusion.csv) · [calibration.csv](runs/modern-direction/rolling-2025/tsla/fold-3/calibration.csv) · [summary.json](runs/modern-direction/rolling-2025/tsla/fold-3/summary.json) · [diagnostics.png](runs/modern-direction/rolling-2025/tsla/fold-3/diagnostics.png)

## 新股票外部检查

规则冻结后只评估六个主要候选；测试期为 2025-06-02 至 2025-12-31。

### ACN

[metrics.csv](runs/external-symbol-check/acn/metrics.csv) · [predictions.csv](runs/external-symbol-check/acn/predictions.csv) · [confusion.csv](runs/external-symbol-check/acn/confusion.csv) · [calibration.csv](runs/external-symbol-check/acn/calibration.csv) · [summary.json](runs/external-symbol-check/acn/summary.json) · [diagnostics.png](runs/external-symbol-check/acn/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gbdt-window` | 68/148 = 45.9% | 45.8% | 0.2762 | 46.6% | [查看](runs/external-symbol-check/acn/diagnostics/models/gbdt-window.png) |
| `gru-attention` | 62/148 = 41.9% | 42.5% | 0.2654 | 65.5% | [查看](runs/external-symbol-check/acn/diagnostics/models/gru-attention.png) |
| `gru` | 69/148 = 46.6% | 46.9% | 0.2568 | 58.1% | [查看](runs/external-symbol-check/acn/diagnostics/models/gru.png) |
| `logit-window` | 68/148 = 45.9% | 45.9% | 0.2663 | 48.0% | [查看](runs/external-symbol-check/acn/diagnostics/models/logit-window.png) |
| `train-majority` | 71/148 = 48.0% | 50.0% | 0.2501 | 100.0% | [查看](runs/external-symbol-check/acn/diagnostics/models/train-majority.png) |
| `tcn-residual` | 68/148 = 45.9% | 45.5% | 0.2508 | 38.5% | [查看](runs/external-symbol-check/acn/diagnostics/models/tcn-residual.png) |

### RMD

[metrics.csv](runs/external-symbol-check/rmd/metrics.csv) · [predictions.csv](runs/external-symbol-check/rmd/predictions.csv) · [confusion.csv](runs/external-symbol-check/rmd/confusion.csv) · [calibration.csv](runs/external-symbol-check/rmd/calibration.csv) · [summary.json](runs/external-symbol-check/rmd/summary.json) · [diagnostics.png](runs/external-symbol-check/rmd/diagnostics.png)

| 模型 | 正确天数 / 准确率 | 平衡准确率 | Brier ↓ | 预测上涨比例 | 图像 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gru-attention` | 74/148 = 50.0% | 50.0% | 0.2500 | 50.7% | [查看](runs/external-symbol-check/rmd/diagnostics/models/gru-attention.png) |
| `gru` | 75/148 = 50.7% | 50.2% | 0.2499 | 14.9% | [查看](runs/external-symbol-check/rmd/diagnostics/models/gru.png) |
| `logit-window` | 63/148 = 42.6% | 42.9% | 0.2766 | 73.0% | [查看](runs/external-symbol-check/rmd/diagnostics/models/logit-window.png) |
| `train-majority` | 73/148 = 49.3% | 50.0% | 0.2509 | 100.0% | [查看](runs/external-symbol-check/rmd/diagnostics/models/train-majority.png) |
| `gbdt-window` | 72/148 = 48.6% | 48.7% | 0.2668 | 56.1% | [查看](runs/external-symbol-check/rmd/diagnostics/models/gbdt-window.png) |
| `tcn-residual` | 68/148 = 45.9% | 46.1% | 0.2505 | 58.8% | [查看](runs/external-symbol-check/rmd/diagnostics/models/tcn-residual.png) |

## 其他实验与旧任务

- GRU 5/20 日 × 收盘收益率/OHLCV × 3 随机种子：[108 组逐折指标](runs/ablation-study/gru/metrics.csv)、[逐日预测](runs/ablation-study/gru/predictions.csv)。
- 主候选相对基线的成对重抽样：[区间表](runs/direction-paired-uncertainty.csv)。
- 旧版价格回归网络：[18 类网络指标](runs/neural-comparison.csv)，解释见 [价格回归比较](NEURAL_COMPARISON.md)。
- 旧版价格特征对照：[滚动指标](runs/feature-study/rolling/all_tickers.csv)，解释见 [分析报告](ANALYSIS.md)。
- 较早的分类对照：[滚动指标](runs/direction-study/rolling/all_tickers.csv)，解释见 [方向结果](DIRECTION_RESULTS.md)。
- 2026 年补充 CSV 已按用户要求删除；其逐日历史归档未放入公开仓库。相关汇总结论见 [分析报告](ANALYSIS.md) 和 [改进报告](IMPROVEMENT_REPORT.md)，不能作为新模型的首次样本外测试。

完整图像目录见 [RESULTS_GALLERY.md](RESULTS_GALLERY.md)。原始 CSV 是本页显示值的依据；所有模型图均由各测试文件夹中已保存的逐日概率生成。

## 全部已发布结果 CSV（236 个）

按目录给出原始文件，便于复核未在上面汇总表展开的旧价格实验与中间结果。

- `runs/aapl/`：[metrics.csv](runs/aapl/metrics.csv) · [predictions.csv](runs/aapl/predictions.csv) · [training_history.csv](runs/aapl/training_history.csv)
- `runs/aapl-all/`：[metrics.csv](runs/aapl-all/metrics.csv) · [predictions.csv](runs/aapl-all/predictions.csv) · [training_history.csv](runs/aapl-all/training_history.csv)
- `runs/ablation-study/gru/`：[metrics.csv](runs/ablation-study/gru/metrics.csv) · [predictions.csv](runs/ablation-study/gru/predictions.csv) · [seed_summary.csv](runs/ablation-study/gru/seed_summary.csv) · [summary.csv](runs/ablation-study/gru/summary.csv)
- `runs/`：[direction-paired-uncertainty.csv](runs/direction-paired-uncertainty.csv) · [neural-comparison.csv](runs/neural-comparison.csv) · [neural-direction-comparison.csv](runs/neural-direction-comparison.csv) · [neural-direction-selected.csv](runs/neural-direction-selected.csv)
- `runs/direction-study/rolling/aapl/`：[fold_metrics.csv](runs/direction-study/rolling/aapl/fold_metrics.csv) · [predictions.csv](runs/direction-study/rolling/aapl/predictions.csv) · [rolling_metrics.csv](runs/direction-study/rolling/aapl/rolling_metrics.csv) · [selected_by_fold.csv](runs/direction-study/rolling/aapl/selected_by_fold.csv)
- `runs/direction-study/rolling/`：[all_tickers.csv](runs/direction-study/rolling/all_tickers.csv)
- `runs/direction-study/rolling/goog/`：[fold_metrics.csv](runs/direction-study/rolling/goog/fold_metrics.csv) · [predictions.csv](runs/direction-study/rolling/goog/predictions.csv) · [rolling_metrics.csv](runs/direction-study/rolling/goog/rolling_metrics.csv) · [selected_by_fold.csv](runs/direction-study/rolling/goog/selected_by_fold.csv)
- `runs/direction-study/rolling/tsla/`：[fold_metrics.csv](runs/direction-study/rolling/tsla/fold_metrics.csv) · [predictions.csv](runs/direction-study/rolling/tsla/predictions.csv) · [rolling_metrics.csv](runs/direction-study/rolling/tsla/rolling_metrics.csv) · [selected_by_fold.csv](runs/direction-study/rolling/tsla/selected_by_fold.csv)
- `runs/external-symbol-check/acn/`：[calibration.csv](runs/external-symbol-check/acn/calibration.csv) · [confusion.csv](runs/external-symbol-check/acn/confusion.csv) · [metrics.csv](runs/external-symbol-check/acn/metrics.csv) · [predictions.csv](runs/external-symbol-check/acn/predictions.csv) · [training_history.csv](runs/external-symbol-check/acn/training_history.csv)
- `runs/external-symbol-check/`：[all_tickers.csv](runs/external-symbol-check/all_tickers.csv)
- `runs/external-symbol-check/rmd/`：[calibration.csv](runs/external-symbol-check/rmd/calibration.csv) · [confusion.csv](runs/external-symbol-check/rmd/confusion.csv) · [metrics.csv](runs/external-symbol-check/rmd/metrics.csv) · [predictions.csv](runs/external-symbol-check/rmd/predictions.csv) · [training_history.csv](runs/external-symbol-check/rmd/training_history.csv)
- `runs/feature-study/rolling/aapl/`：[fold_metrics.csv](runs/feature-study/rolling/aapl/fold_metrics.csv) · [predictions.csv](runs/feature-study/rolling/aapl/predictions.csv) · [rolling_metrics.csv](runs/feature-study/rolling/aapl/rolling_metrics.csv) · [selected_by_fold.csv](runs/feature-study/rolling/aapl/selected_by_fold.csv)
- `runs/feature-study/rolling/`：[all_tickers.csv](runs/feature-study/rolling/all_tickers.csv)
- `runs/feature-study/rolling/goog/`：[fold_metrics.csv](runs/feature-study/rolling/goog/fold_metrics.csv) · [predictions.csv](runs/feature-study/rolling/goog/predictions.csv) · [rolling_metrics.csv](runs/feature-study/rolling/goog/rolling_metrics.csv) · [selected_by_fold.csv](runs/feature-study/rolling/goog/selected_by_fold.csv)
- `runs/feature-study/rolling/tsla/`：[fold_metrics.csv](runs/feature-study/rolling/tsla/fold_metrics.csv) · [predictions.csv](runs/feature-study/rolling/tsla/predictions.csv) · [rolling_metrics.csv](runs/feature-study/rolling/tsla/rolling_metrics.csv) · [selected_by_fold.csv](runs/feature-study/rolling/tsla/selected_by_fold.csv)
- `runs/goog/`：[metrics.csv](runs/goog/metrics.csv) · [predictions.csv](runs/goog/predictions.csv) · [training_history.csv](runs/goog/training_history.csv)
- `runs/goog-all/`：[metrics.csv](runs/goog-all/metrics.csv) · [predictions.csv](runs/goog-all/predictions.csv) · [training_history.csv](runs/goog-all/training_history.csv)
- `runs/modern-direction/fixed-2025/aapl/`：[calibration.csv](runs/modern-direction/fixed-2025/aapl/calibration.csv) · [confusion.csv](runs/modern-direction/fixed-2025/aapl/confusion.csv) · [metrics.csv](runs/modern-direction/fixed-2025/aapl/metrics.csv) · [predictions.csv](runs/modern-direction/fixed-2025/aapl/predictions.csv) · [training_history.csv](runs/modern-direction/fixed-2025/aapl/training_history.csv)
- `runs/modern-direction/fixed-2025/goog/`：[calibration.csv](runs/modern-direction/fixed-2025/goog/calibration.csv) · [confusion.csv](runs/modern-direction/fixed-2025/goog/confusion.csv) · [metrics.csv](runs/modern-direction/fixed-2025/goog/metrics.csv) · [predictions.csv](runs/modern-direction/fixed-2025/goog/predictions.csv) · [training_history.csv](runs/modern-direction/fixed-2025/goog/training_history.csv)
- `runs/modern-direction/fixed-2025/tsla/`：[calibration.csv](runs/modern-direction/fixed-2025/tsla/calibration.csv) · [confusion.csv](runs/modern-direction/fixed-2025/tsla/confusion.csv) · [metrics.csv](runs/modern-direction/fixed-2025/tsla/metrics.csv) · [predictions.csv](runs/modern-direction/fixed-2025/tsla/predictions.csv) · [training_history.csv](runs/modern-direction/fixed-2025/tsla/training_history.csv)
- `runs/modern-direction/rolling-2025/aapl/fold-1/`：[calibration.csv](runs/modern-direction/rolling-2025/aapl/fold-1/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/aapl/fold-1/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/aapl/fold-1/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/aapl/fold-1/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/aapl/fold-1/training_history.csv)
- `runs/modern-direction/rolling-2025/aapl/fold-2/`：[calibration.csv](runs/modern-direction/rolling-2025/aapl/fold-2/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/aapl/fold-2/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/aapl/fold-2/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/aapl/fold-2/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/aapl/fold-2/training_history.csv)
- `runs/modern-direction/rolling-2025/aapl/fold-3/`：[calibration.csv](runs/modern-direction/rolling-2025/aapl/fold-3/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/aapl/fold-3/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/aapl/fold-3/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/aapl/fold-3/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/aapl/fold-3/training_history.csv)
- `runs/modern-direction/rolling-2025/aapl/`：[fold_metrics.csv](runs/modern-direction/rolling-2025/aapl/fold_metrics.csv) · [rolling_metrics.csv](runs/modern-direction/rolling-2025/aapl/rolling_metrics.csv)
- `runs/modern-direction/rolling-2025/goog/fold-1/`：[calibration.csv](runs/modern-direction/rolling-2025/goog/fold-1/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/goog/fold-1/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/goog/fold-1/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/goog/fold-1/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/goog/fold-1/training_history.csv)
- `runs/modern-direction/rolling-2025/goog/fold-2/`：[calibration.csv](runs/modern-direction/rolling-2025/goog/fold-2/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/goog/fold-2/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/goog/fold-2/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/goog/fold-2/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/goog/fold-2/training_history.csv)
- `runs/modern-direction/rolling-2025/goog/fold-3/`：[calibration.csv](runs/modern-direction/rolling-2025/goog/fold-3/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/goog/fold-3/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/goog/fold-3/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/goog/fold-3/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/goog/fold-3/training_history.csv)
- `runs/modern-direction/rolling-2025/goog/`：[fold_metrics.csv](runs/modern-direction/rolling-2025/goog/fold_metrics.csv) · [rolling_metrics.csv](runs/modern-direction/rolling-2025/goog/rolling_metrics.csv)
- `runs/modern-direction/rolling-2025/tsla/fold-1/`：[calibration.csv](runs/modern-direction/rolling-2025/tsla/fold-1/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/tsla/fold-1/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/tsla/fold-1/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/tsla/fold-1/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/tsla/fold-1/training_history.csv)
- `runs/modern-direction/rolling-2025/tsla/fold-2/`：[calibration.csv](runs/modern-direction/rolling-2025/tsla/fold-2/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/tsla/fold-2/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/tsla/fold-2/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/tsla/fold-2/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/tsla/fold-2/training_history.csv)
- `runs/modern-direction/rolling-2025/tsla/fold-3/`：[calibration.csv](runs/modern-direction/rolling-2025/tsla/fold-3/calibration.csv) · [confusion.csv](runs/modern-direction/rolling-2025/tsla/fold-3/confusion.csv) · [metrics.csv](runs/modern-direction/rolling-2025/tsla/fold-3/metrics.csv) · [predictions.csv](runs/modern-direction/rolling-2025/tsla/fold-3/predictions.csv) · [training_history.csv](runs/modern-direction/rolling-2025/tsla/fold-3/training_history.csv)
- `runs/modern-direction/rolling-2025/tsla/`：[fold_metrics.csv](runs/modern-direction/rolling-2025/tsla/fold_metrics.csv) · [rolling_metrics.csv](runs/modern-direction/rolling-2025/tsla/rolling_metrics.csv)
- `runs/neural-classifiers/fixed-2025/aapl/`：[calibration.csv](runs/neural-classifiers/fixed-2025/aapl/calibration.csv) · [confusion.csv](runs/neural-classifiers/fixed-2025/aapl/confusion.csv) · [metrics.csv](runs/neural-classifiers/fixed-2025/aapl/metrics.csv) · [predictions.csv](runs/neural-classifiers/fixed-2025/aapl/predictions.csv) · [training_history.csv](runs/neural-classifiers/fixed-2025/aapl/training_history.csv)
- `runs/neural-classifiers/fixed-2025/`：[all_tickers.csv](runs/neural-classifiers/fixed-2025/all_tickers.csv)
- `runs/neural-classifiers/fixed-2025/goog/`：[calibration.csv](runs/neural-classifiers/fixed-2025/goog/calibration.csv) · [confusion.csv](runs/neural-classifiers/fixed-2025/goog/confusion.csv) · [metrics.csv](runs/neural-classifiers/fixed-2025/goog/metrics.csv) · [predictions.csv](runs/neural-classifiers/fixed-2025/goog/predictions.csv) · [training_history.csv](runs/neural-classifiers/fixed-2025/goog/training_history.csv)
- `runs/neural-classifiers/fixed-2025/tsla/`：[calibration.csv](runs/neural-classifiers/fixed-2025/tsla/calibration.csv) · [confusion.csv](runs/neural-classifiers/fixed-2025/tsla/confusion.csv) · [metrics.csv](runs/neural-classifiers/fixed-2025/tsla/metrics.csv) · [predictions.csv](runs/neural-classifiers/fixed-2025/tsla/predictions.csv) · [training_history.csv](runs/neural-classifiers/fixed-2025/tsla/training_history.csv)
- `runs/neural-classifiers/rolling-2025/aapl/fold-1/`：[calibration.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/aapl/fold-1/training_history.csv)
- `runs/neural-classifiers/rolling-2025/aapl/fold-2/`：[calibration.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/aapl/fold-2/training_history.csv)
- `runs/neural-classifiers/rolling-2025/aapl/fold-3/`：[calibration.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/aapl/fold-3/training_history.csv)
- `runs/neural-classifiers/rolling-2025/aapl/`：[fold_metrics.csv](runs/neural-classifiers/rolling-2025/aapl/fold_metrics.csv) · [rolling_metrics.csv](runs/neural-classifiers/rolling-2025/aapl/rolling_metrics.csv) · [selected_by_fold.csv](runs/neural-classifiers/rolling-2025/aapl/selected_by_fold.csv)
- `runs/neural-classifiers/rolling-2025/`：[all_tickers.csv](runs/neural-classifiers/rolling-2025/all_tickers.csv)
- `runs/neural-classifiers/rolling-2025/goog/fold-1/`：[calibration.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/goog/fold-1/training_history.csv)
- `runs/neural-classifiers/rolling-2025/goog/fold-2/`：[calibration.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/goog/fold-2/training_history.csv)
- `runs/neural-classifiers/rolling-2025/goog/fold-3/`：[calibration.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/goog/fold-3/training_history.csv)
- `runs/neural-classifiers/rolling-2025/goog/`：[fold_metrics.csv](runs/neural-classifiers/rolling-2025/goog/fold_metrics.csv) · [rolling_metrics.csv](runs/neural-classifiers/rolling-2025/goog/rolling_metrics.csv) · [selected_by_fold.csv](runs/neural-classifiers/rolling-2025/goog/selected_by_fold.csv)
- `runs/neural-classifiers/rolling-2025/tsla/fold-1/`：[calibration.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/tsla/fold-1/training_history.csv)
- `runs/neural-classifiers/rolling-2025/tsla/fold-2/`：[calibration.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/tsla/fold-2/training_history.csv)
- `runs/neural-classifiers/rolling-2025/tsla/fold-3/`：[calibration.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/calibration.csv) · [confusion.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/confusion.csv) · [metrics.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/metrics.csv) · [predictions.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/predictions.csv) · [training_history.csv](runs/neural-classifiers/rolling-2025/tsla/fold-3/training_history.csv)
- `runs/neural-classifiers/rolling-2025/tsla/`：[fold_metrics.csv](runs/neural-classifiers/rolling-2025/tsla/fold_metrics.csv) · [rolling_metrics.csv](runs/neural-classifiers/rolling-2025/tsla/rolling_metrics.csv) · [selected_by_fold.csv](runs/neural-classifiers/rolling-2025/tsla/selected_by_fold.csv)
- `runs/rolling-aapl/fold-1/`：[metrics.csv](runs/rolling-aapl/fold-1/metrics.csv) · [predictions.csv](runs/rolling-aapl/fold-1/predictions.csv) · [training_history.csv](runs/rolling-aapl/fold-1/training_history.csv)
- `runs/rolling-aapl/fold-2/`：[metrics.csv](runs/rolling-aapl/fold-2/metrics.csv) · [predictions.csv](runs/rolling-aapl/fold-2/predictions.csv) · [training_history.csv](runs/rolling-aapl/fold-2/training_history.csv)
- `runs/rolling-aapl/fold-3/`：[metrics.csv](runs/rolling-aapl/fold-3/metrics.csv) · [predictions.csv](runs/rolling-aapl/fold-3/predictions.csv) · [training_history.csv](runs/rolling-aapl/fold-3/training_history.csv)
- `runs/rolling-aapl/`：[fold_metrics.csv](runs/rolling-aapl/fold_metrics.csv) · [rolling_metrics.csv](runs/rolling-aapl/rolling_metrics.csv) · [selected_by_fold.csv](runs/rolling-aapl/selected_by_fold.csv)
- `runs/rolling-goog/fold-1/`：[metrics.csv](runs/rolling-goog/fold-1/metrics.csv) · [predictions.csv](runs/rolling-goog/fold-1/predictions.csv) · [training_history.csv](runs/rolling-goog/fold-1/training_history.csv)
- `runs/rolling-goog/fold-2/`：[metrics.csv](runs/rolling-goog/fold-2/metrics.csv) · [predictions.csv](runs/rolling-goog/fold-2/predictions.csv) · [training_history.csv](runs/rolling-goog/fold-2/training_history.csv)
- `runs/rolling-goog/fold-3/`：[metrics.csv](runs/rolling-goog/fold-3/metrics.csv) · [predictions.csv](runs/rolling-goog/fold-3/predictions.csv) · [training_history.csv](runs/rolling-goog/fold-3/training_history.csv)
- `runs/rolling-goog/`：[fold_metrics.csv](runs/rolling-goog/fold_metrics.csv) · [rolling_metrics.csv](runs/rolling-goog/rolling_metrics.csv) · [selected_by_fold.csv](runs/rolling-goog/selected_by_fold.csv)
- `runs/rolling-tsla/fold-1/`：[metrics.csv](runs/rolling-tsla/fold-1/metrics.csv) · [predictions.csv](runs/rolling-tsla/fold-1/predictions.csv) · [training_history.csv](runs/rolling-tsla/fold-1/training_history.csv)
- `runs/rolling-tsla/fold-2/`：[metrics.csv](runs/rolling-tsla/fold-2/metrics.csv) · [predictions.csv](runs/rolling-tsla/fold-2/predictions.csv) · [training_history.csv](runs/rolling-tsla/fold-2/training_history.csv)
- `runs/rolling-tsla/fold-3/`：[metrics.csv](runs/rolling-tsla/fold-3/metrics.csv) · [predictions.csv](runs/rolling-tsla/fold-3/predictions.csv) · [training_history.csv](runs/rolling-tsla/fold-3/training_history.csv)
- `runs/rolling-tsla/`：[fold_metrics.csv](runs/rolling-tsla/fold_metrics.csv) · [rolling_metrics.csv](runs/rolling-tsla/rolling_metrics.csv) · [selected_by_fold.csv](runs/rolling-tsla/selected_by_fold.csv)
- `runs/tsla/`：[metrics.csv](runs/tsla/metrics.csv) · [predictions.csv](runs/tsla/predictions.csv) · [training_history.csv](runs/tsla/training_history.csv)
- `runs/tsla-all/`：[metrics.csv](runs/tsla-all/metrics.csv) · [predictions.csv](runs/tsla-all/predictions.csv) · [training_history.csv](runs/tsla-all/training_history.csv)

## 全部已发布配置与来源 JSON（51 个）

按目录给出原始文件，便于复核未在上面汇总表展开的旧价格实验与中间结果。

- `runs/aapl/`：[summary.json](runs/aapl/summary.json)
- `runs/aapl-all/`：[summary.json](runs/aapl-all/summary.json)
- `runs/ablation-study/gru/`：[summary.json](runs/ablation-study/gru/summary.json)
- `runs/direction-study/rolling/aapl/`：[summary.json](runs/direction-study/rolling/aapl/summary.json)
- `runs/direction-study/rolling/goog/`：[summary.json](runs/direction-study/rolling/goog/summary.json)
- `runs/direction-study/rolling/tsla/`：[summary.json](runs/direction-study/rolling/tsla/summary.json)
- `runs/external-symbol-check/acn/`：[summary.json](runs/external-symbol-check/acn/summary.json)
- `runs/external-symbol-check/rmd/`：[summary.json](runs/external-symbol-check/rmd/summary.json)
- `runs/feature-study/rolling/aapl/`：[summary.json](runs/feature-study/rolling/aapl/summary.json)
- `runs/feature-study/rolling/goog/`：[summary.json](runs/feature-study/rolling/goog/summary.json)
- `runs/feature-study/rolling/tsla/`：[summary.json](runs/feature-study/rolling/tsla/summary.json)
- `runs/goog/`：[summary.json](runs/goog/summary.json)
- `runs/goog-all/`：[summary.json](runs/goog-all/summary.json)
- `runs/modern-direction/fixed-2025/aapl/`：[summary.json](runs/modern-direction/fixed-2025/aapl/summary.json)
- `runs/modern-direction/fixed-2025/goog/`：[summary.json](runs/modern-direction/fixed-2025/goog/summary.json)
- `runs/modern-direction/fixed-2025/tsla/`：[summary.json](runs/modern-direction/fixed-2025/tsla/summary.json)
- `runs/modern-direction/rolling-2025/aapl/fold-1/`：[summary.json](runs/modern-direction/rolling-2025/aapl/fold-1/summary.json)
- `runs/modern-direction/rolling-2025/aapl/fold-2/`：[summary.json](runs/modern-direction/rolling-2025/aapl/fold-2/summary.json)
- `runs/modern-direction/rolling-2025/aapl/fold-3/`：[summary.json](runs/modern-direction/rolling-2025/aapl/fold-3/summary.json)
- `runs/modern-direction/rolling-2025/goog/fold-1/`：[summary.json](runs/modern-direction/rolling-2025/goog/fold-1/summary.json)
- `runs/modern-direction/rolling-2025/goog/fold-2/`：[summary.json](runs/modern-direction/rolling-2025/goog/fold-2/summary.json)
- `runs/modern-direction/rolling-2025/goog/fold-3/`：[summary.json](runs/modern-direction/rolling-2025/goog/fold-3/summary.json)
- `runs/modern-direction/rolling-2025/tsla/fold-1/`：[summary.json](runs/modern-direction/rolling-2025/tsla/fold-1/summary.json)
- `runs/modern-direction/rolling-2025/tsla/fold-2/`：[summary.json](runs/modern-direction/rolling-2025/tsla/fold-2/summary.json)
- `runs/modern-direction/rolling-2025/tsla/fold-3/`：[summary.json](runs/modern-direction/rolling-2025/tsla/fold-3/summary.json)
- `runs/neural-classifiers/fixed-2025/aapl/`：[summary.json](runs/neural-classifiers/fixed-2025/aapl/summary.json)
- `runs/neural-classifiers/fixed-2025/goog/`：[summary.json](runs/neural-classifiers/fixed-2025/goog/summary.json)
- `runs/neural-classifiers/fixed-2025/tsla/`：[summary.json](runs/neural-classifiers/fixed-2025/tsla/summary.json)
- `runs/neural-classifiers/rolling-2025/aapl/fold-1/`：[summary.json](runs/neural-classifiers/rolling-2025/aapl/fold-1/summary.json)
- `runs/neural-classifiers/rolling-2025/aapl/fold-2/`：[summary.json](runs/neural-classifiers/rolling-2025/aapl/fold-2/summary.json)
- `runs/neural-classifiers/rolling-2025/aapl/fold-3/`：[summary.json](runs/neural-classifiers/rolling-2025/aapl/fold-3/summary.json)
- `runs/neural-classifiers/rolling-2025/goog/fold-1/`：[summary.json](runs/neural-classifiers/rolling-2025/goog/fold-1/summary.json)
- `runs/neural-classifiers/rolling-2025/goog/fold-2/`：[summary.json](runs/neural-classifiers/rolling-2025/goog/fold-2/summary.json)
- `runs/neural-classifiers/rolling-2025/goog/fold-3/`：[summary.json](runs/neural-classifiers/rolling-2025/goog/fold-3/summary.json)
- `runs/neural-classifiers/rolling-2025/tsla/fold-1/`：[summary.json](runs/neural-classifiers/rolling-2025/tsla/fold-1/summary.json)
- `runs/neural-classifiers/rolling-2025/tsla/fold-2/`：[summary.json](runs/neural-classifiers/rolling-2025/tsla/fold-2/summary.json)
- `runs/neural-classifiers/rolling-2025/tsla/fold-3/`：[summary.json](runs/neural-classifiers/rolling-2025/tsla/fold-3/summary.json)
- `runs/rolling-aapl/fold-1/`：[summary.json](runs/rolling-aapl/fold-1/summary.json)
- `runs/rolling-aapl/fold-2/`：[summary.json](runs/rolling-aapl/fold-2/summary.json)
- `runs/rolling-aapl/fold-3/`：[summary.json](runs/rolling-aapl/fold-3/summary.json)
- `runs/rolling-aapl/`：[rolling_summary.json](runs/rolling-aapl/rolling_summary.json)
- `runs/rolling-goog/fold-1/`：[summary.json](runs/rolling-goog/fold-1/summary.json)
- `runs/rolling-goog/fold-2/`：[summary.json](runs/rolling-goog/fold-2/summary.json)
- `runs/rolling-goog/fold-3/`：[summary.json](runs/rolling-goog/fold-3/summary.json)
- `runs/rolling-goog/`：[rolling_summary.json](runs/rolling-goog/rolling_summary.json)
- `runs/rolling-tsla/fold-1/`：[summary.json](runs/rolling-tsla/fold-1/summary.json)
- `runs/rolling-tsla/fold-2/`：[summary.json](runs/rolling-tsla/fold-2/summary.json)
- `runs/rolling-tsla/fold-3/`：[summary.json](runs/rolling-tsla/fold-3/summary.json)
- `runs/rolling-tsla/`：[rolling_summary.json](runs/rolling-tsla/rolling_summary.json)
- `runs/tsla/`：[summary.json](runs/tsla/summary.json)
- `runs/tsla-all/`：[summary.json](runs/tsla-all/summary.json)
