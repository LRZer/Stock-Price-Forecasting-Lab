# 教学数据：本地下载与核对

公开仓库只保存来源说明和冻结哈希，不再分发 Pubmarks 股票 CSV。安装项目依赖后，在项目根目录下载 2022—2025 年的小切片：

```powershell
python scripts/download_data.py
python scripts/download_data.py --tickers ACN RMD --output-dir data/external-check
```

| 本地目录 | 股票 | 日期 | 每只行数 | 用途 |
| --- | --- | --- | ---: | --- |
| `raw/` | GOOG、AAPL、TSLA | 2022-01-03—2025-12-31 | 1003 | 主要训练、验证和 2025 年测试 |
| `external-check/` | ACN、RMD | 2022-01-03—2025-12-31 | 1003 | 固定六候选规则的换股票检查 |

每个 CSV 含 `Date`、`Open`、`High`、`Low`、`Close`、`Volume`。主分类任务用过去 20 个交易日的六个 OHLCV 相对特征，预测下一交易日的 `Close` 是否严格高于当日 `Close`。固定测试每只股票有 688 个训练目标、147 个验证目标、148 个测试目标；窗口前 20 行只提供历史输入。三轮滚动测试每轮各有 100 个验证目标和 100 个测试目标。

旧报告在生成时记录了原始文件 SHA-256；三只主股票的冻结值也在 [`FUTURE_BASE_HASHES.json`](../FUTURE_BASE_HASHES.json)。运行 `python -m scripts.verify_direction_reports` 和 `python -m scripts.verify_modern_direction` 可核对下载文件与已保存报告。上游会更新数据，若重下载后的旧行已变，核验会失败；此时旧报告仍可阅读，但不能声称当前文件能精确重现旧训练。不要为了让数值吻合而悄悄改写已保存的测试结果。

2026 年补充 CSV 已按用户要求删除，公开仓库也不包含它的逐日历史归档。2026 年已查看的旧结论只在项目分析文档中作为历史说明；真正未见交易日的评测另按根目录的冻结协议进行。

来源：[Pubmarks Datasets](https://github.com/Pubmarks/datasets)。上游抓取代码使用 yfinance `auto_adjust=False`；本项目采用保存的 `Close`，不是含股息总回报。详情见[数据来源与口径](../DATA_PROVENANCE.md)。
