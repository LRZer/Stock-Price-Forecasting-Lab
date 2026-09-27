# 股票日线数据的来源与价格口径

## 数据怎样来到项目

本项目的 GOOG、AAPL、TSLA、ACN、RMD 日线来自 [Pubmarks Datasets](https://github.com/Pubmarks/datasets) 的每股 `ohlcv.csv`。上游[每日更新工作流](https://github.com/Pubmarks/datasets/blob/main/.github/workflows/stocks-ohlcv-daily.yaml)调用其 `yf ohlcv` 命令；[抓取代码](https://github.com/Pubmarks/datasets/blob/main/scripts/python/yf/lib/ohlcv_fetch.py)用 yfinance 下载日线，参数包括 `auto_adjust=False`、`actions=False`，并保存日期、Open、High、Low、Close、Volume。它没有将 `Adj Close` 写入该 CSV。项目本地下载程序只筛选日期、检查基本数值，并把所需区间另存为教学用小文件，不做价格调整。

上述上游代码在 2026-09-27 核查，保存的[抓取代码版本](https://github.com/Pubmarks/datasets/blob/1665548302f6f396fcae70f73cf7ea0e141446d4/scripts/python/yf/lib/ohlcv_fetch.py)和[写入流程版本](https://github.com/Pubmarks/datasets/blob/1665548302f6f396fcae70f73cf7ea0e141446d4/scripts/python/yf/lib/main.py)可供复核。数据文件没有附逐行生成版本号，因此不能证明所有旧行都由这一个提交生成；本地固定文件的哈希是复现实验数值的依据。

## “上涨”究竟指什么

分类标签是“下一交易日的 `Close` 严格高于当天的 `Close`”。它只对应上游保存的收盘价序列；没有把现金股息加回，也不能当作含股息的总回报方向。相等的收盘价归入“未上涨”。模型输入和标签始终使用同一来源及同一口径。

`auto_adjust=False` 说明上游没有用 `Adj Close` 自动调整 OHLC；它**不保证** Yahoo 历史价格在拆股或更正后绝不变化。上游[写入流程](https://github.com/Pubmarks/datasets/blob/main/scripts/python/yf/lib/main.py)会重取当前年份，保留此前年份的已存行。跨年或公司行动附近可能产生价格衔接问题，未来新日期评测前应先检查边界和异常跳变；若出现问题，保存原数据并单独说明，不能看测试成绩后悄悄换来源或修正标签。

## 可复现边界

- 2022–2025 年三只主要股票的固定 CSV 仍保存在本地并由 `FUTURE_BASE_HASHES.json` 锁定。本地 2026 年补充 CSV 已应用户要求删除；清单只保留其原始哈希。未来如需用作训练前的历史上下文，须临时从相同来源取回且哈希完全一致，否则停止评测。
- 2026-09-27 运行只读 `python -m scripts.audit_source_drift`：GOOG、AAPL、TSLA 各 1187 行旧数据与当前来源的日期及五个 OHLCV 数值列完全一致，缺失日期、额外日期和不同数值单元均为 0。未来正式运行前仍要重查，不能用这次结果替代届时核验。
- 新 100 日评测继续从同一 Pubmarks CSV 获取。下载时间、软件版本、各文件哈希和逐日预测都要随正式结果保存。
- Pubmarks 公开提供文件，但本项目没有据此认定可以任意再分发；公开发布前仍需核对其授权。项目用于模型学习与结果分析，不提供交易信号。
