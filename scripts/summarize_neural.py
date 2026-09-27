"""Combine the three full 18-neural-model reports without retraining."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocklab.models import NEURAL_MODELS


ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("GOOG", "AAPL", "TSLA")


def main() -> None:
    reports = {}
    rows = []
    selected = {}
    for ticker in TICKERS:
        report = pd.read_csv(ROOT / "runs" / f"{ticker.lower()}-all" / "metrics.csv")
        reports[ticker] = report.set_index("model")
        selected[ticker] = report.iloc[0]["model"]
        baseline = float(reports[ticker].loc["naive", "test_mae"])
        for name in NEURAL_MODELS:
            result = reports[ticker].loc[name]
            rows.append({
                "ticker": ticker,
                "model": name,
                "selected_by_validation": name == selected[ticker],
                "validation_mae": result["validation_mae"],
                "test_mae": result["test_mae"],
                "test_rmse": result["test_rmse"],
                "test_mape_pct": result["test_mape_pct"],
                "test_mae_improvement_vs_naive_pct": result["test_mae_improvement_vs_naive_pct"],
                "naive_test_mae": baseline,
                "epochs_run": int(result["epochs_run"]),
            })

    pd.DataFrame(rows).to_csv(ROOT / "runs" / "neural-comparison.csv", index=False,
                              float_format="%.6f")
    lines = [
        "# 18 类神经网络：三只股票的同条件对照",
        "",
        "任务：过去 20 个交易日的收盘价 → 下一交易日收盘价。每只股票各 1003 行，",
        "按时间分为 688 个训练目标、147 个验证目标和 148 个测试目标；",
        "测试日期均为 2025-06-02 至 2025-12-31。神经网络使用相同的隐藏维度 32、",
        "最多 15 轮、随机种子 42，并在同一张 RTX 4060 上训练。",
        "这些模型是统一任务下的教学版实现，并非旧 TensorFlow 笔记本的逐行复刻。",
        "",
        "MAE 是平均绝对价格误差，越低越好。括号内是相对该股票“昨日收盘价”",
        "基线的 MAE 改善率；正数表示更好。不同股票的绝对 MAE 不宜直接比较。",
        "星号 ★ 表示按**验证 MAE**选中的模型，测试成绩不参与选择。",
        "",
        "| 模型 | GOOG 测试 MAE（相对基线） | AAPL 测试 MAE（相对基线） | TSLA 测试 MAE（相对基线） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in NEURAL_MODELS:
        values = []
        for ticker in TICKERS:
            result = reports[ticker].loc[name]
            marker = " ★" if name == selected[ticker] else ""
            values.append(f"{result['test_mae']:.3f} ({result['test_mae_improvement_vs_naive_pct']:+.2f}%){marker}")
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines += ["| 基线 | " + " | ".join(
        f"{reports[ticker].loc['naive', 'test_mae']:.3f}" for ticker in TICKERS) + " |", ""]
    lines += [
        "## 结果边界", "",
        "GOOG 按验证集选出的双向 GRU 在测试期 MAE 为 3.212，基线为 3.068；",
        "AAPL 按验证集选出的双向 LSTM 为 2.185，基线为 2.200；",
        "TSLA 按验证集选出的双向 LSTM 为 9.870，基线为 9.381。",
        "个别模型在测试期取得更低误差，但不能看完测试分数后改选并仍称其为样本外选模结果。",
        "三只股票的模型表现也受具体时期影响，不能仅凭这一次切分得出架构优劣结论。",
        "2025 年的测试时期已经多次用于项目分析，后续改动不能再把它当作全新留出期。",
        "",
        "每只股票的完整验证 MAE、测试 MAE、RMSE、MAPE、训练轮数与预测明细分别在",
        "`runs/goog-all/`、`runs/aapl-all/` 和 `runs/tsla-all/`。",
        "机器可读的 54 行汇总见 `runs/neural-comparison.csv`。",
        "2026 年的 OHLCV 特征实验使用另一套数据与模型，应分别阅读 `ANALYSIS.md`。",
        "",
    ]
    from scripts.document_links import write_report
    write_report(ROOT, "NEURAL_COMPARISON.md", "\n".join(lines))
    print(f"Saved 54 neural results and summary; selected: {selected}")


if __name__ == "__main__":
    main()
