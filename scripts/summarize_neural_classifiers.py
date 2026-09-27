"""Build the learner-facing report from saved direct-classification runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocklab.models import DIRECTION_NEURAL_MODELS
from stocklab.neural_direction import PRIMARY_NAMES, SIMPLE_NAMES


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "neural-classifiers"
TICKERS = ("GOOG", "AAPL", "TSLA")
NAMES = ("train-majority",) + SIMPLE_NAMES + DIRECTION_NEURAL_MODELS


def score(correct: int, total: int) -> str:
    return f"{correct}/{total} = {correct / total:.1%}"


def main() -> None:
    fixed = {}
    rolling = {}
    fixed_summary = {}
    for ticker in TICKERS:
        slug = ticker.lower()
        fixed[ticker] = pd.read_csv(RUNS / "fixed-2025" / slug / "metrics.csv").set_index("model")
        rolling[ticker] = pd.read_csv(RUNS / "rolling-2025" / slug / "rolling_metrics.csv").set_index("model")
        fixed_summary[ticker] = json.loads((RUNS / "fixed-2025" / slug / "summary.json")
                                           .read_text(encoding="utf-8"))
        if set(fixed[ticker].index) != set(NAMES) or set(rolling[ticker].index) != set(NAMES):
            raise ValueError(f"Incomplete classifier report for {ticker}")

    lines = [
        "# 20 类神经网络：直接预测次日上涨概率",
        "",
        "## 任务与读法",
        "",
        "每个网络使用预测日前已知的 20 个交易日 OHLCV 历史特征，直接学习次日收盘价是否高于当日收盘价。",
        "这里的收盘价是来源保存的 yfinance `auto_adjust=False` 下的 `Close`，不是 `Adj Close`；股息不计入标签。详见[数据来源与口径](DATA_PROVENANCE.md)。",
        "持平归入“未上涨”。网络输出上涨概率，达到 50% 判为上涨；准确率是预测正确天数除以测试天数。",
        "训练期多数方向基线每天猜训练期更常见的方向。数据按时间先后切分，归一化只用训练期数据。",
        "每个网络以验证期 Brier 误差选择训练轮次；候选以验证期准确率选择，同分再比 Brier。",
        "原有 18 种结构继续保留，新增残差时间卷积网络与带注意力汇总的 GRU；另加入使用相同日期和输入历史的逻辑回归、梯度提升树。",
        "主要比较名单事先固定为多数方向、两个传统模型、普通 GRU、新增的两个网络；其余结构用于教学观察。",
        "所有模型和基线使用同一测试日期，测试集不参与选模。",
        "",
        "## 固定切分：2025-06-02 至 2025-12-31",
        "",
        "每只股票测试 148 天。下面列出全部候选的成绩；不要按测试列挑模型。",
        "",
        "| 模型 | GOOG | AAPL | TSLA |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in NAMES:
        cells = []
        for ticker in TICKERS:
            row = fixed[ticker].loc[name]
            cells.append(score(int(row.test_correct_days), int(row.test_days)))
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines.extend([
        "",
        "主要候选名单：`" + "`, `".join(PRIMARY_NAMES) + "`。下表只在这六者之间按验证集选模。",
        "",
        "| 股票 | 主要候选中验证集选中 | 测试正确天数 / 准确率 | 多数方向基线 | 所选模型 Brier / 基线 Brier |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for ticker in TICKERS:
        chosen = fixed_summary[ticker]["primary_selected_by_validation"]
        row = fixed[ticker].loc[chosen]
        base = fixed[ticker].loc["train-majority"]
        lines.append(f"| {ticker} | {chosen} | {score(int(row.test_correct_days), int(row.test_days))} "
                     f"| {score(int(base.test_correct_days), int(base.test_days))} "
                     f"| {row.test_brier:.3f} / {base.test_brier:.3f} |")
    lines.extend([
        "",
        "## 三轮滚动测试",
        "",
        "每轮先训练、再验证、最后测试 100 个交易日；三个测试窗口不重叠。",
        "下表把同名模型在三轮中的正确天数相加；它展示结构的跨窗口表现，不能据测试分数倒过来选模。",
        "",
        "| 模型 | GOOG（300 天） | AAPL（300 天） | TSLA（300 天） |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in NAMES:
        cells = []
        for ticker in TICKERS:
            row = rolling[ticker].loc[name]
            cells.append(score(int(row.total_correct_days), int(row.total_test_days)))
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines.extend([
        "",
        "逐轮只在六个主要候选中根据验证期选模后，实际得到：",
        "",
        "| 股票 | 三轮所选候选正确天数 / 准确率 | 多数方向基线 |",
        "| --- | ---: | ---: |",
    ])
    selected = pd.read_csv(RUNS / "rolling-2025" / "all_tickers.csv").set_index("ticker")
    for ticker in TICKERS:
        row = selected.loc[ticker]
        lines.append(f"| {ticker} | {score(int(row.primary_correct_days), int(row.test_days))} "
                     f"| {score(int(row.baseline_correct_days), int(row.test_days))} |")
    lines.extend([
        "",
        "完整候选名单中的验证选模结果另见 `all_tickers.csv`，用于观察比较范围变大后的选择变化；主要结论以上面的固定六候选为准。",
        "固定切分中 GOOG 的注意力 GRU 较基线多对 7 天，但 AAPL、TSLA 落后；三轮滚动中三只股票的主要候选都没有超过基线。",
        "差距随时期改变，新增结构没有给出稳定优势。很多网络的预测概率集中在 50% 附近，或几乎总猜同一方向。",
        "每个测试文件夹的 `confusion.csv` 有所有候选的四格计数；`calibration.csv` 有 10 个概率分箱的样本数、平均预测概率与实际上涨比例；",
        "`diagnostics.png` 展示主要候选选中者与基线的混淆矩阵、校准图和概率分布。样本较少的分箱不宜过度解读。",
        "`diagnostics/models/<模型名>.png` 为每个候选单独展示混淆矩阵、实际与预测上涨比例、概率校准图和概率分布；滚动测试每轮也有对应图。",
        "这些 2025 年数据已在项目之前的实验中查看，所以本报告是回顾性学习比较，不能当作全新样本外验证。",
        "",
        "## 复现与逐日数据",
        "",
        "```powershell",
        "python -m stocklab.neural_direction --mode fixed-2025",
        "python -m stocklab.neural_direction --mode rolling-2025",
        "python -m scripts.render_all_model_diagnostics",
        "python -m scripts.verify_direction_reports",
        "python -m scripts.summarize_neural_classifiers",
        "```",
        "",
        "`runs/neural-classifiers/fixed-2025/<股票小写>/metrics.csv` 有每个模型的验证和测试指标；",
        "`predictions.csv` 有每日真实标签、上涨概率和判断；`summary.json` 有数据哈希、日期、候选名单和设置；",
        "`models/` 存放网络权重。滚动测试的逐轮文件在 `runs/neural-classifiers/rolling-2025/`。",
        "旧版价格模型换算涨跌准确率的结果仍单独保留在 [NEURAL_DIRECTION.md](NEURAL_DIRECTION.md)。",
        "",
    ])
    from scripts.document_links import write_report
    path = write_report(ROOT, "NEURAL_CLASSIFIERS.md", "\n".join(lines))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
