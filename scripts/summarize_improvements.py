"""Summarize diagnostics, small ablations, and frozen external-symbol check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocklab.neural_direction import PRIMARY_NAMES


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def cell(correct: int, total: int) -> str:
    return f"{correct}/{total} = {correct / total:.1%}"


def main() -> None:
    fixed = pd.read_csv(RUNS / "neural-classifiers" / "fixed-2025" / "all_tickers.csv")
    rolling = pd.read_csv(RUNS / "neural-classifiers" / "rolling-2025" / "all_tickers.csv")
    ablation = pd.read_csv(RUNS / "ablation-study" / "gru" / "summary.csv")
    external = pd.read_csv(RUNS / "external-symbol-check" / "all_tickers.csv")
    uncertainty = pd.read_csv(RUNS / "direction-paired-uncertainty.csv")
    fixed_summary = json.loads((RUNS / "neural-classifiers" / "fixed-2025" / "goog" /
                                "summary.json").read_text(encoding="utf-8"))
    fixed_dates = fixed_summary["splits"]["test"]
    lines = [
        "# 涨跌模型的下一轮改进与检查",
        "",
        "这轮继续保持小数据集、短训练。原有 18 种结构仍在；新增 `tcn-residual`（残差时间卷积）和 `gru-attention`（注意力汇总 GRU）。",
        "另加入与网络使用相同 20 日历史输入的逻辑回归和小型梯度提升树，便于比较复杂度。新增结构是待检验候选，不预设它们更准。",
        "主要比较名单事先固定为 `" + "`, `".join(PRIMARY_NAMES) + "`；每轮按验证准确率、同分时 Brier 选模。",
        "这里的上涨标签来自来源保存的 `Close`，不包含股息回报；上游价格参数与局限见[数据来源与口径](DATA_PROVENANCE.md)。",
        "完整 20 类网络与简单模型逐项结果见 [NEURAL_CLASSIFIERS.md](NEURAL_CLASSIFIERS.md)。",
        "",
        "## 主要候选的结果",
        "",
        "| 股票 | 固定期所选候选 | 固定期正确天数 | 固定期基线 | 三轮滚动正确天数 | 滚动基线 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    roll = rolling.set_index("ticker")
    for row in fixed.itertuples(index=False):
        other = roll.loc[row.ticker]
        lines.append(f"| {row.ticker} | {row.primary_selected_models} | "
                     f"{cell(row.primary_correct_days, row.test_days)} | "
                     f"{cell(row.baseline_correct_days, row.test_days)} | "
                     f"{cell(int(other.primary_correct_days), int(other.test_days))} | "
                     f"{cell(int(other.baseline_correct_days), int(other.test_days))} |")
    lines.extend([
        "",
        "GOOG 固定期的注意力 GRU 比基线多对 7 天，但同一选择规则在三轮滚动合计少对 1 天。",
        "AAPL、TSLA 的主要候选在固定和滚动测试中均未超过基线。因此目前没有一致的预测优势。",
        f"固定期的测试日期（{fixed_dates['first_date']} 至 {fixed_dates['last_date']}）与滚动期第二、三轮重叠，不能把它们当作独立重复验证。",
        "2025 年的测试时期此前已被查看，上述结果属于回顾性分析。",
        "",
        "### 滚动三轮分别选中了什么",
        "",
        "每轮沿时间向前重新扩大训练期，再用当轮验证期从预定六候选中选一个。",
        "上表的 300 天是三次选择规则各预测 100 天的合计，不能理解为同一个模型权重连续预测 300 天。",
        "",
        "| 股票 | 测试期 | 验证期所选主要候选 | 正确天数 | 多数方向基线 | 相差天数 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for ticker in ("GOOG", "AAPL", "TSLA"):
        for fold in (1, 2, 3):
            folder = RUNS / "neural-classifiers" / "rolling-2025" / ticker.lower() / f"fold-{fold}"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            metrics = pd.read_csv(folder / "metrics.csv").set_index("model")
            chosen = summary["primary_selected_by_validation"]
            selected_correct = int(metrics.loc[chosen].test_correct_days)
            baseline_correct = int(metrics.loc["train-majority"].test_correct_days)
            days = int(metrics.loc[chosen].test_days)
            dates = summary["splits"]["test"]
            lines.append(f"| {ticker} | {dates['first_date']} 至 {dates['last_date']} | "
                         f"{chosen} | {selected_correct}/{days} | {baseline_correct}/{days} | "
                         f"{selected_correct - baseline_correct:+d} |")
    lines.extend([
        "",
        "GOOG 第二轮所选的梯度提升树比基线少对 6 天，抵消了另外两轮的小幅领先。",
        "AAPL 第一轮和 TSLA 第一轮都选中了基线；这两轮的“所选模型”并不是神经网络。",
        "",
        "### 与基线差距的波动范围",
        "",
        "对同一天的‘所选模型是否正确’减去‘多数方向基线是否正确’，采用相邻 5 日移动区块、10000 次重抽样、种子 42。",
        "下表的 95% 区间是描述性的；这些历史测试期已被查看，不能把区间解释为未来盈利保证或新的确认性检验。",
        "",
        "| 测试期 | 股票 | 准确率差（模型减基线） | 描述性 95% 区间 |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in uncertainty.itertuples(index=False):
        lines.append(f"| {row.period} | {row.ticker} | {100 * row.accuracy_gain:+.1f} 个百分点 | "
                     f"{100 * row.block_bootstrap_95pct_low:+.1f} 至 "
                     f"{100 * row.block_bootstrap_95pct_high:+.1f} 个百分点 |")
    lines.extend([
        "",
        "GOOG 固定期虽多对 7 天，区间仍跨过零；RMD 只多对 1 天，区间更宽。逐日成对结果在 `runs/direction-paired-uncertainty.csv`。",
        "",
        "## 模型到底在猜什么",
        "",
        "每个测试文件夹新增 `confusion.csv`（所有模型的判断四格表）、`calibration.csv`（10 个概率分箱的样本数和真实上涨频率）、",
        "`diagnostics.png`（验证期所选主要候选与基线的混淆矩阵、概率校准图和预测概率分布），",
        "以及 `diagnostics/models/<模型名>.png`（每个候选单独的混淆矩阵、预测上涨比例、校准图和概率分布）。",
        "14 个测试文件夹的 288 张单模型图由已保存的逐日预测生成，无需重新训练。",
        "例如 GOOG 固定期的多数方向基线每天都猜涨；注意力 GRU 在 148 天中猜涨 115 天，",
        "把 20 个真实未上涨日判断正确，同时仍把 43 个未上涨日误判成上涨。",
        "RMD 外部检查中，注意力 GRU 的概率仅约 0.498–0.505；这类接近 50% 的输出不宜读作有把握的判断。",
        "分箱里的天数可能很少，Brier 较低也不单独证明概率已校准。",
        "",
        "### 初学者可以怎样读结果",
        "",
        "“预测上涨”指模型给出的概率达到 50% 的天数；“实际上涨”指次日收盘价真的高于当天的天数。",
        "正确率是判断对的天数除以测试天数。平衡准确率把上涨日和未上涨日各自判断对的比例取平均，",
        "能看出只猜一个方向造成的偏差。Brier 是上涨概率与真实结果（涨为 1、不涨为 0）的平方误差平均，越小越好。",
        "下表只展示每只股票固定期或外部检查中**由验证期选出的主要候选**；括号内是同日期多数方向基线的 Brier。",
        "",
        "| 测试 | 股票 | 验证期所选 | 实际上涨 | 预测上涨 | 判断正确 | 平衡准确率 | Brier（基线） |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    selected_rows = []
    for period, parent in (("固定期", RUNS / "neural-classifiers" / "fixed-2025"),
                           ("外部股票", RUNS / "external-symbol-check")):
        for ticker in (("GOOG", "AAPL", "TSLA") if period == "固定期" else ("ACN", "RMD")):
            folder = parent / ticker.lower()
            metrics = pd.read_csv(folder / "metrics.csv").set_index("model")
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            selected = summary["primary_selected_by_validation"]
            item = metrics.loc[selected]
            baseline = metrics.loc["train-majority"]
            days = int(item.test_days)
            actual_up = round(float(item.test_actual_up_rate) * days)
            predicted_up = round(float(item.test_predicted_up_rate) * days)
            selected_rows.append((item, baseline))
            lines.append(f"| {period} | {ticker} | {selected} | {actual_up}/{days} | "
                         f"{predicted_up}/{days} | {int(item.test_correct_days)}/{days} | "
                         f"{item.test_balanced_accuracy:.1%} | "
                         f"{item.test_brier:.4f}（{baseline.test_brier:.4f}） |")
    lines.extend([
        "",
        "例如 GOOG 固定期所选模型比基线多对 7 天，且平衡准确率超过 50%；但其他股票和滚动期没有重现稳定优势。",
    ])
    if all(float(baseline.test_predicted_up_rate) == 1.0
           for _, baseline in selected_rows):
        lines.append("这五段测试中的多数方向基线每天都猜涨，因此它的平衡准确率均为 50%；"
                     "只看普通正确率容易高估这种单方向判断的本事。")
    calibration = pd.read_csv(RUNS / "neural-classifiers" / "fixed-2025" / "goog" /
                              "calibration.csv")
    goog_selected = json.loads((RUNS / "neural-classifiers" / "fixed-2025" / "goog" /
                                "summary.json").read_text(encoding="utf-8"))[
                                    "primary_selected_by_validation"]
    example = calibration.loc[(calibration.model == goog_selected) &
                              (calibration.bin == 6) & (calibration.days > 0)]
    if not example.empty:
        row = example.iloc[0]
        observed_up = round(float(row.actual_up_rate) * int(row.days))
        lines.extend([
            f"概率校准可以这样读：GOOG 的 {goog_selected} 给出 60%–70% 上涨概率的日子有 "
            f"{int(row.days)} 天，平均预测 {row.mean_predicted_probability:.1%}，实际 "
            f"{observed_up}/{int(row.days)} 天上涨（{row.actual_up_rate:.1%}）。",
            "这是一个历史分箱，不足以证明未来预测概率可靠；还要看其他分箱、样本量和新的未见日期。",
        ])
    lines.extend([
        "",
        "## 小规模受控对照：GRU",
        "",
        "固定相同的目标日期、三个滚动测试期、网络宽度和训练规则，仅改变输入为 5/20 日、收盘收益率/完整 OHLCV，",
        "并在每种配置重复随机种子 42、123、2026。下表是**每个种子各测试 300 天后**的准确率均值与种子范围；",
        "三个种子重复使用同一批日期，不应当成 900 个独立测试日。",
        "",
        "| 股票 | 输入 | 历史天数 | 三种子平均准确率（范围） | 多数方向基线 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for ticker in ("GOOG", "AAPL", "TSLA"):
        for feature in ("close-return", "ohlcv"):
            for days in (5, 20):
                row = ablation.loc[(ablation.ticker == ticker) &
                                   (ablation.feature_set == feature) &
                                   (ablation.lookback == days)].iloc[0]
                lines.append(f"| {ticker} | {feature} | {days} | "
                             f"{row.mean_accuracy:.1%}（{row.min_accuracy:.1%}–{row.max_accuracy:.1%}） "
                             f"| {row.baseline_accuracy:.1%} |")
    lines.extend([
        "",
        "这组数据没有显示 OHLCV 一定优于只用收盘收益率，也没有显示 20 日窗口一定优于 5 日。",
        "例如 GOOG 的 20 日两种输入平均仅差约 0.1 个百分点；TSLA 在不同配置与种子间波动较大。",
        "本对照用于诊断敏感性，不按表中测试成绩反过来改主要候选或外部检查规则。",
        "逐轮指标、每个种子的 300 日汇总和逐日概率在 `runs/ablation-study/gru/`。",
        "",
        "## 规则冻结后的新股票检查",
        "",
        "在查看 ACN、RMD 结果前写定了 [外部检查规则](EXTERNAL_CHECK_PROTOCOL.md)，只评上述六个主要候选。",
        "最初计划的 JPM、XOM 不在原数据源的可用名单中；因文件可用性改用 ACN、RMD，不依据模型结果换股票。",
        "每只仅增加约 1000 行历史数据，测试期各 148 天。",
        "",
        "| 股票 | 验证期选中 | 测试正确天数 | 多数方向基线 |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in external.itertuples(index=False):
        lines.append(f"| {row.ticker} | {row.primary_selected_models} | "
                     f"{cell(row.primary_correct_days, row.test_days)} | "
                     f"{cell(row.baseline_correct_days, row.test_days)} |")
    lines.extend([
        "",
        "ACN 比基线少对 3 天，RMD 多对 1 天；新增模型没有在两只新股票上形成一致优势。",
        "这是换股票的历史检查，不是未来新交易日的前瞻验证。数据和逐日预测保存在 `data/external-check/` 与 `runs/external-symbol-check/`。",
        "此轮结束后应保留当前配置，等待新的交易日再做时间上真正向前的检查。",
        "",
        "## 复现",
        "",
        "```powershell",
        "python -m stocklab.neural_direction --mode fixed-2025",
        "python -m stocklab.neural_direction --mode rolling-2025",
        "python -m stocklab.ablation_study",
        "python -m scripts.render_all_model_diagnostics",
        "python -m scripts.direction_paired_uncertainty",
        "python -m scripts.verify_direction_reports",
        "python -m scripts.summarize_neural_classifiers",
        "python -m scripts.summarize_improvements",
        "```",
        "",
        "`stocklab.external_symbol_check` 默认保护原始外部结果，不重复覆盖。",
        "",
    ])
    from scripts.document_links import write_report
    path = write_report(ROOT, "IMPROVEMENT_REPORT.md", "\n".join(lines))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
