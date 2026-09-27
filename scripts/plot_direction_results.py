"""Plot learner-friendly accuracy versus the train-majority baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CHINESE_FONT = Path("C:/Windows/Fonts/msyh.ttc")


def main() -> None:
    if CHINESE_FONT.is_file():
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=CHINESE_FONT).get_name()
    base = ROOT / "runs" / "direction-study"
    periods = (("rolling", "2025 年滚动评测：300 天"),
               ("retrospective-2026", "2026 年回顾性评测：184 天"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (folder, title) in zip(axes, periods):
        report = pd.read_csv(base / folder / "all_tickers.csv")
        positions = np.arange(len(report))
        width = 0.36
        selected = ax.bar(positions - width / 2, report["selected_accuracy"] * 100,
                          width, label="验证集选中的模型", color="#3377bb")
        baseline = ax.bar(positions + width / 2, report["baseline_accuracy"] * 100,
                          width, label="训练期多数方向基线", color="#999999")
        ax.bar_label(selected, fmt="%.1f%%", padding=3, fontsize=9)
        ax.bar_label(baseline, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_xticks(positions, report["ticker"])
        ax.set_ylim(0, 70)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("次日涨跌判断正确率（%）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("判断正确的天数 ÷ 测试天数；同时对照简单基线")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(base / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
