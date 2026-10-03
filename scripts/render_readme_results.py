"""Render README result charts and complete model-level test tables from saved CSVs.

This reads archived reports only. It never trains models or changes frozen results.
Run: python -m scripts.render_readme_results
Check: python -m scripts.render_readme_results --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ASSETS = ROOT / "docs" / "assets" / "results"
BEGIN = "<!-- BEGIN GENERATED README RESULTS -->"
END = "<!-- END GENERATED README RESULTS -->"
MAIN = ROOT / "runs" / "neural-classifiers"
MODERN = ROOT / "runs" / "modern-direction"
EXTERNAL = ROOT / "runs" / "external-symbol-check"
TICKERS = ("GOOG", "AAPL", "TSLA")
CONFUSION = ("true_non_up", "false_up", "missed_up", "true_up")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def read_case(folder: Path) -> pd.DataFrame:
    metrics = pd.read_csv(folder / "metrics.csv").set_index("model", drop=False)
    confusion = pd.read_csv(folder / "confusion.csv").set_index("model")
    assert set(metrics.index) == set(confusion.index), folder
    result = metrics.copy()
    for key in CONFUSION:
        result[key] = confusion.loc[result.index, key]
    n = result["test_days"].astype(int)
    assert ((result.true_non_up + result.false_up + result.missed_up + result.true_up) == n).all()
    assert ((result.true_non_up + result.true_up) == result.test_correct_days).all()
    return result


def read_rolling(folder: Path) -> pd.DataFrame:
    cases = [read_case(folder / f"fold-{index}") for index in (1, 2, 3)]
    order = cases[0].index
    assert all(set(case.index) == set(order) for case in cases)
    frames = [case.loc[order] for case in cases]
    result = frames[0].copy()
    for key in (*CONFUSION, "test_correct_days", "test_days"):
        result[key] = sum(frame[key] for frame in frames)
    total = result.test_days.astype(int)
    result["test_accuracy"] = result.test_correct_days / total
    result["test_brier"] = sum(frame.test_brier * frame.test_days for frame in frames) / total
    result["test_predicted_up_rate"] = (result.false_up + result.true_up) / total
    result["test_actual_up_rate"] = (result.missed_up + result.true_up) / total
    result["test_balanced_accuracy"] = 0.5 * (
        result.true_up / (result.true_up + result.missed_up)
        + result.true_non_up / (result.true_non_up + result.false_up)
    )
    result["validation_accuracy"] = sum(frame.validation_accuracy for frame in frames) / 3
    rollup = pd.read_csv(folder / "rolling_metrics.csv").set_index("model")
    for model in order:
        assert int(result.loc[model, "test_correct_days"]) == int(rollup.loc[model, "total_correct_days"])
        assert abs(result.loc[model, "test_brier"] - rollup.loc[model, "mean_test_brier"]) < 0.000002
    return result


def plot_direction(groups: dict[str, pd.DataFrame], title: str, output: Path) -> None:
    tickers = tuple(groups)
    models = list(next(iter(groups.values())).index)
    assert all(set(group.index) == set(models) for group in groups.values())
    specs = (
        ("Accuracy ↑", "test_accuracy", "YlGn", 0.35, 0.70, percent),
        ("Balanced accuracy ↑", "test_balanced_accuracy", "YlGn", 0.35, 0.70, percent),
        ("Brier ↓", "test_brier", "RdYlGn_r", 0.20, 0.31, lambda x: f"{x:.3f}"),
        ("Predicted up", "test_predicted_up_rate", "PuBu", 0, 1, percent),
    )
    fig, grid = plt.subplots(2, 2, figsize=(12.8, max(8.0, 0.64 * len(models) + 2.2)))
    axes = grid.ravel()
    fig.patch.set_facecolor("#ffffff")
    for ax, (label, key, cmap, low, high, formatter) in zip(axes, specs):
        matrix = np.array([[groups[ticker].loc[model, key] for ticker in tickers]
                           for model in models], dtype=float)
        image = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=high, aspect="auto")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=14)
        ax.set_xticks(range(len(tickers)), tickers, fontsize=9.5)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(models)), models, fontsize=9.2)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(tickers), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(models)):
            for j in range(len(tickers)):
                red, green, blue, _ = image.cmap(image.norm(matrix[i, j]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(j, i, formatter(matrix[i, j]), ha="center", va="center",
                        fontsize=10.5, color="white" if luminance < 0.46 else "#132238")
    fig.suptitle(title.replace(" | ", "\n", 1), fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.98, 0.008, "Saved test reports · colors compare within each metric",
             ha="right", fontsize=8, color="#475569")
    fig.subplots_adjust(left=0.21, right=0.985,
                        top=0.85 if len(models) <= 9 else 0.925,
                        bottom=0.04, wspace=0.72,
                        hspace=0.38 if len(models) <= 9 else 0.20)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def case_table(frame: pd.DataFrame, folder: Path, *, rolling: bool = False) -> str:
    lines = [
        "| Model | Val. acc | Correct / days | Balanced acc | Brier ↓ | Pred. up | TN / FP / FN / TP | Chart |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model, row in frame.iterrows():
        counts = " / ".join(str(int(row[key])) for key in CONFUSION)
        if rolling:
            links = " ".join(
                f"[F{fold}]({relative(folder / f'fold-{fold}' / 'diagnostics' / 'models' / f'{model}.png')})"
                for fold in (1, 2, 3)
            )
        else:
            links = f"[View]({relative(folder / 'diagnostics' / 'models' / f'{model}.png')})"
        lines.append(
            f"| `{model}` | {percent(row.validation_accuracy)} | "
            f"{int(row.test_correct_days)}/{int(row.test_days)} ({percent(row.test_accuracy)}) | "
            f"{percent(row.test_balanced_accuracy)} | {row.test_brier:.4f} | "
            f"{percent(row.test_predicted_up_rate)} | {counts} | {links} |"
        )
    return "\n".join(lines)


def details(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def direction_section(root: Path, label: str, slug: str) -> str:
    fixed = {ticker: read_case(root / "fixed-2025" / ticker.lower()) for ticker in TICKERS}
    rolling = {ticker: read_rolling(root / "rolling-2025" / ticker.lower()) for ticker in TICKERS}
    order = json.loads((root / "fixed-2025" / "goog" / "summary.json").read_text(encoding="utf-8"))["models"]
    assert all(set(frame.index) == set(order) for frame in (*fixed.values(), *rolling.values()))
    fixed = {ticker: frame.loc[order] for ticker, frame in fixed.items()}
    rolling = {ticker: frame.loc[order] for ticker, frame in rolling.items()}
    fixed_chart = ASSETS / f"{slug}-fixed-metrics.png"
    rolling_chart = ASSETS / f"{slug}-rolling-metrics.png"
    plot_direction(fixed, f"{label} | Fixed 2025 test | 148 days per stock", fixed_chart)
    plot_direction(rolling, f"{label} | Rolling 2025 test | 3 × 100 days per stock", rolling_chart)
    blocks = [f"#### {label}: fixed test", "",
              f"![{label} fixed test metrics]({relative(fixed_chart)})", ""]
    fixed_tables = []
    for ticker, frame in fixed.items():
        folder = root / "fixed-2025" / ticker.lower()
        fixed_tables.extend((
            f"##### {ticker} · actual up {percent(frame.iloc[0].test_actual_up_rate)}",
            "",
            f"Source: [metrics.csv]({relative(folder / 'metrics.csv')}) · "
            f"[predictions.csv]({relative(folder / 'predictions.csv')}) · "
            f"[confusion.csv]({relative(folder / 'confusion.csv')}) · "
            f"[calibration.csv]({relative(folder / 'calibration.csv')})",
            "",
            case_table(frame, folder), "",
        ))
    blocks.extend((details(f"Show every {label} fixed-test model and metric", "\n".join(fixed_tables)),
                   "", f"#### {label}: rolling test", "",
                   f"![{label} rolling test metrics]({relative(rolling_chart)})", ""))
    rolling_tables = []
    fold_tables = []
    for ticker, frame in rolling.items():
        folder = root / "rolling-2025" / ticker.lower()
        rolling_tables.extend((
            f"##### {ticker} · 300 test days · actual up {percent(frame.iloc[0].test_actual_up_rate)}",
            "",
            f"Source: [rolling_metrics.csv]({relative(folder / 'rolling_metrics.csv')}) · "
            f"[fold_metrics.csv]({relative(folder / 'fold_metrics.csv')})",
            "",
            case_table(frame, folder, rolling=True), "",
        ))
        for fold in (1, 2, 3):
            fold_folder = folder / f"fold-{fold}"
            fold_frame = read_case(fold_folder)
            fold_tables.extend((
                f"##### {ticker} · fold {fold} · actual up {percent(fold_frame.iloc[0].test_actual_up_rate)}",
                "",
                f"Source: [metrics.csv]({relative(fold_folder / 'metrics.csv')}) · "
                f"[predictions.csv]({relative(fold_folder / 'predictions.csv')}) · "
                f"[confusion.csv]({relative(fold_folder / 'confusion.csv')}) · "
                f"[calibration.csv]({relative(fold_folder / 'calibration.csv')})",
                "",
                case_table(fold_frame, fold_folder), "",
            ))
    blocks.extend((details(f"Show every {label} rolling aggregate model and metric",
                           "\n".join(rolling_tables)),
                   "", details(f"Show all nine {label} rolling fold tables",
                               "\n".join(fold_tables))))
    return "\n".join(blocks)


def external_section() -> str:
    groups = {ticker: read_case(EXTERNAL / ticker.lower()) for ticker in ("ACN", "RMD")}
    order = json.loads((EXTERNAL / "acn" / "summary.json").read_text(encoding="utf-8"))["models"]
    groups = {ticker: frame.loc[order] for ticker, frame in groups.items()}
    chart = ASSETS / "external-fixed-metrics.png"
    plot_direction(groups, "External symbols | Fixed 2025 test | 148 days per stock", chart)
    blocks = ["#### External stock check", "",
              f"![External stock check metrics]({relative(chart)})", ""]
    tables = []
    for ticker, frame in groups.items():
        folder = EXTERNAL / ticker.lower()
        tables.extend((
            f"##### {ticker} · actual up {percent(frame.iloc[0].test_actual_up_rate)}",
            "",
            f"Source: [metrics.csv]({relative(folder / 'metrics.csv')}) · "
            f"[predictions.csv]({relative(folder / 'predictions.csv')}) · "
            f"[confusion.csv]({relative(folder / 'confusion.csv')}) · "
            f"[calibration.csv]({relative(folder / 'calibration.csv')})",
            "",
            case_table(frame, folder), "",
        ))
    blocks.append(details("Show all external-check models and metrics", "\n".join(tables)))
    return "\n".join(blocks)


def price_section() -> str:
    frames = []
    for ticker in TICKERS:
        folder = ROOT / "runs" / f"{ticker.lower()}-all"
        item = pd.read_csv(folder / "metrics.csv")
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        item["ticker"] = ticker
        item["selected_by_validation"] = item.model == summary["selected_by_validation_mae"]
        frames.append(item.set_index("model", drop=False).loc[summary["settings"]["models"]])
    frame = pd.concat(frames, ignore_index=True)
    models = list(frame[frame.ticker == TICKERS[0]].model)
    assert len(models) == 22 and len(frame) == 66
    matrix = np.array([[frame[(frame.model == model) & (frame.ticker == ticker)].iloc[0].test_mae
                        for ticker in TICKERS] for model in models])
    baseline = [frame[(frame.ticker == ticker) & (frame.model == "naive")].iloc[0].test_mae
                for ticker in TICKERS]
    relative_gain = np.array([[100 * (baseline[j] - matrix[i, j]) / baseline[j]
                               for j in range(3)] for i in range(len(models))])
    fig, ax = plt.subplots(figsize=(8.5, 11.9))
    image = ax.imshow(relative_gain, cmap="RdYlGn", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(3), TICKERS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(models)):
        for j in range(3):
            red, green, blue, _ = image.cmap(image.norm(relative_gain[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(j, i, f"{matrix[i, j]:.3f}\n({relative_gain[i, j]:+.1f}%)",
                    ha="center", va="center", fontsize=7,
                    color="white" if luminance < 0.46 else "#132238")
    fig.suptitle("Archived price task | Test MAE and change vs last-close baseline",
                 fontsize=12, fontweight="bold")
    fig.text(0.99, 0.01, "Lower MAE is better · positive change means lower error",
             ha="right", fontsize=8)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.93, bottom=0.05)
    chart = ASSETS / "archived-price-mae.png"
    fig.savefig(chart, dpi=160, facecolor="white")
    plt.close(fig)
    lines = [
        "| Stock | Model | Selected on validation | Val. MAE | Test MAE | RMSE | MAPE | vs last-close MAE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.ticker} | `{row.model}` | {'Yes' if row.selected_by_validation else 'No'} | "
            f"{row.validation_mae:.3f} | {row.test_mae:.3f} | {row.test_rmse:.3f} | "
            f"{row.test_mape_pct:.2f}% | {row.test_mae_improvement_vs_naive_pct:+.2f}% |"
        )
    return "\n".join((
        "#### Archived next-close price task",
        "",
        f"![Archived price task MAE]({relative(chart)})",
        "",
        "The chart shows all 18 networks and four baselines on the separate next-close price task. "
        "GOOG/AAPL/TSLA last-close baselines are "
        + "/".join(f"{x:.3f}" for x in baseline) + " MAE. "
        "Sources: " + " · ".join(
            f"[{ticker}]({relative(ROOT / 'runs' / (ticker.lower() + '-all') / 'metrics.csv')})"
            for ticker in TICKERS) + ".",
        "",
        details("Show all 66 archived price-model test rows", "\n".join(lines)),
    ))


def ablation_section() -> str:
    source = ROOT / "runs" / "ablation-study" / "gru" / "metrics.csv"
    frame = pd.read_csv(source)
    labels = [(feature, lookback) for feature in ("close-return", "ohlcv") for lookback in (5, 20)]
    matrix = np.array([[frame[(frame.ticker == ticker) & (frame.feature_set == feature)
                               & (frame.lookback == lookback)].test_accuracy.mean()
                        for ticker in TICKERS] for feature, lookback in labels])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.imshow(matrix, cmap="YlGn", vmin=0.4, vmax=0.65, aspect="auto")
    ax.set_xticks(range(3), TICKERS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(4), [f"{feature} / {lookback}d" for feature, lookback in labels])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, percent(matrix[i, j]), ha="center", va="center", fontsize=10)
    fig.suptitle("GRU ablation | Mean test accuracy across 3 folds × 3 seeds",
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.29, right=0.98, top=0.77, bottom=0.08)
    chart = ASSETS / "gru-ablation-accuracy.png"
    fig.savefig(chart, dpi=160, facecolor="white")
    plt.close(fig)
    lines = [
        "| Stock | Fold | Features | Window | Seed | Val. acc | Test correct | Brier ↓ | Pred. up | Baseline correct |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.ticker} | {int(row.fold)} | `{row.feature_set}` | {int(row.lookback)} | "
            f"{int(row.seed)} | {percent(row.validation_accuracy)} | "
            f"{int(row.test_correct_days)}/{int(row.test_days)} ({percent(row.test_accuracy)}) | "
            f"{row.test_brier:.4f} | {percent(row.test_predicted_up_rate)} | "
            f"{int(row.baseline_correct_days)}/{int(row.test_days)} |"
        )
    return "\n".join((
        "#### GRU feature/window/seed ablation",
        "",
        f"![GRU ablation mean accuracy]({relative(chart)})",
        "",
        "Means are descriptive: the folds and seeds reuse historical periods. "
        f"Source: [metrics.csv]({relative(source)}).",
        "",
        details("Show all 108 GRU ablation rows", "\n".join(lines)),
    ))


def simple_heatmap(models: list[str], values: np.ndarray, title: str, output: Path,
                   fmt: str, *, low: float, high: float, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, max(3.8, len(models) * 0.43 + 1.7)))
    image = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
    ax.set_xticks(range(3), TICKERS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(models)):
        for j in range(3):
            red, green, blue, _ = image.cmap(image.norm(values[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if luminance < 0.46 else "#132238")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.32, right=0.98, top=0.88 if len(models) < 10 else 0.93,
                        bottom=0.05)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def archived_auxiliary_section() -> str:
    direction_source = ROOT / "runs" / "neural-direction-comparison.csv"
    direction = pd.read_csv(direction_source)
    models = list(direction[direction.ticker == TICKERS[0]].model)
    assert len(models) == 19 and len(direction) == 57
    matrix = np.array([[direction[(direction.model == model) & (direction.ticker == ticker)]
                        .iloc[0].test_accuracy for ticker in TICKERS] for model in models])
    direction_chart = ASSETS / "archived-price-to-direction.png"
    simple_heatmap(models, 100 * matrix,
                   "Archived price models | Derived direction accuracy (%)",
                   direction_chart, "{:.1f}%", low=35, high=65, cmap="YlGn")
    direction_lines = [
        "| Stock | Model | Val. acc | Correct / days | Balanced acc | Pred. up |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in direction.iterrows():
        direction_lines.append(
            f"| {row.ticker} | `{row.model}` | {percent(row.validation_accuracy)} | "
            f"{int(row.test_correct_days)}/{int(row.test_days)} ({percent(row.test_accuracy)}) | "
            f"{percent(row.test_balanced_accuracy)} | {percent(row.test_predicted_up_rate)} |"
        )

    old_direction_root = ROOT / "runs" / "direction-study" / "rolling"
    old_direction = {ticker: pd.read_csv(old_direction_root / ticker.lower() / "rolling_metrics.csv")
                     .set_index("model") for ticker in TICKERS}
    simple_models = list(old_direction[TICKERS[0]].index)
    matrix = np.array([[old_direction[ticker].loc[model, "overall_test_accuracy"]
                        for ticker in TICKERS] for model in simple_models])
    old_direction_chart = ASSETS / "archived-simple-direction.png"
    simple_heatmap(simple_models, 100 * matrix,
                   "Archived simple direction study | Rolling accuracy (%)",
                   old_direction_chart, "{:.1f}%", low=35, high=65, cmap="YlGn")
    simple_lines = [
        "| Stock | Model | Correct / days | Brier ↓ | Mean balanced acc |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for ticker, frame in old_direction.items():
        for model, row in frame.iterrows():
            simple_lines.append(
                f"| {ticker} | `{model}` | {int(row.total_correct_days)}/{int(row.total_test_days)} "
                f"({percent(row.overall_test_accuracy)}) | {row.mean_test_brier:.4f} | "
                f"{percent(row.mean_test_balanced_accuracy)} |"
            )

    feature_root = ROOT / "runs" / "feature-study" / "rolling"
    feature = {ticker: pd.read_csv(feature_root / ticker.lower() / "rolling_metrics.csv")
               .set_index("model") for ticker in TICKERS}
    feature_models = list(feature[TICKERS[0]].index)
    matrix = np.array([[feature[ticker].loc[model, "mean_test_mae_improvement_vs_naive_pct"]
                        for ticker in TICKERS] for model in feature_models])
    feature_chart = ASSETS / "archived-feature-price.png"
    simple_heatmap(feature_models, matrix,
                   "Archived price feature study | MAE improvement vs last close (%)",
                   feature_chart, "{:+.1f}%", low=-8, high=8, cmap="RdYlGn")
    feature_lines = [
        "| Stock | Model | Mean val. MAE | Mean test MAE | vs last-close | Folds beating baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for ticker, frame in feature.items():
        for model, row in frame.iterrows():
            feature_lines.append(
                f"| {ticker} | `{model}` | {row.mean_validation_mae:.3f} | "
                f"{row.mean_test_mae:.3f} | {row.mean_test_mae_improvement_vs_naive_pct:+.2f}% | "
                f"{int(row.folds_beating_naive)}/3 |"
            )
    return "\n".join((
        "#### Earlier auxiliary experiments",
        "",
        "These are separate historical tasks. The first converts **price forecasts** to up/down decisions, "
        "so it has no forecast probability or Brier score. The other two use earlier feature/model rules; "
        "their scores do not belong in the current 28-network ranking.",
        "",
        f"![Price-derived direction accuracy]({relative(direction_chart)})",
        "",
        f"Source: [neural-direction-comparison.csv]({relative(direction_source)}).",
        "",
        details("Show all 57 price-derived direction rows", "\n".join(direction_lines)),
        "",
        f"![Earlier simple direction study]({relative(old_direction_chart)})",
        "",
        "Sources: " + " · ".join(
            f"[{ticker}]({relative(old_direction_root / ticker.lower() / 'rolling_metrics.csv')})"
            for ticker in TICKERS
        ) + ".",
        "",
        details("Show all 15 earlier direction-study rows", "\n".join(simple_lines)),
        "",
        f"![Earlier price feature study]({relative(feature_chart)})",
        "",
        "Sources: " + " · ".join(
            f"[{ticker}]({relative(feature_root / ticker.lower() / 'rolling_metrics.csv')})"
            for ticker in TICKERS
        ) + ".",
        "",
        details("Show all 15 earlier price-feature rows", "\n".join(feature_lines)),
    ))


def archived_price_rolling_section() -> str:
    frames = {ticker: pd.read_csv(ROOT / "runs" / f"rolling-{ticker.lower()}" / "rolling_metrics.csv")
              .set_index("model") for ticker in TICKERS}
    models = ["naive", "ridge", "gru", "attention-is-all-you-need"]
    gains = np.array([[100 * (1 - frames[ticker].loc[model, "mean_test_mae"]
                              / frames[ticker].loc["naive", "mean_test_mae"])
                       for ticker in TICKERS] for model in models])
    chart = ASSETS / "archived-price-rolling.png"
    simple_heatmap(models, gains,
                   "Archived close-only price study | Rolling MAE improvement (%)",
                   chart, "{:+.2f}%", low=-3, high=3, cmap="RdYlGn")
    lines = [
        "| Stock | Model | Mean val. MAE | Mean test MAE | Fold MAE std. | vs last-close |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    folds = [
        "| Stock | Fold | Model | Val. MAE | Test MAE | RMSE | MAPE | vs last-close MAE |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ticker, frame in frames.items():
        for model, row in frame.iterrows():
            gain = 100 * (1 - row.mean_test_mae / frame.loc["naive", "mean_test_mae"])
            lines.append(f"| {ticker} | `{model}` | {row.mean_validation_mae:.3f} | "
                         f"{row.mean_test_mae:.3f} | {row.std_test_mae:.3f} | {gain:+.2f}% |")
        fold_frame = pd.read_csv(ROOT / "runs" / f"rolling-{ticker.lower()}" / "fold_metrics.csv")
        for _, row in fold_frame.iterrows():
            folds.append(f"| {ticker} | {int(row.fold)} | `{row.model}` | {row.validation_mae:.3f} | "
                         f"{row.test_mae:.3f} | {row.test_rmse:.3f} | {row.test_mape_pct:.2f}% | "
                         f"{row.test_mae_improvement_vs_naive_pct:+.2f}% |")
    return "\n".join((
        "#### Earlier close-only price rolling study",
        "",
        f"![Archived price rolling study]({relative(chart)})",
        "",
        "Positive values mean lower MAE than the last-close baseline. Sources: "
        + " · ".join(f"[{ticker}]({relative(ROOT / 'runs' / ('rolling-' + ticker.lower()) / 'fold_metrics.csv')})"
                     for ticker in TICKERS) + ".",
        "",
        details("Show all 12 rolling price aggregates and 36 fold rows",
                "\n".join(lines) + "\n\n" + "\n".join(folds)),
    ))


def generated_content() -> str:
    ASSETS.mkdir(parents=True, exist_ok=True)
    blocks = [
        "### 全部模型结果与指标图",
        "",
        "下列图表直接由已保存的 CSV 生成，图内使用英文。每张热图的行是模型，列是股票；"
        "色深只帮助比较，**格内数字才是实际结果**。固定测试为每只 148 日，滚动测试为每只三折共 300 日；"
        "滚动汇总的平衡准确率、预测上涨比例及混淆矩阵由三折计数合并计算，Brier 按测试天数加权。"
        "各折以及每个模型的完整数字和诊断图可在下方展开。模型行是事后展示，不能按测试结果重新选模型。",
        "",
        "表内 TN/FP/FN/TP 依次是正确判不涨、误判涨、漏判涨、正确判涨。"
        "`Val. acc` 是验证期准确率；滚动汇总取三折平均。"
        "校准分箱及逐日概率保存在对应的 `calibration.csv`、`predictions.csv`，"
        "单模型图可查看概率校准和混淆矩阵。",
        "",
        direction_section(MAIN, "20 teaching networks + 3 baselines", "core"),
        "",
        direction_section(MODERN, "8 modern adaptations + baseline", "modern"),
        "",
        external_section(),
        "",
        price_section(),
        "",
        ablation_section(),
        "",
        archived_auxiliary_section(),
        "",
        archived_price_rolling_section(),
    ]
    return "\n".join(blocks).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README numbers differ")
    args = parser.parse_args()
    original = README.read_text(encoding="utf-8")
    if BEGIN not in original or END not in original:
        raise SystemExit("README generated section markers are missing")
    before, rest = original.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    updated = before + BEGIN + "\n\n" + generated_content() + "\n\n" + END + after
    if args.check:
        if updated != original:
            raise SystemExit("README result section differs from saved CSVs")
        print("README result section matches saved CSVs")
    else:
        README.write_bytes(updated.encode("utf-8"))
        print(f"Updated {README}; charts in {ASSETS}")


if __name__ == "__main__":
    main()
