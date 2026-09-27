"""Report next-day up/down accuracy for the saved 18 price-forecast networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from stocklab.data import prepare_data
from stocklab.models import NEURAL_MODELS, build_model
from stocklab.training import _torch_predict


ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("GOOG", "AAPL", "TSLA")


def direction_scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "correct_days": int((actual == predicted).sum()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "predicted_up_rate": float(predicted.mean()),
    }


def main() -> None:
    rows = []
    selected_rows = []
    for ticker in TICKERS:
        report_dir = ROOT / "runs" / f"{ticker.lower()}-all"
        saved_prices = pd.read_csv(report_dir / "predictions.csv")
        data = prepare_data(ROOT / "data" / "raw" / f"{ticker}.csv")
        train_up = data.train.y_raw > data.train.last_close
        validation_up = data.validation.y_raw > data.validation.last_close
        test_up = data.test.y_raw > data.test.last_close
        majority_up = bool(train_up.mean() >= 0.5)
        names = ("train-majority",) + NEURAL_MODELS
        for name in names:
            if name == "train-majority":
                validation_prediction = np.full(len(data.validation), majority_up)
                test_prediction = np.full(len(data.test), majority_up)
            else:
                model = build_model(name, lookback=data.lookback, hidden_size=32)
                weights = torch.load(report_dir / "models" / f"{name}.pt",
                                     map_location="cpu", weights_only=True)
                model.load_state_dict(weights)
                validation_price = data.inverse(_torch_predict(model, data.validation.x_scaled,
                                                                torch.device("cpu"), 64))
                test_price = data.inverse(_torch_predict(model, data.test.x_scaled,
                                                          torch.device("cpu"), 64))
                if not np.allclose(test_price, saved_prices[name].to_numpy(), atol=1e-3):
                    raise ValueError(f"{ticker} {name}: reloaded weights disagree with saved forecast")
                validation_prediction = validation_price > data.validation.last_close
                test_prediction = test_price > data.test.last_close
            validation = direction_scores(validation_up, validation_prediction)
            test = direction_scores(test_up, test_prediction)
            rows.append({"ticker": ticker, "model": name,
                         "train_up_rate": float(train_up.mean()),
                         "validation_up_rate": float(validation_up.mean()),
                         "test_up_rate": float(test_up.mean()),
                         "validation_accuracy": validation["accuracy"],
                         "test_correct_days": test["correct_days"],
                         "test_days": len(data.test),
                         "test_accuracy": test["accuracy"],
                         "test_balanced_accuracy": test["balanced_accuracy"],
                         "test_predicted_up_rate": test["predicted_up_rate"]})
        ticker_rows = [row for row in rows if row["ticker"] == ticker]
        selected = max(ticker_rows, key=lambda row: row["validation_accuracy"])
        majority = ticker_rows[0]
        selected_rows.append({"ticker": ticker, "selected_by_validation_accuracy": selected["model"],
                              "selected_test_correct_days": selected["test_correct_days"],
                              "selected_test_accuracy": selected["test_accuracy"],
                              "majority_test_accuracy": majority["test_accuracy"]})

    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "runs" / "neural-direction-comparison.csv", index=False,
                  float_format="%.6f")
    chosen = pd.DataFrame(selected_rows)
    chosen.to_csv(ROOT / "runs" / "neural-direction-selected.csv", index=False,
                   float_format="%.6f")
    lines = [
        "# 18 类神经网络的次日涨跌准确率（2025 年）", "",
        "定义：次日收盘价高于今日收盘价记为上涨；持平归入未上涨。",
        "预测价格高于今日实际收盘价就判断上涨，否则判断未上涨。",
        "准确率 = 判断正确的交易日数 ÷ 测试交易日数。每只股票测试期 148 天，",
        "均为 2025-06-02 至 2025-12-31。", "",
        "这些神经网络原本按**价格误差**训练，输出价格而不是经过校准的上涨概率。",
        "下表只是把保存的价格预测换算成涨跌判断，不应解释为概率预测。",
        "基线在训练期确定较常见的方向，之后每天都预测这个方向。",
        "星号 ★ 表示按验证期涨跌准确率选中的模型；验证准确率相同时优先保留表中靠前的模型。", "",
        "| 模型 | GOOG 测试准确率 | AAPL 测试准确率 | TSLA 测试准确率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("train-majority",) + NEURAL_MODELS:
        parts = []
        for ticker in TICKERS:
            row = result[(result["ticker"] == ticker) & (result["model"] == name)].iloc[0]
            marker = " ★" if chosen.loc[chosen["ticker"] == ticker,
                                        "selected_by_validation_accuracy"].iloc[0] == name else ""
            parts.append(f"{int(row['test_correct_days'])}/148 = {row['test_accuracy']:.1%}{marker}")
        lines.append(f"| {name} | {' | '.join(parts)} |")
    lines += ["", "按验证准确率选中的模型在 2025 年测试期的表现见",
              "`runs/neural-direction-selected.csv`。所有模型的验证准确率、",
              "测试平衡准确率与预测上涨比例见 `runs/neural-direction-comparison.csv`。",
              "该 2025 年测试时期已被多次查看，这是一份回顾性学习报告。", ""]
    (ROOT / "NEURAL_DIRECTION.md").write_text("\n".join(lines), encoding="utf-8")
    print(chosen.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
