"""Analyze training results - writes output to an artifact markdown file."""
import json
import numpy as np
from pathlib import Path

from config import MODELS_DIR

with open(MODELS_DIR / "training_summary.json") as f:
    data = json.load(f)

lines = []
lines.append("# AlphaStock Training Results\n")

for horizon in ["1d", "5d", "20d"]:
    lines.append(f"\n## Horizon: {horizon}\n")

    rows = []
    for ticker, models in data["horizons"].get(horizon, {}).items():
        best_clf = 0
        best_model = ""
        for m in ["lightgbm_clf", "xgboost_clf", "ensemble_clf"]:
            acc = (models.get(m, {}).get("test_accuracy", 0) or 0)
            if acc > best_clf:
                best_clf = acc
                best_model = m

        best_reg_dir = 0
        for m in ["lightgbm", "xgboost", "ensemble_reg"]:
            d = models.get(m, {}).get("test_dir_acc", 0) or 0
            ed = models.get(m, {}).get("test_ensemble_dir_acc", 0) or 0
            best_reg_dir = max(best_reg_dir, d, ed)

        best_auc = 0
        for m in ["lightgbm_clf", "xgboost_clf", "ensemble_clf"]:
            a = (models.get(m, {}).get("test_auc", 0) or 0)
            best_auc = max(best_auc, a)

        rows.append((ticker, best_clf, best_reg_dir, best_auc, best_model))

    rows.sort(key=lambda x: x[1], reverse=True)

    lines.append("| Rank | Ticker | CLF Acc | Reg Dir | AUC | Best Model |")
    lines.append("|------|--------|---------|---------|-----|------------|")
    for i, (t, c, r, a, m) in enumerate(rows):
        lines.append(f"| {i+1} | {t} | {c:.2%} | {r:.2%} | {a:.4f} | {m} |")

    accs = [r[1] for r in rows]
    lines.append(f"\n**Mean: {np.mean(accs):.2%} | Median: {np.median(accs):.2%} | Best: {np.max(accs):.2%} | Worst: {np.min(accs):.2%}**")
    above55 = sum(1 for a in accs if a > 0.55)
    above52 = sum(1 for a in accs if a > 0.52)
    lines.append(f"\nStocks above 55%: **{above55}/{len(accs)}** | Above 52%: **{above52}/{len(accs)}**")

output = "\n".join(lines)

out_path = Path(__file__).parent / "training_results.md"
out_path.write_text(output, encoding="utf-8")
print("Results written to training_results.md")
print(output)
