"""Turn experiment JSON into publication-ready Markdown tables."""
from __future__ import annotations
from pathlib import Path
import argparse
import json


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def build(metrics_path: Path, deid_path: Path | None = None) -> str:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    lines = []
    lines.append("## TRTR/TSTR metrics\n")
    lines.append("| Model | F1 | Precision | Recall | FPR | FNR | PR-AUC | ROC-AUC | F1 retention |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, row in metrics["results"].items():
        lines.append(
            f"| {key} | {fmt(row.get('f1'))} | {fmt(row.get('precision'))} | "
            f"{fmt(row.get('recall'))} | {fmt(row.get('fpr'))} | {fmt(row.get('fnr'))} | "
            f"{fmt(row.get('pr_auc'))} | {fmt(row.get('roc_auc'))} | {fmt(row.get('f1_retention_percent'))}% |"
        )

    lines.append("\n## Confusion matrices\n")
    lines.append("| Model | TN | FP | FN | TP | Test normal | Test threat |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, row in metrics["results"].items():
        lines.append(
            f"| {key} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} | "
            f"{row['test_negative']} | {row['test_positive']} |"
        )

    if deid_path:
        deid = json.loads(deid_path.read_text(encoding="utf-8"))
        lines.append("\n## 195-target de-identification benchmark\n")
        lines.append("| Type | Ground truth | Regex detected | Hybrid detected | Regex recall | Hybrid recall |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for t, row in deid["by_type"].items():
            gt = row["ground_truth"]
            rr = row["regex_detected"] / gt if gt else 0
            hr = row["hybrid_detected"] / gt if gt else 0
            lines.append(f"| {t} | {gt} | {row['regex_detected']} | {row['hybrid_detected']} | {rr:.4f} | {hr:.4f} |")
        gt = deid["ground_truth"]
        lines.append(f"| **Total** | **{gt}** | **{deid['regex_detected']}** | **{deid['hybrid_detected']}** | **{deid['regex_recall']:.4f}** | **{deid['hybrid_recall']:.4f}** |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--deid", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = build(args.metrics, args.deid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()
