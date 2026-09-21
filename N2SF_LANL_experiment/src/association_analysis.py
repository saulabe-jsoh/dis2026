"""Association analysis that does not treat categorical label codes as continuous.

For numeric variables, Pearson correlation is reported.
For categorical variables, Cramer's V and mutual information are reported.
For binary label vs categorical feature, Cramer's V is used.
For numeric feature vs binary label, Pearson is used as a descriptive statistic.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

CAT = ["src_user", "dst_user", "src_comp", "dst_comp", "process_name", "dns_query"]
NUM = ["pkt_count", "byte_count"]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x.astype(str), y.astype(str)).to_numpy(dtype=float)
    if tab.size == 0:
        return float("nan")
    n = tab.sum()
    if n == 0:
        return float("nan")
    row_sum = tab.sum(axis=1, keepdims=True)
    col_sum = tab.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    mask = expected > 0
    chi2 = ((tab - expected) ** 2 / np.where(mask, expected, 1)).sum()
    phi2 = chi2 / n
    r, k = tab.shape
    # Bias correction from Bergsma & Wicher style adjustment.
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = max(min(kcorr - 1, rcorr - 1), 1e-12)
    return float(np.sqrt(phi2corr / denom))


def analyze(df: pd.DataFrame, sample_for_cat: int = 100_000, seed: int = 42) -> dict:
    if len(df) > sample_for_cat:
        d = df.sample(sample_for_cat, random_state=seed)
    else:
        d = df.copy()

    numeric = d[NUM + ["label"]].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson").round(6).to_dict()

    cat_vs_label = {}
    mi_vs_label = {}
    for c in CAT:
        x = d[c].astype(str)
        y = d["label"].astype(str)
        cat_vs_label[c] = cramers_v(x, y)
        mi_vs_label[c] = float(mutual_info_score(x, y))

    cat_pair = {}
    for i, c1 in enumerate(CAT):
        for c2 in CAT[i+1:]:
            cat_pair[f"{c1}__{c2}"] = cramers_v(d[c1], d[c2])

    return {
        "n": len(d),
        "pearson_numeric_and_label": pearson,
        "cramers_v_category_vs_label": cat_vs_label,
        "mutual_information_category_vs_label": mi_vs_label,
        "cramers_v_category_pairs": cat_pair,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.input, compression="gzip")
    result = analyze(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
