"""Publication-grade TRTR/TSTR evaluation with fixed real holdout data.

The train/test split is created once by split_data.py and every synthesizer
receives REAL TRAIN ONLY. All models are evaluated on the identical REAL TEST.
Categorical features are one-hot encoded; no ordinal label coding is used as a
numeric feature representation.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

CAT = ["src_user", "dst_user", "src_comp", "dst_comp", "process_name", "dns_query"]
NUM = ["pkt_count", "byte_count"]
FEATURES = CAT + NUM


def make_pipeline(pos_weight: float, seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), NUM),
    ])
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("prep", pre), ("model", clf)])


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, name: str, seed: int) -> dict:
    X_train, y_train = train_df[FEATURES], train_df["label"].astype(int)
    X_test, y_test = test_df[FEATURES], test_df["label"].astype(int)
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError(f"Both classes must be present in training data for {name}")
    model = make_pipeline(neg / pos, seed)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "name": name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_positive": pos,
        "train_negative": neg,
        "test_positive": int(y_test.sum()),
        "test_negative": int((y_test == 0).sum()),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) == 2 else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "class_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }


def evaluate_all(real_train_csv_gz: Path, real_test_csv_gz: Path,
                 synthetic_ctgan_csv_gz: Path | None, synthetic_llama_csv_gz: Path | None,
                 output_json: Path, seed: int) -> dict:
    train = pd.read_csv(real_train_csv_gz, compression="gzip")
    test = pd.read_csv(real_test_csv_gz, compression="gzip")
    results = {"TRTR": evaluate_model(train, test, "TRTR", seed)}
    if synthetic_ctgan_csv_gz:
        ctgan = pd.read_csv(synthetic_ctgan_csv_gz, compression="gzip")
        results["TSTR_CTGAN"] = evaluate_model(ctgan, test, "TSTR_CTGAN", seed)
    if synthetic_llama_csv_gz:
        llama = pd.read_csv(synthetic_llama_csv_gz, compression="gzip")
        results["TSTR_Llama3"] = evaluate_model(llama, test, "TSTR_Llama3", seed)

    baseline = results["TRTR"]["f1"]
    for row in results.values():
        row["f1_retention_percent"] = float(row["f1"] / baseline * 100) if baseline else None

    payload = {
        "seed": seed,
        "train_total": len(train),
        "test_total": len(test),
        "train_positive": int(train["label"].sum()),
        "train_negative": int((train["label"] == 0).sum()),
        "test_positive": int(test["label"].sum()),
        "test_negative": int((test["label"] == 0).sum()),
        "results": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-train", required=True, type=Path)
    parser.add_argument("--real-test", required=True, type=Path)
    parser.add_argument("--ctgan", type=Path, default=None)
    parser.add_argument("--llama", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(evaluate_all(args.real_train, args.real_test, args.ctgan, args.llama, args.output, args.seed), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
