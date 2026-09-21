"""Create one fixed train/test split used by every synthesis/evaluation stage."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save(input_csv_gz: Path, train_out: Path, test_out: Path, test_size: float, seed: int, mode: str):
    df = pd.read_csv(input_csv_gz, compression="gzip")
    if mode == "stratified":
        train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    elif mode == "chronological":
        d = df.sort_values("time", key=lambda s: pd.to_numeric(s, errors="coerce"))
        cut = int(len(d) * (1 - test_size))
        train, test = d.iloc[:cut].copy(), d.iloc[cut:].copy()
    else:
        raise ValueError("mode must be stratified or chronological")
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_out, index=False, compression="gzip")
    test.to_csv(test_out, index=False, compression="gzip")
    meta = {
        "seed": seed,
        "mode": mode,
        "test_size": test_size,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive": int(train["label"].sum()),
        "train_negative": int((train["label"] == 0).sum()),
        "test_positive": int(test["label"].sum()),
        "test_negative": int((test["label"] == 0).sum()),
    }
    (train_out.parent / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--test-output", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["stratified", "chronological"], default="stratified")
    args = parser.parse_args()
    split_and_save(args.input, args.train_output, args.test_output, args.test_size, args.seed, args.mode)

if __name__ == "__main__":
    main()
