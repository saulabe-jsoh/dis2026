"""Real class-conditional SDV-CTGAN synthesis; no noise-based fallback."""
from __future__ import annotations
from pathlib import Path
import argparse
import json

import pandas as pd

CATEGORICAL = ["src_user", "dst_user", "src_comp", "dst_comp", "process_name", "dns_query"]
NUMERIC = ["pkt_count", "byte_count"]
FEATURES = CATEGORICAL + NUMERIC


def _fit_ctgan(df: pd.DataFrame, epochs: int, seed: int):
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer
    metadata = SingleTableMetadata()
    model_df = df[FEATURES].copy()
    for c in CATEGORICAL:
        model_df[c] = model_df[c].astype(str)
    for c in NUMERIC:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce").fillna(0.0)
    metadata.detect_from_dataframe(data=model_df)
    synth = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)
    synth.set_random_state(seed)
    synth.fit(model_df)
    return synth


def train_and_sample(train_csv_gz: Path, output_csv_gz: Path, epochs: int, seed: int,
                     positive_size: int | None = None, negative_size: int | None = None) -> dict:
    try:
        import sdv  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install SDV before running real CTGAN synthesis") from exc

    train = pd.read_csv(train_csv_gz, compression="gzip")
    if positive_size is None:
        positive_size = int(train["label"].sum())
    if negative_size is None:
        negative_size = int((train["label"] == 0).sum())

    pos = train[train["label"].astype(int) == 1].copy()
    neg = train[train["label"].astype(int) == 0].copy()
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Both classes must exist in the real training set")

    # Class-conditional synthesis removes uncertainty about synthetic class prevalence.
    pos_synth = _fit_ctgan(pos, epochs, seed + 1).sample(num_rows=positive_size)
    neg_synth = _fit_ctgan(neg, epochs, seed + 2).sample(num_rows=negative_size)
    pos_synth["label"] = 1
    neg_synth["label"] = 0
    synthetic = pd.concat([pos_synth, neg_synth], ignore_index=True)
    synthetic = synthetic[FEATURES + ["label"]]
    for c in CATEGORICAL:
        synthetic[c] = synthetic[c].astype(str)
    for c in NUMERIC:
        synthetic[c] = pd.to_numeric(synthetic[c], errors="coerce").fillna(0.0)

    output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output_csv_gz, index=False, compression="gzip")
    meta = {
        "train_rows_used": len(train),
        "output_rows": len(synthetic),
        "output_positive": int(synthetic["label"].sum()),
        "output_negative": int((synthetic["label"] == 0).sum()),
        "epochs_per_class": epochs,
        "seed": seed,
        "class_conditional": True,
    }
    output_csv_gz.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-size", type=int, default=None)
    parser.add_argument("--negative-size", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(train_and_sample(args.input_train, args.output, args.epochs, args.seed,
                                      args.positive_size, args.negative_size), indent=2))

if __name__ == "__main__":
    main()
