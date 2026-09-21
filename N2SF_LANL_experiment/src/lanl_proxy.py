"""LANL Cyber1 -> reproducible 100k proxy dataset builder.

Design choices for the paper revision:
- auth events are the anchor rows because redteam events are defined as known
  compromise events taken from the authentication stream.
- labels are assigned by exact event identity:
    (time, user@domain, source computer, destination computer)
- 95k negatives + 5k positives are sampled from the full 58-day corpus to make
  the severe 19:1 class imbalance explicit as an experimental design choice.
- proc/dns/flows are attached by deterministic exact-second keys with bounded
  one-to-many aggregation; this avoids accidental many-to-many row explosion.
- NO synthetic/random fallback is used. Missing source files raise an error.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import gzip
import logging
import math
import random

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

AUTH_COLS = [
    "time", "src_user", "dst_user", "src_comp", "dst_comp",
    "auth_type", "logon_type", "auth_orientation", "status"
]
PROC_COLS = ["time", "src_user", "src_comp", "process_name", "action"]
DNS_COLS = ["time", "src_comp", "dns_query"]
FLOW_COLS = [
    "time", "duration", "src_comp", "src_port", "dst_comp",
    "dst_port", "protocol", "pkt_count", "byte_count"
]
RED_COLS = ["time", "user", "src_comp", "dst_comp"]

FINAL_CATEGORICAL = [
    "src_user", "dst_user", "src_comp", "dst_comp", "process_name", "dns_query"
]
FINAL_NUMERIC = ["pkt_count", "byte_count"]
FINAL_FEATURES = FINAL_CATEGORICAL + FINAL_NUMERIC

@dataclass(frozen=True)
class BuildConfig:
    data_dir: Path
    output_csv_gz: Path
    sample_size: int = 100_000
    positive_size: int = 5_000
    negative_size: int = 95_000
    seed: int = 42
    chunk_size: int = 250_000


def _check_config(cfg: BuildConfig) -> None:
    if cfg.positive_size + cfg.negative_size != cfg.sample_size:
        raise ValueError("positive_size + negative_size must equal sample_size")
    for name in ["auth.txt.gz", "proc.txt.gz", "flows.txt.gz", "dns.txt.gz", "redteam.txt.gz"]:
        p = cfg.data_dir / name
        if not p.exists():
            raise FileNotFoundError(f"LANL source file missing: {p}")


def read_gzip_chunks(path: Path, names: list[str], usecols: list[int], chunk_size: int) -> Iterable[pd.DataFrame]:
    """Read a gzipped, headerless comma-delimited LANL source incrementally."""
    return pd.read_csv(
        path,
        compression="gzip",
        header=None,
        names=names,
        usecols=usecols,
        chunksize=chunk_size,
        dtype="string",
        na_filter=False,
        low_memory=False,
        on_bad_lines="skip",
    )


def load_redteam_keys(path: Path) -> set[tuple[str, str, str, str]]:
    """Load the small redteam file into an exact event-key set."""
    df = pd.read_csv(
        path,
        compression="gzip",
        header=None,
        names=RED_COLS,
        dtype="string",
        na_filter=False,
        on_bad_lines="skip",
    )
    df = df.drop_duplicates()
    return set(zip(df["time"], df["user"], df["src_comp"], df["dst_comp"]))


def _reservoir_update(reservoir: list[tuple], seen: int, item: tuple, capacity: int, rng: random.Random) -> int:
    """Reservoir-sampling update; returns incremented number of seen items."""
    seen += 1
    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        j = rng.randrange(seen)
        if j < capacity:
            reservoir[j] = item
    return seen


def sample_auth_anchors(cfg: BuildConfig, red_keys: set[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Stream auth.txt.gz and create a controlled 95/5 labeled anchor sample.

    Positive rows must exactly match a redteam ground-truth event.
    Negatives are sampled by reservoir sampling from non-redteam auth events.
    """
    rng = random.Random(cfg.seed)
    positives: list[tuple] = []
    negative_reservoir: list[tuple] = []
    negative_seen = 0

    auth_path = cfg.data_dir / "auth.txt.gz"
    columns = ["time", "src_user", "dst_user", "src_comp", "dst_comp"]

    for chunk in read_gzip_chunks(auth_path, AUTH_COLS, [0, 1, 2, 3, 4], cfg.chunk_size):
        for row in chunk.itertuples(index=False, name=None):
            key = (row[0], row[1], row[3], row[4])
            if key in red_keys:
                positives.append(row)
            else:
                negative_seen = _reservoir_update(
                    negative_reservoir, negative_seen, row, cfg.negative_size, rng
                )

    # Remove duplicate positive anchors before sampling.
    pos_df = pd.DataFrame(positives, columns=columns).drop_duplicates()
    if len(pos_df) < cfg.positive_size:
        raise RuntimeError(
            f"Only {len(pos_df):,} unique positive auth events matched redteam; "
            f"cannot create requested {cfg.positive_size:,} positives. "
            "Lower positive_size or build the dataset from a different sampling frame."
        )
    if len(negative_reservoir) < cfg.negative_size:
        raise RuntimeError(
            f"Only {len(negative_reservoir):,} negative auth events retained; "
            f"cannot create requested {cfg.negative_size:,} negatives."
        )

    pos = pos_df.sample(n=cfg.positive_size, random_state=cfg.seed)
    neg = pd.DataFrame(negative_reservoir, columns=columns).sample(
        n=cfg.negative_size, random_state=cfg.seed
    )
    pos["label"] = 1
    neg["label"] = 0
    anchors = pd.concat([pos, neg], ignore_index=True)
    # Shuffle before downstream processing, but preserve original time column.
    anchors = anchors.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    LOG.info(
        "Anchor sample built: total=%d, positive=%d, negative=%d, positive_rate=%.4f",
        len(anchors), int(anchors["label"].sum()), int((anchors["label"] == 0).sum()),
        anchors["label"].mean(),
    )
    return anchors


def _anchor_key_set(anchors: pd.DataFrame, cols: list[str]) -> set[tuple]:
    return set(map(tuple, anchors[cols].astype(str).itertuples(index=False, name=None)))


def attach_process(anchors: pd.DataFrame, path: Path, chunk_size: int) -> pd.DataFrame:
    """Attach one deterministic process name per exact (time, src_user, src_comp)."""
    keys = _anchor_key_set(anchors, ["time", "src_user", "src_comp"])
    values: dict[tuple, Counter] = defaultdict(Counter)
    for chunk in read_gzip_chunks(path, PROC_COLS, [0, 1, 2, 3, 4], chunk_size):
        for time_, user, comp, proc, _action in chunk.itertuples(index=False, name=None):
            key = (str(time_), str(user), str(comp))
            if key in keys and proc not in (None, "", "?"):
                values[key][str(proc)] += 1

    out = anchors.copy()
    out["process_name"] = [
        values.get((str(t), str(u), str(c)), Counter()).most_common(1)[0][0]
        if values.get((str(t), str(u), str(c))) else "unknown_proc"
        for t, u, c in out[["time", "src_user", "src_comp"]].itertuples(index=False, name=None)
    ]
    return out


def attach_dns(anchors: pd.DataFrame, path: Path, chunk_size: int) -> pd.DataFrame:
    """Attach one deterministic DNS query per exact (time, src_comp)."""
    keys = _anchor_key_set(anchors, ["time", "src_comp"])
    values: dict[tuple, Counter] = defaultdict(Counter)
    for chunk in read_gzip_chunks(path, DNS_COLS, [0, 1, 2], chunk_size):
        for time_, comp, qry in chunk.itertuples(index=False, name=None):
            key = (str(time_), str(comp))
            if key in keys and qry not in (None, "", "?"):
                values[key][str(qry)] += 1

    out = anchors.copy()
    out["dns_query"] = [
        values.get((str(t), str(c)), Counter()).most_common(1)[0][0]
        if values.get((str(t), str(c))) else "none"
        for t, c in out[["time", "src_comp"]].itertuples(index=False, name=None)
    ]
    return out


def attach_flows(anchors: pd.DataFrame, path: Path, chunk_size: int) -> pd.DataFrame:
    """Aggregate flow packet/byte counts per exact (time, src_comp) key."""
    keys = _anchor_key_set(anchors, ["time", "src_comp"])
    sums: dict[tuple, tuple[int, int]] = {}
    for chunk in read_gzip_chunks(path, FLOW_COLS, [0, 2, 7, 8], chunk_size):
        chunk["pkt_count"] = pd.to_numeric(chunk["pkt_count"], errors="coerce").fillna(0)
        chunk["byte_count"] = pd.to_numeric(chunk["byte_count"], errors="coerce").fillna(0)
        for time_, comp, pkts, bytes_ in chunk.itertuples(index=False, name=None):
            key = (str(time_), str(comp))
            if key not in keys:
                continue
            prev = sums.get(key, (0, 0))
            sums[key] = (prev[0] + int(pkts), prev[1] + int(bytes_))

    out = anchors.copy()
    out["pkt_count"] = [
        sums.get((str(t), str(c)), (0, 0))[0]
        for t, c in out[["time", "src_comp"]].itertuples(index=False, name=None)
    ]
    out["byte_count"] = [
        sums.get((str(t), str(c)), (0, 0))[1]
        for t, c in out[["time", "src_comp"]].itertuples(index=False, name=None)
    ]
    return out


def build_proxy_dataset(cfg: BuildConfig) -> pd.DataFrame:
    _check_config(cfg)
    red_keys = load_redteam_keys(cfg.data_dir / "redteam.txt.gz")
    LOG.info("Loaded %d unique redteam ground-truth event keys", len(red_keys))

    df = sample_auth_anchors(cfg, red_keys)
    df = attach_process(df, cfg.data_dir / "proc.txt.gz", cfg.chunk_size)
    df = attach_dns(df, cfg.data_dir / "dns.txt.gz", cfg.chunk_size)
    df = attach_flows(df, cfg.data_dir / "flows.txt.gz", cfg.chunk_size)

    # Explicit string handling, preserving LANL '?' markers as a category.
    for col in FINAL_CATEGORICAL:
        df[col] = df[col].astype("string").fillna("?")
    for col in FINAL_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    df["label"] = df["label"].astype(int)

    # Exact publication-ready field order.
    df = df[["time"] + FINAL_CATEGORICAL + FINAL_NUMERIC + ["label"]]

    if len(df) != cfg.sample_size:
        raise AssertionError(f"Unexpected dataset size: {len(df)} != {cfg.sample_size}")
    if df["label"].sum() != cfg.positive_size:
        raise AssertionError("Positive label count does not match configuration")
    if (df["label"] == 0).sum() != cfg.negative_size:
        raise AssertionError("Negative label count does not match configuration")

    cfg.output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv_gz, index=False, compression="gzip")
    LOG.info("Saved proxy dataset: %s", cfg.output_csv_gz)
    return df


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build reproducible LANL 100k proxy dataset")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--positive-size", type=int, default=5_000)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = BuildConfig(
        data_dir=args.data_dir,
        output_csv_gz=args.output,
        sample_size=args.sample_size,
        positive_size=args.positive_size,
        negative_size=args.sample_size - args.positive_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    build_proxy_dataset(cfg)


if __name__ == "__main__":
    main()
